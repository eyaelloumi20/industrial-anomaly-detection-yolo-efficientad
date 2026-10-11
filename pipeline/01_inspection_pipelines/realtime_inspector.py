#!/usr/bin/env python3
"""Real-time YOLO + EfficientAD inspection with a LAN MJPEG dashboard."""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import collections
import io
import json
import socket
import sys
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Reuse the project's crop/scoring code unchanged.  Both public spellings are
# supported because the supplied helper exports score_crop_batch, while some
# repository revisions export _score_crop_batch.
try:
    from compare_pipeline_methods_batched import _score_crop_batch, get_crop
except ImportError:
    from compare_pipeline_methods_batched import score_crop_batch as _score_crop_batch, get_crop

# Reuse the supplied loader unchanged.  The attached file calls it
# load_efficientad_model; newer repository revisions may call it load_bundle.
try:
    from utils.load_efficientad import load_bundle
except ImportError:
    from utils.load_efficientad import load_efficientad_model as _load_efficientad_model

    def load_bundle(method, ckpt_dir, category, model_size, device):
        if method not in ("baseline", "lowres"):
            raise ValueError("method must be baseline or lowres")
        models = _load_efficientad_model(ckpt_dir, category, model_size, device)
        image_size = int(models.get("image_size", 256 if method == "baseline" else 128))
        return models, image_size, device

import utils.models as _models_module

RED = (0, 0, 255)
GREEN = (0, 200, 0)
GREY = (128, 128, 128)


def _dtype_preserving_norm(x):
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype,
                        device=x.device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype,
                       device=x.device)[None, :, None, None]
    return (x - mean) / (std + 1e-11)


def patch_model_forward_dtype_safe(model, kind):
    if kind in ("teacher", "student"):
        def forward(self, x):
            return self.pdn(_dtype_preserving_norm(x))
    elif kind == "ae":
        def forward(self, x):
            return self.decoder(self.encoder(_dtype_preserving_norm(x)))
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    model.forward = types.MethodType(forward, model)
    return model


_models_module.imagenet_norm_batch = _dtype_preserving_norm


def parse_camera(value):
    try:
        return int(value)
    except ValueError:
        return value


def is_video_file(source):
    if not isinstance(source, str) or source.lower().startswith(("rtsp://", "http://", "https://")):
        return False
    return Path(source).is_file()


class LatestFrameCamera:
    """Background reader holding exactly one frame; overwritten frames drop."""
    def __init__(self, source, width=None, height=None, loop_video=False):
        self.source = source
        self.loop_video = loop_video and is_video_file(source)
        self.cap = cv2.VideoCapture(source)
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera source: {source}")
        self.file_source = is_video_file(source)
        fps = float(self.cap.get(cv2.CAP_PROP_FPS)) if self.file_source else 0.0
        self.file_period = 1.0 / fps if fps > 0.0 else 0.0
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.frame = None
        self.sequence = 0
        self.stopped = False
        self.eof = False
        self.thread = threading.Thread(target=self._read_loop, name="camera-reader", daemon=True)

    def start(self):
        self.thread.start()

    def _read_loop(self):
        next_file_frame = time.perf_counter()
        while not self.stopped:
            if self.file_period:
                delay = next_file_frame - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            ok, frame = self.cap.read()
            if not ok:
                if self.loop_video:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    next_file_frame = time.perf_counter()
                    continue
                self.eof = True
                with self.cond:
                    self.cond.notify_all()
                break
            with self.cond:
                self.frame = frame
                self.sequence += 1
                self.cond.notify_all()
            if self.file_period:
                next_file_frame += self.file_period

    def latest_after(self, previous_sequence, timeout=1.0):
        with self.cond:
            self.cond.wait_for(lambda: self.sequence != previous_sequence or self.stopped or self.eof,
                               timeout=timeout)
            if self.sequence == previous_sequence:
                return None, previous_sequence
            return self.frame.copy(), self.sequence

    def stop(self):
        self.stopped = True
        with self.cond:
            self.cond.notify_all()
        self.cap.release()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)


class SharedState:
    def __init__(self, fire_window_seconds):
        self.lock = threading.Lock()
        self.jpeg = None
        self.frame_version = 0
        self.fps = 0.0
        self.n_frames = 0
        self.n_visible = 0
        self.frame_alarm = False
        self.fire_events = collections.deque()
        self.fire_window = fire_window_seconds

    def prune_locked(self, now):
        while self.fire_events and now - self.fire_events[0][0] > self.fire_window:
            self.fire_events.popleft()

    def publish(self, frame, fps, n_frames, n_visible, frame_alarm):
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return
        with self.lock:
            self.jpeg = encoded.tobytes()
            self.frame_version += 1
            self.fps, self.n_frames, self.n_visible = fps, n_frames, n_visible
            self.frame_alarm = bool(frame_alarm)

    def add_fire(self, timestamp, track_id, score):
        with self.lock:
            self.fire_events.append((timestamp, int(track_id), float(score)))

    def snapshot(self):
        with self.lock:
            self.prune_locked(time.monotonic())
            latest = {}
            for ts, tid, score in self.fire_events:
                latest[tid] = (ts, score)
            return {
                "fps": round(self.fps, 2), "n_frames": self.n_frames,
                "n_visible": self.n_visible,
                "recent_fires": [{"track_id": tid, "score": round(v[1], 4)}
                                 for tid, v in sorted(latest.items())],
                "alarm": bool(self.frame_alarm or latest),
            }

    def recent_map(self):
        with self.lock:
            self.prune_locked(time.monotonic())
            return {tid: score for _, tid, score in self.fire_events}


DASHBOARD = r'''<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Real-time Inspector</title><style>
*{box-sizing:border-box}body{margin:0;background:#0e0e0e;color:#eee;font-family:monospace}
#banner{padding:14px 190px 14px 18px;background:#087a32;font-size:20px;font-weight:bold;min-height:52px}
#banner.alarm{background:#b00020}.qr{position:absolute;right:14px;top:6px;width:150px;height:150px;background:#fff;padding:5px;border-radius:4px}
.qr.off{display:none}.main{height:calc(100vh - 122px);display:flex;align-items:center;justify-content:center;padding:10px}
.main img{max-width:100%;max-height:100%;object-fit:contain}.stats{height:70px;display:grid;grid-template-columns:repeat(5,1fr);gap:1px;background:#333}
.cell{background:#181818;text-align:center;padding:9px}.label{color:#999;font-size:12px}.value{font-size:20px;margin-top:4px}.ok{color:#35e66b}.bad{color:#ff405d}
@media(max-width:700px){#banner{padding-right:12px;font-size:15px}.qr{display:none}.stats{grid-template-columns:repeat(5,1fr)}.value{font-size:14px}.cell{padding:8px 2px}}
</style></head><body><div id="banner">MONITORING — no defects in last 5s</div>
__QR__<div class="main"><img src="/stream" alt="live inspection stream"></div>
<div class="stats"><div class="cell"><div class="label">FPS</div><div class="value" id="fps">0.0</div></div>
<div class="cell"><div class="label">Frames</div><div class="value" id="frames">0</div></div>
<div class="cell"><div class="label">Visible</div><div class="value" id="visible">0</div></div>
<div class="cell"><div class="label">Recent fires (5s)</div><div class="value" id="fires">0</div></div>
<div class="cell"><div class="label">Status</div><div class="value ok" id="status">OK</div></div></div>
<script>async function update(){try{let r=await fetch('/stats',{cache:'no-store'}),s=await r.json();
fps.textContent=s.fps.toFixed(1);frames.textContent=s.n_frames;visible.textContent=s.n_visible;
let ids=s.recent_fires.map(x=>x.track_id);fires.textContent=ids.length;let a=ids.length>0;
banner.textContent=a?'DEFECT DETECTED — ID '+ids.join(', '):'MONITORING — no defects in last 5s';
banner.className=a?'alarm':'';status.textContent=s.alarm?'ALARM':'OK';status.className='value '+(s.alarm?'bad':'ok');}catch(e){}}
setInterval(update,500);update();</script></body></html>'''


class DashboardServer:
    def __init__(self, host, port, state, qr_png):
        self.state, self.qr_png = state, qr_png
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                return

            def send_bytes(self, code, content_type, payload):
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                path = urlparse(self.path).path
                if path == "/":
                    qr = '<img class="qr" src="/qr" alt="phone access QR">' if outer.qr_png else ''
                    self.send_bytes(200, "text/html; charset=utf-8",
                                    DASHBOARD.replace("__QR__", qr).encode())
                elif path == "/stats":
                    self.send_bytes(200, "application/json",
                                    json.dumps(outer.state.snapshot()).encode())
                elif path == "/qr" and outer.qr_png:
                    self.send_bytes(200, "image/png", outer.qr_png)
                elif path == "/stream":
                    self.send_response(200)
                    self.send_header("Cache-Control", "no-cache, private")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    version = -1
                    try:
                        while not outer.stopping.is_set():
                            with outer.state.lock:
                                jpeg, current = outer.state.jpeg, outer.state.frame_version
                            if jpeg is None or current == version:
                                time.sleep(1.0 / 30.0)
                                continue
                            version = current
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n" +
                                             f"Content-Length: {len(jpeg)}\r\n\r\n".encode() + jpeg + b"\r\n")
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                        pass
                else:
                    self.send_error(404)

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, name="http-server", daemon=True)
        self.stopping = threading.Event()

    def start(self):
        self.thread.start()

    def stop(self):
        self.stopping.set()
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2.0)


def detect_lan_ip():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError as exc:
        print(f"[QR] Warning: LAN IP detection failed ({exc}); using 127.0.0.1")
        return "127.0.0.1"
    finally:
        sock.close()


def make_qr(url, disabled):
    if disabled:
        return None
    try:
        import qrcode
    except ImportError:
        print("[QR] Install qrcode[pil] to enable QR codes.")
        return None
    qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L,
                       box_size=1, border=1)
    qr.add_data(url)
    try:
        qr.make(fit=False)
    except qrcode.exceptions.DataOverflowError:
        qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_L,
                           box_size=1, border=1)
        qr.add_data(url); qr.make(fit=True)
    print()
    qr.print_ascii(out=sys.stdout, invert=True)
    image = qr.make_image(fill_color="black", back_color="white")
    buffer = io.BytesIO(); image.save(buffer, format="PNG")
    png = buffer.getvalue()
    Path.cwd().joinpath("live_qr.png").write_bytes(png)
    return png


def classify_clipped(dets, width, height, margin):
    if not dets:
        return []
    areas = np.asarray([(x2-x1)*(y2-y1) for _, x1, y1, x2, y2 in dets], dtype=float)
    median = float(np.median(areas))
    return [x1 <= margin or y1 <= margin or x2 >= width-margin or y2 >= height-margin
            or area < 0.5*median for (_, x1, y1, x2, y2), area in zip(dets, areas)]


def call_score_batch(models, crops, device, image_size, ratio, fp16):
    # Supplied helper revisions differ only in whether AMP is named use_amp or
    # a model_dtype argument is used. The helper itself remains untouched.
    try:
        return _score_crop_batch(models, crops, device, image_size,
                                 ratio=ratio, use_amp=fp16)
    except TypeError as exc:
        if "use_amp" not in str(exc):
            raise
        dtype = torch.float16 if fp16 else torch.float32
        return _score_crop_batch(models, crops, device, image_size,
                                 model_dtype=dtype, ratio=ratio)


def draw_label(frame, text, x, y, color, thickness):
    y = max(20, y)
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x, y-th-base-5), (x+tw+6, y), color, -1)
    cv2.putText(frame, text, (x+3, y-base-2), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2, cv2.LINE_AA)


def load_thresholds(path, fallback):
    if not path:
        return fallback, fallback
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tau_p = float(data.get("tau_p", fallback))
    tau_f = float(data.get("tau_f", tau_p))
    print(f"[Run] Calibration display: tau_p={tau_p:.4f}, tau_f={tau_f:.4f}")
    return tau_p, tau_f


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camera", required=True, type=parse_camera)
    p.add_argument("--yolo_weights", required=True)
    p.add_argument("--method", choices=("baseline", "lowres"), default="baseline")
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--thresholds_json")
    p.add_argument("--category", default="chewinggum")
    p.add_argument("--model_size", choices=("S", "M"), default="S")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--tau_track", type=float, default=0.45)
    p.add_argument("--inactive_frames", type=int, default=30)
    p.add_argument("--min_samples", type=int, default=10)
    p.add_argument("--ema_alpha", type=float, default=0.3)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--pad_ratio", type=float, default=0.02)
    p.add_argument("--denoise", action="store_true")
    p.add_argument("--ratio", type=float, default=0.1)
    p.add_argument("--clip_margin", type=int, default=5)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--loop_video", action="store_true")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--fire_window_seconds", type=float, default=5.0)
    p.add_argument("--no_qr", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    if not 0.0 < args.ema_alpha <= 1.0:
        raise ValueError("--ema_alpha must be in (0, 1]")
    print("[Run] Loading YOLO and EfficientAD models...")
    yolo = YOLO(args.yolo_weights)
    models, image_size, actual_device = load_bundle(
        args.method, args.ckpt_dir, args.category, args.model_size, args.device)
    device = str(actual_device)
    use_fp16 = bool(args.fp16 and device.startswith("cuda"))
    if use_fp16:
        for key, kind in (("teacher", "teacher"), ("student", "student"), ("ae", "ae")):
            models[key] = patch_model_forward_dtype_safe(models[key].half(), kind)
        for key in ("mean", "std", "qa_st", "qb_st", "qa_ae", "qb_ae"):
            if isinstance(models.get(key), torch.Tensor):
                models[key] = models[key].half()
    # Calibration values are reported for operator reference only, as requested;
    # the real-time decision threshold remains --tau_track.
    load_thresholds(args.thresholds_json, args.tau_track)

    camera = LatestFrameCamera(args.camera, args.width, args.height, args.loop_video)
    camera.start()
    first, sequence = camera.latest_after(0, timeout=10.0)
    if first is None:
        camera.stop()
        raise RuntimeError("Camera opened but no frame arrived within 10 seconds")
    print(f"[Run] Camera opened: {args.camera} ({first.shape[1]}x{first.shape[0]})")

    # Warm-up happens before the public server starts.
    with torch.inference_mode():
        dummy = torch.zeros((1, 3, image_size, image_size), device=device,
                            dtype=torch.float16 if use_fp16 else torch.float32)
        models["teacher"](dummy); models["student"](dummy); models["ae"](dummy)
    print("[Run] Warm-up done")

    lan_ip = detect_lan_ip()
    url = f"http://{lan_ip}:{args.port}/"
    qr_png = make_qr(url, args.no_qr)
    print(f"[QR] URL: {url}")
    if qr_png:
        print(f"[QR] PNG: {Path.cwd() / 'live_qr.png'}")

    state = SharedState(args.fire_window_seconds)
    server = DashboardServer(args.host, args.port, state, qr_png)
    server.start()
    print(f"[Web] Server ready: {url}")

    ema = {}
    buffers = {}
    last_seen = {}
    final_scores = {}
    decided = set()
    n_frames = 0
    start = time.perf_counter()
    last_tick = start
    smooth_fps = 0.0

    try:
        frame = first
        while frame is not None:
            n_frames += 1
            now_tick = time.perf_counter()
            instant = 1.0 / max(now_tick-last_tick, 1e-9)
            smooth_fps = instant if n_frames == 1 else 0.1*instant + 0.9*smooth_fps
            last_tick = now_tick

            results = yolo.track(frame, persist=True, conf=args.conf, iou=args.iou, verbose=False)
            dets = []
            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                xyxy = results[0].boxes.xyxy.detach().cpu().numpy().astype(int)
                ids = results[0].boxes.id.detach().cpu().numpy().astype(int)
                dets = [(int(tid), int(b[0]), int(b[1]), int(b[2]), int(b[3]))
                        for tid, b in zip(ids, xyxy)]
            clipped = classify_clipped(dets, frame.shape[1], frame.shape[0], args.clip_margin)
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            crops = [get_crop(pil, (x1, y1, x2, y2), image_size, args.pad_ratio, args.denoise)
                     for (_, x1, y1, x2, y2), clip in zip(dets, clipped) if not clip]
            with torch.inference_mode():
                scores = call_score_batch(models, crops, device, image_size, args.ratio, use_fp16)

            emas = [None] * len(dets)
            score_index = 0
            for index, ((tid, *_), clip) in enumerate(zip(dets, clipped)):
                if clip:
                    continue
                raw = float(scores[score_index]); score_index += 1
                value = args.ema_alpha*raw + (1.0-args.ema_alpha)*ema.get(tid, raw)
                ema[tid] = value
                buffers.setdefault(tid, collections.deque(maxlen=200)).append(value)
                last_seen[tid] = n_frames
                emas[index] = value

            # inactive_frames deliberately counts PROCESSED inference frames,
            # not captured or wall-clock frames. At 10 processed fps, 30 means
            # about 3 seconds even when the camera itself produces 30 fps.
            for tid, seen_frame in list(last_seen.items()):
                if tid in decided or n_frames-seen_frame < args.inactive_frames:
                    continue
                buf = buffers.get(tid, ())
                decided.add(tid)
                if len(buf) < args.min_samples:
                    continue
                final = float(max(buf) if len(buf) < 25 else np.median(buf))
                final_scores[tid] = final
                if final >= args.tau_track:
                    state.add_fire(time.monotonic(), tid, final)
                    print(f"[Track] FIRED frame {n_frames}: ID{tid} score={final:.3f} (n={len(buf)})")

            recent = state.recent_map()
            annotated = frame.copy()
            for (tid, x1, y1, x2, y2), clip, value in zip(dets, clipped, emas):
                if clip:
                    color, thickness, label = GREY, 2, f"ID{tid} CLIP"
                elif tid in recent:
                    color, thickness, label = RED, 4, f"ID{tid} FIRED {recent[tid]:.2f}"
                else:
                    color, thickness = GREEN, 2
                    label = f"ID{tid} {value:.2f}" if value is not None else f"ID{tid}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
                draw_label(annotated, label, x1, y1, color, thickness)

            frame_alarm = any(v is not None and v >= args.tau_track for v in emas)
            ids_recent = sorted(recent)
            banner_color = RED if ids_recent else GREEN
            banner = (f"frame {n_frames} | visible={len(dets)} | "
                      f"recent fires={ids_recent} | fps={smooth_fps:.1f}")
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 38), banner_color, -1)
            cv2.putText(annotated, banner, (10, 27), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (255, 255, 255), 2, cv2.LINE_AA)
            state.publish(annotated, smooth_fps, n_frames, len(dets), frame_alarm)
            if n_frames % 60 == 0:
                print(f"frame {n_frames} | visible={len(dets)} | recent_fires={len(ids_recent)} | fps={smooth_fps:.1f}")

            frame, sequence = camera.latest_after(sequence, timeout=1.0)
            if frame is None and not camera.eof:
                continue
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        server.stop()
        elapsed = max(time.perf_counter()-start, 1e-9)
        print(f"[Run] Stopped after {n_frames} frames (avg FPS: {n_frames/elapsed:.1f})")


if __name__ == "__main__":
    main()