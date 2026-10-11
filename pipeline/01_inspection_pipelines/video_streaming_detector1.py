"""
video_streaming_detector1.py
============================
Streaming video anomaly detector — single model, track-end decision.

Track-level decision rule
-------------------------
For each track, we keep a cumulative buffer of every EMA value seen so
far (capped at 200 for memory). A track fires when, AT TRACK END (after
--inactive_frames of no detection):

    n_seen >= min_samples  AND  score >= tau_track

Score selection:
  * n <  min_samples         -> skipped (too few samples to trust)
  * min_samples <= n < 25    -> max(cumulative buffer)  [short-track rule]
  * n >= 25                  -> median(cumulative buffer)  [stable]

Why track-end decision:
  Firing on the fly causes transient crossings to lock a track as fired
  forever. The track-end decision waits for the track to finish, then
  evaluates its whole-life median. Matches the offline evaluation.

Operating point
---------------
  tau_track = 0.55 -> retrain ckpt_cropped_multi (from the sweep)
  tau_track = 0.883 -> old ckpt_192 median
  tau_track = 1.335 -> new ckpt_cropped_large1 median

Usage
-----
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python video_streaming_detector1.py \
        --video ./video_test/video_A.mp4 \
        --yolo_det_weights ./runs/detect/.../best.pt \
        --method baseline --ckpt_dir ./ckpt_cropped_multi \
        --thresholds_json ./video_calibration_new/video_thresholds_baseline.json \
        --category chewinggum --model_size S \
        --tau_track 0.55 --inactive_frames 30 --min_samples 10 \
        --clip_margin 5 --min_box_frac 0.5 \
        --batch_frames 16 --batch_crops 96 --crop_workers 12 \
        --fp16 --device cuda:0 \
        --save_dir ./options/M/video_A
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import json
import time
import types
import threading
import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from torchvision import transforms

try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except ImportError:
    pass

torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = False

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


# ===========================================================================
# dtype-safe imagenet_norm_batch patches
# ===========================================================================
import utils.models as _models_module


def _dtype_preserving_norm(x):
    mean = torch.tensor([0.485, 0.456, 0.406],
                        dtype=x.dtype, device=x.device)[None, :, None, None]
    std  = torch.tensor([0.229, 0.224, 0.225],
                        dtype=x.dtype, device=x.device)[None, :, None, None]
    return (x - mean) / (std + 1e-11)


_models_module.imagenet_norm_batch = _dtype_preserving_norm
print("[Stream] Patched imagenet_norm_batch (module-level).")


def patch_model_forward_dtype_safe(model, kind):
    if kind in ("teacher", "student"):
        def _f(self, x):
            x = _dtype_preserving_norm(x)
            return self.pdn(x)
    elif kind == "ae":
        def _f(self, x):
            x = _dtype_preserving_norm(x)
            x = self.encoder(x)
            x = self.decoder(x)
            return x
    else:
        raise ValueError(f"Unknown kind '{kind}'")
    model.forward = types.MethodType(_f, model)
    return model


from compare_pipeline_methods_batched import (
    load_bundle, get_crop, reencode_h264,
    draw_unknown_contours, run_yolo_once,
    METHOD_COLOURS, OUT_CHANNELS,
)


# ===========================================================================
# Clipped-box filter
# ===========================================================================
def classify_boxes(frame_dets, src_w, src_h, clip_margin, min_box_frac):
    flags = []
    if not frame_dets:
        return flags
    areas = np.asarray(
        [(x2 - x1) * (y2 - y1) for _, x1, y1, x2, y2 in frame_dets],
        dtype=np.float64)
    use_size_test = (min_box_frac > 0) and (len(frame_dets) >= 3)
    med = float(np.median(areas)) if use_size_test else 0.0
    for i, (_tid, x1, y1, x2, y2) in enumerate(frame_dets):
        edge = (x1 <= clip_margin or
                y1 <= clip_margin or
                x2 >= src_w - clip_margin or
                y2 >= src_h - clip_margin)
        size = use_size_test and (areas[i] < min_box_frac * med)
        flags.append(bool(edge or size))
    return flags


# ===========================================================================
# Web MJPEG server
# ===========================================================================
class MJPEGServer:
    def __init__(self, port: int, jpeg_quality: int = 75):
        self.port = port
        self.quality = jpeg_quality
        self._frame = None
        self._lock = threading.Lock()
        self._server = None
        self._thread = None

    def set_frame(self, frame_bgr: np.ndarray) -> None:
        with self._lock:
            self._frame = frame_bgr.copy()

    def _get_frame(self):
        with self._lock:
            return None if self._frame is None else self._frame

    def _make_handler(self):
        srv = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args, **kwargs):
                pass

            def do_GET(self):
                if self.path == "/":
                    html = (
                        b"<!doctype html><html><head><meta charset=utf-8>"
                        b"<title>EfficientAD live</title>"
                        b"<style>html,body{margin:0;background:#111;height:100%}"
                        b"img{display:block;margin:0 auto;max-width:100vw;"
                        b"max-height:100vh;object-fit:contain}</style></head>"
                        b"<body><img src='/stream'></body></html>"
                    )
                    self.send_response(200)
                    self.send_header("Content-Type",
                                     "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(html)))
                    self.end_headers()
                    self.wfile.write(html)
                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Age", "0")
                    self.send_header("Cache-Control", "no-cache, private")
                    self.send_header("Pragma", "no-cache")
                    self.send_header(
                        "Content-Type",
                        "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    try:
                        while True:
                            f = srv._get_frame()
                            if f is None:
                                time.sleep(0.02)
                                continue
                            ok, jpg = cv2.imencode(
                                ".jpg", f,
                                [cv2.IMWRITE_JPEG_QUALITY, srv.quality])
                            if not ok:
                                continue
                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(
                                f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                            self.wfile.write(jpg.tobytes())
                            self.wfile.write(b"\r\n")
                            time.sleep(0.03)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                else:
                    self.send_response(404)
                    self.end_headers()

        return Handler

    def start(self):
        self._server = HTTPServer(
            ("0.0.0.0", self.port), self._make_handler())
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(f"[Web] MJPEG stream live at http://0.0.0.0:{self.port}/")

    def stop(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()


# ===========================================================================
# PyTorch scoring
# ===========================================================================
@torch.no_grad()
def _score_crop_batch(models, crops_pil, device, image_size,
                      model_dtype, ratio=0.1):
    if not crops_pil:
        return []
    tensors = [transforms.ToTensor()(c) for c in crops_pil]
    t = torch.stack(tensors, dim=0).to(device)
    if t.dtype != model_dtype:
        t = t.to(model_dtype)
    t = t.to(memory_format=torch.channels_last)

    t_out = models["teacher"](t)
    s_out = models["student"](t)
    a_out = models["ae"](t)

    y_st   = s_out[:, :OUT_CHANNELS,  :, :]
    y_stae = s_out[:, -OUT_CHANNELS:, :, :]

    norm_t = (t_out - models["mean"]) / (models["std"] + 1e-8)
    d_st   = torch.pow(norm_t - y_st,   2)
    d_stae = torch.pow(a_out  - y_stae, 2)

    fm_st   = F.interpolate(torch.mean(d_st,   dim=1, keepdim=True),
                            size=(image_size, image_size),
                            mode="bilinear", align_corners=False)
    fm_stae = F.interpolate(torch.mean(d_stae, dim=1, keepdim=True),
                            size=(image_size, image_size),
                            mode="bilinear", align_corners=False)

    fm_st, fm_stae = fm_st.float(), fm_stae.float()
    qa_st = models["qa_st"].float(); qb_st = models["qb_st"].float()
    qa_ae = models["qa_ae"].float(); qb_ae = models["qb_ae"].float()

    nm_st = (ratio * (fm_st   - qa_st)) / (qb_st - qa_st + 1e-8)
    nm_ae = (ratio * (fm_stae - qa_ae)) / (qb_ae - qa_ae + 1e-8)
    amap  = (0.5 * nm_st + 0.5 * nm_ae)[:, 0].cpu().numpy()
    return [float(np.max(amap[i])) for i in range(amap.shape[0])]


# ===========================================================================
# Streaming detector
# ===========================================================================
class VideoStreamingDetector:
    def __init__(self, method, ckpt_dir, thresholds, args, src_w, src_h):
        self.method = method
        self.ckpt_dir = ckpt_dir
        self.args = args
        self.src_w = src_w
        self.src_h = src_h

        self.tau_p = float(thresholds["tau_p"])
        self.tau_f = float(thresholds.get("tau_f", thresholds["tau_p"]))
        self.tau_track = float(args.tau_track)
        self.ema_window = int(args.ema_window)
        self.inactive_frames = int(args.inactive_frames)
        self.min_samples = int(args.min_samples)

        self.models, self.image_size, self.device = load_bundle(
            method, ckpt_dir, args.category, args.model_size, args.device)

        self.use_fp16 = args.fp16 and self.device.startswith("cuda")
        if self.use_fp16:
            print("[Stream] Converting models to FP16...")
            for key in ("teacher", "student", "ae"):
                self.models[key] = self.models[key].half()
            for key in ("mean", "std", "qa_st", "qb_st", "qa_ae", "qb_ae"):
                if isinstance(self.models.get(key), torch.Tensor):
                    self.models[key] = self.models[key].half()
            for key, kind in (("teacher", "teacher"),
                              ("student", "student"),
                              ("ae", "ae")):
                self.models[key] = patch_model_forward_dtype_safe(
                    self.models[key], kind)
            print("[Stream] Converting models to channels_last...")
            for key in ("teacher", "student", "ae"):
                self.models[key] = self.models[key].to(
                    memory_format=torch.channels_last)

        self.model_dtype = torch.float16 if self.use_fp16 else torch.float32

        # Per-track state
        self.ema = {}
        self.ema_history = {}

        # Track-end decision state
        self.track_last_seen = {}
        self.track_final_score = {}
        self.decided_track_ids = set()
        self.fired_track_ids = set()

        self.alpha = args.ema_alpha

        # Frame-level state machine
        self.alarm = 0
        self.persist = 0
        self.clear = 0
        self.frame_idx = 0
        self.records = []

        # Diagnostics
        self.n_boxes_total   = 0
        self.n_boxes_clipped = 0
        self.n_boxes_scored  = 0
        self.n_alarm_frame   = 0
        self.n_alarm_track   = 0

        self._crop_pool = ThreadPoolExecutor(
            max_workers=max(1, args.crop_workers))
        self._last_chunk_crop_ms = 0.0
        self._last_chunk_fwd_ms  = 0.0

    # -------------------------------------------------------------------
    def _compute_track_score(self, tid):
        """
        Compute the final score for a track using its cumulative EMA buffer.
          n <  min_samples       -> None (skip)
          min_samples <= n < 25  -> max(buffer)       [short-track rule]
          n >= 25                -> median(buffer)
        """
        buf = self.ema_history.get(tid)
        if buf is None or len(buf) < self.min_samples:
            return None
        if len(buf) < 25:
            return float(np.max(buf))
        return float(np.median(buf))

    # -------------------------------------------------------------------
    def _update_state(self, frame_idx, frame_max, n_alarm_crops, n_crops,
                      raw_alarm, raw_scores, fired_tracks):
        if raw_alarm:
            self.persist += 1; self.clear = 0
        else:
            self.clear += 1; self.persist = 0

        just_started = just_ended = False
        if self.alarm == 0 and self.persist >= self.args.persist_K:
            self.alarm = 1; just_started = True
        elif (self.alarm == 1
              and self.clear >= self.args.clear_M
              and frame_max < self.args.gamma * self.tau_f):
            self.alarm = 0; just_ended = True

        rec = {
            "frame":         frame_idx,
            "frame_max":     float(frame_max),
            "n_alarm_crops": n_alarm_crops,
            "n_crops":       n_crops,
            "raw_alarm":     bool(raw_alarm),
            "alarm":         int(self.alarm),
            "just_started":  just_started,
            "just_ended":    just_ended,
            "fired_tracks":  fired_tracks,
        }
        self.records.append(rec)
        return rec

    # -------------------------------------------------------------------
    def step_chunk(self, chunk):
        tasks = []
        frame_slices = []
        clipped_per_frame = []

        for (fi, frame_bgr, dets) in chunk:
            clipped = classify_boxes(
                dets, self.src_w, self.src_h,
                self.args.clip_margin, self.args.min_box_frac)
            clipped_per_frame.append(clipped)

            self.n_boxes_total   += len(dets)
            self.n_boxes_clipped += int(sum(clipped))

            pil_frame = Image.fromarray(
                cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            start = len(tasks)
            for (_tid, x1, y1, x2, y2), is_clip in zip(dets, clipped):
                if is_clip:
                    continue
                tasks.append((pil_frame, (x1, y1, x2, y2)))
            frame_slices.append((start, len(tasks)))

        t_crop = time.perf_counter()
        if tasks:
            futures = [
                self._crop_pool.submit(
                    get_crop, pil, box, self.image_size,
                    self.args.pad_ratio, self.args.denoise)
                for pil, box in tasks
            ]
            all_crops = [f.result() for f in futures]
        else:
            all_crops = []
        self._last_chunk_crop_ms = (time.perf_counter() - t_crop) * 1000.0

        t_fwd = time.perf_counter()
        all_scores = _score_crop_batch(
            self.models, all_crops, self.device, self.image_size,
            model_dtype=self.model_dtype, ratio=self.args.ratio)
        self._last_chunk_fwd_ms = (time.perf_counter() - t_fwd) * 1000.0

        out = []
        for (fi, _fb, dets), (a, b), clipped in zip(
                chunk, frame_slices, clipped_per_frame):
            raw = all_scores[a:b]
            emas = [None] * len(dets)

            score_i = 0
            for k, ((tid, *_box), is_clip) in enumerate(zip(dets, clipped)):
                if is_clip:
                    continue
                s = raw[score_i]; score_i += 1

                # Update EMA
                prev = self.ema.get(tid, s)
                e = self.alpha * s + (1 - self.alpha) * prev
                self.ema[tid] = e
                emas[k] = e

                # Cumulative buffer, capped at 200
                buf = self.ema_history.setdefault(
                    tid, collections.deque(maxlen=200))
                buf.append(e)

                # Record last-seen frame (used by track-end decision)
                self.track_last_seen[tid] = fi

            self.n_boxes_scored += len(raw)
            valid_emas = [e for e in emas if e is not None]
            frame_max = float(max(valid_emas)) if valid_emas else 0.0

            # ---- Frame-level trigger ----
            frame_alarm = (frame_max >= self.tau_f)

            # ---- Track-end decisions ----
            # For each unseen-for-inactive_frames track, evaluate its
            # cumulative score and lock in the decision.
            newly_fired = []
            for tid, last_f in list(self.track_last_seen.items()):
                if tid in self.decided_track_ids:
                    continue
                if (fi - last_f) < self.inactive_frames:
                    continue
                score = self._compute_track_score(tid)
                if score is None:
                    score = 0.0
                self.track_final_score[tid] = score
                self.decided_track_ids.add(tid)
                if score >= self.tau_track:
                    self.fired_track_ids.add(int(tid))
                    newly_fired.append((int(tid), score))
                    print(f"[Track] FIRED frame {fi}: ID{tid} "
                          f"score={score:.3f} >= tau={self.tau_track}")

            # Build fired_tracks for this frame's record
            fired_tracks = []
            for tid in sorted(self.fired_track_ids):
                fired_tracks.append({
                    "track_id": int(tid),
                    "score":    self.track_final_score.get(tid, 0.0),
                })

            track_alarm = bool(self.fired_track_ids)
            raw_alarm = frame_alarm or track_alarm

            if frame_alarm:
                self.n_alarm_frame += 1
            if track_alarm:
                self.n_alarm_track += 1

            rec = self._update_state(fi, frame_max, len(fired_tracks),
                                     len(valid_emas), raw_alarm, valid_emas,
                                     fired_tracks)
            rec["n_clipped"] = int(sum(clipped))
            rec["frame_alarm"] = bool(frame_alarm)
            rec["track_alarm"] = bool(track_alarm)
            out.append((rec, raw, emas, clipped, fired_tracks))
        self.frame_idx = chunk[-1][0] + 1 if chunk else self.frame_idx
        return out

    # -------------------------------------------------------------------
    def finalize(self):
        """
        Flush tracks still active at end of video. Called before the last
        chunk so those tracks can be drawn/scored in the final frames.
        """
        for tid, last_f in list(self.track_last_seen.items()):
            if tid in self.decided_track_ids:
                continue
            score = self._compute_track_score(tid)
            if score is None:
                self.decided_track_ids.add(tid)
                continue
            self.track_final_score[tid] = score
            self.decided_track_ids.add(tid)
            if score >= self.tau_track:
                self.fired_track_ids.add(int(tid))
                print(f"[Track] FIRED end-of-video: ID{tid} "
                      f"score={score:.3f} >= tau={self.tau_track}")


# ===========================================================================
# Chunk splitter
# ===========================================================================
def _make_chunks(detections, n_frames, max_frames_per_chunk,
                 max_crops_per_chunk, src_w, src_h, clip_margin,
                 min_box_frac):
    start = 0
    while start < n_frames:
        end = start
        crops = 0
        while end < n_frames and (end - start) < max_frames_per_chunk:
            dets = detections.get(end, [])
            clipped = classify_boxes(dets, src_w, src_h,
                                     clip_margin, min_box_frac)
            n_this = sum(1 for c in clipped if not c)
            if crops + n_this > max_crops_per_chunk and end > start:
                break
            crops += n_this
            end += 1
        if end == start:
            end = start + 1
        yield start, end
        start = end


# ===========================================================================
# CLI
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Streaming video anomaly detection — single model, "
                    "track-end decision.")
    p.add_argument("--video", required=True)
    p.add_argument("--yolo_det_weights", required=True)
    p.add_argument("--method", required=True,
                   choices=["baseline", "lowres", "thin", "qat", "pruned"])
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--thresholds_json", required=True)
    p.add_argument("--category", default="chewinggum")
    p.add_argument("--model_size", default="S", choices=["S", "M"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ratio",     type=float, default=0.1)
    p.add_argument("--conf",      type=float, default=0.25)
    p.add_argument("--iou",       type=float, default=0.45)
    p.add_argument("--pad_ratio", type=float, default=0.02)
    p.add_argument("--denoise",   action="store_true")
    p.add_argument("--fp16",      action="store_true")
    p.add_argument("--ema_alpha", type=float, default=0.3)
    p.add_argument("--persist_K", type=int, default=3)
    p.add_argument("--clear_M",   type=int, default=5)
    p.add_argument("--gamma",     type=float, default=0.5)
    p.add_argument("--max_frames", type=int, default=0)

    p.add_argument("--batch_frames", type=int, default=16)
    p.add_argument("--batch_crops", type=int, default=96)
    p.add_argument("--crop_workers", type=int, default=12)

    p.add_argument("--clip_margin",  type=int,   default=5)
    p.add_argument("--min_box_frac", type=float, default=0.5)

    p.add_argument("--tau_track", type=float, default=1.29,
                   help="Track-level threshold on cumulative median.")
    p.add_argument("--inactive_frames", type=int, default=30,
                   help="A track is 'ended' after this many frames of no "
                        "detection. The final decision is made at that "
                        "moment.")
    p.add_argument("--min_samples", type=int, default=10,
                   help="Minimum samples in the EMA buffer. Tracks with "
                        "fewer are skipped. 10 recommended.")
    p.add_argument("--ema_window", type=int, default=15,
                   help="Legacy flag. Unused.")

    p.add_argument("--no_detect_unknown", action="store_true")
    p.add_argument("--otsu_max_area_frac", type=float, default=0.25)
    p.add_argument("--otsu_min_unknown_area", type=int, default=20)
    p.add_argument("--no_video", action="store_true")
    p.add_argument("--save_dir", required=True)

    p.add_argument("--display", action="store_true")
    p.add_argument("--display_scale", type=float, default=0.5)
    p.add_argument("--web_port", type=int, default=0)
    p.add_argument("--sleep_between_frames", type=float, default=0.0)
    p.add_argument("--hold_after_run", action="store_true")

    return p.parse_args()


# ===========================================================================
# Main
# ===========================================================================
def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    with open(args.thresholds_json) as f:
        thr = json.load(f)

    print(f"[Stream] method={thr['method']}  tau_p={thr['tau_p']:.4f}  "
          f"tau_f={thr['tau_f']:.4f}")
    print(f"[Stream] tau_track={args.tau_track:.4f}")
    print(f"[Stream] inactive_frames={args.inactive_frames}  "
          f"min_samples={args.min_samples}")
    print(f"[Stream] Track decision: cumulative MEDIAN at track end "
          f"(max for n < 25, skip n < {args.min_samples})")
    print(f"[Stream] batch_frames={args.batch_frames}  "
          f"batch_crops={args.batch_crops}  "
          f"crop_workers={args.crop_workers}  "
          f"fp16={args.fp16}  device={args.device}")

    web_server = None
    if args.web_port > 0:
        web_server = MJPEGServer(args.web_port)
        web_server.start()

    detections, src_w, src_h, src_fps, n_frames, frame_cache = run_yolo_once(
        args.video, args.yolo_det_weights, args.conf, args.iou,
        device=args.device,
        max_frames=(args.max_frames or None),
        detect_unknown=not args.no_detect_unknown,
        otsu_max_area_frac=args.otsu_max_area_frac,
        cache_frames=True)

    det = VideoStreamingDetector(
        args.method, args.ckpt_dir, thr, args, src_w, src_h)
    colour = METHOD_COLOURS.get(args.method, (255, 255, 255))

    writer = None
    video_raw = None
    if not args.no_video:
        video_raw = os.path.join(
            args.save_dir, f"annotated_stream_{args.method}_raw.mp4")
        writer = cv2.VideoWriter(video_raw,
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 src_fps, (src_w, src_h))

    K = max(1, args.batch_frames)

    try:
        warm_n = min(args.batch_crops, max(K * 4, 8))
        dummy = Image.new("RGB", (det.image_size, det.image_size))
        t_warm = time.perf_counter()
        _ = _score_crop_batch(
            det.models, [dummy] * warm_n, det.device, det.image_size,
            model_dtype=det.model_dtype, ratio=args.ratio)
        _ = det._crop_pool.submit(
            get_crop, dummy, (0, 0, 64, 64), det.image_size,
            args.pad_ratio, args.denoise).result()
        print(f"[Stream] Warm-up done in "
              f"{(time.perf_counter() - t_warm) * 1000:.0f}ms "
              f"(batch={warm_n}).")
    except Exception as e:
        print(f"[Stream] Warm-up failed: {e}")
        raise

    t0 = time.perf_counter()
    n_processed = 0
    user_quit = False

    all_chunks = list(_make_chunks(
        detections, n_frames,
        max_frames_per_chunk=K,
        max_crops_per_chunk=args.batch_crops,
        src_w=src_w, src_h=src_h,
        clip_margin=args.clip_margin,
        min_box_frac=args.min_box_frac))

    for ci, (chunk_start, chunk_end) in enumerate(all_chunks):
        # Before the last chunk, flush remaining tracks so the final
        # frames can draw them.
        if ci == len(all_chunks) - 1:
            det.finalize()

        chunk = [(i, frame_cache.bgr_frames[i], detections.get(i, []))
                 for i in range(chunk_start, chunk_end)]

        chunk_results = det.step_chunk(chunk)

        for (i, frame_bgr, dets), (rec, raw, emas, clipped, fired) in zip(
                chunk, chunk_results):
            annot = frame_bgr.copy()
            mask = frame_cache.unknown_masks[i] if frame_cache else None

            if not args.no_detect_unknown and mask is not None:
                draw_unknown_contours(annot, mask,
                                      args.otsu_min_unknown_area)

            score_i = 0
            for (tid, x1, y1, x2, y2), e, is_clip in zip(dets, emas, clipped):
                tid_int = int(tid)
                if is_clip:
                    cv2.rectangle(annot, (x1, y1), (x2, y2),
                                  (128, 128, 128), 2)
                    label = f"ID{tid} CLIPPED"
                    color = (128, 128, 128)
                elif tid_int in det.fired_track_ids:
                    score = det.track_final_score.get(tid_int, 0.0)
                    cv2.rectangle(annot, (x1, y1), (x2, y2), (0, 0, 255), 4)
                    label = f"ID{tid} FIRED {score:.2f}"
                    color = (0, 0, 255)
                else:
                    s = raw[score_i]; score_i += 1
                    if e >= det.tau_p:
                        label = f"ID{tid} ANOMALY {e:.2f}"
                        color = (0, 0, 255)
                    else:
                        label = f"ID{tid} NORMAL {e:.2f}"
                        color = (0, 200, 0)
                    cv2.rectangle(annot, (x1, y1), (x2, y2), color, 2)

                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                tx, ty = max(0, x1 + 3), max(th + 6, y1 + 3)
                cv2.rectangle(annot, (tx - 2, ty - th - 4),
                              (tx + tw + 4, ty + 2), color, -1)
                cv2.putText(annot, label, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 1, cv2.LINE_AA)

            fired_list = sorted(det.fired_track_ids)
            banner_c = (0, 0, 255) if fired_list else colour
            banner = (f"{args.method} | frame {i} | "
                      f"FIRED={fired_list if fired_list else 'none'}")
            cv2.putText(annot, banner, (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, banner_c, 2)

            if writer is not None:
                writer.write(annot)

            display_frame = None
            if args.display or web_server is not None:
                if args.display_scale != 1.0:
                    display_frame = cv2.resize(
                        annot, None,
                        fx=args.display_scale, fy=args.display_scale,
                        interpolation=cv2.INTER_LINEAR)
                else:
                    display_frame = annot

            if args.display and display_frame is not None:
                cv2.imshow("EfficientAD stream (press q to quit)",
                           display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    user_quit = True
                    break

            if web_server is not None and display_frame is not None:
                web_server.set_frame(display_frame)

            n_processed += 1
            if args.sleep_between_frames > 0:
                time.sleep(args.sleep_between_frames)

        if user_quit:
            break

        dt = time.perf_counter() - t0
        fps = n_processed / max(dt, 1e-6)
        print(f"  chunk [{chunk_start:5d}-{chunk_end-1:5d}] "
              f"({len(chunk)} frames) "
              f"| crop={det._last_chunk_crop_ms:6.1f}ms  "
              f"fwd={det._last_chunk_fwd_ms:6.1f}ms  "
              f"| {n_processed}/{n_frames} | {fps:.1f} fps")

    out_video = None
    if writer is not None:
        writer.release()
        h264 = video_raw.replace("_raw.mp4", "_h264.mp4")
        if reencode_h264(video_raw, h264):
            os.remove(video_raw)
            out_video = h264
        else:
            out_video = video_raw

    if args.display:
        cv2.destroyAllWindows()
    if web_server is not None:
        if args.hold_after_run:
            input(f"\n[Web] Server still at http://<host>:{args.web_port}/ "
                  f"-- press Enter to exit.\n")
        web_server.stop()

    records_path = os.path.join(args.save_dir,
                                f"records_{args.method}.jsonl")
    with open(records_path, "w") as f:
        for r in det.records:
            f.write(json.dumps(r) + "\n")

    dt = time.perf_counter() - t0
    print(f"\nDone. {n_processed} frames in {dt:.1f}s "
          f"({n_processed / dt:.1f} fps).")
    print(f"  tau_track           : {det.tau_track:.4f}")
    print(f"  inactive_frames     : {det.inactive_frames}")
    print(f"  min_samples         : {det.min_samples}")
    print(f"  Track IDs that fired: {sorted(det.fired_track_ids)}")
    if out_video:
        print(f"  Video               : {out_video}")
    print(f"  Records             : {records_path}")

    det._crop_pool.shutdown(wait=False)


if __name__ == "__main__":
    main()