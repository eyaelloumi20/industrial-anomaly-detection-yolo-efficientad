"""Offline YOLO + EfficientAD detector with final-verdict video annotations.

Save this file beside compare_pipeline_methods_batched.py in the existing
inspection project. Keep the project's utils/ directory: load_bundle() uses
utils.load_efficientad_adaptive for baseline/lowres, exactly as before.

Checkpoint variants supported (auto-detected from --ckpt_dir contents):
  * baseline (256px full width)     — compare_pipeline_methods_batched.load_bundle
  * lowres   (192 / 224px)          — utils.load_efficientad_adaptive
  * thin     (width_mult < 1.0)     — utils.load_efficientad_thin
  * pruned   (channel-pruned)       — utils.load_efficientad_pruned
  * qat      (int8 CPU)             — utils.load_efficientad_qat

The saved annotated.mp4 is a retrospective review. Confirmed anomalous track
IDs are red throughout their recorded appearances, including clipped boxes.
Green means no final anomaly flag; grey means an unflagged clipped box.
The optional processing preview shows decisions known at that moment.
Actual decision times remain in events.jsonl; they are never backdated.

Frames are not skipped. A bounded chunk is scored during one input pass.
Video rendering uses recorded boxes and does not run either model again.
--no_video skips rendering and box metadata. --early_fire is optional and
provisional; final track decisions keep the original max/median EMA rule.

--web_port publishes the optimized processing preview as an MJPEG stream.
--web_final_replay additionally streams the retrospective final review, where
confirmed anomalous tracks are red throughout their recorded appearances.

Example (run from the existing project):
    python video_streaming_detector_optimized.py \
        --video ./video_test/video_A.mp4 \
        --yolo_det_weights ./runs/detect/yolo_runs/chewinggum_yolov8n_v2/weights/best.pt \
        --ckpt_dir ./ckpt_cropped_multi \
        --thresholds_json ./video_calibration_new/video_thresholds_baseline.json \
        --tau_track 0.45 --fp16 --device cuda:0 --web_port 8080 \
        --save_dir ./video_outputs
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import importlib
import json
import math
import tempfile
import threading
import time
import types
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO
import utils.models as models_module
from compare_pipeline_methods_batched import OUT_CHANNELS, get_crop, load_bundle

RED, GREEN, GREY = (0, 0, 255), (0, 200, 0), (128, 128, 128)
_NORM_CACHE = {}


class MJPEGServer:
    """Small latest-frame MJPEG server; frames are never queued."""

    def __init__(self, host, port, jpeg_quality=75):
        self.host = host
        self.port = int(port)
        self.jpeg_quality = int(jpeg_quality)
        self._condition = threading.Condition()
        self._jpeg = None
        self._sequence = 0
        self._server = None
        self._thread = None

    def set_frame(self, frame):
        ok, encoded = cv2.imencode(
            ".jpg", frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if not ok:
            return
        jpeg = encoded.tobytes()
        with self._condition:
            self._jpeg = jpeg
            self._sequence += 1
            self._condition.notify_all()

    def _next_jpeg(self, previous_sequence):
        with self._condition:
            self._condition.wait_for(
                lambda: self._sequence != previous_sequence,
                timeout=1.0
            )
            return self._jpeg, self._sequence

    def _handler(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path == "/":
                    page = (
                        "<!doctype html><html><head><meta charset='utf-8'>"
                        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
                        "<title>YOLO + EfficientAD</title><style>"
                        "html,body{margin:0;width:100%;height:100%;background:#0e0e0e;"
                        "color:#eee;font-family:monospace;overflow:hidden}"
                        "body{display:grid;place-items:center}"
                        "img{display:block;max-width:100vw;max-height:100vh;object-fit:contain}"
                        "</style></head><body><img src='/stream' alt='Detection stream'>"
                        "</body></html>"
                    ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(page)))
                    self.end_headers()
                    self.wfile.write(page)
                    return
                if path != "/stream":
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header(
                    "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                )
                self.end_headers()
                sequence = -1
                try:
                    while True:
                        jpeg, new_sequence = owner._next_jpeg(sequence)
                        if jpeg is None or new_sequence == sequence:
                            continue
                        sequence = new_sequence
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(
                            f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                        )
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

        return Handler

    def start(self):
        self._server = ThreadingHTTPServer(
            (self.host, self.port), self._handler()
        )
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="mjpeg-server", daemon=True
        )
        self._thread.start()
        shown_host = "<server-ip>" if self.host == "0.0.0.0" else self.host
        print(f"[Web] Detection stream: http://{shown_host}:{self.port}/")
        if self.host == "0.0.0.0":
            print("[Web] Warning: no authentication or HTTPS; use a firewall or SSH tunnel.")

    def stop(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None


def _dtype_preserving_norm(x):
    key = (x.device, x.dtype)
    if key not in _NORM_CACHE:
        mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        _NORM_CACHE[key] = mean, std + 1e-11
    mean, denom = _NORM_CACHE[key]
    return (x - mean) / denom


models_module.imagenet_norm_batch = _dtype_preserving_norm


def patch_model_forward_dtype_safe(model, kind):
    if kind in ("teacher", "student"):
        if not hasattr(model, "pdn"):
            raise TypeError("Expected the original teacher/student .pdn architecture.")

        def forward(self, x):
            return self.pdn(_dtype_preserving_norm(x))

    elif kind == "ae":
        if not (hasattr(model, "encoder") and hasattr(model, "decoder")):
            raise TypeError("Expected the original autoencoder architecture.")

        def forward(self, x):
            return self.decoder(self.encoder(_dtype_preserving_norm(x)))

    else:
        raise ValueError(kind)
    model.forward = types.MethodType(forward, model)


def load_any_bundle(ckpt_dir, category, model_size, device):
    """
    Auto-dispatch to the right EfficientAD loader based on the contents of
    ckpt_dir. Works for every variant trained in this project:
      - baseline  (256px, full width)          → compare_pipeline_methods_batched.load_bundle
      - lowres    (192 / 224px)                → utils.load_efficientad_adaptive
      - thin      (width_mult < 1.0)           → utils.load_efficientad_thin
      - pruned    (channel-pruned)             → utils.load_efficientad_pruned
      - qat       (int8 CPU)                   → utils.load_efficientad_qat

    Returns (models_dict, image_size, device).
    """
    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.is_dir():
        raise FileNotFoundError(f"ckpt_dir not found: {ckpt_dir}")

    quantiles_file = ckpt_path / f"{category}_quantiles.npy"
    quantiles = None
    if quantiles_file.is_file():
        try:
            quantiles = np.load(quantiles_file, allow_pickle=True).item()
        except Exception as exc:
            print(f"[Load] Warning: could not read quantiles ({exc})")

    # 1) QAT int8 — unique to the qat loader, detected by the .jit files
    if (ckpt_path / f"{category}_student_int8.jit").is_file() or \
       (ckpt_path / f"{category}_student_int8.pth").is_file():
        print("[Load] Detected int8 QAT checkpoint → load_efficientad_qat")
        mod = importlib.import_module("utils.load_efficientad_qat")
        models = mod.load_efficientad_model(
            ckpt_dir=str(ckpt_path), category=category,
            model_size=model_size, device="cpu")
        return models, 256, "cpu"

    # 2) Pruned — full pickled nn.Module, unique filename
    if (ckpt_path / f"{category}_student_pruned.pt").is_file():
        print("[Load] Detected pruned checkpoint → load_efficientad_pruned")
        mod = importlib.import_module("utils.load_efficientad_pruned")
        models = mod.load_efficientad_model(
            ckpt_dir=str(ckpt_path), category=category,
            model_size=model_size, device=device)
        return models, models.get("image_size", 256), device

    # 3) Thin — width_mult key in quantiles
    if quantiles is not None and "width_mult" in quantiles \
            and float(quantiles["width_mult"]) != 1.0:
        print(f"[Load] Detected thin checkpoint "
              f"(width_mult={quantiles['width_mult']}) → load_efficientad_thin")
        mod = importlib.import_module("utils.load_efficientad_thin")
        models = mod.load_efficientad_model(
            ckpt_dir=str(ckpt_path), category=category,
            model_size=model_size, device=device)
        return models, models.get("image_size", 256), device

    # 4) Adaptive low-res — image_size key != 256
    if quantiles is not None and "image_size" in quantiles \
            and int(quantiles["image_size"]) != 256:
        print(f"[Load] Detected low-res checkpoint "
              f"(image_size={quantiles['image_size']}) → load_efficientad_adaptive")
        mod = importlib.import_module("utils.load_efficientad_adaptive")
        models = mod.load_efficientad_model(
            ckpt_dir=str(ckpt_path), category=category,
            model_size=model_size, device=device)
        return models, models.get("image_size", 256), device

    # 5) Fallback: baseline full-width 256px via the existing load_bundle
    print("[Load] No variant markers found — using default baseline loader")
    return load_bundle("baseline", ckpt_dir, category, model_size, device)


class Scorer:
    def __init__(self, args):
        self.args = args
        self.models, self.image_size, device = load_any_bundle(
            args.ckpt_dir, args.category, args.model_size, args.device
        )
        self.device = torch.device(device)
        self.image_size = int(self.image_size)
        self.dtype = (
            torch.float16 if args.fp16 and self.device.type == "cuda"
            else torch.float32
        )
        if args.fp16 and self.dtype != torch.float16:
            print("[Run] FP16 disabled on non-CUDA device.")
        for key in ("teacher", "student", "ae"):
            model = self.models[key]
            model.eval().requires_grad_(False)
            if self.dtype == torch.float16:
                model.half().to(memory_format=torch.channels_last)
                patch_model_forward_dtype_safe(model, key)

        # Preserve the original statistic casting, including its FP16 mode.
        for key in ("mean", "std", "qa_st", "qb_st", "qa_ae", "qb_ae"):
            self.models[key] = torch.as_tensor(
                self.models[key], device=self.device, dtype=self.dtype
            )
        self.mean = self.models["mean"]
        self.std_denom = self.models["std"] + 1e-8
        self.qa_st = self.models["qa_st"].float()
        self.qa_ae = self.models["qa_ae"].float()
        self.st_denom = self.models["qb_st"].float() - self.qa_st + 1e-8
        self.ae_denom = self.models["qb_ae"].float() - self.qa_ae + 1e-8
        self.verified = False

    @torch.inference_mode()
    def __call__(self, crops, verify=False):
        if not crops:
            return []
        # Batched equivalent of the original RGB uint8 -> float32 / 255.
        pixels = np.stack([np.asarray(c, dtype=np.uint8) for c in crops])
        x = torch.from_numpy(pixels).permute(0, 3, 1, 2).float().div_(255.0)
        x = x.to(
            device=self.device, dtype=self.dtype,
            memory_format=torch.channels_last
        )
        teacher = self.models["teacher"](x)
        student = self.models["student"](x)
        ae = self.models["ae"](x)
        norm_teacher = (teacher - self.mean) / self.std_denom
        fm_st = F.interpolate(
            (norm_teacher - student[:, :OUT_CHANNELS]).pow(2).mean(1, keepdim=True),
            size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False
        ).float()
        fm_ae = F.interpolate(
            (ae - student[:, -OUT_CHANNELS:]).pow(2).mean(1, keepdim=True),
            size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False
        ).float()
        nm_st = (self.args.ratio * (fm_st - self.qa_st)) / self.st_denom
        nm_ae = (self.args.ratio * (fm_ae - self.qa_ae)) / self.ae_denom
        amap = 0.5 * nm_st + 0.5 * nm_ae
        # Transfer one scalar per crop rather than full-resolution maps.
        scores = amap.flatten(1).amax(1).cpu().tolist()
        if not all(math.isfinite(s) for s in scores):
            raise RuntimeError(
                "Non-finite EfficientAD scores. Check checkpoint statistics; "
                "retry without --fp16. No decisions were made for this batch."
            )
        del x, teacher, student, ae, norm_teacher, fm_st, fm_ae, nm_st, nm_ae, amap

        if verify and not self.verified:
            from video_streaming_detector1 import _score_crop_batch as reference
            models_module.imagenet_norm_batch = _dtype_preserving_norm
            expected = reference(
                self.models, crops, str(self.device), self.image_size,
                model_dtype=self.dtype, ratio=self.args.ratio
            )
            np.testing.assert_allclose(
                scores, expected, rtol=1e-3, atol=1e-5, equal_nan=False,
                err_msg="Optimized scores differ from the original scorer."
            )
            self.verified = True
            print("[Check] First real crop batch agrees with original scorer.")
        return scores


def classify_boxes(dets, width, height, margin, min_frac):
    if not dets:
        return []
    areas = np.asarray(
        [(x2 - x1) * (y2 - y1) for _, x1, y1, x2, y2 in dets],
        dtype=np.float64
    )
    use_size = min_frac > 0 and len(dets) >= 3
    median = float(np.median(areas)) if use_size else 0.0
    return [
        bool(x1 <= margin or y1 <= margin or x2 >= width - margin
             or y2 >= height - margin
             or (use_size and areas[i] < min_frac * median))
        for i, (_, x1, y1, x2, y2) in enumerate(dets)
    ]


class TrackState:
    def __init__(self, args, thresholds, events):
        self.args = args
        self.tau_p = float(thresholds["tau_p"])
        self.tau_f = float(thresholds.get("tau_f", self.tau_p))
        self.events = events
        self.active = OrderedDict()
        self.closed_ema = {}
        self.fired = {}
        self.early = {}
        self.early_count = 0
        self.alarm = self.persist = self.clear = 0

    def score(self, track):
        buf = track["buffer"]
        if len(buf) < self.args.min_samples:
            return None
        return float(max(buf) if len(buf) < 25 else np.median(buf))

    def event(self, record):
        self.events.write(json.dumps(record, allow_nan=False) + "\n")

    def finish_one(self, tid, fi, reason):
        track = self.active.pop(tid)
        score = self.score(track)
        fired = score is not None and score >= self.args.tau_track
        was_early = tid in self.early
        self.early.pop(tid, None)
        self.closed_ema[tid] = track["ema"]
        if fired:
            self.fired[tid] = score
            print(f"[Track] FIRED frame {fi}: ID{tid} score={score:.3f} "
                  f"(n={track['seen']}, {reason})")
        self.event({
            "event": "track_decision", "frame": fi, "track_id": tid,
            "reason": reason, "n_samples": track["seen"],
            "n_buffer": len(track["buffer"]), "score": score,
            "skipped": score is None, "fired": fired,
            "had_early_alert": was_early
        })

    def finish(self, fi, reason):
        for tid in list(self.active):
            self.finish_one(tid, fi, reason)
        self.events.flush()

    def step(self, fi, dets, clipped, raw):
        if len(dets) != len(clipped) or len(raw) != sum(not c for c in clipped):
            raise RuntimeError("Crop-score alignment error.")
        emas = [None] * len(dets)
        score_i = 0
        alpha = self.args.ema_alpha
        for i, ((tid, *_), is_clip) in enumerate(zip(dets, clipped)):
            if is_clip:
                continue
            s = float(raw[score_i])
            score_i += 1
            if tid in self.closed_ema:
                # A decided ID stays decided if ByteTrack returns it again.
                e = alpha * s + (1 - alpha) * self.closed_ema[tid]
                self.closed_ema[tid] = e
            else:
                track = self.active.get(tid)
                if track is None:
                    track = {"ema": s, "buffer": deque(maxlen=200),
                             "last": fi - 1, "seen": 0, "streak": 0}
                    self.active[tid] = track
                if fi != track["last"] + 1:
                    track["streak"] = 0
                e = alpha * s + (1 - alpha) * track["ema"]
                track["ema"], track["last"] = e, fi
                track["seen"] += 1
                track["buffer"].append(e)
                self.active.move_to_end(tid)
                if self.args.early_fire and tid not in self.early:
                    candidate = self.score(track)
                    qualifies = (candidate is not None
                                 and candidate >= self.args.tau_track
                                 and e >= self.args.tau_track)
                    track["streak"] = track["streak"] + 1 if qualifies else 0
                    if track["streak"] >= self.args.early_consecutive:
                        self.early[tid] = candidate
                        self.early_count += 1
                        self.event({
                            "event": "early_alert", "frame": fi,
                            "track_id": tid, "score": candidate,
                            "n_samples": track["seen"], "provisional": True
                        })
                        print(f"[Track] EARLY frame {fi}: ID{tid} "
                              f"score={candidate:.3f} (provisional)")
            emas[i] = e

        # Preserve original semantics: only scored, non-clipped sightings
        # refresh last. inactive_frames counts PROCESSED frames, not seconds.
        while self.active:
            tid = next(iter(self.active))
            if fi - self.active[tid]["last"] < self.args.inactive_frames:
                break
            self.finish_one(tid, fi, "inactive")

        valid = [e for e in emas if e is not None]
        frame_max = max(valid, default=0.0)
        frame_alarm = frame_max >= self.tau_f
        track_alarm = bool(self.fired)  # Original video-wide final-fire latch.
        early_alarm = bool(self.early)
        raw_alarm = frame_alarm or track_alarm or early_alarm
        if raw_alarm:
            self.persist += 1
            self.clear = 0
        else:
            self.clear += 1
            self.persist = 0
        previous_alarm = self.alarm
        if not self.alarm and self.persist >= self.args.persist_K:
            self.alarm = 1
        elif (self.alarm and self.clear >= self.args.clear_M
              and frame_max < self.args.gamma * self.tau_f):
            self.alarm = 0
        return {
            "frame": fi, "frame_max": frame_max,
            "n_crops": len(valid), "n_clipped": sum(clipped),
            "n_alarm_crops": sum(e >= self.tau_p for e in valid),
            "frame_alarm": frame_alarm, "track_alarm": track_alarm,
            "early_alarm": early_alarm, "raw_alarm": raw_alarm,
            "alarm": self.alarm, "n_fired_tracks": len(self.fired),
            "just_started": previous_alarm == 0 and self.alarm == 1,
            "just_ended": previous_alarm == 1 and self.alarm == 0
        }, emas


def score_chunk(chunk, scorer, pool, args):
    tasks, spans = [], []
    for _, frame, dets, clipped in chunk:
        start = len(tasks)
        if any(not c for c in clipped):
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            tasks.extend((pil, tuple(box))
                         for (_, *box), is_clip in zip(dets, clipped)
                         if not is_clip)
        spans.append((start, len(tasks)))

    def make_crop(task):
        return get_crop(task[0], task[1], scorer.image_size,
                        args.pad_ratio, args.denoise)

    scores = []
    # Bound GPU batches even when one frame contains more than batch_crops.
    for start in range(0, len(tasks), args.batch_crops):
        subset = tasks[start:start + args.batch_crops]
        crops = (list(pool.map(make_crop, subset)) if pool is not None
                 else [make_crop(t) for t in subset])
        scores.extend(scorer(crops, verify=args.verify_scores))
    return scores, spans


def draw_box(frame, box, label, color, thickness):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    (tw, th), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
    )
    tx = max(0, min(x1, frame.shape[1] - tw - 8))
    ty = max(th + 5, min(max(55, y1 - 8), frame.shape[0] - baseline - 4))
    cv2.rectangle(frame, (tx, ty - th - 5),
                  (tx + tw + 6, ty + baseline + 3), color, -1)
    cv2.putText(frame, label, (tx + 3, ty), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2, cv2.LINE_AA)


def draw_banner(frame, text, color):
    cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, 32), color, -1)
    cv2.putText(frame, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)


def draw_unknown(frame, dets, args):
    if args.detect_unknown:
        from compare_pipeline_methods_batched import draw_unknown_overlay
        draw_unknown_overlay(frame, frame, dets, args.otsu_max_area_frac,
                             args.otsu_min_unknown_area)


def annotate_preview(frame, dets, clipped, emas, state, record, args):
    """Optional GUI preview: decisions available at the current frame only."""
    out = frame.copy()
    draw_unknown(out, dets, args)
    for (tid, *box), clip, ema in zip(dets, clipped, emas):
        if tid in state.fired:
            color, thickness = RED, 4
            label = f"ID{tid} ANOMALY {state.fired[tid]:.3f}"
        elif clip:
            color, thickness, label = GREY, 2, f"ID{tid} CLIP"
        elif tid in state.early:
            color, thickness, label = RED, 3, f"ID{tid} EARLY {ema:.2f}"
        else:
            color = RED if ema >= state.tau_p else GREEN
            thickness, label = 2, f"ID{tid} {ema:.2f}"
        draw_box(out, box, label, color, thickness)
    draw_banner(
        out, f"PROCESSING | frame {record['frame']} | "
        f"fires={len(state.fired)} | early={len(state.early)}",
        RED if record["raw_alarm"] else GREEN
    )
    return out


def final_box_style(tid, is_clipped, fired_scores):
    """A confirmed track takes precedence over a clipped-box display flag."""
    if tid in fired_scores:
        return RED, 4, f"ID{tid} ANOMALY {fired_scores[tid]:.3f}"
    if is_clipped:
        return GREY, 2, f"ID{tid} CLIP"
    return GREEN, 2, f"ID{tid}"


def render_final_video(args, run_dir, fired_scores, web_server=None):
    """Replay source pixels and saved boxes; never rerun YOLO or EfficientAD."""
    output_path = run_dir / "annotated.mp4"
    partial_path = run_dir / "annotated.partial.mp4"
    cap = cv2.VideoCapture(args.video)
    writer = None
    rendered = 0
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot reopen source video: {args.video}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            fps = 25.0
        print("[Video] Rendering confirmed anomalous tracks in red...")
        with open(run_dir / "records.jsonl", encoding="utf-8") as records_file:
            for frame_index, line in enumerate(records_file):
                frame_started = time.perf_counter()
                record = json.loads(line)
                if record["frame"] != frame_index:
                    raise RuntimeError("Video and detection records are misaligned.")
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Cannot read source frame {frame_index}.")
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(partial_path), cv2.VideoWriter_fourcc(*"mp4v"),
                        fps, (width, height)
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Cannot create video: {partial_path}")
                dets = record.get("detections")
                clipped = record.get("clipped")
                if dets is None or clipped is None or len(dets) != len(clipped):
                    raise RuntimeError("Missing or inconsistent saved bounding boxes.")
                draw_unknown(frame, dets, args)
                visible_anomalies = set()
                for detection, is_clipped in zip(dets, clipped):
                    tid, x1, y1, x2, y2 = map(int, detection)
                    color, thickness, label = final_box_style(
                        tid, is_clipped, fired_scores
                    )
                    if tid in fired_scores:
                        visible_anomalies.add(tid)
                    draw_box(frame, (x1, y1, x2, y2), label, color, thickness)
                draw_banner(
                    frame, f"FINAL REVIEW | frame {frame_index} | "
                    f"anomaly IDs: {sorted(visible_anomalies)}",
                    RED if visible_anomalies else GREEN
                )
                writer.write(frame)
                rendered += 1
                if web_server is not None and args.web_final_replay:
                    web_frame = frame
                    if args.display_scale != 1:
                        web_frame = cv2.resize(
                            frame, None, fx=args.display_scale,
                            fy=args.display_scale,
                            interpolation=cv2.INTER_LINEAR
                        )
                    web_server.set_frame(web_frame)
                    # Only the optional final network replay is paced. Normal
                    # rendering stays as fast as the machine can write it.
                    remaining = (1.0 / fps) - (time.perf_counter() - frame_started)
                    if remaining > 0:
                        time.sleep(remaining)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if not rendered:
        raise RuntimeError("No frames available for final video rendering.")
    os.replace(partial_path, output_path)
    print(f"[Video] Saved {rendered} frames: {output_path.resolve()}")
    return output_path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True)
    p.add_argument("--yolo_det_weights", "--yolo_weights", required=True)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--thresholds_json", required=True)
    p.add_argument("--method", default="baseline",
                   choices=["baseline", "lowres", "thin", "qat", "pruned"],
                   help="Kept for backwards compatibility — auto-detected from --ckpt_dir.")
    p.add_argument("--category", default="chewinggum")
    p.add_argument("--model_size", choices=["S", "M"], default="S")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--fp16", action="store_true", help="EfficientAD FP16, as before.")
    p.add_argument("--yolo_fp16", action="store_true",
                   help="Separate opt-in; can change boxes/tracks. Revalidate.")
    p.add_argument("--tau_track", type=float, default=0.45)
    p.add_argument("--ema_alpha", type=float, default=0.3)
    p.add_argument("--inactive_frames", type=int, default=30)
    p.add_argument("--min_samples", type=int, default=10)
    p.add_argument("--early_fire", action="store_true",
                   help="Provisional early alerts; final decisions remain separate.")
    p.add_argument("--early_consecutive", type=int, default=3)
    p.add_argument("--persist_K", type=int, default=3)
    p.add_argument("--clear_M", type=int, default=5)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--pad_ratio", type=float, default=0.02)
    p.add_argument("--ratio", type=float, default=0.1)
    p.add_argument("--denoise", action="store_true")
    p.add_argument("--clip_margin", type=int, default=5)
    p.add_argument("--min_box_frac", type=float, default=0.5)
    p.add_argument("--batch_frames", type=int, default=16)
    p.add_argument("--batch_crops", type=int, default=96)
    p.add_argument("--crop_workers", type=int, default=12,
                   help="0 disables crop threads.")
    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--log_every", type=int, default=60,
                   help="0 disables progress logs.")
    p.add_argument("--save_dir", default="./video_outputs")
    p.add_argument("--no_video", action="store_true",
                   help="Skip final video rendering and bounding-box records.")
    p.add_argument("--display", action="store_true",
                   help="GUI preview of currently known decisions during processing.")
    p.add_argument("--display_scale", type=float, default=0.5)
    p.add_argument("--web_port", type=int, default=0,
                   help="MJPEG port; 0 disables network streaming.")
    p.add_argument("--web_host", default="127.0.0.1",
                   help="Use 0.0.0.0 for LAN access; keep 127.0.0.1 for SSH tunnels.")
    p.add_argument("--web_quality", type=int, default=75,
                   help="MJPEG JPEG quality from 1 to 100.")
    p.add_argument("--web_final_replay", action="store_true",
                   help="After inference, stream final red-box review at source FPS.")
    p.add_argument("--sleep_between_frames", type=float, default=0.0,
                   help="Optional processing-stream pacing delay in seconds.")
    p.add_argument("--hold_after_run", action="store_true",
                   help="Keep the last network frame available until Enter/Ctrl+C.")
    unknown = p.add_mutually_exclusive_group()
    unknown.add_argument("--detect_unknown", action="store_true",
                         help="Optional Otsu overlay, independent of EfficientAD.")
    unknown.add_argument("--no_detect_unknown", dest="detect_unknown", action="store_false")
    p.set_defaults(detect_unknown=False)
    p.add_argument("--otsu_max_area_frac", type=float, default=0.25)
    p.add_argument("--otsu_min_unknown_area", type=int, default=20)
    p.add_argument("--verify_scores", action="store_true",
                   help="Compare first real batch with original video_streaming_detector1.py.")
    a = p.parse_args()
    for name in ("batch_frames", "batch_crops", "early_consecutive", "inactive_frames",
                 "persist_K", "clear_M"):
        if getattr(a, name) < 1:
            p.error(f"--{name} must be at least 1")
    if not 1 <= a.min_samples <= 200:
        p.error("--min_samples must be between 1 and 200")
    for name in ("crop_workers", "max_frames", "clip_margin", "log_every", "web_port",
                 "otsu_min_unknown_area"):
        if getattr(a, name) < 0:
            p.error(f"--{name} cannot be negative")
    for name in ("tau_track", "ema_alpha", "gamma", "conf", "iou", "pad_ratio",
                 "ratio", "min_box_frac", "display_scale", "otsu_max_area_frac",
                 "sleep_between_frames"):
        if not math.isfinite(getattr(a, name)):
            p.error(f"--{name} must be finite")
    if not 0 < a.ema_alpha <= 1 or not 0 < a.gamma <= 1:
        p.error("--ema_alpha and --gamma must be in (0, 1]")
    if not 0 <= a.conf <= 1 or not 0 <= a.iou <= 1:
        p.error("--conf and --iou must be in [0, 1]")
    if (a.pad_ratio < 0 or a.min_box_frac < 0 or a.ratio <= 0
            or a.display_scale <= 0 or a.sleep_between_frames < 0):
        p.error("Check pad_ratio, min_box_frac, ratio, and display_scale.")
    if not 0 < a.otsu_max_area_frac <= 1:
        p.error("--otsu_max_area_frac must be in (0, 1]")
    if not 1 <= a.web_quality <= 100:
        p.error("--web_quality must be between 1 and 100")
    if a.web_final_replay and a.no_video:
        p.error("--web_final_replay cannot be combined with --no_video")
    if (a.web_final_replay or a.hold_after_run) and a.web_port == 0:
        p.error("--web_final_replay/--hold_after_run require --web_port")
    return a


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select --device cpu without FP16.")
    if args.yolo_fp16 and not args.device.startswith("cuda"):
        raise ValueError("--yolo_fp16 requires CUDA.")
    with open(args.thresholds_json, encoding="utf-8") as f:
        thresholds = json.load(f)
    tau_p = float(thresholds["tau_p"])
    tau_f = float(thresholds.get("tau_f", tau_p))
    if not (math.isfinite(tau_p) and math.isfinite(tau_f)):
        raise ValueError("tau_p and tau_f must be finite.")
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False

    with ExitStack() as stack:
        web_server = None
        if args.web_port:
            web_server = MJPEGServer(
                args.web_host, args.web_port, args.web_quality
            )
            web_server.start()
            stack.callback(web_server.stop)
        cap = cv2.VideoCapture(args.video)
        stack.callback(cap.release)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {args.video}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("[Run] Loading YOLO + one EfficientAD bundle...")
        yolo = YOLO(args.yolo_det_weights)
        yolo.to(args.device)
        scorer = Scorer(args)
        dummy = Image.new("RGB", (scorer.image_size, scorer.image_size))
        scorer([dummy] * min(4, args.batch_crops))
        print("[Run] EfficientAD warm-up complete.")
        if args.early_fire:
            print("[Run] Early alerts ON (provisional); final decisions remain separate.")

        root = Path(args.save_dir)
        root.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix="run_", dir=str(root)))
        records = stack.enter_context(open(run_dir / "records.jsonl", "w", encoding="utf-8"))
        events = stack.enter_context(open(run_dir / "events.jsonl", "w", encoding="utf-8"))
        with open(run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "thresholds": thresholds,
                       "image_size": scorer.image_size,
                       "video_annotation": "retrospective_final_track_verdicts"}, f, indent=2)
        state = TrackState(args, thresholds, events)
        pool = ThreadPoolExecutor(max_workers=args.crop_workers) if args.crop_workers else None
        if pool is not None:
            stack.callback(pool.shutdown, wait=True)
        if args.display:
            stack.callback(cv2.destroyAllWindows)

        t0 = time.perf_counter()
        processed = decoded = 0
        fallback_id = -1
        stopped = False
        reason = "end_of_video"

        def consume(chunk):
            nonlocal processed
            scores, spans = score_chunk(chunk, scorer, pool, args)
            for (fi, frame, dets, clipped), (start, end) in zip(chunk, spans):
                record, emas = state.step(fi, dets, clipped, scores[start:end])
                if not args.no_video:
                    # Small per-frame metadata permits exact later rendering
                    # without retaining the video or rerunning tracking.
                    record["detections"] = dets
                    record["clipped"] = clipped
                records.write(json.dumps(record, allow_nan=False) + "\n")
                processed += 1
                if args.display or web_server is not None:
                    view = annotate_preview(frame, dets, clipped, emas, state, record, args)
                    if args.display_scale != 1:
                        view = cv2.resize(view, None, fx=args.display_scale,
                                          fy=args.display_scale,
                                          interpolation=cv2.INTER_LINEAR)
                    if web_server is not None:
                        web_server.set_frame(view)
                if args.display:
                    cv2.imshow("YOLO + EfficientAD (q to stop)", view)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        return False
                if args.sleep_between_frames:
                    time.sleep(args.sleep_between_frames)
                if args.log_every and processed % args.log_every == 0:
                    rate = processed / max(time.perf_counter() - t0, 1e-9)
                    print(f"frame {fi} | visible={len(dets)} | "
                          f"fires={len(state.fired)} | fps={rate:.1f}")
                    records.flush()
                    events.flush()
            return True

        chunk, chunk_crops = [], 0
        try:
            while not args.max_frames or decoded < args.max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                fi = decoded
                decoded += 1
                if frame.shape[:2] != (height, width):
                    raise RuntimeError("Video resolution changed during processing.")
                results = yolo.track(
                    source=frame, persist=True, classes=[0],
                    tracker="bytetrack.yaml", conf=args.conf, iou=args.iou,
                    device=args.device, half=args.yolo_fp16, verbose=False
                )
                dets = []
                for result in results:
                    if result.boxes is None or len(result.boxes) == 0:
                        continue
                    boxes = result.boxes.cpu()
                    xyxy = boxes.xyxy.numpy().astype(int)
                    ids = boxes.id.numpy().astype(int) if boxes.id is not None else None
                    for i, (x1, y1, x2, y2) in enumerate(xyxy):
                        x1, y1 = max(0, int(x1)), max(0, int(y1))
                        x2, y2 = min(width, int(x2)), min(height, int(y2))
                        if x2 <= x1 or y2 <= y1:
                            continue
                        if ids is None:
                            tid = fallback_id
                            fallback_id -= 1
                        else:
                            tid = int(ids[i])
                        dets.append((tid, x1, y1, x2, y2))
                clipped = classify_boxes(
                    dets, width, height, args.clip_margin, args.min_box_frac
                )
                n_crops = len(clipped) - sum(clipped)
                if chunk and chunk_crops + n_crops > args.batch_crops:
                    if not consume(chunk):
                        stopped = True
                        break
                    chunk, chunk_crops = [], 0
                chunk.append((fi, frame, dets, clipped))
                chunk_crops += n_crops
                if len(chunk) >= args.batch_frames or chunk_crops >= args.batch_crops:
                    if not consume(chunk):
                        stopped = True
                        break
                    chunk, chunk_crops = [], 0
            else:
                reason = "max_frames"
            if not stopped and chunk:
                stopped = not consume(chunk)
        except KeyboardInterrupt:
            stopped = True

        if not stopped:
            # ALL processed crops must contribute before EOF finalization.
            state.finish(processed - 1, reason)
        else:
            reason = "interrupted"
            print("[Run] Interrupted: unfinished tracks were NOT finalized.")
        cap.release()
        records.flush()
        events.flush()
        processing_seconds = time.perf_counter() - t0

        output_video = None
        render_seconds = 0.0
        if not args.no_video and processed > 0:
            render_start = time.perf_counter()
            output_video = render_final_video(
                args, run_dir, state.fired, web_server=web_server
            )
            render_seconds = time.perf_counter() - render_start

        elapsed = time.perf_counter() - t0
        summary = {
            "reason": reason, "n_frames": processed,
            "processing_seconds": processing_seconds,
            "render_seconds": render_seconds,
            "elapsed_seconds": elapsed,
            "processing_fps": processed / max(processing_seconds, 1e-9),
            "fps": processed / max(elapsed, 1e-9),
            "fired_track_ids": sorted(state.fired),
            "final_scores_of_fired_tracks": state.fired,
            "early_alerts_emitted": state.early_count,
            "unfinished_tracks": len(state.active),
            "video_path": str(output_video.resolve()) if output_video else None,
            "video_annotation": "retrospective_confirmed_track_verdicts"
        }
        with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, allow_nan=False)
        print(f"[Run] {processed} frames | processing FPS={summary['processing_fps']:.2f} "
              "(decode + YOLO + EfficientAD + records; excludes startup)")
        print(f"[Run] Video render={render_seconds:.2f}s | "
              f"total FPS including render={summary['fps']:.2f}")
        print(f"[Run] Final fired IDs: {summary['fired_track_ids']}")
        print(f"[Run] Outputs: {run_dir.resolve()}")
        if web_server is not None and args.hold_after_run:
            try:
                input(f"[Web] Stream held at http://{args.web_host}:{args.web_port}/ "
                      "— press Enter or Ctrl+C to stop.\n")
            except (KeyboardInterrupt, EOFError):
                print("\n[Web] Stopping.")


if __name__ == "__main__":
    main()