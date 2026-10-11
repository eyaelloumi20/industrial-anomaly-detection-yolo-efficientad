"""
video_streaming_detector.py
===========================
Dual-model streaming detector for modes F (union) and G (intersection).

Speedups
--------
  * Model A and model C forward passes run on parallel CUDA streams.
  * Crop preprocessing for both sizes is submitted concurrently.
  * Larger per-chunk crop counts give the GPU more work to amortize
    kernel-launch overhead.

Short-track handling
--------------------
The --min_samples flag (default 10) controls the minimum buffer size
before a track can be scored.
  * n < min_samples          -> skipped entirely
  * min_samples <= n < 25    -> scored with max() instead of median()
                                 (median is unstable with few samples)
  * n >= 25                  -> scored with the configured mode

Usage
-----
 # F (union)
 python video_streaming_detector.py \
   --mode F \
   --video ./video_test/video_A2.mp4 \
   --yolo_det_weights ./runs/detect/.../best.pt \
   --ckpt_A /home/ahmed/framework/ckpt_cropped_large1 \
   --thresholds_A ./video_calibration_new/video_thresholds_baseline.json \
   --ckpt_C ./ckpt_192 \
   --thresholds_C ./video_calibration/video_thresholds_lowres.json \
   --category chewinggum --model_size S --inactive_frames 30 \
   --min_samples 10 --batch_frames 16 --batch_crops 96 --crop_workers 12 \
   --fp16 --device cuda:0 \
   --web_port 8080 --display_scale 0.5 --hold_after_run \
   --save_dir ./options/F/video_A2
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
import webbrowser
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

import utils.models as _models_module


def _dtype_preserving_norm(x):
    mean = torch.tensor([0.485, 0.456, 0.406],
                        dtype=x.dtype, device=x.device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225],
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


COLOR_RED   = (0, 0, 255)
COLOR_GREEN = (0, 200, 0)


MODE_CONFIGS = {
    "F": dict(use_A=True,  use_C=True,
              mode_A="median",    tau_A=1.335,
              mode_C="frame_rel", tau_C=0.373,
              combine="union",
              label="union"),
    "G": dict(use_A=True,  use_C=True,
              mode_A="median", tau_A=1.335,
              mode_C="median", tau_C=0.883,
              combine="intersection",
              label="intersection"),
}


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


class MJPEGServer:
    def __init__(self, port, jpeg_quality=75):
        self.port = port
        self.quality = jpeg_quality
        self._frame = None
        self._lock = threading.Lock()
        self._server = None
        self._thread = None

    def set_frame(self, frame_bgr):
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
                    html = (b"<!doctype html><html><head>"
                            b"<meta charset=utf-8><title>EfficientAD live</title>"
                            b"<style>html,body{margin:0;background:#111;height:100%}"
                            b"img{display:block;margin:0 auto;max-width:100vw;"
                            b"max-height:100vh;object-fit:contain}</style></head>"
                            b"<body><img src='/stream'></body></html>")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(html)))
                    self.end_headers()
                    self.wfile.write(html)
                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Age", "0")
                    self.send_header("Cache-Control", "no-cache, private")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Content-Type",
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
        self._server = HTTPServer(("0.0.0.0", self.port), self._make_handler())
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(f"[Web] MJPEG stream live at http://0.0.0.0:{self.port}/")

    def stop(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()


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
    y_st = s_out[:, :OUT_CHANNELS, :, :]
    y_stae = s_out[:, -OUT_CHANNELS:, :, :]
    norm_t = (t_out - models["mean"]) / (models["std"] + 1e-8)
    d_st = torch.pow(norm_t - y_st, 2)
    d_stae = torch.pow(a_out - y_stae, 2)
    fm_st = F.interpolate(torch.mean(d_st, dim=1, keepdim=True),
                          size=(image_size, image_size),
                          mode="bilinear", align_corners=False)
    fm_stae = F.interpolate(torch.mean(d_stae, dim=1, keepdim=True),
                            size=(image_size, image_size),
                            mode="bilinear", align_corners=False)
    fm_st, fm_stae = fm_st.float(), fm_stae.float()
    qa_st = models["qa_st"].float(); qb_st = models["qb_st"].float()
    qa_ae = models["qa_ae"].float(); qb_ae = models["qb_ae"].float()
    nm_st = (ratio * (fm_st - qa_st)) / (qb_st - qa_st + 1e-8)
    nm_ae = (ratio * (fm_stae - qa_ae)) / (qb_ae - qa_ae + 1e-8)
    amap = (0.5 * nm_st + 0.5 * nm_ae)[:, 0].cpu().numpy()
    return [float(np.max(amap[i])) for i in range(amap.shape[0])]


# ---------------------------------------------------------------------------
# Dual-model streaming detector
# ---------------------------------------------------------------------------
class DualStreamingDetector:
    def __init__(self, cfg, args, ckpt_A, json_A, ckpt_C, json_C,
                 src_w, src_h):
        self.cfg = cfg
        self.args = args
        self.src_w = src_w
        self.src_h = src_h

        # ---- Model A ----
        self.models_A = None
        self.image_size_A = None
        self.model_dtype_A = None
        self.tau_A = cfg["tau_A"]
        self.score_mode_A = cfg["mode_A"]

        with open(json_A) as f:
            thr_A = json.load(f)
        self.tau_p_A = float(thr_A["tau_p"])
        self.tau_f_A = float(thr_A.get("tau_f", thr_A["tau_p"]))

        print(f"[Load] Model A from {ckpt_A}")
        self.models_A, self.image_size_A, _ = load_bundle(
            "baseline", ckpt_A, args.category, args.model_size, args.device)
        self._setup_model(self.models_A, args.fp16, args.device)
        self.model_dtype_A = (torch.float16
                              if (args.fp16 and args.device.startswith("cuda"))
                              else torch.float32)

        # ---- Model C ----
        self.models_C = None
        self.image_size_C = None
        self.model_dtype_C = None
        self.tau_C = cfg["tau_C"]
        self.score_mode_C = cfg["mode_C"]

        with open(json_C) as f:
            thr_C = json.load(f)
        self.tau_p_C = float(thr_C["tau_p"])
        self.tau_f_C = float(thr_C.get("tau_f", thr_C["tau_p"]))

        print(f"[Load] Model C from {ckpt_C}")
        self.models_C, self.image_size_C, _ = load_bundle(
            "lowres", ckpt_C, args.category, args.model_size, args.device)
        self._setup_model(self.models_C, args.fp16, args.device)
        self.model_dtype_C = (torch.float16
                              if (args.fp16 and args.device.startswith("cuda"))
                              else torch.float32)

        # ---- Parallel CUDA streams ----
        if torch.cuda.is_available() and args.device.startswith("cuda"):
            self.stream_A = torch.cuda.Stream(device=args.device)
            self.stream_C = torch.cuda.Stream(device=args.device)
        else:
            self.stream_A = None
            self.stream_C = None

        # ---- Per-track state ----
        self.ema_A = {}
        self.ema_history_A = {}
        self.rel_history_A = {}
        self.ema_C = {}
        self.ema_history_C = {}
        self.rel_history_C = {}
        self.alpha = args.ema_alpha

        # ---- Track-end decision ----
        self.inactive_frames = int(args.inactive_frames)
        self.min_samples = int(args.min_samples)
        self.track_last_seen = {}
        self.track_score_A = {}
        self.track_score_C = {}
        self.decided_track_ids = set()
        self.fired_track_ids = set()

        self.frame_idx = 0
        self.records = []

        self._crop_pool = ThreadPoolExecutor(
            max_workers=max(1, args.crop_workers))
        self._last_chunk_crop_ms = 0.0
        self._last_chunk_fwd_ms = 0.0

    def _setup_model(self, models, fp16, device):
        use_fp16 = fp16 and device.startswith("cuda")
        if use_fp16:
            for key in ("teacher", "student", "ae"):
                models[key] = models[key].half()
            for key in ("mean", "std", "qa_st", "qb_st", "qa_ae", "qb_ae"):
                if isinstance(models.get(key), torch.Tensor):
                    models[key] = models[key].half()
            for key, kind in (("teacher", "teacher"),
                              ("student", "student"),
                              ("ae", "ae")):
                models[key] = patch_model_forward_dtype_safe(models[key], kind)
        for key in ("teacher", "student", "ae"):
            models[key] = models[key].to(memory_format=torch.channels_last)

    # -------------------------------------------------------------------
    def _live_score(self, tid, mode, ema_history, rel_history):
        """Compute the current cumulative score for a track.

        Track lifecycle:
          n <  min_samples       -> return None (skip)
          n <  25                -> use MAX (median unstable)
          n >= 25                -> use the requested mode
        """
        buf = ema_history.get(tid)
        if buf is None or len(buf) < self.min_samples:
            return None

        # Short tracks: use max (median is unstable with <25 samples).
        if len(buf) < 25:
            if mode == "frame_rel":
                rbuf = rel_history.get(tid)
                if rbuf is None or len(rbuf) < self.min_samples:
                    return None
                return float(np.max(rbuf))
            return float(np.max(buf))

        # Long tracks: use the requested mode.
        if mode == "median":
            return float(np.median(buf))
        if mode == "mean":
            return float(np.mean(buf))
        if mode == "max":
            return float(np.max(buf))
        if mode == "frame_rel":
            rbuf = rel_history.get(tid)
            if rbuf is None or len(rbuf) < 25:
                return None
            return float(np.median(rbuf))
        return None

    # -------------------------------------------------------------------
    def step_chunk(self, chunk):
        # ---- 1) Classify boxes once ----
        clipped_per_frame = []
        for (fi, frame_bgr, dets) in chunk:
            clipped = classify_boxes(
                dets, self.src_w, self.src_h,
                self.args.clip_margin, self.args.min_box_frac)
            clipped_per_frame.append(clipped)

        # ---- 2) Build ALL crop tasks for BOTH models ----
        all_tasks = []
        frame_slices_A = []
        frame_slices_C = []

        for fidx, (fi, frame_bgr, dets) in enumerate(chunk):
            clipped = clipped_per_frame[fidx]
            pil_frame = Image.fromarray(
                cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

            start_A = sum(1 for t in all_tasks if t[2] == 'A')
            start_C = sum(1 for t in all_tasks if t[2] == 'C')

            for (_tid, x1, y1, x2, y2), is_clip in zip(dets, clipped):
                if is_clip:
                    continue
                all_tasks.append((pil_frame, (x1, y1, x2, y2), 'A'))
                all_tasks.append((pil_frame, (x1, y1, x2, y2), 'C'))

            end_A = sum(1 for t in all_tasks if t[2] == 'A')
            end_C = sum(1 for t in all_tasks if t[2] == 'C')
            frame_slices_A.append((start_A, end_A))
            frame_slices_C.append((start_C, end_C))

        # ---- 3) Build crops CONCURRENTLY ----
        t_crop = time.perf_counter()
        if all_tasks:
            futures = [
                self._crop_pool.submit(
                    get_crop, pil, box,
                    self.image_size_A if tag == 'A' else self.image_size_C,
                    self.args.pad_ratio, self.args.denoise)
                for pil, box, tag in all_tasks
            ]
            results = [f.result() for f in futures]
            all_crops_A = [r for r, t in zip(results, all_tasks) if t[2] == 'A']
            all_crops_C = [r for r, t in zip(results, all_tasks) if t[2] == 'C']
        else:
            all_crops_A = []
            all_crops_C = []
        self._last_chunk_crop_ms = (time.perf_counter() - t_crop) * 1000.0

        # ---- 4) Forward passes on PARALLEL CUDA STREAMS ----
        t_fwd = time.perf_counter()
        if self.stream_A is not None and self.stream_C is not None:
            with torch.cuda.stream(self.stream_A):
                all_scores_A = _score_crop_batch(
                    self.models_A, all_crops_A, self.args.device,
                    self.image_size_A, self.model_dtype_A,
                    ratio=self.args.ratio)
            with torch.cuda.stream(self.stream_C):
                all_scores_C = _score_crop_batch(
                    self.models_C, all_crops_C, self.args.device,
                    self.image_size_C, self.model_dtype_C,
                    ratio=self.args.ratio)
            self.stream_A.synchronize()
            self.stream_C.synchronize()
        else:
            all_scores_A = _score_crop_batch(
                self.models_A, all_crops_A, self.args.device,
                self.image_size_A, self.model_dtype_A,
                ratio=self.args.ratio)
            all_scores_C = _score_crop_batch(
                self.models_C, all_crops_C, self.args.device,
                self.image_size_C, self.model_dtype_C,
                ratio=self.args.ratio)
        self._last_chunk_fwd_ms = (time.perf_counter() - t_fwd) * 1000.0

        # ---- 5) Per-frame state machine ----
        out = []
        for fidx, ((fi, _fb, dets), clipped) in enumerate(
                zip(chunk, clipped_per_frame)):

            aA, bA = frame_slices_A[fidx]
            aC, bC = frame_slices_C[fidx]
            raw_A = all_scores_A[aA:bA]
            raw_C = all_scores_C[aC:bC]

            emas_A = [None] * len(dets)
            emas_C = [None] * len(dets)

            si = 0
            for k, ((tid, *_box), is_clip) in enumerate(zip(dets, clipped)):
                if is_clip:
                    continue
                s = raw_A[si]
                prev = self.ema_A.get(tid, s)
                e = self.alpha * s + (1 - self.alpha) * prev
                self.ema_A[tid] = e
                emas_A[k] = e
                self.ema_history_A.setdefault(
                    tid, collections.deque(maxlen=200)).append(e)
                si += 1

            si = 0
            for k, ((tid, *_box), is_clip) in enumerate(zip(dets, clipped)):
                if is_clip:
                    continue
                s = raw_C[si]
                prev = self.ema_C.get(tid, s)
                e = self.alpha * s + (1 - self.alpha) * prev
                self.ema_C[tid] = e
                emas_C[k] = e
                self.ema_history_C.setdefault(
                    tid, collections.deque(maxlen=200)).append(e)
                si += 1

            vis_A = [e for e in emas_A if e is not None]
            vis_C = [e for e in emas_C if e is not None]
            fm_A = float(np.median(vis_A)) if vis_A else 0.0
            fm_C = float(np.median(vis_C)) if vis_C else 0.0

            for k, ((tid, *_box), is_clip) in enumerate(zip(dets, clipped)):
                if is_clip or emas_A[k] is None:
                    continue
                self.rel_history_A.setdefault(
                    tid, collections.deque(maxlen=200)).append(
                        emas_A[k] - fm_A)
            for k, ((tid, *_box), is_clip) in enumerate(zip(dets, clipped)):
                if is_clip or emas_C[k] is None:
                    continue
                self.rel_history_C.setdefault(
                    tid, collections.deque(maxlen=200)).append(
                        emas_C[k] - fm_C)

            for (tid, *_box), is_clip in zip(dets, clipped):
                if is_clip:
                    continue
                self.track_last_seen[tid] = fi

            # ---- Track-end decisions ----
            for tid, last_f in list(self.track_last_seen.items()):
                if tid in self.decided_track_ids:
                    continue
                if (fi - last_f) < self.inactive_frames:
                    continue
                sa = self._live_score(tid, self.score_mode_A,
                                      self.ema_history_A, self.rel_history_A)
                sc = self._live_score(tid, self.score_mode_C,
                                      self.ema_history_C, self.rel_history_C)
                sa = sa if sa is not None else 0.0
                sc = sc if sc is not None else 0.0
                self.track_score_A[tid] = sa
                self.track_score_C[tid] = sc

                fired_A = sa >= self.tau_A
                fired_C = sc >= self.tau_C
                combine = self.cfg["combine"]
                is_fired = (fired_A or fired_C) if combine == "union" \
                    else (fired_A and fired_C)

                self.decided_track_ids.add(tid)
                if is_fired:
                    self.fired_track_ids.add(int(tid))
                    print(f"[Track] FIRED frame {fi}: ID{tid} "
                          f"[{combine}] A={sa:.3f}{'>' if fired_A else ''} "
                          f"C={sc:.3f}{'>' if fired_C else ''}")

            fired_tracks = []
            for tid in sorted(self.fired_track_ids):
                fired_tracks.append({
                    "track_id": int(tid),
                    "score_A": self.track_score_A.get(tid, 0.0),
                    "score_C": self.track_score_C.get(tid, 0.0),
                })

            alarm = int(bool(self.fired_track_ids))
            rec = {"frame": fi, "alarm": alarm, "fired_tracks": fired_tracks}
            self.records.append(rec)
            out.append((rec, emas_A, emas_C, clipped, fired_tracks))

        self.frame_idx = chunk[-1][0] + 1 if chunk else self.frame_idx
        return out

    def finalize(self):
        for tid, last_f in list(self.track_last_seen.items()):
            if tid in self.decided_track_ids:
                continue
            sa = self._live_score(tid, self.score_mode_A,
                                  self.ema_history_A,
                                  self.rel_history_A) or 0.0
            sc = self._live_score(tid, self.score_mode_C,
                                  self.ema_history_C,
                                  self.rel_history_C) or 0.0
            self.track_score_A[tid] = sa
            self.track_score_C[tid] = sc
            fired_A = sa >= self.tau_A
            fired_C = sc >= self.tau_C
            is_fired = (fired_A or fired_C) \
                if self.cfg["combine"] == "union" \
                else (fired_A and fired_C)
            self.decided_track_ids.add(tid)
            if is_fired:
                self.fired_track_ids.add(int(tid))
                print(f"[Track] FIRED end: ID{tid} A={sa:.3f} C={sc:.3f}")


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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=list(MODE_CONFIGS.keys()))
    p.add_argument("--video", required=True)
    p.add_argument("--yolo_det_weights", required=True)
    p.add_argument("--ckpt_A", required=True)
    p.add_argument("--thresholds_A", required=True)
    p.add_argument("--ckpt_C", required=True)
    p.add_argument("--thresholds_C", required=True)
    p.add_argument("--category", default="chewinggum")
    p.add_argument("--model_size", default="S", choices=["S", "M"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ratio", type=float, default=0.1)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--pad_ratio", type=float, default=0.02)
    p.add_argument("--denoise", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--ema_alpha", type=float, default=0.3)
    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--batch_frames", type=int, default=16,
                   help="Frames per chunk.")
    p.add_argument("--batch_crops", type=int, default=96,
                   help="Hard cap on crops per chunk per model.")
    p.add_argument("--crop_workers", type=int, default=12)
    p.add_argument("--clip_margin", type=int, default=5)
    p.add_argument("--min_box_frac", type=float, default=0.5)
    p.add_argument("--inactive_frames", type=int, default=30)
    p.add_argument("--min_samples", type=int, default=10,
                   help="Minimum samples in the EMA buffer before a track "
                        "can be scored. Tracks with fewer samples are "
                        "skipped. 25 = old behavior (misses short defects), "
                        "10 = recommended.")
    p.add_argument("--no_detect_unknown", action="store_true")
    p.add_argument("--otsu_max_area_frac", type=float, default=0.25)
    p.add_argument("--otsu_min_unknown_area", type=int, default=20)
    p.add_argument("--no_video", action="store_true")
    p.add_argument("--save_dir", required=True)
    p.add_argument("--display", action="store_true")
    p.add_argument("--display_scale", type=float, default=0.5)
    p.add_argument("--web_port", type=int, default=0)
    p.add_argument("--no_open_browser", action="store_true")
    p.add_argument("--sleep_between_frames", type=float, default=0.0)
    p.add_argument("--hold_after_run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = MODE_CONFIGS[args.mode]

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"[Stream] mode={args.mode}  combine={cfg['combine']}")
    print(f"[Stream]   Model A: baseline  score={cfg['mode_A']}  "
          f"tau={cfg['tau_A']}")
    print(f"[Stream]   Model C: lowres    score={cfg['mode_C']}  "
          f"tau={cfg['tau_C']}")
    print(f"[Stream] batch_frames={args.batch_frames}  "
          f"batch_crops={args.batch_crops}  "
          f"crop_workers={args.crop_workers}")
    print(f"[Stream] min_samples={args.min_samples}  "
          f"inactive_frames={args.inactive_frames}")
    print(f"[Stream] Parallel CUDA streams: "
          f"{'ENABLED' if torch.cuda.is_available() else 'DISABLED'}")

    web_server = None
    if args.web_port > 0:
        web_server = MJPEGServer(args.web_port)
        web_server.start()
        if not args.no_open_browser:
            url = f"http://localhost:{args.web_port}/"
            try:
                if webbrowser.open(url, new=2):
                    print(f"[Web] Opened browser to {url}")
            except Exception as e:
                print(f"[Web] Browser open failed: {e}")

    detections, src_w, src_h, src_fps, n_frames, frame_cache = run_yolo_once(
        args.video, args.yolo_det_weights, args.conf, args.iou,
        device=args.device,
        max_frames=(args.max_frames or None),
        detect_unknown=not args.no_detect_unknown,
        otsu_max_area_frac=args.otsu_max_area_frac,
        cache_frames=True)

    det = DualStreamingDetector(
        cfg, args, args.ckpt_A, args.thresholds_A,
        args.ckpt_C, args.thresholds_C, src_w, src_h)
    colour = METHOD_COLOURS.get("baseline", (255, 255, 255))

    writer = None
    video_raw = None
    if not args.no_video:
        video_raw = os.path.join(
            args.save_dir, f"annotated_stream_{args.mode}_raw.mp4")
        writer = cv2.VideoWriter(video_raw,
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 src_fps, (src_w, src_h))

    K = max(1, args.batch_frames)

    # Warm-up
    try:
        warm_n = min(args.batch_crops, max(K * 4, 8))
        dummy = Image.new("RGB", (det.image_size_A, det.image_size_A))
        _ = _score_crop_batch(det.models_A, [dummy] * warm_n, args.device,
                              det.image_size_A, det.model_dtype_A,
                              ratio=args.ratio)
        dummy = Image.new("RGB", (det.image_size_C, det.image_size_C))
        _ = _score_crop_batch(det.models_C, [dummy] * warm_n, args.device,
                              det.image_size_C, det.model_dtype_C,
                              ratio=args.ratio)
        print(f"[Stream] Warm-up done (batch={warm_n}).")
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
        if ci == len(all_chunks) - 1:
            det.finalize()

        chunk = [(i, frame_cache.bgr_frames[i], detections.get(i, []))
                 for i in range(chunk_start, chunk_end)]
        chunk_results = det.step_chunk(chunk)

        for (i, frame_bgr, dets), (rec, emas_A, emas_C, clipped, fired) in zip(
                chunk, chunk_results):
            annot = frame_bgr.copy()
            mask = frame_cache.unknown_masks[i] if frame_cache else None
            if not args.no_detect_unknown and mask is not None:
                draw_unknown_contours(annot, mask,
                                      args.otsu_min_unknown_area)

            for (tid, x1, y1, x2, y2), is_clip in zip(dets, clipped):
                tid_int = int(tid)
                if tid_int in det.fired_track_ids:
                    sa = det.track_score_A.get(tid_int, 0.0)
                    sc = det.track_score_C.get(tid_int, 0.0)
                    label = f"ID{tid} FIRED A={sa:.2f} C={sc:.2f}"
                    cv2.rectangle(annot, (x1, y1), (x2, y2), COLOR_RED, 4)
                    cx = (x1 + x2) // 2
                    cy = max(14, y1 - 14)
                    cv2.circle(annot, (cx, cy), 9, COLOR_RED, -1)
                    cv2.circle(annot, (cx, cy), 9, (255, 255, 255), 2)
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    tx = max(0, x1 + 3)
                    ty = max(th + 10, y1 + 3)
                    cv2.rectangle(annot, (tx - 4, ty - th - 8),
                                  (tx + tw + 8, ty + 4), COLOR_RED, -1)
                    cv2.putText(annot, label, (tx, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    cv2.rectangle(annot, (x1, y1), (x2, y2), COLOR_GREEN, 2)
                    label = f"ID{tid}"
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    tx, ty = max(0, x1 + 3), max(th + 6, y1 + 3)
                    cv2.rectangle(annot, (tx - 2, ty - th - 4),
                                  (tx + tw + 4, ty + 2), COLOR_GREEN, -1)
                    cv2.putText(annot, label, (tx, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (255, 255, 255), 1, cv2.LINE_AA)

            fired_list = sorted(det.fired_track_ids)
            banner_c = COLOR_RED if fired_list else COLOR_GREEN
            banner = (f"mode={args.mode} ({cfg['combine']}) | frame {i} | "
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
        print(f" chunk [{chunk_start:5d}-{chunk_end-1:5d}] "
              f"({len(chunk)} frames) "
              f"| crop={det._last_chunk_crop_ms:6.1f}ms "
              f"fwd={det._last_chunk_fwd_ms:6.1f}ms "
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
            try:
                input(f"\n[Web] Server still at "
                      f"http://localhost:{args.web_port}/ -- press Enter.\n")
            except KeyboardInterrupt:
                pass
        web_server.stop()

    records_path = os.path.join(args.save_dir,
                                f"records_{args.mode}.jsonl")
    with open(records_path, "w") as f:
        for r in det.records:
            f.write(json.dumps(r) + "\n")

    dt = time.perf_counter() - t0
    print(f"\nDone. {n_processed} frames in {dt:.1f}s "
          f"({n_processed / dt:.1f} fps).")
    print(f" mode={args.mode} ({cfg['combine']})")
    print(f" min_samples={det.min_samples}")
    print(f" Track IDs FIRED : {sorted(det.fired_track_ids)}")
    if out_video:
        print(f" Video   : {out_video}")
    print(f" Records : {records_path}")

    det._crop_pool.shutdown(wait=False)


if __name__ == "__main__":
    main()