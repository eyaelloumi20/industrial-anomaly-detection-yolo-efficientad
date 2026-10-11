"""
combined_gum_inspection_pipeline.py
====================================
Standalone script version of combined_gum_inspection_pipeline_fixed.ipynb.

Runs one or more YOLO detect/segment + ByteTrack + EfficientAD methods over
a video, EMA-smooths per-track anomaly scores, overlays foreign-object
("UNKNOWN") detection, writes an annotated H.264 video + per-track/per-frame
CSVs + a timing report per method, and finishes with a single HTML
comparison report across all methods run.

Same pipeline logic as the notebook -- only the Colab-specific plumbing
(!pip installs, drive.mount, file upload/download widgets, inline video
playback) has been removed and replaced with CLI arguments + local output
files.

Usage (raw-crop + Otsu-BG methods, no segmentation):
    python combined_gum_inspection_pipeline.py \
        --video /path/to/video.mp4 \
        --yolo_det_weights /path/to/best_det.pt \
        --ckpt_dir /path/to/ckpt_cropped_large \
        --save_dir ./results \
        --device cuda

Add segmentation method:
    ... --yolo_seg_weights /path/to/best_seg.pt

Quick smoke-test on first 200 frames only, no video output:
    ... --max_frames 200 --no_video
"""

from __future__ import annotations

import os
import sys
import csv
import json
import time
import zipfile
import warnings
import argparse
import traceback
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless -- no display, just save PNGs
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from utils.load_efficientad import load_efficientad_model

IMAGE_SIZE   = 192
OUT_CHANNELS = 384


# ── dataclasses ─────────────────────────────────────────────────────────────

@dataclass
class TimingReport:
    """Time / inference evaluation for one method, saved as timing_<method>.json."""
    method_name:         str
    tracker:             str
    yolo_load_ms:        float
    efficientad_load_ms: float
    num_frames:          int
    num_tracks:          int
    avg_yolo_track_ms:   float
    avg_ead_ms_per_crop: float
    avg_total_ms_per_frame: float
    avg_fps:             float
    total_runtime_s:     float
    error:               Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


@dataclass
class VideoMethodResult:
    """Everything produced by run_video_method() for one pre-processing method."""
    method_name:       str
    category:          str          # 'detection' | 'bg_removal' | 'segmentation'
    frame_csv_path:    Optional[str] = None
    track_csv_path:    Optional[str] = None
    video_path:        Optional[str] = None
    timing_path:       Optional[str] = None
    timing:            Optional[TimingReport] = None
    track_rows:        List[Dict[str, Any]] = field(default_factory=list)
    error:             Optional[str] = None


# ── crop preprocessing ──────────────────────────────────────────────────────

def resize_with_padding(crop: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    """Aspect-ratio-preserving resize + black padding. Matches training pipeline."""
    ow, oh = crop.size
    scale  = size / max(ow, oh)
    nw, nh = int(ow * scale), int(oh * scale)
    resized = crop.resize((nw, nh), Image.LANCZOS)
    canvas  = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def get_crop(pil_img: Image.Image, box: tuple, pad_ratio: float = 0.02) -> Image.Image:
    """Crop with padding from a PIL image given an (x1,y1,x2,y2) box."""
    W, H = pil_img.size
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad_ratio), int(bh * pad_ratio)
    x1 = max(0, x1 - px);  y1 = max(0, y1 - py)
    x2 = min(W, x2 + px);  y2 = min(H, y2 + py)
    crop = pil_img.crop((x1, y1, x2, y2)).convert("RGB")
    return resize_with_padding(crop, IMAGE_SIZE)


# ── EfficientAD wrapper ─────────────────────────────────────────────────────

class EfficientADRunner:
    """Wraps load_efficientad_model + per-crop scoring, keeps load_ms separate.
    Produces a raw anomaly score only -- no threshold, no verdict."""

    def __init__(self):
        self.models      = None
        self.device      = "cpu"
        self.ratio       = 0.1
        self.load_ms     = 0.0
        self.fp16        = False
        self._available  = False
        self._error      = ""

    def load(self, ckpt_dir: str, category: str, model_size: str,
              device: str, ratio: float = 0.1, fp16: bool = False) -> bool:
        try:
            t0 = time.perf_counter()
            self.models = load_efficientad_model(
                ckpt_dir=ckpt_dir, category=category,
                model_size=model_size, device=device,
            )
            # Defensive: make sure the networks are in eval mode. If they were
            # accidentally left in train mode, BatchNorm/Dropout layers add
            # both extra compute AND run-to-run score noise -- eval() is a
            # safe no-op if load_efficientad_model already handled this.
            for key in ("teacher", "student", "ae"):
                m = self.models.get(key)
                if m is not None and hasattr(m, "eval"):
                    m.eval()

            self.fp16 = fp16 and device.startswith("cuda")
            # NOTE: models/stats are deliberately left in fp32 here. fp16 is
            # applied via torch.autocast around the forward pass in
            # score_batch() instead of permanently converting weights with
            # .half() -- manual .half() conversion turned out to be fragile
            # (some parameters didn't convert consistently, causing
            # "Input type (float) and bias type (c10::Half)" mismatches).
            # autocast handles per-op casting internally and can't desync.

            self.load_ms  = (time.perf_counter() - t0) * 1000
            self.device   = device
            self.ratio    = ratio
            self._available = True
            return True
        except Exception as e:
            self._available = False
            self._error = f"{type(e).__name__}: {e}"
            print(f"  [EfficientAD] load failed -> {self._error}")
            return False

    def score(self, crop_pil: Image.Image) -> Tuple[np.ndarray, float, float]:
        """Returns (anomaly_map [H,W], scalar_score, infer_ms). No thresholding."""
        amaps, scores, ms = self.score_batch([crop_pil])
        return amaps[0], scores[0], ms

    def score_batch(self, crops_pil: List[Image.Image]) -> Tuple[List[np.ndarray], List[float], float]:
        """Batched version: stacks all N crops into ONE forward pass through
        teacher/student/autoencoder instead of N separate GPU calls.

        This amortizes fixed kernel-launch/memory-transfer overhead across
        all detections in the frame instead of paying it once per crop.

        Numerically IDENTICAL to calling score() once per crop when fp16 is
        off (same ops, just batched) -- anomaly scores don't change, so any
        threshold calibrated against the old per-crop score() is still valid.
        With fp16 on, scores may drift very slightly (usually negligible,
        but re-verify your calibrated threshold before trusting it in fp16).

        Returns (list_of_amaps, list_of_scores, total_ms_for_the_whole_batch).
        """
        if not crops_pil:
            return [], [], 0.0

        t0 = time.perf_counter()
        tensors = [transforms.ToTensor()(c) for c in crops_pil]
        batch = torch.stack(tensors, dim=0).to(self.device)  # stays fp32

        # fp16 via autocast: PyTorch casts each eligible op (conv, matmul,
        # etc.) to fp16 on the fly for the duration of this block, without
        # ever permanently changing the stored weight dtype. This is the
        # standard, robust way to do mixed-precision inference -- unlike
        # manually calling .half() on the model (which turned out to leave
        # some parameters inconsistently converted and caused "Input type
        # (float) and bias type (c10::Half)" errors), autocast can't desync
        # because input and weights are never separately tracked dtypes.
        autocast_device = "cuda" if self.device.startswith("cuda") else "cpu"
        with torch.no_grad(), torch.autocast(device_type=autocast_device,
                                             dtype=torch.float16, enabled=self.fp16):
            t_out = self.models['teacher'](batch)
            s_out = self.models['student'](batch)
            a_out = self.models['ae'](batch)

            C = OUT_CHANNELS
            y_st   = s_out[:, :C,  :, :]
            y_stae = s_out[:, C:,  :, :]

            norm_t = (t_out - self.models['mean']) / (self.models['std'] + 1e-8)
            d_st   = torch.pow(norm_t - y_st,   2)
            d_stae = torch.pow(a_out  - y_stae, 2)

            fm_st   = F.interpolate(torch.mean(d_st,   dim=1, keepdim=True),
                                    size=(IMAGE_SIZE, IMAGE_SIZE),
                                    mode='bilinear', align_corners=False)
            fm_stae = F.interpolate(torch.mean(d_stae, dim=1, keepdim=True),
                                    size=(IMAGE_SIZE, IMAGE_SIZE),
                                    mode='bilinear', align_corners=False)

            nm_st = (self.ratio * (fm_st   - self.models['qa_st'])) / (self.models['qb_st'] - self.models['qa_st'] + 1e-8)
            nm_ae = (self.ratio * (fm_stae - self.models['qa_ae'])) / (self.models['qb_ae'] - self.models['qa_ae'] + 1e-8)

        amap_batch = (0.5 * nm_st + 0.5 * nm_ae)[:, 0].float().cpu().numpy()  # (N, H, W)
        total_ms = (time.perf_counter() - t0) * 1000

        amaps  = [amap_batch[i] for i in range(amap_batch.shape[0])]
        scores = [float(np.max(a)) for a in amaps]
        return amaps, scores, total_ms


# ── Otsu foreground / foreign-object detection ──────────────────────────────

def otsu_mask_global(gray: np.ndarray, max_area_frac: float) -> np.ndarray:
    """Single global Otsu threshold. Flips polarity if it flags the majority of the frame as foreground."""
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (mask.mean() / 255.0) > max_area_frac:
        mask = cv2.bitwise_not(mask)
    return mask


def clean_mask(mask: np.ndarray, kernel_size: int = 3, close_kernel_size: int = 7) -> np.ndarray:
    """Open (remove speckle noise) then close (fill small holes / bridge gaps)."""
    open_kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    close_kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    return mask


def interior_roi_mask(h: int, w: int,
                       margin_x_frac: float = 0.0,
                       margin_y_frac: float = 0.0) -> np.ndarray:
    """White interior rectangle, black margin band around all four sides.
    ANDing this with the unknown mask drops any foreign-object contour that
    lives in the excluded border (e.g. the conveyor's side rails)."""
    roi = np.zeros((h, w), np.uint8)
    mx = int(w * margin_x_frac)
    my = int(h * margin_y_frac)
    roi[my:h - my if my > 0 else h, mx:w - mx if mx > 0 else w] = 255
    return roi


def draw_unknown_contours(frame: np.ndarray, unknown_mask: np.ndarray, min_area: int) -> None:
    """Draws foreign-object ('UNKNOWN') contours in-place."""
    contours, _ = cv2.findContours(unknown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.putText(frame, "UNKNOWN", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)


def _rects_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    """True if two (x1, y1, x2, y2) rects intersect."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


def _place_label(x_left: int, preferred_ty: int, tw: int, th: int,
                  placed_rects: List[Tuple[int, int, int, int]],
                  frame_h: int) -> Tuple[int, Tuple[int, int, int, int]]:
    """Picks a y-position for a label that doesn't overlap any rect already placed
    this frame. Tries the preferred spot first, then steps straight down in
    label-height increments (dodging other labels on a crowded belt) until it
    finds a free slot, hits the bottom of the frame, or runs out of attempts."""
    step = th + 10
    ty = preferred_ty
    rect = (x_left - 2, ty - th - 4, x_left + tw + 4, ty + 4)
    for _ in range(8):
        if not any(_rects_overlap(rect, r) for r in placed_rects):
            return ty, rect
        ty += step
        if ty - th - 4 > frame_h:
            break
        rect = (x_left - 2, ty - th - 4, x_left + tw + 4, ty + 4)
    ty = preferred_ty
    rect = (x_left - 2, ty - th - 4, x_left + tw + 4, ty + 4)
    return ty, rect


def reencode_h264(input_path: str, output_path: str) -> bool:
    """Re-encode mp4v output to H.264 so it plays back everywhere (browsers,
    QuickTime, Windows Media Player -- most players can't open raw mp4v)."""
    import subprocess

    if not os.path.exists(input_path) or os.path.getsize(input_path) < 1024:
        print(f"  [WARN] reencode_h264: input missing or too small ({input_path}) -- "
              f"the mp4v file was likely never finalized (writer not released).")
        return False

    cmd = ["ffmpeg", "-y", "-i", input_path,
           "-vcodec", "libx264", "-pix_fmt", "yuv420p",
           "-loglevel", "error", output_path]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0 or not os.path.exists(output_path):
        print(f"  [WARN] ffmpeg re-encode failed (code {proc.returncode}): {proc.stderr.strip()[:500]}")
        return False
    return True


# ── per-method crop preprocessing ───────────────────────────────────────────

def preprocess_raw(crop_bgr: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """No pre-processing -- the tracked crop as-is."""
    return crop_bgr


def preprocess_otsu_bg(crop_bgr: np.ndarray, mask: Optional[np.ndarray] = None,
                       otsu_max_area_frac: float = 0.25) -> np.ndarray:
    """Per-crop Otsu foreground extraction: blacks out whatever Otsu calls background
    (typically the conveyor belt) inside the crop before it reaches EfficientAD."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    fg_mask = otsu_mask_global(gray, otsu_max_area_frac)
    fg_mask = clean_mask(fg_mask)
    return cv2.bitwise_and(crop_bgr, crop_bgr, mask=fg_mask)


def preprocess_seg_mask(crop_bgr: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Blacks out everything outside the YOLO-segmentation polygon for this instance.
    Falls back to the raw crop if no mask was produced (e.g. segmentation didn't fire)."""
    if mask is None:
        return crop_bgr
    if mask.ndim == 3:
        mask = mask[..., 0]
    bin_mask = (mask > 0).astype(np.uint8) * 255
    if bin_mask.shape[:2] != crop_bgr.shape[:2]:
        bin_mask = cv2.resize(bin_mask, (crop_bgr.shape[1], crop_bgr.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
    return cv2.bitwise_and(crop_bgr, crop_bgr, mask=bin_mask)


# ── main per-method video runner ────────────────────────────────────────────

def run_video_method(video_path: str,
                      yolo_weights: str,
                      method_name: str,
                      category: str,
                      preprocess_key: str,
                      ead: "EfficientADRunner",
                      save_dir: str,
                      use_masks: bool = False,
                      conf: float = 0.5,
                      iou: float = 0.45,
                      tracker_yaml: Optional[str] = "bytetrack.yaml",
                      ema_alpha: float = 0.5,
                      grace_period: int = 5,
                      max_frames: Optional[int] = None,
                      device: str = "cuda",
                      detect_unknown: bool = True,
                      otsu_max_area_frac: float = 0.25,
                      otsu_min_unknown_area: int = 20,
                      anomaly_display_threshold: float = 0.5,
                      pad_ratio: float = 0.02,
                      yolo_half: bool = False,
                      save_video: bool = True) -> VideoMethodResult:

    safe_name = method_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "").replace("__", "_")
    result = VideoMethodResult(method_name=method_name, category=category)
    tracker_label = tracker_yaml if tracker_yaml else "default (Ultralytics native)"

    try:
        from ultralytics import YOLO
        from ultralytics.trackers.basetrack import BaseTrack
        BaseTrack.reset_id()  # each method's track IDs start at 1 (counter is process-global otherwise)

        print(f"\n{'='*70}\n  Method: {method_name}  |  tracker: {tracker_label}\n{'='*70}")

        t0 = time.perf_counter()
        yolo = YOLO(yolo_weights)
        yolo.to(device if torch.cuda.is_available() else "cpu")
        yolo_load_ms = (time.perf_counter() - t0) * 1000
        print(f"  YOLO loaded in {yolo_load_ms:.1f} ms  ({yolo_weights})")

        preprocess_fn = PREPROCESSORS[preprocess_key]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
        src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  Source: {src_w}x{src_h} @ {src_fps:.1f} fps")

        video_writer = None
        video_out_path = None
        if save_video:
            video_out_path = os.path.join(save_dir, f"annotated_{safe_name}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(video_out_path, fourcc, src_fps, (src_w, src_h))
            if not video_writer.isOpened():
                print(f"  [WARN] Could not open video writer for {video_out_path}; skipping video output.")
                video_writer = None
                video_out_path = None

        frame_csv_path = os.path.join(save_dir, f"frame_scores_{safe_name}.csv")
        frame_csv = open(frame_csv_path, "w", newline="")
        frame_w = csv.writer(frame_csv)
        frame_w.writerow(["frame", "track_id", "raw_score", "ema_score",
                          "x1", "y1", "x2", "y2",
                          "yolo_track_ms", "ead_ms", "frame_total_ms"])

        ema_state, last_seen, raw_hist, ema_hist = {}, {}, {}, {}
        yolo_ms_log, ead_ms_log, total_ms_log = [], [], []
        ead_total_dets = 0   # total crops scored, across all batches -- needed
                              # because ead_ms_log now holds per-frame BATCH
                              # totals (one batched call/frame), not per-crop times

        frame_idx = 0
        run_t0 = time.perf_counter()

        # The whole loop is wrapped in try/finally so cap.release() /
        # frame_csv.close() / video_writer.release() ALWAYS run, even if
        # something raises partway through -- an mp4 whose writer is never
        # released has no finalized moov atom/trailer and will not play in
        # virtually any player.
        try:
            while True:
                ret, frame = cap.read()
                if not ret or (max_frames is not None and frame_idx >= max_frames):
                    break

                t_frame0 = time.perf_counter()
                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_max_score = 0.0
                active_ids = set()

                t_y0 = time.perf_counter()
                track_kwargs = dict(source=frame, conf=conf, iou=iou, classes=[0],
                                    persist=True, verbose=False, half=yolo_half)
                if tracker_yaml:
                    track_kwargs["tracker"] = tracker_yaml
                results = yolo.track(**track_kwargs)
                yolo_ms = (time.perf_counter() - t_y0) * 1000
                yolo_ms_log.append(yolo_ms)

                # -- Pass 1: gather + preprocess every crop in this frame, but
                # -- DON'T score yet -- collecting them lets us send the whole
                # -- frame's detections through EfficientAD in a single batched
                # -- forward pass instead of one GPU call per crop.
                frame_dets = []  # list of dicts: tid, box, proc_pil
                for r in results:
                    if r.boxes is None or len(r.boxes) == 0:
                        continue
                    boxes = r.boxes.xyxy.cpu().numpy().astype(int)
                    track_ids = (r.boxes.id.cpu().numpy().astype(int)
                                if r.boxes.id is not None else list(range(len(boxes))))
                    seg_masks = None
                    if use_masks and getattr(r, "masks", None) is not None:
                        seg_masks = r.masks.data.cpu().numpy()  # N x h x w

                    for i, (box, tid) in enumerate(zip(boxes, track_ids)):
                        x1 = max(0, box[0]); y1 = max(0, box[1])
                        x2 = min(src_w, box[2]); y2 = min(src_h, box[3])
                        if x2 <= x1 or y2 <= y1:
                            continue

                        active_ids.add(tid)
                        last_seen[tid] = frame_idx

                        crop_pil = get_crop(pil_frame, (x1, y1, x2, y2), pad_ratio)
                        crop_bgr = cv2.cvtColor(np.array(crop_pil), cv2.COLOR_RGB2BGR)

                        mask_crop = None
                        if use_masks and seg_masks is not None and i < len(seg_masks):
                            full_mask = (seg_masks[i] > 0.5).astype(np.uint8) * 255
                            full_mask = cv2.resize(full_mask, (src_w, src_h), interpolation=cv2.INTER_NEAREST)
                            mask_crop = full_mask[y1:y2, x1:x2]
                            mask_crop = np.array(resize_with_padding(Image.fromarray(mask_crop), IMAGE_SIZE))

                        if preprocess_key == "otsu":
                            proc_bgr = preprocess_fn(crop_bgr, mask_crop, otsu_max_area_frac=otsu_max_area_frac)
                        else:
                            proc_bgr = preprocess_fn(crop_bgr, mask_crop)
                        proc_pil = Image.fromarray(cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB))

                        frame_dets.append(dict(tid=tid, x1=x1, y1=y1, x2=x2, y2=y2, proc_pil=proc_pil))

                # -- Pass 2: one batched EfficientAD call for every detection
                # -- in this frame (instead of len(frame_dets) separate calls).
                if frame_dets:
                    if ead._available:
                        _, raw_scores, batch_ms = ead.score_batch([d["proc_pil"] for d in frame_dets])
                        ead_ms_log.append(batch_ms)               # per-frame batch cost
                        ead_total_dets += len(frame_dets)
                        per_det_ms = batch_ms / len(frame_dets)   # for the CSV / avg-per-crop stat
                    else:
                        raw_scores = [0.0] * len(frame_dets)
                        per_det_ms = 0.0

                    for det, raw in zip(frame_dets, raw_scores):
                        tid = det["tid"]
                        prev_ema = ema_state.get(tid, raw)
                        ema_score = ema_alpha * raw + (1 - ema_alpha) * prev_ema
                        ema_state[tid] = ema_score
                        frame_max_score = max(frame_max_score, ema_score)

                        raw_hist.setdefault(tid, []).append(raw)
                        ema_hist.setdefault(tid, []).append(ema_score)

                        frame_total_ms = (time.perf_counter() - t_frame0) * 1000
                        frame_w.writerow([frame_idx, tid, f"{raw:.6f}", f"{ema_score:.6f}",
                                          det["x1"], det["y1"], det["x2"], det["y2"],
                                          f"{yolo_ms:.1f}", f"{per_det_ms:.1f}", f"{frame_total_ms:.1f}"])

                for tid in list(ema_state.keys()):
                    if tid not in active_ids and frame_idx - last_seen.get(tid, 0) > grace_period:
                        ema_state.pop(tid, None)

                if video_writer is not None:
                    annotated = frame.copy()
                    try:
                        # -- Foreign-object ("UNKNOWN") overlay: global Otsu foreground minus
                        # -- every YOLO-known region (polygon if this method segments, else box).
                        if detect_unknown:
                            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            non_bg_mask = otsu_mask_global(gray_full, otsu_max_area_frac)
                            known_mask = np.zeros_like(non_bg_mask)
                            for r in results:
                                if r.boxes is None or len(r.boxes) == 0:
                                    continue
                                if use_masks and getattr(r, "masks", None) is not None:
                                    for poly in r.masks.xy:
                                        if len(poly) >= 3:
                                            cv2.fillPoly(known_mask, [poly.astype(np.int32)], 255)
                                else:
                                    for box in r.boxes.xyxy.cpu().numpy().astype(int):
                                        cv2.rectangle(known_mask, (box[0], box[1]), (box[2], box[3]), 255, -1)
                            unknown_mask = cv2.bitwise_and(non_bg_mask, cv2.bitwise_not(known_mask))
                            unknown_mask = clean_mask(unknown_mask)
                            # Drop anything sitting in the fixed left/right (rail) margin band --
                            # those are structure, not foreign objects on the belt.
                            unknown_mask = cv2.bitwise_and(unknown_mask, interior_roi_mask(src_h, src_w))
                            draw_unknown_contours(annotated, unknown_mask, otsu_min_unknown_area)

                        placed_label_rects: List[Tuple[int, int, int, int]] = []  # reset per frame

                        for r in results:
                            if r.boxes is None or len(r.boxes) == 0:
                                continue
                            boxes = r.boxes.xyxy.cpu().numpy().astype(int)
                            track_ids = (r.boxes.id.cpu().numpy().astype(int)
                                        if r.boxes.id is not None else list(range(len(boxes))))
                            polys = None
                            if use_masks and getattr(r, "masks", None) is not None:
                                polys = r.masks.xy  # list of (N_i, 2) float arrays, one per instance
                            confs = (r.boxes.conf.cpu().numpy()
                                     if getattr(r.boxes, "conf", None) is not None else None)

                            for i, (box, tid) in enumerate(zip(boxes, track_ids)):
                                x1b = max(0, box[0]); y1b = max(0, box[1])
                                x2b = min(src_w, box[2]); y2b = min(src_h, box[3])
                                score = ema_state.get(tid, 0.0)
                                conf_val = float(confs[i]) if (confs is not None and i < len(confs)) else None
                                is_anomaly = score >= anomaly_display_threshold
                                color = (0, 0, 220) if is_anomaly else (0, 200, 60)   # red / green

                                poly = polys[i] if (polys is not None and i < len(polys)) else None
                                drew_poly = False
                                if poly is not None and len(poly) >= 3 and np.isfinite(poly).all():
                                    try:
                                        pts = poly.astype(np.int32).reshape((-1, 1, 2))
                                        overlay = annotated.copy()
                                        cv2.fillPoly(overlay, [pts], color)
                                        cv2.addWeighted(overlay, 0.30, annotated, 0.70, 0, annotated)
                                        cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2)
                                        drew_poly = True
                                    except Exception:
                                        drew_poly = False
                                if not drew_poly:
                                    cv2.rectangle(annotated, (x1b, y1b), (x2b, y2b), color, 2)

                                if conf_val is not None:
                                    tag = f"ID{tid}  conf:{conf_val:.2f} | score:{score:.2f}"
                                else:
                                    tag = f"ID{tid}  score:{score:.2f}"
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                (tw, th), _ = cv2.getTextSize(tag, font, 0.5, 2)

                                tx = max(0, x1b + 4)
                                preferred_ty = max(th + 8, y1b + 4)
                                ty, label_rect = _place_label(tx, preferred_ty, tw, th,
                                                               placed_label_rects, src_h)
                                placed_label_rects.append(label_rect)

                                cv2.rectangle(annotated, (tx - 2, ty - th - 4), (tx + tw + 4, ty + 4), color, -1)
                                cv2.putText(annotated, tag, (tx, ty), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(annotated, f"{method_name} | frame {frame_idx}", (10, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
                    except Exception as draw_err:
                        if frame_idx % 30 == 0 or frame_idx < 5:
                            print(f"  [WARN] frame {frame_idx}: draw/annotate failed ({draw_err}); writing raw frame instead.")
                        annotated = frame.copy()
                    video_writer.write(annotated)

                t_frame1 = time.perf_counter()
                frame_ms = (t_frame1 - t_frame0) * 1000.0
                total_ms_log.append(frame_ms)
                cur_fps = 1000.0 / max(frame_ms, 1e-3)

                frame_idx += 1
                if frame_idx % 30 == 0:
                    print(f"    [{frame_idx:05d}] fps={cur_fps:.1f} max_score={frame_max_score:.4f} tracks={len(active_ids)}")
        finally:
            cap.release()
            frame_csv.close()
            if video_writer is not None:
                video_writer.release()
                # mp4v is unreliable outside OpenCV/VLC -- re-encode to H.264 so the
                # file that gets saved actually plays everywhere. Always runs, even
                # after a mid-loop exception, so a partial video still gets finalized.
                h264_path = video_out_path.replace(".mp4", "_h264.mp4")
                if reencode_h264(video_out_path, h264_path):
                    os.remove(video_out_path)
                    video_out_path = h264_path
                    print(f"  Annotated video saved (H.264) -> {video_out_path}")
                else:
                    print(f"  [WARN] H.264 re-encode failed, keeping mp4v -> {video_out_path}")

        total_runtime_s = time.perf_counter() - run_t0

        # ── Per-track score summary (no verdict -- scores only) ─────────────
        track_csv_path = os.path.join(save_dir, f"per_track_summary_{safe_name}.csv")
        track_rows = []
        with open(track_csv_path, "w", newline="") as sf:
            sw = csv.writer(sf)
            sw.writerow(["track_id", "total_frames",
                        "raw_min", "raw_max", "raw_mean",
                        "ema_min", "ema_max", "ema_mean",
                        "peak_ema_score", "peak_ema_frame"])
            for tid in sorted(raw_hist.keys()):
                raws, emas = raw_hist[tid], ema_hist[tid]
                n = len(emas)
                peak_frame = int(np.argmax(emas))
                row = dict(track_id=tid, total_frames=n,
                          raw_min=min(raws), raw_max=max(raws), raw_mean=float(np.mean(raws)),
                          ema_min=min(emas), ema_max=max(emas), ema_mean=float(np.mean(emas)),
                          peak_ema_score=max(emas), peak_ema_frame=peak_frame)
                track_rows.append(row)
                sw.writerow([tid, n,
                            f"{min(raws):.6f}", f"{max(raws):.6f}", f"{np.mean(raws):.6f}",
                            f"{min(emas):.6f}", f"{max(emas):.6f}", f"{np.mean(emas):.6f}",
                            f"{max(emas):.6f}", peak_frame])

        # ── Timing / inference evaluation report ────────────────────────────
        timing = TimingReport(
            method_name=method_name,
            tracker=tracker_label,
            yolo_load_ms=yolo_load_ms,
            efficientad_load_ms=ead.load_ms if ead._available else 0.0,
            num_frames=frame_idx,
            num_tracks=len(raw_hist),
            avg_yolo_track_ms=float(np.mean(yolo_ms_log)) if yolo_ms_log else 0.0,
            avg_ead_ms_per_crop=float(sum(ead_ms_log) / ead_total_dets) if ead_total_dets else 0.0,
            avg_total_ms_per_frame=float(np.mean(total_ms_log)) if total_ms_log else 0.0,
            avg_fps=float(1000.0 / np.mean(total_ms_log)) if total_ms_log else 0.0,
            total_runtime_s=total_runtime_s,
        )
        timing_path = os.path.join(save_dir, f"timing_{safe_name}.json")
        with open(timing_path, "w") as tf:
            json.dump(timing.to_dict(), tf, indent=2)

        result.frame_csv_path    = frame_csv_path
        result.track_csv_path    = track_csv_path
        result.video_path        = video_out_path
        result.timing_path       = timing_path
        result.timing            = timing
        result.track_rows        = track_rows

        print(f"  Done: {frame_idx} frames | {len(raw_hist)} tracks | {timing.avg_fps:.1f} fps avg")

    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        print(f"  [ERROR] {method_name}: {result.error}")
        traceback.print_exc()

    return result


PREPROCESSORS = {
    "raw":      preprocess_raw,
    "otsu":     preprocess_otsu_bg,
    "seg_mask": preprocess_seg_mask,
}


# ── HTML report ──────────────────────────────────────────────────────────────

def _fmt(v, nd=2):
    try:
        return f"{v:.{nd}f}"
    except Exception:
        return str(v)


def _timing_table(t: Optional[TimingReport]) -> str:
    if t is None:
        return "<p class='muted'>No timing data.</p>"
    rows = [
        ("Tracker",                    t.tracker),
        ("YOLO load time",             f"{_fmt(t.yolo_load_ms)} ms"),
        ("EfficientAD load time",      f"{_fmt(t.efficientad_load_ms)} ms"),
        ("Frames processed",           f"{t.num_frames}"),
        ("Unique tracks",              f"{t.num_tracks}"),
        ("Avg YOLO+track time/frame",  f"{_fmt(t.avg_yolo_track_ms)} ms"),
        ("Avg EfficientAD time/crop",  f"{_fmt(t.avg_ead_ms_per_crop)} ms"),
        ("Avg total time/frame",       f"{_fmt(t.avg_total_ms_per_frame)} ms"),
        ("Average FPS",                f"{_fmt(t.avg_fps)}"),
        ("Total runtime",              f"{_fmt(t.total_runtime_s)} s"),
    ]
    trs = "".join(f"<tr><td>{k}</td><td><b>{v}</b></td></tr>" for k, v in rows)
    return f"<table class='timing-table'>{trs}</table>"


def _track_table(rows: List[Dict[str, Any]], max_rows: int = 25) -> str:
    if not rows:
        return "<p class='muted'>No tracks.</p>"
    head = ("<tr><th>Track</th><th>Frames</th><th>Raw min/mean/max</th>"
            "<th>EMA min/mean/max</th><th>Peak EMA score</th><th>Peak frame</th></tr>")
    body = ""
    for r in rows[:max_rows]:
        body += (f"<tr><td>{r['track_id']}</td><td>{r['total_frames']}</td>"
                f"<td>{_fmt(r['raw_min'],3)} / {_fmt(r['raw_mean'],3)} / {_fmt(r['raw_max'],3)}</td>"
                f"<td>{_fmt(r['ema_min'],3)} / {_fmt(r['ema_mean'],3)} / {_fmt(r['ema_max'],3)}</td>"
                f"<td><b>{_fmt(r['peak_ema_score'],4)}</b></td><td>{r['peak_ema_frame']}</td></tr>")
    more = f"<p class='muted'>...and {len(rows) - max_rows} more tracks.</p>" if len(rows) > max_rows else ""
    return f"<table class='track-table'>{head}{body}</table>{more}"


CATEGORY_COLOURS = {"detection": "#3b82f6", "bg_removal": "#f59e0b", "segmentation": "#a855f7"}
CATEGORY_LABELS  = {"detection": "Detection", "bg_removal": "Otsu BG-Removal", "segmentation": "Segmentation"}

METHOD_EXPLANATIONS = {
    "YOLO-Detect + Track + EfficientAD (Raw Crop)": (
        "YOLO detects and tracks each gum piece with a rectangular box "
        "(ByteTrack keeps the same ID across frames). The box is cropped "
        "exactly as YOLO drew it -- no background removal -- and sent "
        "straight to EfficientAD, which returns a raw per-track anomaly "
        "score, EMA-smoothed across frames."
    ),
    "YOLO-Detect + Otsu BG-Removal + EfficientAD": (
        "Same detection + ByteTrack pipeline as the raw-crop method, but "
        "each crop is passed through a per-crop Otsu threshold first, "
        "blacking out the conveyor-belt background so EfficientAD only "
        "sees the gum piece itself."
    ),
    "YOLO-Segment + Mask Crop + EfficientAD": (
        "YOLO's segmentation head outputs a polygon per gum piece instead "
        "of a box; everything outside the polygon is blacked out before "
        "the crop reaches EfficientAD, so the mask follows the piece's "
        "actual shape rather than its rectangular bounding box."
    ),
}


def build_html_report(results: List[VideoMethodResult], save_dir: str, video_path: str) -> str:
    method_cards = ""
    for r in results:
        colour = CATEGORY_COLOURS.get(r.category, "#64748b")
        label  = CATEGORY_LABELS.get(r.category, r.category)
        explanation = METHOD_EXPLANATIONS.get(r.method_name, "")
        explanation_html = f"<p class='explanation'>{explanation}</p>" if explanation else ""
        if r.error:
            body = f"{explanation_html}<p class='error'>Failed: {r.error}</p>"
        else:
            body = (f"{explanation_html}"
                    f"{_timing_table(r.timing)}"
                    f"<h4>Per-track EfficientAD score summary</h4>{_track_table(r.track_rows)}")
        method_cards += (
            f"<section class='method-card'>"
            f"<div class='method-header' style='border-left:6px solid {colour}'>"
            f"<span class='badge' style='background:{colour}'>{label}</span>"
            f"<h3>{r.method_name}</h3></div>{body}</section>"
        )

    html = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>Detect + Track + EfficientAD Report</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; background:#0f172a; color:#e2e8f0; margin:0; padding:24px; }}
h1 {{ font-size:1.6rem; }} h2 {{ margin-top:2.5rem; }}
.muted {{ color:#94a3b8; font-style:italic; }}
.error {{ color:#f87171; font-weight:600; }}
.explanation {{ color:#cbd5e1; background:#0f172a; border:1px solid #334155; border-radius:8px;
               padding:10px 14px; margin:10px 0 16px; line-height:1.5; font-size:0.92rem; }}
.method-card {{ background:#1e293b; border-radius:12px; padding:18px 22px; margin-bottom:22px; }}
.method-header {{ padding-left:12px; display:flex; align-items:center; gap:10px; }}
.badge {{ color:#0f172a; font-weight:700; font-size:0.72rem; padding:3px 8px; border-radius:6px; text-transform:uppercase; }}
table {{ border-collapse:collapse; width:100%; font-size:0.85rem; }}
table td, table th {{ padding:6px 10px; border-bottom:1px solid #334155; text-align:left; }}
table th {{ color:#93c5fd; }}
</style></head>
<body>
<h1>YOLO Detect/Segment + Track + EfficientAD — Method Comparison</h1>
<p class='muted'>Source video: {os.path.basename(video_path)}. Scores are raw EfficientAD
anomaly-detection scores per track -- no anomaly/normal classification is applied. Each
video also carries an UNKNOWN foreign-object overlay from the Otsu-vs-known-region diff.</p>

<h2>Result</h2>
{method_cards}

</body></html>"""

    out_path = os.path.join(save_dir, "report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML report written -> {out_path}")
    return out_path


# ── CLI / main ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="YOLO detect/segment + track + EfficientAD video inspection pipeline "
                    "(standalone script version of combined_gum_inspection_pipeline.ipynb).")
    p.add_argument("--video",             required=True, help="Path to input video")
    p.add_argument("--yolo_det_weights",  required=True, help="Trained YOLO detection weights (boxes)")
    p.add_argument("--yolo_seg_weights",  default=None,
                   help="Trained YOLO segmentation weights (polygons). Omit to skip the "
                        "segmentation method entirely.")
    p.add_argument("--ckpt_dir",          required=True, help="EfficientAD checkpoint directory")
    p.add_argument("--category",          default="chewinggum")
    p.add_argument("--model_size",        default="S", choices=["S", "M"])
    p.add_argument("--image_size",        type=int, default=192,
                   help="EfficientAD crop resolution (must match the resolution the "
                        "checkpoint in --ckpt_dir was trained at, e.g. 192 or 256)")
    p.add_argument("--save_dir",          default="./results")
    p.add_argument("--conf",              type=float, default=0.5,
                   help="YOLO confidence threshold (default 0.5)")
    p.add_argument("--iou",               type=float, default=0.45)
    p.add_argument("--pad_ratio",         type=float, default=0.02)
    p.add_argument("--yolo_half",         action="store_true",
                   help="Run YOLO detection/tracking in fp16 for extra speed on GPU. "
                        "Does NOT affect EfficientAD scores/thresholds (those still run "
                        "fp32) -- only changes YOLO's own detection numerics slightly. "
                        "Off by default; try it if YOLO+tracking (not EfficientAD) is "
                        "your bottleneck per the timing_*.json breakdown.")
    p.add_argument("--ratio",             type=float, default=0.1,
                   help="EfficientAD normalisation ratio")
    p.add_argument("--device",            default="cuda")
    p.add_argument("--no_efficientad",    action="store_true",
                   help="Skip EfficientAD scoring entirely (scores will be 0.0 for every crop)")
    p.add_argument("--fp16",              action="store_true",
                   help="Run EfficientAD (teacher/student/autoencoder) in fp16 for extra "
                        "speed on GPU -- usually the biggest remaining speedup after "
                        "batching. Scores may drift very slightly from fp32; re-verify "
                        "your calibrated threshold still holds before trusting it in "
                        "production. Off by default.")

    # tracking / smoothing
    p.add_argument("--tracker_yaml",      default="bytetrack.yaml",
                   help="ByteTrack config for the detection/otsu methods "
                        "(segmentation method always uses Ultralytics' native tracker)")
    p.add_argument("--ema_alpha",         type=float, default=0.5,
                   help="EMA weight for score smoothing, higher = more reactive")
    p.add_argument("--grace_period",      type=int, default=5,
                   help="Frames to keep a track's EMA alive if briefly lost")
    p.add_argument("--anomaly_display_threshold", type=float, default=0.5,
                   help="EMA score >= this draws the box/label red in the annotated video "
                        "(display only -- not a calibrated threshold)")

    # Otsu foreign-object detection
    p.add_argument("--otsu_max_area_frac",    type=float, default=0.25,
                   help="Polarity guard: flip Otsu mask if this fraction of the frame is flagged foreground")
    p.add_argument("--otsu_min_unknown_area", type=int, default=20,
                   help="Ignore UNKNOWN contours smaller than this (px)")
    p.add_argument("--no_detect_unknown",     action="store_true",
                   help="Disable UNKNOWN foreign-object contour overlay")

    # speed / output control
    p.add_argument("--max_frames",   type=int, default=0, help="0 = full video")
    p.add_argument("--no_video",     action="store_true", help="Skip annotated video output (faster)")
    p.add_argument("--make_zip",     action="store_true",
                   help="Also zip everything in --save_dir at the end")

    return p.parse_args()


def main():
    global IMAGE_SIZE
    args = parse_args()
    IMAGE_SIZE = args.image_size
    os.makedirs(args.save_dir, exist_ok=True)
    print("Configuration set. Output ->", args.save_dir)
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")

    if not os.path.isfile(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if device.startswith("cuda"):
        # Every EfficientAD crop is a fixed 192x192 -- cuDNN can cache the
        # fastest conv algorithm for that shape instead of re-selecting one
        # on every call. Free speed for fixed-size repeated inference.
        torch.backends.cudnn.benchmark = True

    max_frames = args.max_frames if args.max_frames > 0 else None
    detect_unknown = not args.no_detect_unknown
    save_video = not args.no_video

    # ── Build method list ────────────────────────────────────────────────────
    METHODS = [
        dict(name="YOLO-Detect + Track + EfficientAD (Raw Crop)", category="detection",
             yolo_weights=args.yolo_det_weights, preprocess_key="raw", use_masks=False,
             tracker_yaml=args.tracker_yaml, detect_unknown=detect_unknown),
        dict(name="YOLO-Detect + Otsu BG-Removal + EfficientAD", category="bg_removal",
             yolo_weights=args.yolo_det_weights, preprocess_key="otsu", use_masks=False,
             tracker_yaml=args.tracker_yaml, detect_unknown=detect_unknown),
    ]
    if args.yolo_seg_weights and os.path.isfile(args.yolo_seg_weights):
        METHODS.append(
            dict(name="YOLO-Segment + Mask Crop + EfficientAD", category="segmentation",
                 yolo_weights=args.yolo_seg_weights, preprocess_key="seg_mask", use_masks=True,
                 tracker_yaml=None, detect_unknown=detect_unknown)
        )
    elif args.yolo_seg_weights:
        print(f"--yolo_seg_weights not found ({args.yolo_seg_weights}) -> skipping "
              "'YOLO-Segment + Mask Crop + EfficientAD' method.")

    print(f"{len(METHODS)} method(s) queued:")
    for m in METHODS:
        tname = m["tracker_yaml"] or "default (native)"
        print(f"  - {m['name']}  (tracker: {tname})")

    # ── Load EfficientAD once, reused across all methods ────────────────────
    ead = EfficientADRunner()
    if not args.no_efficientad:
        print("\nLoading EfficientAD ...")
        ead.load(ckpt_dir=args.ckpt_dir, category=args.category, model_size=args.model_size,
                 device=device, ratio=args.ratio, fp16=args.fp16)
        if args.fp16 and ead._available:
            print("  [EfficientAD] fp16 enabled -- scores may drift slightly vs a "
                  "threshold calibrated in fp32. Re-verify before trusting in production.")
    else:
        print("--no_efficientad set -> scores will be 0.0 for every crop.")

    results: List[VideoMethodResult] = []
    for m in METHODS:
        if not os.path.isfile(m["yolo_weights"]):
            print(f"[SKIP] {m['name']}: weights not found -> {m['yolo_weights']}")
            r = VideoMethodResult(method_name=m["name"], category=m["category"],
                                  error=f"weights not found: {m['yolo_weights']}")
            results.append(r)
            continue

        r = run_video_method(
            video_path=args.video,
            yolo_weights=m["yolo_weights"],
            method_name=m["name"],
            category=m["category"],
            preprocess_key=m["preprocess_key"],
            ead=ead,
            save_dir=args.save_dir,
            use_masks=m["use_masks"],
            conf=args.conf,
            iou=args.iou,
            tracker_yaml=m["tracker_yaml"],
            ema_alpha=args.ema_alpha,
            grace_period=args.grace_period,
            max_frames=max_frames,
            device=device,
            detect_unknown=m.get("detect_unknown", detect_unknown),
            otsu_max_area_frac=args.otsu_max_area_frac,
            otsu_min_unknown_area=args.otsu_min_unknown_area,
            anomaly_display_threshold=args.anomaly_display_threshold,
            pad_ratio=args.pad_ratio,
            yolo_half=args.yolo_half,
            save_video=save_video,
        )
        results.append(r)

    print(f"\n{'='*70}\nDone: {sum(1 for r in results if r.error is None)}/{len(results)} succeeded\n{'='*70}")

    # ── Timing comparison chart ──────────────────────────────────────────────
    timing_rows = [
        {"method": r.method_name,
         "avg_ms_per_frame": r.timing.avg_total_ms_per_frame,
         "avg_fps": r.timing.avg_fps}
        for r in results if r.timing is not None
    ]
    if not timing_rows:
        print("No timing data to plot (every method errored before completing).")
    else:
        timing_df = pd.DataFrame(timing_rows)
        print(timing_df.to_string(index=False))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].bar(timing_df["method"], timing_df["avg_ms_per_frame"], color="#4C72B0")
        axes[0].set_ylabel("ms / frame")
        axes[0].set_title("Average processing time per frame")
        axes[0].tick_params(axis='x', rotation=20)

        axes[1].bar(timing_df["method"], timing_df["avg_fps"], color="#55A868")
        axes[1].set_ylabel("FPS")
        axes[1].set_title("Achieved throughput (FPS)")
        axes[1].tick_params(axis='x', rotation=20)

        plt.tight_layout()
        chart_path = os.path.join(args.save_dir, "timing_comparison.png")
        plt.savefig(chart_path, dpi=150)
        plt.close(fig)
        print(f"Timing chart saved -> {chart_path}")

    # ── HTML report ───────────────────────────────────────────────────────────
    build_html_report(results, args.save_dir, args.video)

    # ── Optional zip of everything ───────────────────────────────────────────
    if args.make_zip:
        zip_path = os.path.join(os.path.dirname(args.save_dir.rstrip("/")) or ".",
                                 "video_comparison_results.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files_ in os.walk(args.save_dir):
                for file in files_:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, os.path.dirname(args.save_dir)))
        print(f"Created '{zip_path}' containing all CSVs, timing JSONs, videos, "
              f"the timing chart, and the HTML report.")

    video_results = [r for r in results if r.video_path and os.path.isfile(r.video_path)]
    if video_results:
        print("\nAnnotated videos:")
        for r in video_results:
            size_mb = os.path.getsize(r.video_path) / (1024 * 1024)
            print(f"  - {r.method_name}: {r.video_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()