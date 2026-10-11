"""
calibrate_threshold_all.py
===========================
Calibrates a per-method anomaly-score threshold from a FOLDER of videos that
show only normal pieces (no defects). Produces, for every checkpoint type
(baseline / lowres / thin / qat / pruned) that has a valid --*_dir:

    <save_dir>/<method>/threshold.txt              -- a single float
    <save_dir>/<method>/calibration_meta.json       -- compatibility metadata
    <save_dir>/<method>/score_distribution.png      -- score histogram + threshold line
    <save_dir>/<method>/normal_scores.csv           -- every raw/ema score collected

plus a top-level <save_dir>/calibration_summary.json.

This is designed to be a drop-in producer for the threshold files consumed by
compare_pipeline_methods.py's --*_threshold_file / check_calibration_compatibility
/ resolve_threshold machinery, so it deliberately mirrors that script's
preprocessing, scoring, and EMA logic exactly (same get_crop, score_crop,
load_bundle, fingerprint_ckpt_dir) rather than reimplementing it.

Why the meta fields matter (mirrors compare_pipeline_methods.py)
------------------------------------------------------------------
- HARD_COMPAT_FIELDS (category, model_size, image_size, ratio, yolo_weights):
  if any of these differ between calibration time and run time, the threshold
  is refused unless --force_threshold_mismatch is passed at run time.
- SOFT_COMPAT_FIELDS (conf, iou, pad_ratio, denoise, ema_alpha): mismatches
  only trigger a warning at run time.
- checkpoint_fingerprint: a hash of the checkpoint directory's file listing
  (same algorithm as compare_pipeline_methods.py's fingerprint_ckpt_dir), so
  "this checkpoint was retrained/replaced since it was calibrated" is caught
  even when the --*_dir path itself hasn't changed.
- recommended_basis / ema_reliable: EMA is a per-track running average over
  CONSECUTIVE frames. If this script is run with --sample_stride > 1 (to
  speed up calibration by skipping frames), the "EMA" computed here is over
  sparse, non-consecutive samples and is not representative of the EMA the
  live pipeline will compute -- so calibration automatically falls back to
  thresholding on RAW per-crop scores and marks ema_reliable=False,
  recommended_basis="raw". compare_pipeline_methods.py reads this field and
  verdicts against the matching score at run time instead of always
  assuming EMA.

Fairness / efficiency design
------------------------------
Exactly like compare_pipeline_methods.py's Phase 1 / Phase 2 split: YOLO
detection + ByteTrack tracking runs ONCE per video (cached per frame),
shared across every method. Each method then re-seeks to just those cached
frame indices and scores the SAME crops, so score-distribution differences
between methods reflect only the EfficientAD model, never detector
variance -- and YOLO's cost is paid once per video instead of once per
video per method.

Usage
-----
    python calibrate_threshold_all.py \
        --normal_dir /path/to/normal_videos/ \
        --yolo_det_weights /path/to/best.pt \
        --baseline_dir /home/ahmed/framework/ckpt_cropped_large \
        --lowres_dir ./ckpt_192 \
        --thin_dir ./ckpt_thin_025 \
        --qat_dir ./ckpt_qat \
        --pruned_dir ./ckpt_pruned_030 \
        --category chewinggum --model_size S \
        --percentile 99.5 \
        --save_dir ./calibration_output

Then, in compare_pipeline_methods.py:
    --thin_threshold_file ./calibration_output/thin/threshold.txt
    --pruned_threshold_file ./calibration_output/pruned/threshold.txt
    ...
"""

import os
import sys
import csv
import json
import time
import glob
import hashlib
import argparse
import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

OUT_CHANNELS = 384
IMAGE_SIZE_FALLBACK = 256

METHODS = ['baseline', 'lowres', 'thin', 'qat', 'pruned']

# Same tiering as compare_pipeline_methods.py -- kept identical on purpose so
# the two scripts always agree on what counts as a compatibility-breaking
# field. If you edit one list, edit the other.
HARD_COMPAT_FIELDS = ["category", "model_size", "ratio", "yolo_weights"]
SOFT_COMPAT_FIELDS = ["conf", "iou", "pad_ratio", "denoise", "ema_alpha"]

VIDEO_EXTS_DEFAULT = ".mp4,.avi,.mov,.mkv,.m4v"


# ── loader dispatch (identical to compare_pipeline_methods.py) ───────────────

def load_bundle(method: str, ckpt_dir: str, category: str, model_size: str, device: str):
    if method in ('baseline', 'lowres'):
        from utils.load_efficientad_adaptive import load_efficientad_model
        models = load_efficientad_model(ckpt_dir, category, model_size, device)
        return models, models.get('image_size', IMAGE_SIZE_FALLBACK), device
    elif method == 'thin':
        from utils.load_efficientad_thin import load_efficientad_model
        models = load_efficientad_model(ckpt_dir, category, model_size, device)
        return models, IMAGE_SIZE_FALLBACK, device
    elif method == 'qat':
        from utils.load_efficientad_qat import load_efficientad_model
        models = load_efficientad_model(ckpt_dir, category, model_size, device)
        return models, IMAGE_SIZE_FALLBACK, 'cpu'
    elif method == 'pruned':
        from utils.load_efficientad_pruned import load_efficientad_model
        models = load_efficientad_model(ckpt_dir, category, model_size, device)
        return models, IMAGE_SIZE_FALLBACK, device
    else:
        raise ValueError(f"Unknown method '{method}'")


def fingerprint_ckpt_dir(ckpt_dir: str) -> str:
    """Identical algorithm to compare_pipeline_methods.py's fingerprint_ckpt_dir
    -- MUST stay byte-for-byte identical, or fingerprints computed here will
    never match fingerprints compared against at run time."""
    if not ckpt_dir or not os.path.isdir(ckpt_dir):
        return ""
    h = hashlib.sha1()
    for root, _dirs, files in os.walk(ckpt_dir):
        for fn in sorted(files):
            fp = os.path.join(root, fn)
            try:
                st = os.stat(fp)
            except OSError:
                continue
            rel = os.path.relpath(fp, ckpt_dir)
            h.update(f"{rel}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16]


# ── crop preprocessing + scoring (identical to compare_pipeline_methods.py) ──

def resize_with_padding(crop: Image.Image, size: int) -> Image.Image:
    ow, oh = crop.size
    scale  = size / max(ow, oh)
    nw, nh = int(ow * scale), int(oh * scale)
    resized = crop.resize((nw, nh), Image.LANCZOS)
    canvas  = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def get_crop(pil_img: Image.Image, box: tuple, size: int,
            pad_ratio: float, denoise: bool) -> Image.Image:
    W, H = pil_img.size
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad_ratio), int(bh * pad_ratio)
    x1 = max(0, x1 - px);  y1 = max(0, y1 - py)
    x2 = min(W, x2 + px);  y2 = min(H, y2 + py)
    crop = pil_img.crop((x1, y1, x2, y2)).convert("RGB")
    crop = resize_with_padding(crop, size)
    if denoise:
        arr = np.array(crop, dtype=np.uint8)
        arr = cv2.GaussianBlur(arr, (3, 3), sigmaX=0.8, sigmaY=0.8)
        arr = cv2.bilateralFilter(arr, d=5, sigmaColor=15, sigmaSpace=15)
        crop = Image.fromarray(arr)
    return crop


@torch.no_grad()
def score_crop(models: dict, crop_pil: Image.Image, device: str,
               image_size: int, ratio: float = 0.1) -> float:
    t = transforms.ToTensor()(crop_pil).unsqueeze(0).to(device)
    t_out = models['teacher'](t)
    s_out = models['student'](t)
    a_out = models['ae'](t)

    y_st   = s_out[:, :OUT_CHANNELS,  :, :]
    y_stae = s_out[:, -OUT_CHANNELS:, :, :]
    norm_t = (t_out - models['mean']) / (models['std'] + 1e-8)
    d_st   = torch.pow(norm_t - y_st,   2)
    d_stae = torch.pow(a_out  - y_stae, 2)

    fm_st   = F.interpolate(torch.mean(d_st,   dim=1, keepdim=True),
                            size=(image_size, image_size), mode='bilinear', align_corners=False)
    fm_stae = F.interpolate(torch.mean(d_stae, dim=1, keepdim=True),
                            size=(image_size, image_size), mode='bilinear', align_corners=False)

    nm_st = (ratio * (fm_st   - models['qa_st'])) / (models['qb_st'] - models['qa_st'] + 1e-8)
    nm_ae = (ratio * (fm_stae - models['qa_ae'])) / (models['qb_ae'] - models['qa_ae'] + 1e-8)
    amap  = (0.5 * nm_st + 0.5 * nm_ae)[0, 0].cpu().numpy()
    return float(np.max(amap))


# ── find normal videos ────────────────────────────────────────────────────────

def find_videos(normal_dir: str, exts: List[str], recursive: bool) -> List[str]:
    exts = {e.lower() if e.startswith('.') else f'.{e.lower()}' for e in exts}
    found = []
    if recursive:
        for root, _dirs, files in os.walk(normal_dir):
            for fn in sorted(files):
                if os.path.splitext(fn)[1].lower() in exts:
                    found.append(os.path.join(root, fn))
    else:
        for fn in sorted(os.listdir(normal_dir)):
            fp = os.path.join(normal_dir, fn)
            if os.path.isfile(fp) and os.path.splitext(fn)[1].lower() in exts:
                found.append(fp)
    return sorted(found)


# ── Phase 1 (per video): YOLO once, shared across every method ───────────────

def detect_video(video_path: str, yolo, conf: float, iou: float,
                 stride: int, max_frames: int) -> Tuple[Dict[int, List[Tuple[int, int, int, int, int]]], int, int]:
    """Detect+track on every `stride`-th frame of one video. Returns
    (detections: {frame_idx: [(track_id,x1,y1,x2,y2), ...]}, n_frames_seen, n_frames_processed)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detections: Dict[int, List[Tuple[int, int, int, int, int]]] = {}
    frame_idx = 0
    processed = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            if max_frames and processed >= max_frames:
                break
            results = yolo.track(source=frame, conf=conf, iou=iou, classes=[0],
                                 persist=True, tracker="bytetrack.yaml", verbose=False)
            boxes_this_frame = []
            for r in results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
                ids  = (r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None
                       else list(range(len(xyxy))))
                for box, tid in zip(xyxy, ids):
                    x1 = max(0, box[0]); y1 = max(0, box[1])
                    x2 = min(src_w, box[2]); y2 = min(src_h, box[3])
                    if x2 > x1 and y2 > y1:
                        boxes_this_frame.append((int(tid), x1, y1, x2, y2))
            detections[frame_idx] = boxes_this_frame
            processed += 1
        frame_idx += 1

    cap.release()
    return detections, frame_idx, processed


# ── Phase 2 (per video, per method): re-seek cached frames, score crops ──────

def score_video_for_method(video_path: str, detections: Dict[int, List[Tuple[int, int, int, int, int]]],
                           models: dict, device: str, image_size: int, args) -> Tuple[List[float], List[float], List[float]]:
    """Returns (raw_scores, ema_scores, per_crop_ms) for one video, one method.
    EMA state (ema_hist) is local to this video/track set -- tracks never
    carry across videos."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    raw_scores: List[float] = []
    ema_scores: List[float] = []
    per_crop_ms: List[float] = []
    ema_hist: Dict[int, float] = {}
    alpha = args.ema_alpha

    for frame_idx in sorted(detections.keys()):
        frame_dets = detections[frame_idx]
        if not frame_dets:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        for tid, x1, y1, x2, y2 in frame_dets:
            crop = get_crop(pil_frame, (x1, y1, x2, y2), image_size,
                            args.pad_ratio, args.denoise)
            t0 = time.perf_counter()
            score = score_crop(models, crop, device, image_size, ratio=args.ratio)
            per_crop_ms.append((time.perf_counter() - t0) * 1000.0)

            prev = ema_hist.get(tid, score)
            ema = alpha * score + (1 - alpha) * prev
            ema_hist[tid] = ema

            raw_scores.append(score)
            ema_scores.append(ema)

    cap.release()
    return raw_scores, ema_scores, per_crop_ms


# ── threshold computation ─────────────────────────────────────────────────────

def compute_threshold(scores: List[float], calibration_method: str,
                      percentile: float, k_std: float) -> float:
    arr = np.asarray(scores, dtype=np.float64)
    if calibration_method == "percentile":
        return float(np.percentile(arr, percentile))
    elif calibration_method == "mean_std":
        return float(arr.mean() + k_std * arr.std())
    elif calibration_method == "max":
        return float(arr.max())
    else:
        raise ValueError(f"Unknown --calibration_method '{calibration_method}'")


def save_histogram(raw_scores: List[float], ema_scores: List[float],
                   threshold: float, basis: str, method: str, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chosen = ema_scores if basis == "ema" else raw_scores
    other  = raw_scores if basis == "ema" else ema_scores
    other_label = "raw" if basis == "ema" else "ema"

    fig, ax = plt.subplots(figsize=(7, 4), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.hist(chosen, bins=60, color="#e94560", alpha=0.85, label=f"{basis} (used for threshold)")
    ax.hist(other, bins=60, color="#60a5fa", alpha=0.35, label=other_label)
    ax.axvline(threshold, color="#facc15", linestyle="--", linewidth=2,
              label=f"threshold = {threshold:.4f}")
    ax.set_title(f"{method} — normal-video score distribution", color="#eaeaea")
    ax.set_xlabel("EfficientAD score", color="#eaeaea")
    ax.set_ylabel("count", color="#eaeaea")
    ax.tick_params(colors="#eaeaea")
    for s in ax.spines.values():
        s.set_edgecolor("#444")
    leg = ax.legend(facecolor="#1a1a2e", edgecolor="#444")
    for text in leg.get_texts():
        text.set_color("#eaeaea")
    plt.tight_layout()
    plt.savefig(path, dpi=120, facecolor="#1a1a2e")
    plt.close()


# ── args ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Calibrate per-method anomaly thresholds from a folder of normal-piece videos. "
                    "Output is fully compatible with compare_pipeline_methods.py's "
                    "--*_threshold_file / calibration_meta.json compatibility checker.")
    p.add_argument("--normal_dir",        required=True,
                   help="Folder containing videos that show ONLY normal (defect-free) pieces.")
    p.add_argument("--video_exts",        default=VIDEO_EXTS_DEFAULT,
                   help=f"Comma-separated video extensions to include (default: {VIDEO_EXTS_DEFAULT})")
    p.add_argument("--recursive",         action="store_true",
                   help="Search --normal_dir recursively for videos (default: top-level only).")
    p.add_argument("--yolo_det_weights",  required=True)

    p.add_argument("--baseline_dir", default="/home/ahmed/framework/ckpt_cropped_large")
    p.add_argument("--lowres_dir",   default="./ckpt_192")
    p.add_argument("--thin_dir",     default="./ckpt_thin_025")
    p.add_argument("--qat_dir",      default="./ckpt_qat")
    p.add_argument("--pruned_dir",   default="./ckpt_pruned_030")
    p.add_argument("--skip_methods", nargs="*", default=[],
                   help="Method names to skip even if their --*_dir exists, e.g. --skip_methods qat")

    # These MUST match what you intend to run compare_pipeline_methods.py with --
    # they're written into calibration_meta.json and checked against at run time.
    p.add_argument("--category",     default="chewinggum")
    p.add_argument("--model_size",   default="S", choices=["S", "M"])
    p.add_argument("--device",       default="cuda",
                   help="Used for baseline/lowres/thin/pruned. qat always forces cpu regardless.")
    p.add_argument("--ratio",        type=float, default=0.1)
    p.add_argument("--conf",         type=float, default=0.25)
    p.add_argument("--iou",          type=float, default=0.45)
    p.add_argument("--pad_ratio",    type=float, default=0.02)
    p.add_argument("--denoise",      action="store_true",
                   help="Apply the same Gaussian+bilateral crop denoising compare_pipeline_methods.py "
                        "would apply. Must match the run-time flag or you'll get a SOFT mismatch warning.")
    p.add_argument("--ema_alpha",    type=float, default=0.3)

    p.add_argument("--sample_stride", type=int, default=1,
                   help="Process every Nth frame per video (speeds up calibration on long videos). "
                        "sample_stride>1 breaks EMA continuity (EMA is only meaningful over consecutive "
                        "frames), so calibration automatically falls back to thresholding on RAW scores "
                        "and writes recommended_basis='raw' into calibration_meta.json.")
    p.add_argument("--max_frames_per_video", type=int, default=0,
                   help="Cap processed (post-stride) frames per video, 0 = no cap.")

    p.add_argument("--calibration_method", choices=["percentile", "mean_std", "max"], default="percentile",
                   help="percentile: threshold = --percentile-th percentile of normal scores (default). "
                        "mean_std: threshold = mean + --k_std * std. max: threshold = max observed score.")
    p.add_argument("--percentile",   type=float, default=99.5,
                   help="Used when --calibration_method percentile. Leaves roughly (100-p)%% of normal "
                        "crops above threshold on the calibration set itself -- pick lower for a stricter "
                        "(more sensitive) threshold, higher for a looser one.")
    p.add_argument("--k_std",        type=float, default=3.0,
                   help="Used when --calibration_method mean_std.")
    p.add_argument("--min_samples",  type=int, default=200,
                   help="Warn if fewer than this many crops were collected for a method -- "
                        "a threshold from too few normal samples is unreliable.")

    p.add_argument("--no_plots",     action="store_true", help="Skip saving score_distribution.png per method")
    p.add_argument("--save_dir",     default="./calibration_output")
    return p.parse_args()


def build_calibration_meta(args, method: str, image_size: int, ckpt_dir: str,
                           ema_reliable: bool, recommended_basis: str,
                           n_videos: int, n_raw: int, n_ema: int,
                           raw_threshold: float, ema_threshold: float,
                           chosen_threshold: float) -> dict:
    return {
        # HARD fields -- must match at run time or the threshold is refused
        "category":   args.category,
        "model_size": args.model_size,
        "image_size": image_size,
        "ratio":      args.ratio,
        "yolo_weights": os.path.basename(args.yolo_det_weights),
        # SOFT fields -- mismatch only warns at run time
        "conf":       args.conf,
        "iou":        args.iou,
        "pad_ratio":  args.pad_ratio,
        "denoise":    args.denoise,
        "ema_alpha":  args.ema_alpha,
        # checkpoint drift detection
        "checkpoint_fingerprint": fingerprint_ckpt_dir(ckpt_dir),
        # basis selection
        "ema_reliable":      ema_reliable,
        "recommended_basis": recommended_basis,
        # provenance / diagnostics (not read by the compatibility checker,
        # but useful when auditing a threshold later)
        "method":               method,
        "sample_stride":        args.sample_stride,
        "calibration_method":   args.calibration_method,
        "percentile":           args.percentile if args.calibration_method == "percentile" else None,
        "k_std":                args.k_std if args.calibration_method == "mean_std" else None,
        "n_normal_videos":      n_videos,
        "n_raw_samples":        n_raw,
        "n_ema_samples":        n_ema,
        "raw_threshold":        raw_threshold,
        "ema_threshold":        ema_threshold,
        "threshold":            chosen_threshold,
        "calibrated_at":        datetime.datetime.now().isoformat(timespec="seconds"),
        "normal_dir":           os.path.abspath(args.normal_dir),
        "generated_by":         "calibrate_threshold_all.py",
    }


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if not os.path.isdir(args.normal_dir):
        raise SystemExit(f"--normal_dir not found: {args.normal_dir}")

    exts = [e.strip() for e in args.video_exts.split(",") if e.strip()]
    videos = find_videos(args.normal_dir, exts, args.recursive)
    if not videos:
        raise SystemExit(f"No videos found in {args.normal_dir} with extensions {exts} "
                         f"(pass --recursive if they're in subfolders).")
    print(f"Found {len(videos)} normal-piece video(s) in {args.normal_dir}:")
    for v in videos:
        print(f"  - {v}")

    if args.sample_stride > 1:
        print(f"\n[NOTE] --sample_stride={args.sample_stride} > 1: EMA computed during calibration "
              f"will be over non-consecutive frames and is NOT representative of run-time EMA. "
              f"All methods will be calibrated on RAW scores (recommended_basis='raw').")
    ema_reliable = (args.sample_stride == 1)
    recommended_basis = "ema" if ema_reliable else "raw"

    ckpt_dirs = {
        'baseline': args.baseline_dir, 'lowres': args.lowres_dir,
        'thin':     args.thin_dir,     'qat':    args.qat_dir,
        'pruned':   args.pruned_dir,
    }
    active_methods = [m for m in METHODS
                      if m not in args.skip_methods and os.path.isdir(ckpt_dirs[m])]
    skipped = [m for m in METHODS if m not in active_methods]
    print(f"\nActive methods : {active_methods}")
    if skipped:
        print(f"Skipped        : {skipped}  (missing --*_dir or in --skip_methods)")
    if not active_methods:
        raise SystemExit("No checkpoint directories found. Nothing to calibrate.")

    # ── Phase 1: YOLO once per video, shared across every method ─────────────
    from ultralytics import YOLO
    print(f"\n[Phase 1] Loading YOLO: {args.yolo_det_weights}")
    yolo_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    yolo = YOLO(args.yolo_det_weights)
    yolo.to(yolo_device)

    detections_by_video: Dict[str, Dict[int, List[Tuple[int, int, int, int, int]]]] = {}
    for video_path in videos:
        t0 = time.perf_counter()
        dets, n_seen, n_proc = detect_video(video_path, yolo, args.conf, args.iou,
                                            args.sample_stride, args.max_frames_per_video)
        n_boxes = sum(len(v) for v in dets.values())
        print(f"  [Phase 1] {os.path.basename(video_path)}: {n_seen} frames seen, "
              f"{n_proc} processed, {n_boxes} detections  ({time.perf_counter()-t0:.1f}s)")
        detections_by_video[video_path] = dets

    total_boxes = sum(sum(len(v) for v in d.values()) for d in detections_by_video.values())
    if total_boxes == 0:
        raise SystemExit("YOLO found zero detections across every normal video -- check "
                         "--yolo_det_weights / --conf / --iou before calibrating.")

    # ── Phase 2: score with each method ───────────────────────────────────────
    summary = {}
    for method in active_methods:
        print(f"\n{'='*70}\n  Calibrating: {method}  (ckpt={ckpt_dirs[method]})\n{'='*70}")
        try:
            models, image_size, device = load_bundle(
                method, ckpt_dirs[method], args.category, args.model_size, args.device)
        except Exception as e:
            print(f"  [{method}] FAILED to load: {e}")
            summary[method] = {"error": str(e)}
            continue

        all_raw: List[float] = []
        all_ema: List[float] = []
        all_ms:  List[float] = []
        for video_path in videos:
            raw, ema, ms = score_video_for_method(
                video_path, detections_by_video[video_path], models, device, image_size, args)
            all_raw.extend(raw)
            all_ema.extend(ema)
            all_ms.extend(ms)
            print(f"  [{method:<10s}] {os.path.basename(video_path)}: {len(raw)} crops scored")

        del models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not all_raw:
            print(f"  [{method}]   [ERROR] zero crops scored -- skipping (no threshold written).")
            summary[method] = {"error": "zero crops scored"}
            continue
        if len(all_raw) < args.min_samples:
            print(f"  [{method}]   [WARN] only {len(all_raw)} crops collected "
                  f"(< --min_samples={args.min_samples}). Threshold may be unreliable -- "
                  f"add more normal videos or lower --sample_stride.")

        raw_threshold = compute_threshold(all_raw, args.calibration_method, args.percentile, args.k_std)
        ema_threshold = compute_threshold(all_ema, args.calibration_method, args.percentile, args.k_std)
        chosen_threshold = ema_threshold if recommended_basis == "ema" else raw_threshold

        method_dir = os.path.join(args.save_dir, method)
        os.makedirs(method_dir, exist_ok=True)

        threshold_path = os.path.join(method_dir, "threshold.txt")
        with open(threshold_path, "w") as f:
            f.write(f"{chosen_threshold:.6f}\n")

        meta = build_calibration_meta(
            args, method, image_size, ckpt_dirs[method], ema_reliable, recommended_basis,
            len(videos), len(all_raw), len(all_ema), raw_threshold, ema_threshold, chosen_threshold)
        meta_path = os.path.join(method_dir, "calibration_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        csv_path = os.path.join(method_dir, "normal_scores.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["raw_score", "ema_score", "score_ms"])
            for r, e, m in zip(all_raw, all_ema, all_ms):
                w.writerow([f"{r:.6f}", f"{e:.6f}", f"{m:.2f}"])

        if not args.no_plots:
            try:
                save_histogram(all_raw, all_ema, chosen_threshold, recommended_basis, method,
                               os.path.join(method_dir, "score_distribution.png"))
            except Exception as e:
                print(f"  [{method}]   [WARN] could not save histogram: {e}")

        print(f"  [{method:<10s}] n={len(all_raw)} crops  "
              f"raw:  mean={np.mean(all_raw):.4f} max={np.max(all_raw):.4f}  "
              f"ema:  mean={np.mean(all_ema):.4f} max={np.max(all_ema):.4f}")
        print(f"  [{method:<10s}] threshold ({args.calibration_method}, basis={recommended_basis}) "
              f"= {chosen_threshold:.6f}")
        print(f"  [{method:<10s}] wrote: {threshold_path}")
        print(f"  [{method:<10s}] wrote: {meta_path}")

        summary[method] = {
            "threshold": chosen_threshold,
            "basis": recommended_basis,
            "n_samples": len(all_raw),
            "threshold_file": threshold_path,
            "meta_file": meta_path,
        }

    summary_path = os.path.join(args.save_dir, "calibration_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}\n  Calibration summary\n{'='*70}")
    for method, s in summary.items():
        if "error" in s:
            print(f"  {method:<10s} FAILED: {s['error']}")
        else:
            print(f"  {method:<10s} threshold={s['threshold']:.6f}  basis={s['basis']}  "
                 f"n={s['n_samples']}")
    print(f"\nWrote {summary_path}")

    ok_methods = [m for m, s in summary.items() if "error" not in s]
    if ok_methods:
        print("\nUse with compare_pipeline_methods.py, e.g.:")
        print("  python compare_pipeline_methods.py \\")
        print(f"      --video /path/to/test_video.mp4 \\")
        print(f"      --yolo_det_weights {args.yolo_det_weights} \\")
        for m in ok_methods:
            print(f"      --{m}_threshold_file {os.path.join(args.save_dir, m, 'threshold.txt')} \\")
        print(f"      --category {args.category} --model_size {args.model_size} \\")
        print(f"      --save_dir ./results_compare_all")


if __name__ == '__main__':
    main()
