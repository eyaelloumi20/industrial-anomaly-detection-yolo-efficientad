"""
calibrate_threshold_all.py
===========================
Calibrate anomaly thresholds for ALL FIVE checkpoint types
(baseline / lowres / thin / qat / pruned) from ONE video of known-normal
pieces, in a single pass.

Fairness / speed design
------------------------
YOLO detection + ByteTrack tracking runs EXACTLY ONCE for the whole video
(Phase 1), cached per frame -- same approach as
combined_gum_inspection_pipeline_compare_all.py. Every method then scores
the SAME cached crops (Phase 2), so YOLO's cost is paid once instead of
five times, and every method is calibrated against identical detections.

Each method uses the correct loader for its checkpoint type (mirrors
benchmark_efficientad.py / combined_gum_inspection_pipeline_compare_all.py):
    baseline / lowres -> utils.load_efficientad_adaptive  (GPU, any image_size)
    thin              -> utils.load_efficientad_thin       (GPU, 256px)
    qat               -> utils.load_efficientad_qat         (CPU forced, 256px)
    pruned            -> utils.load_efficientad_pruned      (GPU, 256px)

Threshold = mean score across all detections for that method (same
definition as calibrate_threshold3.py).

Any --*_dir that doesn't exist is skipped, same as --compare_all, so you
don't need every checkpoint trained to use this.

Usage
-----
    python calibrate_threshold_all.py \
        --source /path/to/normal_only_video.mp4 \
        --yolo_weights /path/to/best.pt \
        --baseline_dir /home/ahmed/framework/ckpt_cropped_large \
        --lowres_dir ./ckpt_192 \
        --thin_dir ./ckpt_thin_025 \
        --qat_dir ./ckpt_qat \
        --pruned_dir ./ckpt_pruned_030 \
        --out_dir ./calibration_all \
        --category chewinggum --model_size S

Output
------
    ./calibration_all/baseline/threshold.txt
    ./calibration_all/baseline/calibration_scores.csv
    ./calibration_all/lowres/threshold.txt
    ...
    (one subfolder per active method -- pass these threshold.txt paths
    straight into combined_gum_inspection_pipeline_compare_all.py's
    --*_threshold_file flags)
"""

import os
import sys
import csv
import json
import argparse
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


# ── loader dispatch (mirrors combined_gum_inspection_pipeline_compare_all.py) ─

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


# ── Phase 1: shared YOLO detection + tracking, run once ───────────────────────

def run_yolo_once(source: str, yolo_weights: str, conf: float, iou: float,
                  device: str) -> Tuple[Dict[int, List[Tuple[int, int, int, int, int]]], int, int, int]:
    from ultralytics import YOLO
    print(f"[Phase 1] Loading YOLO: {yolo_weights}")
    yolo = YOLO(yolo_weights)
    yolo.to(device)

    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Phase 1] Source: {src_w}x{src_h}  (~{total} frames)")

    detections: Dict[int, List[Tuple[int, int, int, int, int]]] = {}
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
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

        frame_idx += 1
        if frame_idx % 60 == 0:
            n_det = sum(len(v) for v in detections.values())
            print(f"  [Phase 1] frame {frame_idx}/{total}  ({n_det} detections so far)")

    cap.release()
    n_frames = frame_idx
    n_det = sum(len(v) for v in detections.values())
    print(f"[Phase 1] Done: {n_frames} frames, {n_det} total detections")
    if n_det == 0:
        print("[Phase 1] WARNING: no detections found at all. Check YOLO weights and --conf.")
    return detections, src_w, src_h, n_frames


# ── Phase 2: calibrate one method against the cached detections ──────────────

def calibrate_method(method: str, ckpt_dir: str, source: str,
                     detections: Dict[int, List[Tuple[int, int, int, int, int]]],
                     src_w: int, src_h: int, n_frames: int, args,
                     stride: int = 1) -> Optional[float]:
    threshold_basis = args.threshold_basis
    if stride > 1 and threshold_basis == "ema":
        print(f"\n  [{method}] NOTE: --sample_stride {stride} > 1 breaks per-track EMA "
              f"continuity (frames get skipped mid-track), so EMA calibration would "
              f"drift from what the pipeline actually computes. Forcing "
              f"threshold_basis='raw' for {method}.")
        threshold_basis = "raw"

    print(f"\n{'='*70}\n  Method: {method}  (ckpt={ckpt_dir})"
          + (f"  [stride={stride}, ~1/{stride} detections]" if stride > 1 else "")
          + f"  [basis={threshold_basis}]"
          + f"\n{'='*70}")

    try:
        models, image_size, device = load_bundle(
            method, ckpt_dir, args.category, args.model_size, args.device)
    except Exception as e:
        print(f"  [{method}] FAILED to load: {e}")
        return None

    method_out_dir = os.path.join(args.out_dir, method)
    os.makedirs(method_out_dir, exist_ok=True)
    log_path = os.path.join(method_out_dir, "calibration_scores.csv")
    log_file = open(log_path, "w", newline="")
    log_w    = csv.writer(log_file)
    log_w.writerow(["frame", "track_id", "raw_score", "ema_score", "x1", "y1", "x2", "y2"])

    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    denoise = args.denoise

    all_raw: List[float] = []
    all_ema: List[float] = []
    ema_state: Dict[int, float] = {}   # per-track EMA, mirrors run_method() in the pipeline
    det_counter = 0  # global index across all detections, for --stride sampling

    for frame_idx in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        dets = detections.get(frame_idx, [])
        if not dets:
            continue
        pil_frame = None  # lazily decoded only if this frame has a sampled detection

        for tid, x1, y1, x2, y2 in dets:
            take = (det_counter % stride == 0)
            det_counter += 1
            if not take:
                continue
            if pil_frame is None:
                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            crop  = get_crop(pil_frame, (x1, y1, x2, y2), image_size,
                             args.pad_ratio, denoise)
            score = score_crop(models, crop, device, image_size, ratio=args.ratio)

            prev = ema_state.get(tid, score)
            ema  = args.ema_alpha * score + (1 - args.ema_alpha) * prev
            ema_state[tid] = ema

            all_raw.append(score)
            all_ema.append(ema)
            log_w.writerow([frame_idx, tid, f"{score:.6f}", f"{ema:.6f}", x1, y1, x2, y2])

        if (frame_idx + 1) % 60 == 0 and all_raw:
            print(f"  [{method}] frame {frame_idx + 1}/{n_frames}  "
                  f"detections={len(all_raw)}  raw_mean={np.mean(all_raw):.4f}  "
                  f"ema_mean={np.mean(all_ema):.4f}")

    cap.release()
    log_file.close()

    del models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not all_raw:
        print(f"  [{method}] ERROR: no detections scored. Skipping threshold.")
        return None

    raw_arr = np.array(all_raw)
    ema_arr = np.array(all_ema)
    threshold = float(ema_arr.mean()) if threshold_basis == "ema" else float(raw_arr.mean())

    print(f"  [{method}] detections={len(raw_arr)}  "
          f"raw_mean={raw_arr.mean():.4f}  raw_max={raw_arr.max():.4f}  "
          f"ema_mean={ema_arr.mean():.4f}  ema_max={ema_arr.max():.4f}")
    print(f"  [{method}] Threshold ({threshold_basis} basis) = {threshold:.6f}")

    thr_path = os.path.join(method_out_dir, "threshold.txt")
    with open(thr_path, "w") as f:
        f.write(f"{threshold:.8f}\n")
    print(f"  [{method}] Saved -> {thr_path}")

    # ── Compatibility metadata: every parameter that must match between this
    # ── calibration run and the eventual combined_gum_inspection_pipeline_
    # ── compare_all.py inference run for the threshold to actually be valid.
    meta = dict(
        method=method, source_video=source, category=args.category,
        model_size=args.model_size, image_size=image_size,
        conf=args.conf, iou=args.iou, pad_ratio=args.pad_ratio,
        denoise=bool(denoise), ratio=args.ratio, ema_alpha=args.ema_alpha,
        threshold_basis=threshold_basis, sample_stride=stride,
        n_detections=len(raw_arr),
        raw_mean=float(raw_arr.mean()), raw_max=float(raw_arr.max()),
        ema_mean=float(ema_arr.mean()), ema_max=float(ema_arr.max()),
        threshold=threshold,
    )
    meta_path = os.path.join(method_out_dir, "calibration_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return threshold


# ── args + main ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Calibrate anomaly thresholds for all methods "
                    "(baseline/lowres/thin/qat/pruned) from one normal-only video.")
    p.add_argument("--source",       required=True,
                   help="Path to normal-only video (or webcam index).")
    p.add_argument("--yolo_weights", required=True)
    p.add_argument("--baseline_dir", default="/home/ahmed/framework/ckpt_cropped_large")
    p.add_argument("--lowres_dir",   default="./ckpt_192")
    p.add_argument("--thin_dir",     default="./ckpt_thin_025")
    p.add_argument("--qat_dir",      default="./ckpt_qat")
    p.add_argument("--pruned_dir",   default="./ckpt_pruned_030")
    p.add_argument("--skip_methods", nargs="*", default=[],
                   help="Method names to skip even if their --*_dir exists, e.g. --skip_methods qat")
    p.add_argument("--category",     default="chewinggum")
    p.add_argument("--model_size",   default="S", choices=["S", "M"])
    p.add_argument("--device",       default="cuda",
                   help="Used for baseline/lowres/thin/pruned. qat always forces cpu regardless.")
    p.add_argument("--ratio",        type=float, default=0.1)
    p.add_argument("--conf",         type=float, default=0.25,
                   help="YOLO confidence threshold. Default 0.25 matches "
                        "combined_gum_inspection_pipeline_compare_all.py's own default -- "
                        "keep this equal to whatever --conf you'll actually run inference "
                        "with, since a different detection confidence changes which boxes "
                        "get scored.")
    p.add_argument("--iou",          type=float, default=0.45)
    p.add_argument("--pad_ratio",    type=float, default=0.02)
    p.add_argument("--denoise",      action="store_true",
                   help="Apply Gaussian+bilateral denoising on crops. OFF by default -- "
                        "matches combined_gum_inspection_pipeline_compare_all.py's own "
                        "--denoise flag/default exactly, so a plain run of each script "
                        "without extra flags already agrees. Pass this here (and pass "
                        "--denoise on the inference side too) if you actually want "
                        "denoising on both.")
    p.add_argument("--ema_alpha",    type=float, default=0.3,
                   help="EMA weight for per-track score smoothing. Default 0.3 matches "
                        "combined_gum_inspection_pipeline_compare_all.py's own "
                        "--ema_alpha default -- this MUST match at inference time, since "
                        "the pipeline's ANOMALY/NORMAL verdict compares the threshold "
                        "against the EMA-smoothed score, not the raw score.")
    p.add_argument("--threshold_basis", choices=["ema", "raw"], default="ema",
                   help="'ema' (default, recommended): threshold = mean of the same "
                        "per-track EMA-smoothed score the pipeline actually verdicts "
                        "against. 'raw': threshold = mean of raw per-crop scores "
                        "(legacy behaviour, matches calibrate_threshold3.py). "
                        "Automatically forced to 'raw' for any method run with "
                        "--sample_stride > 1, since subsampling breaks EMA continuity.")
    p.add_argument("--sample_stride", type=int, default=1,
                   help="Score every Nth cached detection instead of all of them, e.g. "
                        "--sample_stride 10 scores ~10%% of detections. Applies to every "
                        "method unless overridden below. A few hundred samples is usually "
                        "enough for a stable mean-based threshold.")
    p.add_argument("--qat_sample_stride", type=int, default=None,
                   help="Override --sample_stride for qat only (it runs on CPU and is "
                        "the usual bottleneck). Defaults to --sample_stride if not set.")
    p.add_argument("--pruned_sample_stride", type=int, default=None,
                   help="Override --sample_stride for pruned only. Defaults to "
                        "--sample_stride if not set.")
    p.add_argument("--out_dir",      default="./calibration_all")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"{'='*70}\n  Compatibility-critical settings "
          f"(MUST match combined_gum_inspection_pipeline_compare_all.py)\n{'='*70}")
    print(f"  --conf        {args.conf}")
    print(f"  --iou         {args.iou}")
    print(f"  --pad_ratio   {args.pad_ratio}")
    print(f"  --denoise     {args.denoise}")
    print(f"  --ema_alpha   {args.ema_alpha}")
    print(f"  threshold_basis (requested) = {args.threshold_basis}")
    print()

    ckpt_dirs = {
        'baseline': args.baseline_dir, 'lowres': args.lowres_dir,
        'thin':     args.thin_dir,     'qat':    args.qat_dir,
        'pruned':   args.pruned_dir,
    }

    active_methods = [m for m in METHODS
                      if m not in args.skip_methods and os.path.isdir(ckpt_dirs[m])]
    skipped = [m for m in METHODS if m not in active_methods]
    print(f"Active methods : {active_methods}")
    if skipped:
        print(f"Skipped        : {skipped}  (missing --*_dir or in --skip_methods)")
    if not active_methods:
        raise SystemExit("No checkpoint directories found. Nothing to calibrate.")

    # ── Phase 1: YOLO once, shared across every method ────────────────────────
    detections, src_w, src_h, n_frames = run_yolo_once(
        args.source, args.yolo_weights, args.conf, args.iou,
        device='cuda' if torch.cuda.is_available() else 'cpu')

    # ── Phase 2: calibrate each method ─────────────────────────────────────────
    stride_overrides = {'qat': args.qat_sample_stride, 'pruned': args.pruned_sample_stride}

    thresholds: Dict[str, Optional[float]] = {}
    for method in active_methods:
        stride = stride_overrides.get(method) or args.sample_stride
        thresholds[method] = calibrate_method(
            method, ckpt_dirs[method], args.source, detections,
            src_w, src_h, n_frames, args, stride=stride)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}\n  Calibration Summary\n{'='*70}")
    for method in active_methods:
        thr = thresholds[method]
        if thr is None:
            print(f"  {method:<10s}  FAILED / no detections")
        else:
            thr_path = os.path.join(args.out_dir, method, "threshold.txt")
            print(f"  {method:<10s}  threshold={thr:.6f}   -> {thr_path}")

    print(f"\nPass these straight into combined_gum_inspection_pipeline_compare_all.py, e.g.:")
    for method in active_methods:
        if thresholds[method] is not None:
            print(f"  --{method}_threshold_file {os.path.join(args.out_dir, method, 'threshold.txt')}")


if __name__ == '__main__':
    main()
