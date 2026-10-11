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

Compatibility with combined_gum_inspection_pipeline_compare_all.py
--------------------------------------------------------------------
The pipeline verdicts on the EMA-SMOOTHED score per track_id
("ANOMALY" if ema >= threshold), not the raw per-crop score. To
calibrate a threshold that means what it says, this script now:
  1. Replays the SAME per-track EMA recursion (ema_alpha) that the
     pipeline uses, so the statistic being thresholded matches.
  2. Sets threshold = mean(ema_scores) + k_sigma * std(ema_scores)
     instead of a bare mean (a bare mean flags ~half of all normal
     detections -- there's no margin).
  3. Defaults --conf/--iou/--denoise to the SAME values
     combined_gum_inspection_pipeline_compare_all.py defaults to, so
     if you don't override anything on either script, the detection
     set and crop preprocessing your threshold was measured against
     is the one the pipeline will actually see. If you DO pass
     --conf/--denoise/--ema_alpha overrides to the pipeline, pass the
     identical values here too, or the threshold is meaningless.

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
                     stride: int = 1, k_sigma: float = 3.0) -> Optional[float]:
    print(f"\n{'='*70}\n  Method: {method}  (ckpt={ckpt_dir})"
          + (f"  [stride={stride}, per-track ~1/{stride} detections scored]" if stride > 1 else "")
          + f"  [k_sigma={k_sigma}]"
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
    alpha   = args.ema_alpha

    raw_scores: List[float] = []
    ema_scores: List[float] = []
    ema_hist: Dict[int, float] = {}          # last EMA value per track (pipeline-matching recursion)
    track_counters: Dict[int, int] = {}      # per-track sample counter, for --sample_stride

    for frame_idx in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        dets = detections.get(frame_idx, [])
        if not dets:
            continue
        pil_frame = None  # lazily decoded only if this frame has a sampled detection

        for tid, x1, y1, x2, y2 in dets:
            c = track_counters.get(tid, 0)
            track_counters[tid] = c + 1
            take = (c % stride == 0)
            if not take:
                continue
            if pil_frame is None:
                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            crop  = get_crop(pil_frame, (x1, y1, x2, y2), image_size,
                             args.pad_ratio, denoise)
            score = score_crop(models, crop, device, image_size, ratio=args.ratio)

            # Same recursion as combined_gum_inspection_pipeline_compare_all.py's
            # run_method(): ema = alpha*score + (1-alpha)*prev, prev defaults to
            # score on a track's first (sampled) appearance. When stride > 1 this
            # is the EMA of the SAMPLED subsequence per track, not the full one --
            # a reasonable speed/fidelity tradeoff, but stride=1 (default) is exact.
            prev = ema_hist.get(tid, score)
            ema  = alpha * score + (1 - alpha) * prev
            ema_hist[tid] = ema

            raw_scores.append(score)
            ema_scores.append(ema)
            log_w.writerow([frame_idx, tid, f"{score:.6f}", f"{ema:.6f}", x1, y1, x2, y2])

        if (frame_idx + 1) % 60 == 0 and ema_scores:
            print(f"  [{method}] frame {frame_idx + 1}/{n_frames}  "
                  f"detections={len(ema_scores)}  ema_mean={np.mean(ema_scores):.4f}")

    cap.release()
    log_file.close()

    del models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not ema_scores:
        print(f"  [{method}] ERROR: no detections scored. Skipping threshold.")
        return None

    raw  = np.array(raw_scores)
    ema  = np.array(ema_scores)
    threshold = float(ema.mean() + k_sigma * ema.std())

    print(f"  [{method}] detections={len(ema)}  "
          f"raw_mean={raw.mean():.4f} raw_std={raw.std():.4f}  "
          f"ema_mean={ema.mean():.4f} ema_std={ema.std():.4f}")
    print(f"  [{method}] Threshold = ema_mean + {k_sigma}*ema_std = {threshold:.6f}")

    thr_path = os.path.join(method_out_dir, "threshold.txt")
    with open(thr_path, "w") as f:
        f.write(f"{threshold:.8f}\n")

    stats_path = os.path.join(method_out_dir, "calibration_stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"method={method}\n")
        f.write(f"n_detections={len(ema)}\n")
        f.write(f"conf={args.conf} iou={args.iou} denoise={denoise} "
                f"ema_alpha={alpha} k_sigma={k_sigma} stride={stride}\n")
        f.write(f"raw_mean={raw.mean():.6f} raw_std={raw.std():.6f} raw_max={raw.max():.6f}\n")
        f.write(f"ema_mean={ema.mean():.6f} ema_std={ema.std():.6f} ema_max={ema.max():.6f}\n")
        f.write(f"threshold={threshold:.8f}\n")

    print(f"  [{method}] Saved -> {thr_path}")
    print(f"  [{method}] Saved -> {stats_path}")

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
                   help="YOLO confidence threshold. Default matches "
                        "combined_gum_inspection_pipeline_compare_all.py's default (0.25) -- "
                        "if you pass a different --conf to the pipeline, pass the SAME value "
                        "here, or you're calibrating against a different detection set.")
    p.add_argument("--iou",          type=float, default=0.45)
    p.add_argument("--pad_ratio",    type=float, default=0.02)
    p.add_argument("--denoise",      action="store_true",
                   help="Apply Gaussian+bilateral denoising on crops. Off by default, matching "
                        "combined_gum_inspection_pipeline_compare_all.py's default. Pass this "
                        "flag here iff you also pass --denoise to the pipeline.")
    p.add_argument("--ema_alpha",    type=float, default=0.3,
                   help="EMA smoothing factor, replayed per track_id exactly as "
                        "combined_gum_inspection_pipeline_compare_all.py does before "
                        "thresholding. Must match the pipeline's --ema_alpha (default 0.3).")
    p.add_argument("--k_sigma",      type=float, default=3.0,
                   help="Threshold = mean(ema_scores) + k_sigma*std(ema_scores). A bare mean "
                        "(k_sigma=0) flags ~half of normal detections -- keep this > 0 for a "
                        "real margin above the normal-video score distribution. Applies to "
                        "every method unless overridden below -- methods with noisier "
                        "normal-video scores (higher ema_std relative to ema_mean) may need "
                        "a smaller k_sigma to stay sensitive to real defects.")
    p.add_argument("--baseline_k_sigma", type=float, default=None,
                   help="Override --k_sigma for baseline only. Defaults to --k_sigma.")
    p.add_argument("--lowres_k_sigma",   type=float, default=None,
                   help="Override --k_sigma for lowres only. Defaults to --k_sigma.")
    p.add_argument("--thin_k_sigma",     type=float, default=None,
                   help="Override --k_sigma for thin only. Defaults to --k_sigma.")
    p.add_argument("--qat_k_sigma",      type=float, default=None,
                   help="Override --k_sigma for qat only. Defaults to --k_sigma.")
    p.add_argument("--pruned_k_sigma",   type=float, default=None,
                   help="Override --k_sigma for pruned only. Defaults to --k_sigma.")
    p.add_argument("--sample_stride", type=int, default=1,
                   help="Score every Nth detection WITHIN EACH TRACK (not global) instead of "
                        "all of them, e.g. --sample_stride 10 scores ~10%% of each track's "
                        "detections while keeping its EMA recursion self-consistent. Use "
                        "stride=1 (default) for the closest match to the pipeline's real "
                        "per-frame EMA; higher stride trades fidelity for speed.")
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

    print(f"\n{'='*70}\n  Compatibility check -- these MUST match the flags you pass to\n"
          f"  combined_gum_inspection_pipeline_compare_all.py\n{'='*70}")
    print(f"  --conf={args.conf}  --iou={args.iou}  --denoise={args.denoise}  "
          f"--ema_alpha={args.ema_alpha}\n")

    # ── Phase 1: YOLO once, shared across every method ────────────────────────
    detections, src_w, src_h, n_frames = run_yolo_once(
        args.source, args.yolo_weights, args.conf, args.iou,
        device='cuda' if torch.cuda.is_available() else 'cpu')

    # ── Phase 2: calibrate each method ─────────────────────────────────────────
    stride_overrides  = {'qat': args.qat_sample_stride, 'pruned': args.pruned_sample_stride}
    ksigma_overrides  = {'baseline': args.baseline_k_sigma, 'lowres': args.lowres_k_sigma,
                          'thin': args.thin_k_sigma, 'qat': args.qat_k_sigma,
                          'pruned': args.pruned_k_sigma}

    thresholds: Dict[str, Optional[float]] = {}
    for method in active_methods:
        stride  = stride_overrides.get(method) or args.sample_stride
        k_sigma = ksigma_overrides.get(method)
        k_sigma = args.k_sigma if k_sigma is None else k_sigma
        thresholds[method] = calibrate_method(
            method, ckpt_dirs[method], args.source, detections,
            src_w, src_h, n_frames, args, stride=stride, k_sigma=k_sigma)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}\n  Calibration Summary\n{'='*70}")
    for method in active_methods:
        thr = thresholds[method]
        if thr is None:
            print(f"  {method:<10s}  FAILED / no detections")
        else:
            thr_path = os.path.join(args.out_dir, method, "threshold.txt")
            print(f"  {method:<10s}  threshold={thr:.6f}   -> {thr_path}")

    print(f"\nPass these straight into combined_gum_inspection_pipeline_compare_all.py "
          f"(and keep --conf={args.conf} --denoise={args.denoise} "
          f"--ema_alpha={args.ema_alpha} identical there too), e.g.:")
    for method in active_methods:
        if thresholds[method] is not None:
            print(f"  --{method}_threshold_file {os.path.join(args.out_dir, method, 'threshold.txt')}")


if __name__ == '__main__':
    main()