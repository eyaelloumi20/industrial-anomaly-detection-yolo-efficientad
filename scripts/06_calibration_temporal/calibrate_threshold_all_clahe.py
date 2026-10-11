"""
calibrate_threshold_all.py
===========================
Calibrates a per-method anomaly threshold from one or more REFERENCE videos
of KNOWN-GOOD product. STANDALONE: does not import compare_pipeline_methods_t.py
or anything else from that project -- the model loading, cropping, and
scoring logic it needs are reimplemented directly in this file, so it keeps
working even if the pipeline script's filename, location, or internals
change. Uses the SAME loader dispatch, crop preprocessing, and scoring math
as that project's EfficientAD pipeline, so a threshold calibrated here
still means the same thing when applied there.

Always fp16
------------
Every method's forward pass (teacher/student/autoencoder) always runs
under CUDA autocast fp16 here -- unconditionally, no benchmarking, no
--fp16 flag. If you're running the comparison pipeline at fp16 for real
inference, calibrate at fp16 too, so the threshold matches the score
distribution it's actually being compared against. (qat is CPU-only and
always stays fp32 there regardless -- autocast fp16 doesn't apply off-CUDA.)

Multiple reference videos
--------------------------
Pass --video_dir to calibrate off every video in a folder at once: YOLO
detect+track runs once per video (shared across methods), and every
method's raw/EMA scores are POOLED across all videos before the
percentile is taken -- one threshold per method, reflecting the whole
reference set. Track ids reset per video (a track in video 2 has no
relationship to a same-numbered track in video 1), so EMA is never
smoothed across a file boundary. --video still works for a single file.

Memory note: every active method's models are loaded once, up front, and
kept resident for the whole multi-video pass (so YOLO's per-video
detect+track is paid once per video, not once per video per method). For
four small PDN-S bundles this is a modest footprint; if GPU memory is
tight, run this script multiple times with --skip_methods to calibrate a
subset of methods at a time instead.

What it does, per active method (baseline / lowres / thin / qat / pruned):
  1. Loads that checkpoint with the same loader dispatch the pipeline uses:
       baseline / lowres -> utils.load_efficientad_adaptive
       thin              -> utils.load_efficientad_thin
       qat               -> utils.load_efficientad_qat   (CPU forced)
       pruned            -> utils.load_efficientad_pruned
  2. For each reference video: runs YOLO detect+track once (shared across
     methods), then scores every crop (same crop padding/resize/denoise,
     same teacher/student/ae anomaly-map math), building a raw-score
     stream and a per-track EMA-score stream (reset per video).
  3. Pools every video's raw/EMA scores together and sets the threshold to
     a percentile of that pooled (assumed-normal) distribution, times an
     optional safety margin.
  4. Writes threshold.txt + calibration_meta.json into --out_dir/<method>/,
     using the same field names (category, model_size, image_size, ratio,
     yolo_weights, conf, iou, pad_ratio, denoise, ema_alpha,
     checkpoint_fingerprint, recommended_basis, ema_reliable) that the
     comparison pipeline's compatibility check expects to find there.

IMPORTANT: every video (whether via --video or --video_dir) must show GOOD
product only. Real defects in reference footage pull the percentile up and
make the threshold too lenient, i.e. it will miss real defects later.

Usage
-----
Single reference video:
    python calibrate_threshold_all.py \\
        --video /path/to/known_good_reel.mp4 \\
        --yolo_det_weights /path/to/best.pt \\
        --baseline_dir /home/ahmed/framework/ckpt_cropped_large \\
        --lowres_dir ./ckpt_192 \\
        --thin_dir ./ckpt_thin_025 \\
        --pruned_dir ./ckpt_pruned_030 \\
        --skip_methods qat \\
        --category chewinggum --model_size S \\
        --conf 0.25 --iou 0.45 --pad_ratio 0.02 \\
        --out_dir ./calibration_output

Folder of reference videos (pooled calibration):
    python calibrate_threshold_all.py \\
        --video_dir /home/ahmed/two-stage-efficientad-inspection/videos_N \\
        --yolo_det_weights /path/to/best.pt \\
        --baseline_dir /home/ahmed/framework/ckpt_cropped_large \\
        --lowres_dir ./ckpt_192 \\
        --thin_dir ./ckpt_thin_025 \\
        --pruned_dir ./ckpt_pruned_030 \\
        --skip_methods qat \\
        --category chewinggum --model_size S \\
        --conf 0.25 --iou 0.45 --pad_ratio 0.02 \\
        --out_dir ./calibration_output

Then point compare_pipeline_methods_t.py at the results, e.g.:
    --baseline_threshold_file ./calibration_output/baseline/threshold.txt
(printed for every method at the end of this script's run).
"""
import argparse
import json
import os
import sys
import time
import hashlib
from contextlib import nullcontext
from dataclasses import dataclass
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
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}


# ── loader dispatch (mirrors the pipeline's utils.load_efficientad_* choice
# ── per method -- kept identical here so calibration loads the SAME
# ── architecture/weights the comparison run will later score with) ───────────

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
    """Hash of (relative path, size, mtime) for every file in ckpt_dir --
    written into calibration_meta.json so the comparison pipeline can detect
    'this checkpoint was retrained/replaced since it was calibrated' even
    when the --*_dir path is unchanged."""
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


# ── crop preprocessing (same padding/letterbox-resize/denoise as the
# ── comparison pipeline's get_crop, so calibration sees identical crops) ─────

def resize_with_padding(crop: Image.Image, size: int) -> Image.Image:
    ow, oh = crop.size
    scale  = size / max(ow, oh)
    nw, nh = int(ow * scale), int(oh * scale)
    resized = crop.resize((nw, nh), Image.LANCZOS)
    canvas  = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def normalize_illumination(arr: np.ndarray, clip_limit: float = 2.0,
                            tile_grid: int = 8) -> np.ndarray:
    """CLAHE on the L channel in LAB space. Evens out brightness/contrast
    shifts between capture sessions (different lighting, different
    background) before the crop reaches the model, so the anomaly-score
    distribution is less sensitive to which video it came from."""
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def get_crop(pil_img: Image.Image, box: tuple, size: int,
            pad_ratio: float, denoise: bool, normalize_illum: bool = False) -> Image.Image:
    W, H = pil_img.size
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad_ratio), int(bh * pad_ratio)
    x1 = max(0, x1 - px);  y1 = max(0, y1 - py)
    x2 = min(W, x2 + px);  y2 = min(H, y2 + py)
    crop = pil_img.crop((x1, y1, x2, y2)).convert("RGB")
    crop = resize_with_padding(crop, size)
    # Illumination normalization runs before denoise: CLAHE can amplify
    # sensor noise, so smoothing it afterward is the more stable order.
    if normalize_illum:
        arr = np.array(crop, dtype=np.uint8)
        arr = normalize_illumination(arr)
        crop = Image.fromarray(arr)
    if denoise:
        arr = np.array(crop, dtype=np.uint8)
        arr = cv2.GaussianBlur(arr, (3, 3), sigmaX=0.8, sigmaY=0.8)
        arr = cv2.bilateralFilter(arr, d=5, sigmaColor=15, sigmaSpace=15)
        crop = Image.fromarray(arr)
    return crop


# ── scoring (same teacher/student/ae anomaly-map math as the comparison
# ── pipeline's _amp_forward/score_crop_batch -- always fp16 under CUDA) ──────

@torch.no_grad()
def score_crop_batch(models: dict, crops_pil: List[Image.Image], device: str,
                     image_size: int, ratio: float = 0.1) -> List[float]:
    """Stacks all crops for a frame into ONE [N, 3, H, W] tensor and runs
    teacher/student/ae ONCE for the whole frame. Always runs under CUDA
    autocast fp16 when on CUDA (qat runs on CPU and stays fp32 there --
    autocast fp16 doesn't apply off-CUDA). Returns one max-score float per
    crop, in the same order as crops_pil. Empty input returns [] without
    touching the GPU."""
    if not crops_pil:
        return []

    tensors = [transforms.ToTensor()(c) for c in crops_pil]
    t = torch.stack(tensors, dim=0).to(device)   # [N, 3, H, W]

    amp_ctx = (torch.autocast(device_type="cuda", dtype=torch.float16)
              if device.startswith("cuda") else nullcontext())
    with amp_ctx:
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

    # Normalization + max score always in fp32 -- this is a threshold-facing
    # scalar, so keep it numerically stable regardless of the forward pass precision.
    fm_st, fm_stae = fm_st.float(), fm_stae.float()
    nm_st = (ratio * (fm_st   - models['qa_st'])) / (models['qb_st'] - models['qa_st'] + 1e-8)
    nm_ae = (ratio * (fm_stae - models['qa_ae'])) / (models['qb_ae'] - models['qa_ae'] + 1e-8)
    amap  = (0.5 * nm_st + 0.5 * nm_ae)[:, 0].cpu().numpy()   # [N, H, W]
    return [float(np.max(amap[i])) for i in range(amap.shape[0])]


# ── Phase 1: YOLO detect+track, run once per video, shared across methods ────

@dataclass
class FrameCache:
    pil_frames: List[Image.Image]


def run_yolo_once(video_path: str, yolo_weights: str, conf: float, iou: float,
                  device: str, max_frames: Optional[int]):
    """Detect+track every frame of ONE video, decoding each frame exactly
    once. Returns (detections: {frame_idx: [(track_id, x1,y1,x2,y2), ...]},
    n_frames, frame_cache)."""
    from ultralytics import YOLO
    print(f"[Phase 1] Loading YOLO: {yolo_weights}")
    yolo = YOLO(yolo_weights)
    yolo.to(device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Phase 1] Source: {src_w}x{src_h}  (~{total} frames)")

    detections: Dict[int, List[Tuple[int, int, int, int, int]]] = {}
    pil_frames: List[Image.Image] = []
    frame_idx = 0
    n_tracker_resets = 0
    force_tracker_reset = False
    t0 = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and frame_idx >= max_frames:
            break

        used_fallback = False
        try:
            results = yolo.track(source=frame, conf=conf, iou=iou, classes=[0],
                                 persist=not force_tracker_reset,
                                 tracker="bytetrack.yaml", verbose=False)
            force_tracker_reset = False
        except np.linalg.LinAlgError:
            # A degenerate (0-width/0-height) box in this frame breaks the
            # tracker. Retrying on the same frame hits the same crash, so
            # fall back to plain predict() (no tracker) for just this frame,
            # with synthetic collision-safe ids, and reset the tracker next frame.
            n_tracker_resets += 1
            used_fallback = True
            force_tracker_reset = True
            print(f"  [Phase 1] WARNING: degenerate detection box broke the tracker "
                  f"at frame {frame_idx} -- using untracked detections for this "
                  f"frame only, then reinitializing tracker [reset #{n_tracker_resets}]")
            results = yolo.predict(source=frame, conf=conf, iou=iou, classes=[0], verbose=False)

        boxes_this_frame = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
            if used_fallback or r.boxes.id is None:
                ids = [100_000 + frame_idx * 1000 + i for i in range(len(xyxy))]
            else:
                ids = r.boxes.id.cpu().numpy().astype(int)
            for box, tid in zip(xyxy, ids):
                x1 = max(0, box[0]); y1 = max(0, box[1])
                x2 = min(src_w, box[2]); y2 = min(src_h, box[3])
                if x2 > x1 and y2 > y1:
                    boxes_this_frame.append((int(tid), x1, y1, x2, y2))
        detections[frame_idx] = boxes_this_frame
        pil_frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

        frame_idx += 1
        if frame_idx % 60 == 0:
            print(f"  [Phase 1] frame {frame_idx}/{total}  "
                  f"({frame_idx / (time.perf_counter() - t0):.1f} fps)")

    cap.release()
    n_frames = frame_idx
    print(f"[Phase 1] Done: {n_frames} frames, "
          f"{sum(len(v) for v in detections.values())} total detections, "
          f"{time.perf_counter() - t0:.1f}s"
          + (f", {n_tracker_resets} tracker reset(s)" if n_tracker_resets else ""))
    return detections, n_frames, FrameCache(pil_frames)


# ── calibration ────────────────────────────────────────────────────────────

@dataclass
class CalibrationResult:
    method:                str
    n_samples:             int             = 0
    image_size:            int             = 0
    raw_percentile_value:  float           = 0.0
    ema_percentile_value:  float           = 0.0
    median_track_len:      float           = 0.0
    ema_reliable:          bool            = True
    recommended_basis:     str             = "ema"
    threshold:             float           = 0.0
    out_dir:                str             = ""
    error:                 Optional[str]   = None


def resolve_video_list(args) -> List[str]:
    if args.video_dir:
        if not os.path.isdir(args.video_dir):
            raise SystemExit(f"--video_dir not found or not a directory: {args.video_dir}")
        paths = [os.path.join(args.video_dir, fn)
                 for fn in sorted(os.listdir(args.video_dir))
                 if os.path.splitext(fn)[1].lower() in VIDEO_EXTS]
        if not paths:
            raise SystemExit(f"No video files found in --video_dir={args.video_dir} "
                             f"(looked for extensions: {sorted(VIDEO_EXTS)})")
        return paths
    if args.video:
        if not os.path.isfile(args.video):
            raise SystemExit(f"--video not found: {args.video}")
        return [args.video]
    raise SystemExit("Pass either --video (single file) or --video_dir "
                     "(folder of reference videos).")


def track_lengths(detections: Dict[int, List[Tuple[int, int, int, int, int]]]) -> List[int]:
    """Per-track frame counts for ONE video's detections (track ids are only
    meaningful within a single video, never pooled across videos)."""
    counts: Dict[int, int] = {}
    for boxes in detections.values():
        for tid, *_rest in boxes:
            counts[tid] = counts.get(tid, 0) + 1
    return list(counts.values())


def median_of(vals: List[float]) -> float:
    if not vals:
        return 0.0
    vals = sorted(vals)
    n = len(vals)
    mid = n // 2
    return float(vals[mid]) if n % 2 else (vals[mid - 1] + vals[mid]) / 2.0


def load_method_context(method: str, ckpt_dir: str, args) -> dict:
    print(f"\n{'='*70}\n  Loading: {method}  (ckpt={ckpt_dir})\n{'='*70}")
    try:
        models, image_size, device = load_bundle(
            method, ckpt_dir, args.category, args.model_size, args.device)
    except Exception as e:
        print(f"  [{method}] FAILED to load: {e}")
        return dict(error=str(e))
    print(f"  [{method:<10s}] scoring precision: "
          f"{'fp16 (autocast)' if device.startswith('cuda') else 'fp32 (cpu)'}")
    return dict(models=models, image_size=image_size, device=device,
               raw_scores=[], ema_scores=[], error=None)


def score_video_for_method(ctx: dict,
                           detections: Dict[int, List[Tuple[int, int, int, int, int]]],
                           n_frames: int, frame_cache: FrameCache, args) -> None:
    """Scores one already-decoded video's crops with one method's (already
    loaded) models, appending onto that method's pooled raw/ema score lists.
    ema_hist is local to this call -- a fresh EMA per video, since track ids
    don't carry meaning across separate video files."""
    models, image_size, device = ctx["models"], ctx["image_size"], ctx["device"]
    alpha = args.ema_alpha
    ema_hist: Dict[int, float] = {}

    for frame_idx in range(n_frames):
        pil_frame = frame_cache.pil_frames[frame_idx]
        frame_dets = detections.get(frame_idx, [])
        if not frame_dets:
            continue
        crops = [get_crop(pil_frame, (x1, y1, x2, y2), image_size,
                          args.pad_ratio, args.denoise, args.normalize_illum)
                 for _tid, x1, y1, x2, y2 in frame_dets]
        scores = score_crop_batch(models, crops, device, image_size, ratio=args.ratio)
        for (tid, *_rest), score in zip(frame_dets, scores):
            prev = ema_hist.get(tid, score)
            ema = alpha * score + (1 - alpha) * prev
            ema_hist[tid] = ema
            ctx["raw_scores"].append(score)
            ctx["ema_scores"].append(ema)


def finalize_method(method: str, ckpt_dir: str, ctx: dict, med_track_len: float,
                    ema_reliable: bool, recommended_basis: str, args,
                    video_paths: List[str]) -> CalibrationResult:
    result = CalibrationResult(method=method)
    if ctx.get("error"):
        result.error = ctx["error"]
        return result

    result.image_size = ctx["image_size"]
    result.n_samples = len(ctx["raw_scores"])
    result.median_track_len = med_track_len
    result.ema_reliable = ema_reliable
    result.recommended_basis = recommended_basis

    print(f"\n[{method:<10s}] pooled {result.n_samples} crops across "
          f"{len(video_paths)} reference video(s)")

    if result.n_samples < args.min_samples:
        result.error = (f"only {result.n_samples} samples (< --min_samples "
                        f"{args.min_samples}) -- reference footage is too short/sparse "
                        f"for a reliable percentile threshold; not writing one")
        print(f"  [{method:<10s}] [ERROR] {result.error}")
        return result

    if not ema_reliable:
        print(f"  [{method:<10s}] [WARN] median track length {med_track_len:.1f} frames "
              f"< --min_track_len {args.min_track_len} -- EMA barely smooths on tracks "
              f"this short, calibrating on RAW score instead.")

    raw_arr = np.asarray(ctx["raw_scores"])
    ema_arr = np.asarray(ctx["ema_scores"])
    result.raw_percentile_value = float(np.percentile(raw_arr, args.percentile))
    result.ema_percentile_value = float(np.percentile(ema_arr, args.percentile))

    base_value = (result.ema_percentile_value if recommended_basis == "ema"
                 else result.raw_percentile_value)
    result.threshold = base_value * args.margin

    print(f"  [{method:<10s}] p{args.percentile:g}: raw={result.raw_percentile_value:.6f}  "
          f"ema={result.ema_percentile_value:.6f}  margin={args.margin:g}x  "
          f"-> threshold={result.threshold:.6f}  (basis={recommended_basis})")

    out_dir = os.path.join(args.out_dir, method)
    os.makedirs(out_dir, exist_ok=True)
    result.out_dir = out_dir

    threshold_path = os.path.join(out_dir, "threshold.txt")
    with open(threshold_path, "w") as f:
        f.write(f"{result.threshold:.6f}\n")

    meta = dict(
        category=args.category, model_size=args.model_size, ratio=args.ratio,
        yolo_weights=os.path.basename(args.yolo_det_weights),
        image_size=result.image_size,
        conf=args.conf, iou=args.iou, pad_ratio=args.pad_ratio,
        denoise=args.denoise, normalize_illum=args.normalize_illum, ema_alpha=args.ema_alpha,
        checkpoint_fingerprint=fingerprint_ckpt_dir(ckpt_dir),
        recommended_basis=recommended_basis,
        ema_reliable=ema_reliable,
        percentile=args.percentile,
        margin=args.margin,
        n_samples=result.n_samples,
        median_track_length_frames=med_track_len,
        calibrated_precision=("fp16" if ctx["device"].startswith("cuda") else "fp32"),
        calibrated_on_videos=[os.path.abspath(v) for v in video_paths],
        n_calibration_videos=len(video_paths),
        calibrated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    meta_path = os.path.join(out_dir, "calibration_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  [{method:<10s}] wrote {threshold_path}")
    print(f"  [{method:<10s}] wrote {meta_path}")
    return result


def parse_args():
    p = argparse.ArgumentParser(
        description="Calibrate per-method anomaly thresholds from known-good reference "
                    "video(s). Standalone -- does not import compare_pipeline_methods_t.py. "
                    "Always scores under CUDA autocast fp16.")
    p.add_argument("--video",             default=None,
                   help="Single reference video of KNOWN-GOOD product. Use --video_dir "
                        "instead to calibrate off a whole folder of reference videos.")
    p.add_argument("--video_dir",         default=None,
                   help=f"Folder of reference videos of KNOWN-GOOD product ({sorted(VIDEO_EXTS)}). "
                        "Every method's scores are pooled across all of them before the "
                        "threshold percentile is taken. Overrides --video if both are given.")
    p.add_argument("--yolo_det_weights",  required=True)
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
    p.add_argument("--conf",         type=float, default=0.25)
    p.add_argument("--iou",          type=float, default=0.45)
    p.add_argument("--pad_ratio",    type=float, default=0.02)
    p.add_argument("--denoise",      action="store_true")
    p.add_argument("--normalize_illum", action="store_true",
                   help="CLAHE illumination normalization on each crop before scoring. "
                        "Use this when calibrating across videos with different lighting "
                        "or backgrounds (e.g. via --video_dir) so brightness/contrast "
                        "differences between sessions don't shift the score distribution. "
                        "Must match between calibration and inference (tracked as a SOFT "
                        "compat field).")
    p.add_argument("--ema_alpha",    type=float, default=0.3)
    p.add_argument("--max_frames",   type=int, default=0,
                   help="Cap frames read from EACH reference video (0 = no cap).")
    p.add_argument("--percentile",   type=float, default=99.5,
                   help="Threshold = this percentile of the pooled reference (assumed-normal) "
                        "score distribution, times --margin.")
    p.add_argument("--margin",       type=float, default=1.0,
                   help="Multiplies the percentile value for extra headroom, e.g. 1.05 for "
                        "5%% headroom above the pooled distribution's tail. 1.0 = none.")
    p.add_argument("--min_track_len", type=float, default=5.0,
                   help="If the median track survives fewer frames than this (pooled across "
                        "all reference videos), calibrate on RAW score instead of EMA and "
                        "record recommended_basis='raw'.")
    p.add_argument("--min_samples",  type=int, default=200,
                   help="Refuse to write a threshold for a method if fewer than this many "
                        "crops were scored (pooled across all reference videos).")
    p.add_argument("--out_dir",      default="./calibration_output",
                   help="threshold.txt + calibration_meta.json are written to "
                        "--out_dir/<method>/, matching the --<method>_threshold_file layout "
                        "the comparison pipeline expects.")
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

    video_paths = resolve_video_list(args)
    print(f"Reference videos ({len(video_paths)}):")
    for v in video_paths:
        print(f"  - {v}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── load every active method's model ONCE, up front, and keep it resident
    # ── for the whole multi-video pass (see memory note in the module docstring) ─
    method_ctx: Dict[str, dict] = {
        m: load_method_context(m, ckpt_dirs[m], args) for m in active_methods
    }

    all_track_lengths: List[int] = []

    # ── one video at a time: YOLO detect+track runs ONCE per video (shared
    # ── across every method), then each already-loaded method scores that
    # ── video's crops and the frame cache is freed before the next video. ────
    for vi, video_path in enumerate(video_paths, 1):
        print(f"\n{'='*70}\n  Video {vi}/{len(video_paths)}: {video_path}\n{'='*70}")
        detections, n_frames, frame_cache = run_yolo_once(
            video_path, args.yolo_det_weights, args.conf, args.iou,
            device=device, max_frames=(args.max_frames or None))
        all_track_lengths.extend(track_lengths(detections))

        for method in active_methods:
            ctx = method_ctx[method]
            if ctx.get("error"):
                continue
            score_video_for_method(ctx, detections, n_frames, frame_cache, args)
        # frame_cache (this video's decoded frames) is dropped here, before
        # the next video is decoded -- only one video's frames in RAM at a time.

    med_track_len = median_of(all_track_lengths)
    ema_reliable = med_track_len >= args.min_track_len
    recommended_basis = "ema" if ema_reliable else "raw"

    results = [
        finalize_method(m, ckpt_dirs[m], method_ctx[m], med_track_len,
                        ema_reliable, recommended_basis, args, video_paths)
        for m in active_methods
    ]

    print(f"\n{'='*70}\n  Calibration summary  "
          f"({len(video_paths)} reference video(s), median track length "
          f"{med_track_len:.1f} frames, basis={recommended_basis})\n{'='*70}")
    print(f"  {'method':<10s} {'samples':>8s} {'basis':>6s} {'threshold':>10s}")
    for r in results:
        if r.error:
            print(f"  {r.method:<10s}  FAILED/SKIPPED: {r.error}")
            continue
        print(f"  {r.method:<10s} {r.n_samples:>8d} {r.recommended_basis:>6s} "
              f"{r.threshold:>10.6f}")

    ok_results = [r for r in results if not r.error]
    if ok_results:
        print(f"\nUse with compare_pipeline_methods_t.py, e.g.:")
        for r in ok_results:
            print(f"  --{r.method}_threshold_file {os.path.join(r.out_dir, 'threshold.txt')}")


if __name__ == '__main__':
    main()