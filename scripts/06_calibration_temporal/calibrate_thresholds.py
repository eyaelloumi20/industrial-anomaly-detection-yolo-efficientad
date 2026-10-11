"""
calibrate_threshold_all.py
===========================
Calibrate anomaly thresholds for ALL FIVE checkpoint types
(baseline / lowres / thin / qat / pruned) from ONE OR MORE videos of
known-normal pieces, in a single pass.

CHANGELOG vs. the original calibrate_threshold_all.py
-------------------------------------------------------
1. Multiple source videos: --sources takes 1+ paths (recommended: several,
   across different sessions/lighting/camera setups). Detections/scores are
   pooled across all of them, and a per-video threshold is also computed as
   a diagnostic -- if per-video thresholds disagree a lot, that's flagged as
   a warning (session/domain-shift risk), since a single video's normal-score
   distribution does not reliably represent "normal" in general.
2. Threshold statistic is now configurable (--threshold_method): 'percentile'
   (default, e.g. 99th percentile of normal scores) or 'mean_std'
   (mean + k*std). The OLD default -- a plain mean -- flags roughly half of
   even the normal calibration set, so it's kept only as a legacy option.
3. Both the raw-score threshold AND the EMA-score threshold are ALWAYS
   computed now (previously, --sample_stride > 1 silently forced 'raw' and
   discarded the ema value). calibration_meta.json records both, plus
   `ema_reliable` (False when sample_stride broke per-track EMA continuity)
   and `recommended_basis`, so the inference pipeline can pick the right one
   automatically instead of the two scripts silently disagreeing about what
   a threshold means.
4. calibration_meta.json now records everything that can invalidate a
   threshold if it changes at inference time: category, model_size,
   image_size, ratio, yolo_weights, and a cheap filesystem fingerprint of the
   checkpoint directory (to catch "the checkpoint was retrained after this
   was calibrated").

Fairness / speed design
------------------------
YOLO detection + ByteTrack tracking runs EXACTLY ONCE per source video
(Phase 1), cached per frame. Every method then scores the SAME cached
crops (Phase 2), so YOLO's cost is paid once per video (not once per video
per method), and every method is calibrated against identical detections.

Each method uses the correct loader for its checkpoint type (mirrors
benchmark_efficientad.py / combined_gum_inspection_pipeline_compare_all.py):
    baseline / lowres -> utils.load_efficientad_adaptive  (GPU, any image_size)
    thin              -> utils.load_efficientad_thin       (GPU, 256px)
    qat               -> utils.load_efficientad_qat         (CPU forced, 256px)
    pruned            -> utils.load_efficientad_pruned      (GPU, 256px)

Any --*_dir that doesn't exist is skipped, same as --compare_all, so you
don't need every checkpoint trained to use this.

Usage
-----
    python calibrate_threshold_all.py \
        --sources /path/to/normal_session1.mp4 /path/to/normal_session2.mp4 \
        --yolo_weights /path/to/best.pt \
        --baseline_dir /home/ahmed/framework/ckpt_cropped_large \
        --lowres_dir ./ckpt_192 \
        --thin_dir ./ckpt_thin_025 \
        --qat_dir ./ckpt_qat \
        --pruned_dir ./ckpt_pruned_030 \
        --out_dir ./calibration_all \
        --category chewinggum --model_size S \
        --threshold_method percentile --threshold_percentile 99

Output (per active method, under --out_dir/<method>/)
-------------------------------------------------------
    threshold.txt         <- recommended threshold (EMA basis unless
                              ema_reliable is False, then RAW basis)
    threshold_ema.txt      <- explicit EMA-basis threshold
    threshold_raw.txt      <- explicit RAW-basis threshold
    calibration_meta.json  <- everything needed to validate compatibility
    calibration_scores.csv <- per-detection raw/ema scores, tagged by source

    Pass threshold.txt (or the explicit ema/raw variant) straight into
    combined_gum_inspection_pipeline_compare_all.py's --*_threshold_file flags.
"""

import os
import sys
import csv
import json
import hashlib
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


def fingerprint_ckpt_dir(ckpt_dir: str) -> str:
    """Cheap staleness fingerprint: hash of (relative_path, size, mtime) for
    every file under ckpt_dir. Lets a later compatibility check detect 'this
    checkpoint was retrained/replaced after the threshold was calibrated'
    even though the --*_dir path itself hasn't changed."""
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


# ── Sources: expand any directory arg into the video files it contains ────
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".mpg", ".mpeg", ".wmv"}

def expand_sources(sources: List[str]) -> List[str]:
    expanded: List[str] = []
    for s in sources:
        if os.path.isdir(s):
            found = sorted(
                f for f in os.listdir(s)
                if os.path.splitext(f)[1].lower() in VIDEO_EXTS
            )
            if not found:
                raise SystemExit(f"--sources folder '{s}' contains no video files "
                                  f"(looked for {sorted(VIDEO_EXTS)})")
            print(f"  --sources folder '{s}' -> {len(found)} video(s): {found}")
            expanded.extend(os.path.join(s, f) for f in found)
        else:
            expanded.append(s)
    return expanded


# ── Phase 1: shared YOLO detection + tracking, run once PER VIDEO ─────────────

def run_yolo_all(sources: List[str], yolo_weights: str, conf: float, iou: float,
                 device: str) -> Dict[str, dict]:
    """Loads YOLO once, then runs detect+track once per source video, caching
    boxes per frame. Returns {source: {"detections":..., "src_w":, "src_h":,
    "n_frames":}}."""
    from ultralytics import YOLO
    print(f"[Phase 1] Loading YOLO: {yolo_weights}")
    yolo = YOLO(yolo_weights)
    yolo.to(device)

    out: Dict[str, dict] = {}
    for source in sources:
        print(f"\n[Phase 1] Video: {source}")
        src = int(source) if str(source).isdigit() else source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"  {src_w}x{src_h}  (~{total} frames)")

        detections: Dict[int, List[Tuple[int, int, int, int, int]]] = {}
        frame_idx = 0
        n_tracker_resets = 0
        force_tracker_reset = False
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            used_fallback = False
            try:
                results = yolo.track(source=frame, conf=conf, iou=iou, classes=[0],
                                     persist=not force_tracker_reset,
                                     tracker="bytetrack.yaml", verbose=False)
                force_tracker_reset = False
            except np.linalg.LinAlgError:
                # ByteTrack's Kalman filter can be fed a degenerate (0-width or
                # 0-height) box by ultralytics' own postprocessing, which drives
                # a divide-by-zero -> NaN/inf into the track covariance and makes
                # scipy's Cholesky factorization fail. This is NOT stale/corrupted
                # tracker state -- a fresh tracker fails identically on the same
                # bad box (confirmed: retrying track() with persist=False on the
                # same frame raises the exact same LinAlgError). So we never retry
                # track() on this frame. Instead: fall back to plain predict()
                # (no tracker involved, cannot hit this bug) for just this frame,
                # using synthetic large-offset IDs so they can't collide with real
                # persistent track IDs, and force a genuinely fresh tracker on the
                # NEXT frame (persist=False once) rather than immediately.
                n_tracker_resets += 1
                used_fallback = True
                force_tracker_reset = True
                print(f"  WARNING: degenerate detection box broke the tracker at "
                      f"frame {frame_idx} -- using untracked detections for this "
                      f"frame only, then reinitializing tracker "
                      f"[reset #{n_tracker_resets}]")
                results = yolo.predict(source=frame, conf=conf, iou=iou, classes=[0],
                                       verbose=False)
            boxes_this_frame = []
            for r in results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
                if used_fallback or r.boxes.id is None:
                    # Offset well clear of any real track ID range so a fallback
                    # frame's synthetic IDs can't be mistaken for a real track by
                    # the Phase 2 EMA smoothing (keyed on tid).
                    ids = [100_000 + frame_idx * 1000 + i for i in range(len(xyxy))]
                else:
                    ids = r.boxes.id.cpu().numpy().astype(int)
                for box, tid in zip(xyxy, ids):
                    x1 = max(0, box[0]); y1 = max(0, box[1])
                    x2 = min(src_w, box[2]); y2 = min(src_h, box[3])
                    if x2 > x1 and y2 > y1:
                        boxes_this_frame.append((int(tid), x1, y1, x2, y2))
            detections[frame_idx] = boxes_this_frame
            frame_idx += 1
            if frame_idx % 60 == 0:
                n_det = sum(len(v) for v in detections.values())
                print(f"  frame {frame_idx}/{total}  ({n_det} detections so far)")
        cap.release()
        n_frames = frame_idx
        n_det = sum(len(v) for v in detections.values())
        print(f"  Done: {n_frames} frames, {n_det} detections"
              + (f", {n_tracker_resets} tracker reset(s)" if n_tracker_resets else ""))
        if n_det == 0:
            print(f"  WARNING: no detections in {source}. Check YOLO weights / --conf.")
        out[source] = dict(detections=detections, src_w=src_w, src_h=src_h, n_frames=n_frames)

    widths  = {v["src_w"] for v in out.values()}
    heights = {v["src_h"] for v in out.values()}
    if len(widths) > 1 or len(heights) > 1:
        print(f"\n  NOTE: source videos have different resolutions "
              f"{[(v['src_w'], v['src_h']) for v in out.values()]}. Crops are taken relative "
              f"to each frame so this is normally fine -- just flagging in case it's unintended.")
    return out


# ── Phase 2: calibrate one method against the cached detections ──────────────

def compute_threshold_value(scores: np.ndarray, method: str,
                            percentile: float, std_k: float) -> float:
    if scores.size == 0:
        return float("nan")
    if method == "percentile":
        return float(np.percentile(scores, percentile))
    if method == "mean_std":
        return float(scores.mean() + std_k * scores.std())
    return float(scores.mean())  # legacy "mean"


def calibrate_method(method: str, ckpt_dir: str, video_data: Dict[str, dict],
                     args, stride: int = 1) -> Optional[dict]:
    print(f"\n{'='*70}\n  Method: {method}  (ckpt={ckpt_dir})"
          + (f"  [stride={stride}, ~1/{stride} detections]" if stride > 1 else "")
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
    log_w = csv.writer(log_file)
    log_w.writerow(["source", "frame", "track_id", "raw_score", "ema_score", "x1", "y1", "x2", "y2"])

    denoise = args.denoise
    all_raw: List[float] = []
    all_ema: List[float] = []
    per_video_ema: Dict[str, List[float]] = {}
    det_counter = 0  # global index across all videos/detections, for --sample_stride

    for source, vd in video_data.items():
        cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
        ema_state: Dict[int, float] = {}  # per-track EMA; resets per video (track IDs reset per video too)
        v_raw: List[float] = []
        v_ema: List[float] = []

        for frame_idx in range(vd["n_frames"]):
            ret, frame = cap.read()
            if not ret:
                break
            dets = vd["detections"].get(frame_idx, [])
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
                crop = get_crop(pil_frame, (x1, y1, x2, y2), image_size, args.pad_ratio, denoise)
                score = score_crop(models, crop, device, image_size, ratio=args.ratio)

                prev = ema_state.get(tid, score)
                ema = args.ema_alpha * score + (1 - args.ema_alpha) * prev
                ema_state[tid] = ema

                all_raw.append(score); all_ema.append(ema)
                v_raw.append(score); v_ema.append(ema)
                log_w.writerow([source, frame_idx, tid, f"{score:.6f}", f"{ema:.6f}", x1, y1, x2, y2])

            if (frame_idx + 1) % 60 == 0 and v_raw:
                print(f"  [{method}] {os.path.basename(str(source))} "
                      f"frame {frame_idx + 1}/{vd['n_frames']}  detections={len(v_raw)}  "
                      f"raw_mean={np.mean(v_raw):.4f}  ema_mean={np.mean(v_ema):.4f}")

        cap.release()
        per_video_ema[source] = v_ema
        if v_raw:
            print(f"  [{method}] {os.path.basename(str(source))}: {len(v_raw)} detections, "
                  f"raw_mean={np.mean(v_raw):.4f}  ema_mean={np.mean(v_ema):.4f}")
        else:
            print(f"  [{method}] {os.path.basename(str(source))}: WARNING no detections scored")

    log_file.close()
    del models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not all_raw:
        print(f"  [{method}] ERROR: no detections scored across any video. Skipping threshold.")
        return None

    raw_arr = np.array(all_raw)
    ema_arr = np.array(all_ema)

    threshold_raw = compute_threshold_value(raw_arr, args.threshold_method,
                                            args.threshold_percentile, args.threshold_std_k)
    threshold_ema = compute_threshold_value(ema_arr, args.threshold_method,
                                            args.threshold_percentile, args.threshold_std_k)
    ema_reliable = (stride <= 1)  # stride>1 skips frames mid-track, breaking EMA continuity

    per_video_thr_ema = {
        src: compute_threshold_value(np.array(v), args.threshold_method,
                                     args.threshold_percentile, args.threshold_std_k)
        for src, v in per_video_ema.items() if v
    }
    if len(per_video_thr_ema) > 1:
        vals = list(per_video_thr_ema.values())
        spread = (max(vals) / min(vals)) if min(vals) > 0 else float("inf")
        print(f"  [{method}] per-video EMA threshold: "
              f"{ {os.path.basename(str(k)): round(v, 4) for k, v in per_video_thr_ema.items()} }  "
              f"(max/min={spread:.2f})")
        if spread > 1.3:
            print(f"  [{method}] WARNING: >30% spread across source videos -- the normal-score "
                  f"distribution looks session-dependent (lighting/camera/domain shift). Consider "
                  f"adding more source videos across sessions before trusting this pooled threshold.")

    print(f"  [{method}] detections={len(raw_arr)} across {len(video_data)} video(s)  "
          f"raw_mean={raw_arr.mean():.4f} raw_max={raw_arr.max():.4f}  "
          f"ema_mean={ema_arr.mean():.4f} ema_max={ema_arr.max():.4f}")
    print(f"  [{method}] threshold ({args.threshold_method}) -> raw={threshold_raw:.6f}  "
          f"ema={threshold_ema:.6f}"
          + ("" if ema_reliable else "   [EMA UNRELIABLE: stride>1 broke per-track continuity]"))

    recommended_basis = "ema" if ema_reliable else "raw"
    recommended_value = threshold_ema if ema_reliable else threshold_raw

    with open(os.path.join(method_out_dir, "threshold.txt"), "w") as f:
        f.write(f"{recommended_value:.8f}\n")
    with open(os.path.join(method_out_dir, "threshold_ema.txt"), "w") as f:
        f.write(f"{threshold_ema:.8f}\n")
    with open(os.path.join(method_out_dir, "threshold_raw.txt"), "w") as f:
        f.write(f"{threshold_raw:.8f}\n")
    print(f"  [{method}] Saved -> {method_out_dir}/threshold.txt  (basis={recommended_basis})")

    # ── Compatibility metadata: every parameter that must match between this
    # ── calibration run and the eventual inference run for the threshold to
    # ── actually be valid, plus a checkpoint fingerprint to catch retraining.
    meta = dict(
        method=method, sources=list(video_data.keys()), yolo_weights=args.yolo_weights,
        category=args.category, model_size=args.model_size, image_size=image_size,
        conf=args.conf, iou=args.iou, pad_ratio=args.pad_ratio, denoise=bool(denoise),
        ratio=args.ratio, ema_alpha=args.ema_alpha,
        threshold_method=args.threshold_method, threshold_percentile=args.threshold_percentile,
        threshold_std_k=args.threshold_std_k, sample_stride=stride, ema_reliable=ema_reliable,
        recommended_basis=recommended_basis, threshold=recommended_value,
        threshold_raw=threshold_raw, threshold_ema=threshold_ema,
        n_detections=len(raw_arr), n_videos=len(video_data),
        raw_mean=float(raw_arr.mean()), raw_std=float(raw_arr.std()), raw_max=float(raw_arr.max()),
        ema_mean=float(ema_arr.mean()), ema_std=float(ema_arr.std()), ema_max=float(ema_arr.max()),
        per_video_ema_threshold={os.path.basename(str(k)): v for k, v in per_video_thr_ema.items()},
        checkpoint_fingerprint=fingerprint_ckpt_dir(ckpt_dir),
    )
    meta_path = os.path.join(method_out_dir, "calibration_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return meta


# ── args + main ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Calibrate anomaly thresholds for all methods "
                    "(baseline/lowres/thin/qat/pruned) from one or more normal-only videos.")
    p.add_argument("--sources", nargs="+", default=None,
                   help="One or more normal-only videos (or webcam indices). Recommended: "
                        "2+ videos across different sessions/lighting/camera setups -- a "
                        "threshold from a single video captures that session's noise, not "
                        "general 'normal' variation.")
    p.add_argument("--source", default=None,
                   help="Deprecated single-video alias for --sources.")
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
                   help="YOLO confidence threshold. Keep equal to whatever --conf you'll actually "
                        "run inference with -- a different value changes which boxes get scored.")
    p.add_argument("--iou",          type=float, default=0.45)
    p.add_argument("--pad_ratio",    type=float, default=0.02)
    p.add_argument("--denoise",      action="store_true",
                   help="Apply Gaussian+bilateral denoising on crops. Must match the inference "
                        "run's --denoise setting.")
    p.add_argument("--ema_alpha",    type=float, default=0.3,
                   help="EMA weight for per-track score smoothing. MUST match inference time.")
    p.add_argument("--threshold_method", choices=["percentile", "mean_std", "mean"],
                   default="percentile",
                   help="How to turn the pool of normal scores into a threshold. 'percentile' "
                        "(default, recommended): the Nth percentile of normal scores. 'mean_std': "
                        "mean + k*std. 'mean': legacy behaviour -- flags roughly half of even the "
                        "normal calibration set, kept only for backward compatibility.")
    p.add_argument("--threshold_percentile", type=float, default=99.0,
                   help="Percentile used when --threshold_method percentile.")
    p.add_argument("--threshold_std_k", type=float, default=3.0,
                   help="k used when --threshold_method mean_std (threshold = mean + k*std).")
    p.add_argument("--sample_stride", type=int, default=1,
                   help="Score every Nth cached detection instead of all of them. Applies to "
                        "every method unless overridden below.")
    p.add_argument("--qat_sample_stride", type=int, default=None,
                   help="Override --sample_stride for qat only (usual CPU bottleneck).")
    p.add_argument("--pruned_sample_stride", type=int, default=None,
                   help="Override --sample_stride for pruned only.")
    p.add_argument("--out_dir",      default="./calibration_all")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sources = args.sources or ([args.source] if args.source else None)
    if not sources:
        raise SystemExit("Provide --sources (one or more normal-only videos).")
    sources = expand_sources(sources)

    if len(sources) == 1:
        print(f"{'='*70}\n  WARNING: calibrating from a single video\n{'='*70}")
        print("  A threshold fit to one recording session captures that session's "
              "lighting/background/camera-angle noise, not general 'normal' variation.\n"
              "  This project previously hit an 11x domain shift between studio and "
              "smartphone footage from exactly this kind of single-source calibration.\n"
              "  Recommend >=2-3 videos across different sessions/lighting/times of day "
              "before trusting this threshold in production.\n")

    print(f"{'='*70}\n  Compatibility-critical settings "
          f"(MUST match combined_gum_inspection_pipeline_compare_all.py)\n{'='*70}")
    print(f"  --category      {args.category}")
    print(f"  --model_size    {args.model_size}")
    print(f"  --ratio         {args.ratio}")
    print(f"  --conf          {args.conf}")
    print(f"  --iou           {args.iou}")
    print(f"  --pad_ratio     {args.pad_ratio}")
    print(f"  --denoise       {args.denoise}")
    print(f"  --ema_alpha     {args.ema_alpha}")
    print(f"  --yolo_weights  {args.yolo_weights}")
    print(f"  threshold_method={args.threshold_method}"
          + (f" percentile={args.threshold_percentile}" if args.threshold_method == "percentile" else "")
          + (f" std_k={args.threshold_std_k}" if args.threshold_method == "mean_std" else ""))
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

    # ── Phase 1: YOLO once per video, shared across every method ──────────────
    video_data = run_yolo_all(sources, args.yolo_weights, args.conf, args.iou,
                              device='cuda' if torch.cuda.is_available() else 'cpu')

    # ── Phase 2: calibrate each method ─────────────────────────────────────────
    stride_overrides = {'qat': args.qat_sample_stride, 'pruned': args.pruned_sample_stride}

    metas: Dict[str, Optional[dict]] = {}
    for method in active_methods:
        stride = stride_overrides.get(method) or args.sample_stride
        metas[method] = calibrate_method(method, ckpt_dirs[method], video_data, args, stride=stride)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}\n  Calibration Summary\n{'='*70}")
    for method in active_methods:
        meta = metas[method]
        if meta is None:
            print(f"  {method:<10s}  FAILED / no detections")
        else:
            thr_path = os.path.join(args.out_dir, method, "threshold.txt")
            print(f"  {method:<10s}  threshold={meta['threshold']:.6f}  "
                  f"(basis={meta['recommended_basis']})   -> {thr_path}")

    print(f"\nPass these straight into combined_gum_inspection_pipeline_compare_all.py, e.g.:")
    for method in active_methods:
        if metas[method] is not None:
            print(f"  --{method}_threshold_file {os.path.join(args.out_dir, method, 'threshold.txt')}")


if __name__ == '__main__':
    main()