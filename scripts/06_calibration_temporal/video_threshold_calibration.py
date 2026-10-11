"""
video_threshold_calibration.py
==============================
Calibrate per-method thresholds tau_p (crop-level) and tau_f (frame-level)
on a folder of NORMAL-ONLY calibration videos.

Design
------
- Every crop from every frame of every calibration video is scored once.
- Crop scores are pooled across all videos into ONE distribution.
- Per-frame max score is also collected, giving a second distribution.
- tau_p = (1 - alpha_p)-quantile of the pooled crop-score distribution.
- tau_f = (1 - alpha_f)-quantile of the pooled per-frame max distribution.
- Anchors q_a = p90 and q_b = p99.5 of pooled crop scores are kept for
  diagnostics only (mirrors EfficientAD's a=0.90 / b=0.995 mapping).

This is the video analogue of EfficientAD's image-level normalization:
quantiles are computed over the whole calibration set, not per image.

The companion streaming detector (video_streaming_detector.py) reads the
JSON written here without modification.

Usage
-----
    # Folder of normal-only videos:
    python video_threshold_calibration.py \
        --calib_video_dir ./data/normal_line \
        --yolo_det_weights ./yolo_runs/.../best.pt \
        --baseline_dir /home/ahmed/framework/ckpt_cropped_large \
        --thin_dir     ./ckpt_thin_025 \
        --pruned_dir   ./ckpt_pruned_030 \
        --category chewinggum --model_size S \
        --alpha_p 0.01 --alpha_f 0.01 \
        --save_dir ./video_calibration

    # Still accepts a single video:
    python video_threshold_calibration.py --calib_video ./data/one.mp4 ...
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Reuse the exact Phase-1 / Phase-2 machinery from the batch comparison script
from compare_pipeline_methods_batched import (
    METHODS, load_bundle, run_yolo_once, get_crop, score_crop_batch,
)

import cv2
from PIL import Image


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".webm"}


# ── args ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Calibrate video thresholds (crop-level + frame-level) "
                    "on normal-only calibration videos.")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--calib_video",     default=None,
                     help="Single normal-only calibration video.")
    grp.add_argument("--calib_video_dir", default=None,
                     help="Folder of normal-only calibration videos. All "
                          "videos in the folder are treated as calibration "
                          "data and pooled into one threshold.")

    p.add_argument("--yolo_det_weights", required=True)

    p.add_argument("--baseline_dir", default=None)
    p.add_argument("--lowres_dir",   default=None)
    p.add_argument("--thin_dir",     default=None)
    p.add_argument("--qat_dir",      default=None)
    p.add_argument("--pruned_dir",   default=None)
    p.add_argument("--skip_methods", nargs="*", default=[])

    p.add_argument("--category",   default="chewinggum")
    p.add_argument("--model_size", default="S", choices=["S", "M"])
    p.add_argument("--device",     default="cuda")
    p.add_argument("--ratio",      type=float, default=0.1)
    p.add_argument("--conf",       type=float, default=0.25)
    p.add_argument("--iou",        type=float, default=0.45)
    p.add_argument("--pad_ratio",  type=float, default=0.02)
    p.add_argument("--denoise",    action="store_true")
    p.add_argument("--fp16",       action="store_true")
    p.add_argument("--max_frames", type=int, default=0,
                   help="Max frames PER VIDEO (0 = all).")
    p.add_argument("--alpha_p",    type=float, default=0.01,
                   help="Target crop-level false-positive rate.")
    p.add_argument("--alpha_f",    type=float, default=0.01,
                   help="Target frame-level false-positive rate.")
    # Online-update parameters saved for the streaming detector's use
    p.add_argument("--online_eta",   type=float, default=1e-3)
    p.add_argument("--online_recal", type=int,   default=300)
    p.add_argument("--save_dir", default="./video_calibration")
    return p.parse_args()


# ── video discovery ───────────────────────────────────────────────────────────
def gather_videos(args) -> list:
    """Return a list of (video_path_str, video_tag) tuples. The tag is used to
    prefix track IDs so tracks from different videos never collide in the
    per-track diagnostic statistics."""
    if args.calib_video:
        paths = [Path(args.calib_video)]
    else:
        d = Path(args.calib_video_dir)
        if not d.is_dir():
            raise SystemExit(f"--calib_video_dir '{d}' is not a folder.")
        paths = sorted(p for p in d.iterdir()
                       if p.suffix.lower() in VIDEO_EXTS)

    if not paths:
        raise SystemExit("No calibration videos found.")

    out = []
    for i, p in enumerate(paths):
        if not p.is_file():
            raise SystemExit(f"Calibration video not found: {p}")
        # keep tag short, filesystem-safe, unique
        tag = f"v{i:02d}_{p.stem}"[:40]
        out.append((str(p), tag))
    return out


# ── per-video scoring for one method ──────────────────────────────────────────
def score_video_for_method(models, image_size, device, use_amp,
                           frame_cache, detections, n_frames, args,
                           video_tag: str):
    """Return (raw_scores, frame_max_scores, per_track_scores)."""
    raw_scores = []
    frame_max  = []
    per_track  = {}

    for frame_idx in range(n_frames):
        pil_frame  = frame_cache.pil_frames[frame_idx]
        frame_dets = detections.get(frame_idx, [])

        crops = [get_crop(pil_frame, (x1, y1, x2, y2), image_size,
                          args.pad_ratio, args.denoise)
                 for _tid, x1, y1, x2, y2 in frame_dets]

        if not crops:
            frame_max.append(0.0)
            continue

        scores = score_crop_batch(models, crops, device, image_size,
                                  ratio=args.ratio, use_amp=use_amp)

        for (tid, *_box), s in zip(frame_dets, scores):
            raw_scores.append(float(s))
            prefixed = f"{video_tag}::{int(tid)}"
            per_track.setdefault(prefixed, []).append(float(s))

        frame_max.append(float(max(scores)))

    return raw_scores, frame_max, per_track


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Active methods
    ckpt_dirs = {
        "baseline": args.baseline_dir, "lowres": args.lowres_dir,
        "thin":     args.thin_dir,     "qat":    args.qat_dir,
        "pruned":   args.pruned_dir,
    }
    active = [m for m in METHODS
              if m not in args.skip_methods
              and ckpt_dirs[m] and os.path.isdir(ckpt_dirs[m])]
    if not active:
        raise SystemExit("No checkpoint dirs found.")

    videos = gather_videos(args)

    print(f"{'='*70}")
    print(f"  Video Threshold Calibration")
    print(f"{'='*70}")
    print(f"  Methods            : {active}")
    print(f"  Calibration videos : {len(videos)}")
    for path, tag in videos:
        print(f"     [{tag}]  {path}")
    print(f"  Target FPRs        : crop={args.alpha_p}  frame={args.alpha_f}")
    print(f"  Save dir           : {args.save_dir}")
    print(f"{'='*70}\n")

    # Pooled accumulators per method
    per_method = {m: {"raw": [], "frame_max": [], "per_track": {},
                      "per_video": {}} for m in active}

    # Process videos one at a time; frame cache is released before next video.
    for v_idx, (video_path, video_tag) in enumerate(videos):
        print(f"\n{'='*70}")
        print(f"  Video {v_idx + 1}/{len(videos)}: {video_tag}")
        print(f"  {video_path}")
        print(f"{'='*70}")

        detections, src_w, src_h, src_fps, n_frames, frame_cache = run_yolo_once(
            video_path, args.yolo_det_weights, args.conf, args.iou,
            device="cuda" if torch.cuda.is_available() else "cpu",
            max_frames=(args.max_frames or None),
            detect_unknown=False,
            cache_frames=True)

        if n_frames == 0:
            print(f"  [SKIP] no frames decoded from {video_path}")
            del frame_cache
            continue

        for method in active:
            ckpt_dir = ckpt_dirs[method]
            print(f"\n  [{method}] scoring {n_frames} frames "
                  f"(ckpt={ckpt_dir}) ...")

            models, image_size, device = load_bundle(
                method, ckpt_dir, args.category, args.model_size, args.device)
            use_amp = args.fp16 and device.startswith("cuda")

            t0 = time.perf_counter()
            raw, fmax, per_track = score_video_for_method(
                models, image_size, device, use_amp,
                frame_cache, detections, n_frames, args, video_tag)
            dt = time.perf_counter() - t0

            del models
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            n_crops = len(raw)
            print(f"  [{method}] {n_crops} crops, {n_frames} frames "
                  f"in {dt:.1f}s ({n_frames/max(dt,1e-6):.1f} fps, "
                  f"{n_crops/max(dt,1e-6):.1f} crops/s)")

            # Accumulate pooled arrays
            per_method[method]["raw"].extend(raw)
            per_method[method]["frame_max"].extend(fmax)
            for tid, vals in per_track.items():
                per_method[method]["per_track"].setdefault(tid, []).extend(vals)

            # Per-video diagnostic
            if raw:
                per_method[method]["per_video"][video_tag] = {
                    "video_path":    os.path.abspath(video_path),
                    "n_frames":      int(n_frames),
                    "n_crops":       int(n_crops),
                    "raw_mean":      float(np.mean(raw)),
                    "raw_std":       float(np.std(raw)),
                    "raw_p99":       float(np.quantile(raw, 0.99)),
                    "raw_max":       float(np.max(raw)),
                    "frame_max_p99": float(np.quantile(fmax, 0.99))
                                     if len(fmax) else 0.0,
                }

        del frame_cache, detections

    # ── Pooled thresholds per method ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Computing pooled thresholds")
    print(f"{'='*70}\n")

    summary = {}
    for method in active:
        data = per_method[method]
        raw  = np.asarray(data["raw"],       dtype=np.float32)
        fmax = np.asarray(data["frame_max"], dtype=np.float32)

        if raw.size == 0:
            print(f"  [{method}] no pooled crops -- skipping.")
            continue

        # The two thresholds: crop-level and frame-level
        tau_p = float(np.quantile(raw,  1.0 - args.alpha_p))
        tau_f = float(np.quantile(fmax, 1.0 - args.alpha_f))

        # Anchors (diagnostics only)
        q_a = float(np.quantile(raw, 0.90))
        q_b = float(np.quantile(raw, 0.995))

        # Per-track summary
        track_stats = {}
        for tid, vals in data["per_track"].items():
            v = np.asarray(vals, dtype=np.float32)
            if v.size == 0:
                continue
            track_stats[tid] = {
                "n":    int(v.size),
                "mean": float(v.mean()),
                "std":  float(v.std()),
                "p99":  float(np.quantile(v, 0.99)),
            }

        out = {
            "method":      method,
            "category":    args.category,
            "model_size":  args.model_size,
            "ckpt_dir":    os.path.abspath(ckpt_dirs[method]),

            "calibration_videos":  [os.path.abspath(p) for p, _ in videos],
            "calibration_tags":    [t for _, t in videos],
            "n_videos":            len(videos),

            "n_frames_total":      int(sum(v["n_frames"]
                                           for v in data["per_video"].values())),
            "n_crops_total":       int(raw.size),

            "alpha_p":             args.alpha_p,
            "alpha_f":             args.alpha_f,
            "tau_p":               tau_p,
            "tau_f":               tau_f,

            "q_a":                 q_a,
            "q_b":                 q_b,
            "raw_mean":            float(raw.mean()),
            "raw_std":             float(raw.std()),
            "raw_min":             float(raw.min()),
            "raw_max":             float(raw.max()),
            "frame_max_mean":      float(fmax.mean()) if fmax.size else 0.0,
            "frame_max_p99":       float(np.quantile(fmax, 0.99))
                                   if fmax.size else 0.0,

            "per_video":           data["per_video"],
            "per_track":           track_stats,

            "online_eta":          args.online_eta,
            "online_recal":        args.online_recal,
        }

        out_path = os.path.join(args.save_dir,
                                f"video_thresholds_{method}.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)

        print(f"  [{method}]")
        print(f"      videos={len(videos)}  frames={out['n_frames_total']}  "
              f"crops={out['n_crops_total']}")
        print(f"      tau_p = {tau_p:.4f}   tau_f = {tau_f:.4f}   "
              f"(q_a={q_a:.4f}  q_b={q_b:.4f})")
        for tag, st in data["per_video"].items():
            print(f"      [{tag}] crops={st['n_crops']:5d}  "
                  f"mean={st['raw_mean']:.4f}  p99={st['raw_p99']:.4f}  "
                  f"max={st['raw_max']:.4f}")

        p99s = np.array([st["raw_p99"] for st in data["per_video"].values()],
                        dtype=np.float32)
        if p99s.size > 1:
            spread = float(p99s.max() - p99s.min())
            rel    = spread / max(float(p99s.mean()), 1e-6)
            flag   = "  <-- LARGE SPREAD" if rel > 0.5 else ""
            print(f"      per-video p99 spread: {spread:.4f} "
                  f"({100*rel:.1f}% of mean){flag}")

        summary[method] = {
            "tau_p":   tau_p,
            "tau_f":   tau_f,
            "n_crops": int(raw.size),
            "n_videos": len(videos),
        }

    with open(os.path.join(args.save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  Calibration complete -- {len(summary)} method(s)")
    print(f"{'='*70}")
    for m, s in summary.items():
        print(f"  {m:<10s}  tau_p={s['tau_p']:.4f}  tau_f={s['tau_f']:.4f}  "
              f"({s['n_crops']} crops from {s['n_videos']} videos)")
    print(f"\n  Outputs: {args.save_dir}/video_thresholds_*.json")


if __name__ == "__main__":
    main()