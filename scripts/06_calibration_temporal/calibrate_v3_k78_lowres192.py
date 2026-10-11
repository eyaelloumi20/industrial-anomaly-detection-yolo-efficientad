#!/usr/bin/env python3
"""
Original normal-only calibration adapted to v3 + k78 mean-border preprocessing.

Uses:
- video_streaming_detector_simple_optimized_v3.py
- NEW multi-background YOLO
- LowRes192
- three original normal calibration videos:
    videoN1.mp4
    video_N.mp4
    video_N2.mp4
- EMA alpha = 0.3
- calibration basis = EMA
- threshold = 99th percentile of all normal EMA scores

This script calibrates ONLY LowRes192 because that is the currently selected
final anomaly model.
"""

from pathlib import Path
import csv
import json
import subprocess
import sys
import numpy as np

ROOT = Path("/home/ahmed/two-stage-efficientad-inspection")
PY = sys.executable

DETECTOR = ROOT / "video_streaming_detector_simple_optimized_v3.py"
YOLO = ROOT / "runs/detect/runs/detect/yolo_runs/chewinggum_yolov8n_multibg/weights/best.pt"
CKPT = ROOT / "ckpt_lowres_192"

VIDEOS = [
    ROOT / "videos_N/videoN1.mp4",
    ROOT / "videos_N/video_N.mp4",
    ROOT / "videos_N/video_N2.mp4",
]

OUT_SCORES = ROOT / "calibration_v3_k78_lowres192"
OUT_THRESH = ROOT / "video_calibration_v3_k78"

OUT_SCORES.mkdir(parents=True, exist_ok=True)
OUT_THRESH.mkdir(parents=True, exist_ok=True)

for p in [DETECTOR, YOLO, CKPT, *VIDEOS]:
    if not p.exists():
        raise SystemExit(f"ERROR: missing path: {p}")

all_ema = []
all_raw = []
per_video = {}

for video in VIDEOS:
    tag = video.stem
    save_dir = OUT_SCORES / tag
    save_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 88)
    print(f"V3 K78 CALIBRATION: {video.name}")
    print("=" * 88)

    cmd = [
        PY, str(DETECTOR),
        "--video", str(video),
        "--yolo_det_weights", str(YOLO),
        "--method", "lowres",
        "--ckpt_dir", str(CKPT),
        "--threshold", "0.45",
        "--category", "chewinggum",
        "--model_size", "S",
        "--device", "cuda:0",
        "--fp16",
        "--threshold_basis", "ema",
        "--ema_alpha", "0.3",
        "--crop_preprocess", "mean_border",
        "--mean_border_keep_ratio", "0.78",
        "--mean_border_feather_ratio", "0.08",
        "--mean_border_strength", "1.0",
        "--clip_margin", "5",
        "--min_box_frac", "0.5",
        "--batch_frames", "16",
        "--batch_crops", "96",
        "--gpu_batch_crops", "8",
        "--crop_workers", "12",
        "--no_detect_unknown",
        "--no_video",
        "--save_dir", str(save_dir),
    ]

    subprocess.run(cmd, check=True)

    csv_path = save_dir / "scores.csv"
    if not csv_path.exists():
        raise SystemExit(f"ERROR: missing score CSV: {csv_path}")

    raw_vals = []
    ema_vals = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            raw = (r.get("raw_score") or "").strip()
            ema = (r.get("ema_score") or "").strip()

            if raw:
                raw_vals.append(float(raw))
            if ema:
                ema_vals.append(float(ema))

    if not ema_vals:
        raise SystemExit(f"ERROR: no EMA scores in {csv_path}")

    all_raw.extend(raw_vals)
    all_ema.extend(ema_vals)

    per_video[tag] = {
        "video": str(video),
        "n_scores": len(ema_vals),
        "raw_mean": float(np.mean(raw_vals)) if raw_vals else None,
        "raw_p99": float(np.quantile(raw_vals, 0.99)) if raw_vals else None,
        "ema_mean": float(np.mean(ema_vals)),
        "ema_p95": float(np.quantile(ema_vals, 0.95)),
        "ema_p99": float(np.quantile(ema_vals, 0.99)),
        "ema_max": float(np.max(ema_vals)),
    }

all_ema = np.asarray(all_ema, dtype=np.float64)
all_raw = np.asarray(all_raw, dtype=np.float64)

tau = float(np.quantile(all_ema, 0.99))
raw_q = float(np.quantile(all_raw, 0.99)) if len(all_raw) else None

result = {
    "method": "lowres",
    "variant": "lowres192_v3_k78",
    "category": "chewinggum",
    "model_size": "S",
    "ckpt_dir": str(CKPT),
    "calibration_type": "normal_only_quantile",
    "calibration_basis": "ema",
    "recommended_basis": "ema",
    "ema_alpha": 0.3,
    "quantile": 0.99,
    "alpha_p": 0.01,
    "tau_p": tau,
    "tau_f": tau,
    "raw_quantile_same_level": raw_q,
    "n_videos": len(VIDEOS),
    "n_scores_total": int(len(all_ema)),
    "ema_mean": float(np.mean(all_ema)),
    "ema_std": float(np.std(all_ema)),
    "ema_min": float(np.min(all_ema)),
    "ema_max": float(np.max(all_ema)),
    "per_video": per_video,
    "settings": {
        "detector": str(DETECTOR),
        "yolo": str(YOLO),
        "crop_preprocess": "mean_border",
        "mean_border_keep_ratio": 0.78,
        "mean_border_feather_ratio": 0.08,
        "mean_border_strength": 1.0,
        "fp16": True,
        "gpu_batch_crops": 8,
        "batch_frames": 16,
        "batch_crops": 96,
        "crop_workers": 12,
        "clip_margin": 5,
        "min_box_frac": 0.5,
        "threshold_basis": "ema",
        "ema_alpha": 0.3,
    },
}

out_json = OUT_THRESH / "video_thresholds_lowres192_v3_k78_ema.json"
out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

print()
print("=" * 88)
print("V3 K78 CALIBRATION COMPLETE")
print("=" * 88)
print(f"tau_ema       : {tau:.6f}")
print(f"n_scores      : {len(all_ema)}")
print(f"ema_mean      : {np.mean(all_ema):.6f}")
print(f"ema_std       : {np.std(all_ema):.6f}")
print(f"ema_min       : {np.min(all_ema):.6f}")
print(f"ema_max       : {np.max(all_ema):.6f}")
print(f"saved         : {out_json}")
