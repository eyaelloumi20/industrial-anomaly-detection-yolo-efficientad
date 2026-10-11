#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/ahmed/two-stage-efficientad-inspection")
DETECTOR = ROOT / "video_streaming_detector_simple_optimized_v2.py"
YOLO = ROOT / "runs/detect/yolo_runs/chewinggum_yolov8n_v2/weights/best.pt"
CKPT = ROOT / "ckpt_cropped_multi"
OUTROOT = ROOT / "calibration_ema_variants/baseline"
THRDIR = ROOT / "video_calibration_new"

VIDEOS = [
    ROOT / "videos_N/videoN1.mp4",
    ROOT / "videos_N/video_N.mp4",
    ROOT / "videos_N/video_N2.mp4",
]

def main():
    all_ema = []
    all_raw = []
    per_video = {}

    for video in VIDEOS:
        outdir = OUTROOT / video.stem
        outdir.mkdir(parents=True, exist_ok=True)
        csv_path = outdir / "scores.csv"

        if not csv_path.exists():
            cmd = [
                sys.executable, str(DETECTOR),
                "--video", str(video),
                "--yolo_det_weights", str(YOLO),
                "--method", "baseline",
                "--ckpt_dir", str(CKPT),
                "--threshold", "0.45",  # temporary only
                "--category", "chewinggum",
                "--model_size", "S",
                "--device", "cuda:0",
                "--fp16",
                "--threshold_basis", "ema",
                "--ema_alpha", "0.3",
                "--clip_margin", "5",
                "--min_box_frac", "0.5",
                "--batch_frames", "16",
                "--batch_crops", "96",
                "--gpu_batch_crops", "8",
                "--crop_workers", "12",
                "--no_detect_unknown",
                "--no_video",
                "--save_dir", str(outdir),
            ]
            print("\n" + "="*78)
            print(f"BASELINE EMA CALIBRATION: {video.name}")
            print("="*78)
            subprocess.run(cmd, cwd=ROOT, check=True)
        else:
            print(f"[reuse] {csv_path}")

        df = pd.read_csv(csv_path)
        ema = pd.to_numeric(df["ema_score"], errors="coerce").dropna().to_numpy(float)
        raw = pd.to_numeric(df["raw_score"], errors="coerce").dropna().to_numpy(float)

        if len(ema) == 0:
            raise RuntimeError(f"No valid EMA scores in {csv_path}")

        all_ema.append(ema)
        all_raw.append(raw)
        per_video[video.stem] = {
            "n_scores": int(len(ema)),
            "ema_mean": float(np.mean(ema)),
            "ema_p95": float(np.quantile(ema, 0.95)),
            "ema_p99": float(np.quantile(ema, 0.99)),
            "ema_max": float(np.max(ema)),
        }

    ema = np.concatenate(all_ema)
    raw = np.concatenate(all_raw)

    tau = float(np.quantile(ema, 0.99))
    tau_raw = float(np.quantile(raw, 0.99))

    result = {
        "method": "baseline",
        "variant": "baseline",
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
        "raw_quantile_same_level": tau_raw,
        "n_videos": len(VIDEOS),
        "n_scores_total": int(len(ema)),
        "ema_mean": float(np.mean(ema)),
        "ema_std": float(np.std(ema)),
        "ema_min": float(np.min(ema)),
        "ema_max": float(np.max(ema)),
        "per_video": per_video,
        "settings": {
            "fp16": True,
            "gpu_batch_crops": 8,
            "batch_frames": 16,
            "batch_crops": 96,
            "crop_workers": 12,
            "clip_margin": 5,
            "min_box_frac": 0.5,
            "threshold_basis": "ema",
            "ema_alpha": 0.3,
        }
    }

    THRDIR.mkdir(parents=True, exist_ok=True)
    out = THRDIR / "video_thresholds_baseline_ema.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n" + "="*78)
    print("BASELINE EMA CALIBRATION COMPLETE")
    print("="*78)
    print(f"tau_ema = {tau:.6f}")
    print(f"n       = {len(ema)}")
    print(f"saved   = {out}")

if __name__ == "__main__":
    main()
