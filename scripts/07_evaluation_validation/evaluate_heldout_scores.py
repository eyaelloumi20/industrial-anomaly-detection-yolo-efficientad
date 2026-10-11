#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/ahmed/two-stage-efficientad-inspection")
EVAL = ROOT / "heldout_eval"

GT = {
    "video_A":  {1, 4, 12, 18, 34, 36},
    "video_A3": {6, 8, 18, 27, 34, 50},
    "video_A4": {2, 15, 18, 30, 49, 55},
}

METHODS = ["baseline", "lowres192", "lowres224", "thin075", "thin050", "pruned030"]

rows = []
track_rows = []

for method in METHODS:
    for video, anomalous_ids in GT.items():
        csv_path = EVAL / method / video / "scores.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing: {csv_path}")

        df = pd.read_csv(csv_path)

        required = {"track_id", "ema_score", "verdict"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{csv_path}: missing columns {sorted(missing)}")

        df["track_id"] = pd.to_numeric(df["track_id"], errors="coerce")
        df["ema_score"] = pd.to_numeric(df["ema_score"], errors="coerce")
        df["verdict"] = df["verdict"].astype(str).str.strip().str.upper()

        valid = df[
            df["track_id"].notna()
            & df["ema_score"].notna()
            & df["verdict"].isin(["NORMAL", "ANOMALY"])
        ].copy()

        valid["track_id"] = valid["track_id"].astype(int)
        valid["gt_label"] = valid["track_id"].isin(anomalous_ids).astype(int)
        valid["pred_label"] = (valid["verdict"] == "ANOMALY").astype(int)

        gt = valid["gt_label"]
        pred = valid["pred_label"]

        tp = int(((gt == 1) & (pred == 1)).sum())
        fp = int(((gt == 0) & (pred == 1)).sum())
        fn = int(((gt == 1) & (pred == 0)).sum())
        tn = int(((gt == 0) & (pred == 0)).sum())

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        accuracy = (tp + tn) / len(valid) if len(valid) else 0.0

        rows.append({
            "method": method,
            "video": video,
            "n_valid_frames": int(len(valid)),
            "n_gt_anomaly_frames": int((gt == 1).sum()),
            "n_gt_normal_frames": int((gt == 0).sum()),
            "n_pred_anomaly_frames": int((pred == 1).sum()),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        })

        for tid, g in valid.groupby("track_id"):
            gt_anom = int(tid in anomalous_ids)
            anomaly_fraction = float(g["pred_label"].mean())

            track_rows.append({
                "method": method,
                "video": video,
                "track_id": int(tid),
                "gt_anomaly": gt_anom,
                "n_frames": int(len(g)),
                "anomaly_frames": int(g["pred_label"].sum()),
                "anomaly_fraction": anomaly_fraction,
                "ema_mean": float(g["ema_score"].mean()),
                "ema_max": float(g["ema_score"].max()),
            })

frame_df = pd.DataFrame(rows)
track_df = pd.DataFrame(track_rows)

overall = (
    frame_df.groupby("method", as_index=False)[
        ["TP", "FP", "FN", "TN", "n_valid_frames",
         "n_gt_anomaly_frames", "n_gt_normal_frames", "n_pred_anomaly_frames"]
    ]
    .sum()
)

overall["precision"] = (
    overall["TP"] / (overall["TP"] + overall["FP"]).replace(0, np.nan)
)
overall["recall"] = (
    overall["TP"] / (overall["TP"] + overall["FN"]).replace(0, np.nan)
)
overall["f1"] = (
    2 * overall["precision"] * overall["recall"]
    / (overall["precision"] + overall["recall"])
)
overall["accuracy"] = (
    (overall["TP"] + overall["TN"]) / overall["n_valid_frames"].replace(0, np.nan)
)
overall = overall.fillna(0)

gt_tracks = track_df[track_df["gt_anomaly"] == 1].copy()
normal_tracks = track_df[track_df["gt_anomaly"] == 0].copy()

track_summary = (
    gt_tracks.groupby("method", as_index=False)
    .agg(
        anomalous_tracks=("track_id", "count"),
        mean_anomaly_fraction=("anomaly_fraction", "mean"),
        tracks_any_red=("anomaly_fraction", lambda s: int((s > 0).sum())),
        tracks_25pct_red=("anomaly_fraction", lambda s: int((s >= 0.25).sum())),
        tracks_majority_red=("anomaly_fraction", lambda s: int((s >= 0.50).sum())),
    )
)

normal_track_summary = (
    normal_tracks.groupby("method", as_index=False)
    .agg(
        normal_tracks=("track_id", "count"),
        mean_false_red_fraction=("anomaly_fraction", "mean"),
        normal_tracks_any_red=("anomaly_fraction", lambda s: int((s > 0).sum())),
        normal_tracks_majority_red=("anomaly_fraction", lambda s: int((s >= 0.50).sum())),
    )
)

outdir = EVAL
frame_df.to_csv(outdir / "per_video_metrics_v2.csv", index=False)
track_df.to_csv(outdir / "per_track_metrics_v2.csv", index=False)
overall.to_csv(outdir / "overall_metrics_v2.csv", index=False)
track_summary.to_csv(outdir / "anomalous_track_summary_v2.csv", index=False)
normal_track_summary.to_csv(outdir / "normal_track_summary_v2.csv", index=False)

print("\nOVERALL HELD-OUT FRAME-LEVEL METRICS")
print("=" * 110)
print(
    overall[
        ["method", "TP", "FP", "FN", "TN", "precision", "recall", "f1", "accuracy"]
    ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
)

print("\nANOMALOUS-TRACK COVERAGE")
print("=" * 110)
print(
    track_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}")
)

print("\nNORMAL-TRACK FALSE-RED SUMMARY")
print("=" * 110)
print(
    normal_track_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}")
)

print("\nPER-VIDEO METRICS")
print("=" * 110)
print(
    frame_df[
        ["method", "video", "TP", "FP", "FN", "TN", "precision", "recall", "f1", "accuracy"]
    ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
)

print("\nSaved:")
for name in [
    "per_video_metrics_v2.csv",
    "per_track_metrics_v2.csv",
    "overall_metrics_v2.csv",
    "anomalous_track_summary_v2.csv",
    "normal_track_summary_v2.csv",
]:
    print(f"  {outdir / name}")
