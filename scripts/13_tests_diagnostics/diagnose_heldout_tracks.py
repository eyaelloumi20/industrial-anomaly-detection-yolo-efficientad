#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/home/ahmed/two-stage-efficientad-inspection")
EVAL = ROOT / "heldout_eval"

GT = {
    "video_A":  {1, 4, 12, 18, 34, 36},
    "video_A3": {6, 8, 18, 27, 34, 50},
    "video_A4": {2, 15, 18, 30, 49, 55},
}
METHODS = ["baseline", "lowres192", "lowres224", "thin075", "thin050", "pruned030"]

sel_path = ROOT / "threshold_selection/selected_thresholds.csv"
selected = {}
if sel_path.is_file():
    sdf = pd.read_csv(sel_path)
    selected = dict(zip(sdf["method"], sdf["selected_threshold"]))

rows = []
top_rows = []

for method in METHODS:
    for video, gt_ids in GT.items():
        p = EVAL / method / video / "scores.csv"
        df = pd.read_csv(p)
        df["track_id"] = pd.to_numeric(df["track_id"], errors="coerce")
        df["ema_score"] = pd.to_numeric(df["ema_score"], errors="coerce")
        df = df[df["track_id"].notna() & df["ema_score"].notna()].copy()
        df["track_id"] = df["track_id"].astype(int)

        thr = selected.get(method, np.nan)

        # Every GT track
        for tid in sorted(gt_ids):
            g = df[df["track_id"] == tid]
            if len(g) == 0:
                rows.append({
                    "method": method, "video": video, "track_id": tid,
                    "n_frames": 0, "ema_mean": np.nan, "ema_median": np.nan,
                    "ema_p90": np.nan, "ema_max": np.nan,
                    "selected_threshold": thr, "red_fraction": 0.0,
                    "missing_track": True,
                })
            else:
                rows.append({
                    "method": method, "video": video, "track_id": tid,
                    "n_frames": len(g),
                    "ema_mean": g["ema_score"].mean(),
                    "ema_median": g["ema_score"].median(),
                    "ema_p90": g["ema_score"].quantile(0.90),
                    "ema_max": g["ema_score"].max(),
                    "selected_threshold": thr,
                    "red_fraction": float((g["ema_score"] >= thr).mean()) if np.isfinite(thr) else np.nan,
                    "missing_track": False,
                })

        # Rank every track by max and mean EMA, useful for checking ID mismatch.
        agg = (
            df.groupby("track_id")["ema_score"]
            .agg(["count", "mean", "median", "max"])
            .reset_index()
        )
        agg["is_gt"] = agg["track_id"].isin(gt_ids)
        agg = agg.sort_values(["max", "mean"], ascending=False).head(15)

        for rank, (_, r) in enumerate(agg.iterrows(), start=1):
            top_rows.append({
                "method": method, "video": video, "rank": rank,
                "track_id": int(r["track_id"]),
                "n_frames": int(r["count"]),
                "ema_mean": float(r["mean"]),
                "ema_median": float(r["median"]),
                "ema_max": float(r["max"]),
                "is_gt": bool(r["is_gt"]),
            })

gt_df = pd.DataFrame(rows)
top_df = pd.DataFrame(top_rows)

out = ROOT / "threshold_selection"
gt_df.to_csv(out / "diagnostic_gt_tracks.csv", index=False)
top_df.to_csv(out / "diagnostic_top_score_tracks.csv", index=False)

print("\nGROUND-TRUTH TRACK SCORE DIAGNOSTIC")
print("=" * 130)
for video in GT:
    print(f"\n### {video}")
    v = gt_df[gt_df["video"] == video]
    print(v[
        ["method","track_id","n_frames","ema_mean","ema_median","ema_p90","ema_max",
         "selected_threshold","red_fraction","missing_track"]
    ].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\n\nTOP 15 TRACKS BY EMA MAX — CHECK WHETHER GT IDS APPEAR")
print("=" * 130)
for method in METHODS:
    for video in ["video_A3", "video_A4"]:
        print(f"\n### {method} / {video}")
        t = top_df[(top_df["method"] == method) & (top_df["video"] == video)]
        print(t[["rank","track_id","n_frames","ema_mean","ema_max","is_gt"]]
              .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\nSaved:")
print(out / "diagnostic_gt_tracks.csv")
print(out / "diagnostic_top_score_tracks.csv")
