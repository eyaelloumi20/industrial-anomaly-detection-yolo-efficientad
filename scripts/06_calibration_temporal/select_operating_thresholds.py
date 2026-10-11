#!/usr/bin/env python3
"""
select_operating_thresholds.py

Select one EMA threshold per model WITHOUT rerunning inference.

Protocol:
  - Normal calibration videos define the normal-score distribution / FPR.
  - video_A is used ONLY as validation to choose the operating threshold.
  - video_A3 and video_A4 are used ONLY for final testing.

Default selection rule:
  maximize validation F1 subject to normal-calibration frame FPR <= 5%.

Also reports the unconstrained best validation threshold for diagnosis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/ahmed/two-stage-efficientad-inspection")

METHODS = [
    "baseline",
    "lowres192",
    "lowres224",
    "thin075",
    "thin050",
    "pruned030",
]

GT = {
    "video_A":  {1, 4, 12, 18, 34, 36},
    "video_A3": {6, 8, 18, 27, 34, 50},
    "video_A4": {2, 15, 18, 30, 49, 55},
}

NORMAL_VIDEOS = ["videoN1", "video_N", "video_N2"]


def load_score_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    required = {"track_id", "ema_score"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    df["track_id"] = pd.to_numeric(df["track_id"], errors="coerce")
    df["ema_score"] = pd.to_numeric(df["ema_score"], errors="coerce")

    # Same validity rule as the detector: scored crops only.
    df = df[df["track_id"].notna() & df["ema_score"].notna()].copy()
    df["track_id"] = df["track_id"].astype(int)
    return df


def metrics_at_threshold(df: pd.DataFrame, anomalous_ids: set[int], threshold: float):
    gt = df["track_id"].isin(anomalous_ids).astype(int).to_numpy()
    pred = (df["ema_score"].to_numpy(float) >= threshold).astype(int)

    tp = int(((gt == 1) & (pred == 1)).sum())
    fp = int(((gt == 0) & (pred == 1)).sum())
    fn = int(((gt == 1) & (pred == 0)).sum())
    tn = int(((gt == 0) & (pred == 0)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(df) if len(df) else 0.0

    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def anomalous_track_coverage(df: pd.DataFrame, anomalous_ids: set[int], threshold: float):
    rows = []
    for tid in sorted(anomalous_ids):
        g = df[df["track_id"] == tid]
        if len(g) == 0:
            rows.append({
                "track_id": tid,
                "n_frames": 0,
                "red_fraction": 0.0,
                "detected_any": 0,
                "majority_red": 0,
            })
            continue

        frac = float((g["ema_score"] >= threshold).mean())
        rows.append({
            "track_id": tid,
            "n_frames": int(len(g)),
            "red_fraction": frac,
            "detected_any": int(frac > 0),
            "majority_red": int(frac >= 0.5),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--max_normal_fpr",
        type=float,
        default=0.05,
        help="Maximum allowed frame-level false-positive rate on normal calibration data. Default 0.05.",
    )
    ap.add_argument(
        "--min_quantile",
        type=float,
        default=0.70,
        help="Lowest normal-score quantile included in diagnostic sweep. Default 0.70.",
    )
    ap.add_argument(
        "--max_quantile",
        type=float,
        default=0.999,
        help="Highest normal-score quantile included in sweep. Default 0.999.",
    )
    ap.add_argument(
        "--n_quantiles",
        type=int,
        default=300,
        help="Number of quantile candidates. Default 300.",
    )
    args = ap.parse_args()

    if not (0.0 < args.max_normal_fpr < 1.0):
        raise ValueError("--max_normal_fpr must be between 0 and 1")

    outdir = ROOT / "threshold_selection"
    outdir.mkdir(parents=True, exist_ok=True)

    selected_rows = []
    test_rows = []
    coverage_rows = []
    sweep_rows = []

    for method in METHODS:
        # Pool NORMAL calibration EMA scores.
        normal_scores = []
        for video in NORMAL_VIDEOS:
            p = ROOT / "calibration_ema_variants" / method / video / "scores.csv"
            ndf = load_score_csv(p)
            normal_scores.append(ndf["ema_score"].to_numpy(float))
        normal_scores = np.concatenate(normal_scores)

        val = load_score_csv(ROOT / "heldout_eval" / method / "video_A" / "scores.csv")

        quantiles = np.linspace(
            args.min_quantile,
            args.max_quantile,
            args.n_quantiles,
        )
        thresholds = np.unique(np.quantile(normal_scores, quantiles))

        candidates = []
        for q, thr in zip(quantiles, np.quantile(normal_scores, quantiles)):
            thr = float(thr)
            normal_fpr = float((normal_scores >= thr).mean())
            m = metrics_at_threshold(val, GT["video_A"], thr)
            row = {
                "method": method,
                "normal_quantile": float(q),
                "threshold": thr,
                "normal_fpr": normal_fpr,
                **m,
            }
            candidates.append(row)
            sweep_rows.append(row)

        cand = pd.DataFrame(candidates)

        # Diagnostic best F1 without an FPR constraint.
        best_any = cand.sort_values(
            ["f1", "recall", "precision"],
            ascending=False,
        ).iloc[0]

        # Scientifically safer operating point: FPR-constrained.
        feasible = cand[cand["normal_fpr"] <= args.max_normal_fpr].copy()
        if feasible.empty:
            raise RuntimeError(
                f"{method}: no threshold satisfies normal FPR <= {args.max_normal_fpr:.3f}"
            )

        best = feasible.sort_values(
            ["f1", "recall", "precision"],
            ascending=False,
        ).iloc[0]

        selected_rows.append({
            "method": method,
            "selected_threshold": float(best["threshold"]),
            "selected_normal_quantile": float(best["normal_quantile"]),
            "normal_fpr": float(best["normal_fpr"]),
            "val_precision": float(best["precision"]),
            "val_recall": float(best["recall"]),
            "val_f1": float(best["f1"]),
            "val_accuracy": float(best["accuracy"]),
            "unconstrained_best_threshold": float(best_any["threshold"]),
            "unconstrained_normal_fpr": float(best_any["normal_fpr"]),
            "unconstrained_val_precision": float(best_any["precision"]),
            "unconstrained_val_recall": float(best_any["recall"]),
            "unconstrained_val_f1": float(best_any["f1"]),
        })

        threshold = float(best["threshold"])

        # Final test: A3 + A4 only. These do NOT influence threshold selection.
        for video in ["video_A3", "video_A4"]:
            test = load_score_csv(ROOT / "heldout_eval" / method / video / "scores.csv")
            m = metrics_at_threshold(test, GT[video], threshold)
            test_rows.append({
                "method": method,
                "video": video,
                "threshold": threshold,
                **m,
            })

            cov = anomalous_track_coverage(test, GT[video], threshold)
            cov.insert(0, "video", video)
            cov.insert(0, "method", method)
            coverage_rows.extend(cov.to_dict("records"))

        # Save selected threshold JSON for later detector use.
        payload = {
            "method": method,
            "tau_p": threshold,
            "tau_f": threshold,
            "recommended_basis": "ema",
            "ema_alpha": 0.3,
            "selection_protocol": "video_A_validation_max_f1_with_normal_fpr_constraint",
            "max_normal_fpr": args.max_normal_fpr,
            "normal_quantile": float(best["normal_quantile"]),
            "normal_fpr": float(best["normal_fpr"]),
            "validation_video": "video_A",
            "final_test_videos": ["video_A3", "video_A4"],
            "validation_metrics": {
                "precision": float(best["precision"]),
                "recall": float(best["recall"]),
                "f1": float(best["f1"]),
                "accuracy": float(best["accuracy"]),
            },
        }
        (outdir / f"operating_threshold_{method}.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    selected_df = pd.DataFrame(selected_rows)
    test_df = pd.DataFrame(test_rows)
    coverage_df = pd.DataFrame(coverage_rows)
    sweep_df = pd.DataFrame(sweep_rows)

    # Aggregate A3+A4 only.
    overall_test = (
        test_df.groupby("method", as_index=False)[["TP", "FP", "FN", "TN"]].sum()
    )
    overall_test["precision"] = (
        overall_test["TP"]
        / (overall_test["TP"] + overall_test["FP"]).replace(0, np.nan)
    )
    overall_test["recall"] = (
        overall_test["TP"]
        / (overall_test["TP"] + overall_test["FN"]).replace(0, np.nan)
    )
    overall_test["f1"] = (
        2 * overall_test["precision"] * overall_test["recall"]
        / (overall_test["precision"] + overall_test["recall"])
    )
    total = overall_test[["TP", "FP", "FN", "TN"]].sum(axis=1)
    overall_test["accuracy"] = (
        (overall_test["TP"] + overall_test["TN"]) / total.replace(0, np.nan)
    )
    overall_test = overall_test.fillna(0)

    selected_df.to_csv(outdir / "selected_thresholds.csv", index=False)
    sweep_df.to_csv(outdir / "validation_threshold_sweep.csv", index=False)
    test_df.to_csv(outdir / "final_test_per_video.csv", index=False)
    overall_test.to_csv(outdir / "final_test_overall.csv", index=False)
    coverage_df.to_csv(outdir / "final_test_track_coverage.csv", index=False)

    print("\nSELECTED OPERATING THRESHOLDS — VIDEO_A VALIDATION ONLY")
    print("=" * 125)
    print(selected_df[
        [
            "method",
            "selected_threshold",
            "selected_normal_quantile",
            "normal_fpr",
            "val_precision",
            "val_recall",
            "val_f1",
            "unconstrained_normal_fpr",
            "unconstrained_val_f1",
        ]
    ].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nFINAL TEST — VIDEO_A3 + VIDEO_A4 ONLY")
    print("=" * 125)
    print(overall_test[
        ["method", "TP", "FP", "FN", "TN", "precision", "recall", "f1", "accuracy"]
    ].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nFINAL TEST — PER VIDEO")
    print("=" * 125)
    print(test_df[
        ["method", "video", "TP", "FP", "FN", "TN", "precision", "recall", "f1", "accuracy"]
    ].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nFiles saved under:")
    print(outdir)


if __name__ == "__main__":
    main()
