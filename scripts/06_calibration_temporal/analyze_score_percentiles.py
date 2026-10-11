"""
analyze_score_percentiles.py
==============================
STEP 3 DIAGNOSTIC INSTRUMENTATION -- purely additive, read-only.

evaluate_pipeline.py / metrics.py already compute mean and std for the
normal-score and anomaly-score distributions, but not percentiles
(P90/P95/P99/P99.5/median/max). This script does NOT modify metrics.py or
evaluate_pipeline.py -- it simply reads the per_image_results.csv that
evaluate_pipeline.py already writes, and reports the additional percentile
statistics from it.

Usage
-----
    python analyze_score_percentiles.py \
        --csv ./results_baselineA_original/per_image_results.csv \
        --score_col image_score --label_col label \
        --out_dir ./results_baselineA_original

If your evaluate_pipeline.py's CSV uses different column names than the
defaults below, pass --score_col / --label_col explicitly -- the script
will list the columns it actually found if the given names don't match, so
nothing is silently guessed.

Convention assumed (matching PipelineResult / VisA-style datasets):
    label == 0  -> normal
    label != 0  -> anomaly
Override with --anomaly_label if your CSV uses a different convention
(e.g. label is a string like "good"/"bad").
"""

import argparse
import csv
import json
import os

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Compute percentile statistics for normal vs "
                                            "anomaly scores from evaluate_pipeline.py's CSV output.")
    p.add_argument("--csv", required=True, help="Path to per_image_results.csv from evaluate_pipeline.py")
    p.add_argument("--score_col", default="image_score", help="Column holding the per-image score")
    p.add_argument("--label_col", default="label", help="Column holding the ground-truth label")
    p.add_argument("--anomaly_label", default="1",
                   help="Value in --label_col that means 'anomaly' (default '1'); everything else is 'normal'")
    p.add_argument("--out_dir", default=".", help="Where to write score_percentiles.json")
    return p.parse_args()


def percentile(arr, q):
    return float(np.percentile(arr, q)) if len(arr) else float("nan")


def stats_block(arr):
    arr = np.asarray(arr, dtype=float)
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()) if len(arr) else float("nan"),
        "median": float(np.median(arr)) if len(arr) else float("nan"),
        "std": float(arr.std()) if len(arr) else float("nan"),
        "p90": percentile(arr, 90),
        "p95": percentile(arr, 95),
        "p99": percentile(arr, 99),
        "p99_5": percentile(arr, 99.5),
        "max": float(arr.max()) if len(arr) else float("nan"),
        "min": float(arr.min()) if len(arr) else float("nan"),
    }


def main():
    args = parse_args()

    if not os.path.isfile(args.csv):
        raise SystemExit(f"CSV not found: {args.csv}")

    with open(args.csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if not rows:
        raise SystemExit(f"No rows found in {args.csv}")

    if args.score_col not in fieldnames or args.label_col not in fieldnames:
        raise SystemExit(
            f"--score_col={args.score_col!r} or --label_col={args.label_col!r} not found in CSV.\n"
            f"Columns actually present: {fieldnames}\n"
            f"Re-run with the correct --score_col / --label_col."
        )

    normal_scores, anomaly_scores = [], []
    skipped = 0
    for row in rows:
        raw_score = row[args.score_col]
        raw_label = row[args.label_col]
        if raw_score in ("", None):
            skipped += 1
            continue
        score = float(raw_score)
        if str(raw_label) == str(args.anomaly_label):
            anomaly_scores.append(score)
        else:
            normal_scores.append(score)

    result = {
        "source_csv": os.path.abspath(args.csv),
        "score_col": args.score_col,
        "label_col": args.label_col,
        "anomaly_label_value": args.anomaly_label,
        "rows_skipped_missing_score": skipped,
        "normal": stats_block(normal_scores),
        "anomaly": stats_block(anomaly_scores),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "score_percentiles.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*60}\n  Score percentile statistics\n{'='*60}")
    for group in ("normal", "anomaly"):
        s = result[group]
        print(f"  {group.upper():<8s} n={s['n']}")
        print(f"    mean={s['mean']:.5f}  median={s['median']:.5f}  std={s['std']:.5f}")
        print(f"    p90={s['p90']:.5f}  p95={s['p95']:.5f}  p99={s['p99']:.5f}  "
              f"p99.5={s['p99_5']:.5f}  max={s['max']:.5f}  min={s['min']:.5f}")
    print(f"{'='*60}\n  Written -> {out_path}\n{'='*60}")


if __name__ == "__main__":
    main()
