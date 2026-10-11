"""
analyze_defects_v2.py
=====================
Deeper defect analysis using per-frame score trajectories.

For each track, computes:
  - median_ema, mean_ema      (robust to single-frame spikes)
  - p90_ema                   (near-max but less spike-sensitive)
  - frac_above_{3,4,5}        (how much of the track's life was suspicious)
  - box position and size stats

Then reports which feature best separates defects from normals.
"""

import os
import re
import csv
import json
import argparse
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def parse_defects(path):
    out = {}
    current = None
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            m = re.match(r"^in\s+(.+?)\s*$", line, re.IGNORECASE)
            if m:
                current = m.group(1).strip()
                out.setdefault(current, set())
                continue
            ids = re.findall(r"ID(\d+)", line, re.IGNORECASE)
            if ids and current:
                out[current].update(int(i) for i in ids)
    return out


def load_track_scores(tracks_root):
    """
    Read per-track scores.csv from every track folder. Returns:
        [{video, track_id, frames[], scores[], emas[], boxes[]}, ...]
    """
    tracks_root = Path(tracks_root)
    out = []
    for video_dir in sorted(tracks_root.iterdir()):
        if not video_dir.is_dir():
            continue
        tdir = video_dir / "tracks"
        if not tdir.is_dir():
            continue
        for track_dir in sorted(tdir.iterdir()):
            if not track_dir.name.startswith("track_"):
                continue
            csv_path = track_dir / "scores.csv"
            if not csv_path.exists():
                continue
            tid = int(track_dir.name.split("_")[1])

            frames, scores, emas, boxes, clipped = [], [], [], [], []
            with csv_path.open() as f:
                rdr = csv.DictReader(f)
                for row in rdr:
                    frames.append(int(row["frame"]))
                    scores.append(float(row["score"]) if row["score"] else None)
                    emas.append(float(row["ema"]) if row["ema"] else None)
                    boxes.append((int(row["box_x1"]), int(row["box_y1"]),
                                  int(row["box_x2"]), int(row["box_y2"])))
                    clipped.append(int(row["clipped"]))

            # Keep only non-clipped entries with valid scores
            valid = [(fr, s, e, b)
                     for fr, s, e, b, c in zip(frames, scores, emas, boxes, clipped)
                     if not c and e is not None]
            if not valid:
                continue
            frames_v, scores_v, emas_v, boxes_v = zip(*valid)

            out.append({
                "video": video_dir.name,
                "track_id": tid,
                "frames": np.array(frames_v),
                "scores": np.array(scores_v, dtype=np.float32),
                "emas":   np.array(emas_v,   dtype=np.float32),
                "boxes":  np.array(boxes_v,  dtype=np.int32),
            })
    return out


def compute_features(track):
    """Return a dict of derived features for one track."""
    e = track["emas"]
    s = track["scores"]
    b = track["boxes"]

    w = (b[:, 2] - b[:, 0]).astype(np.float32)
    h = (b[:, 3] - b[:, 1]).astype(np.float32)
    aspect = w / np.maximum(h, 1e-6)

    feat = {
        "n_frames_seen": len(e),
        "first_frame":   int(track["frames"][0]),
        "last_frame":    int(track["frames"][-1]),
        "max_ema":   float(e.max()),
        "p90_ema":   float(np.percentile(e, 90)),
        "median_ema": float(np.median(e)),
        "mean_ema":  float(e.mean()),
        "std_ema":   float(e.std()),
        "frac_above_3": float((e > 3).mean()),
        "frac_above_4": float((e > 4).mean()),
        "frac_above_5": float((e > 5).mean()),
        "mean_box_w": float(w.mean()),
        "mean_box_h": float(h.mean()),
        "mean_aspect": float(aspect.mean()),
        "mean_center_x": float(((b[:, 0] + b[:, 2]) / 2).mean()),
        "mean_center_y": float(((b[:, 1] + b[:, 3]) / 2).mean()),
    }
    return feat


def separation_quality(defects, normals, metric):
    """Youden J for a single metric across given defect and normal sets."""
    d = np.array([r[metric] for r in defects], dtype=float)
    n = np.array([r[metric] for r in normals],  dtype=float)
    if len(d) == 0 or len(n) == 0:
        return None
    thresholds = np.linspace(min(d.min(), n.min()),
                             max(d.max(), n.max()), 300)
    best_j, best_thr = -1.0, None
    tpr_b, fpr_b = 0.0, 1.0
    for t in thresholds:
        tp = (d >= t).sum()
        fp = (n >= t).sum()
        tpr = tp / len(d)
        fpr = fp / len(n)
        j = tpr - fpr
        if j > best_j:
            best_j, best_thr, tpr_b, fpr_b = j, float(t), float(tpr), float(fpr)
    return {
        "metric": metric,
        "best_threshold": best_thr,
        "best_tpr": tpr_b,
        "best_fpr": fpr_b,
        "youden_J": best_j,
        "defect_mean":  float(d.mean()),
        "defect_median": float(np.median(d)),
        "normal_mean":  float(n.mean()),
        "normal_median": float(np.median(n)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracks_root", required=True)
    p.add_argument("--defects_file", required=True)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--video_filter", default=None,
                   help="If given, restrict to this video only.")
    args = p.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    defects_map = parse_defects(args.defects_file)
    print(f"[V2] Defect map: {[(v, len(s)) for v, s in defects_map.items()]}")

    tracks = load_track_scores(args.tracks_root)
    print(f"[V2] Loaded {len(tracks)} tracks with score CSVs")

    if args.video_filter:
        tracks = [t for t in tracks if t["video"] == args.video_filter]
        print(f"[V2] Filtered to {args.video_filter}: {len(tracks)} tracks")

    # Build feature rows
    rows = []
    for t in tracks:
        f = compute_features(t)
        f["video"] = t["video"]
        f["track_id"] = t["track_id"]
        f["defective"] = (t["video"] in defects_map
                          and t["track_id"] in defects_map[t["video"]])
        rows.append(f)

    defects = [r for r in rows if r["defective"]]
    normals = [r for r in rows if not r["defective"]]
    print(f"[V2] {len(defects)} defects, {len(normals)} normals")

    # Evaluate every feature for separability
    features = ["max_ema", "p90_ema", "median_ema", "mean_ema",
                "frac_above_3", "frac_above_4", "frac_above_5",
                "std_ema", "n_frames_seen",
                "mean_box_w", "mean_box_h", "mean_aspect"]
    print(f"\n[V2] Separability by feature (Youden J, higher is better):")
    print(f"  {'feature':<18} {'J':>6}  {'thr':>7}  {'TPR':>5}  {'FPR':>5}  "
          f"{'def_med':>8}  {'norm_med':>9}")
    results = {}
    for f in features:
        r = separation_quality(defects, normals, f)
        if r is None:
            continue
        results[f] = r
        print(f"  {f:<18} {r['youden_J']:>6.2f}  "
              f"{r['best_threshold']:>7.2f}  "
              f"{r['best_tpr']:>5.2f}  {r['best_fpr']:>5.2f}  "
              f"{r['defect_median']:>8.2f}  {r['normal_median']:>9.2f}")

    # Dump raw features to CSV for external analysis
    csv_out = os.path.join(args.save_dir, "track_features.csv")
    with open(csv_out, "w", newline="") as f:
        keys = ["video", "track_id", "defective"] + features
        w = csv.writer(f)
        w.writerow(keys)
        for r in rows:
            w.writerow([r.get(k, "") for k in keys])
    print(f"\n[V2] Features CSV: {csv_out}")

    # Write the report
    report = {
        "n_defects": len(defects),
        "n_normals": len(normals),
        "separation": results,
    }
    with open(os.path.join(args.save_dir, "separation.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Plot: median_ema histogram (the most useful alternative to max)
    if HAS_MPL:
        for metric in ["median_ema", "mean_ema", "frac_above_4"]:
            d_vals = [r[metric] for r in defects]
            n_vals = [r[metric] for r in normals]
            if not (d_vals and n_vals):
                continue
            fig, ax = plt.subplots(figsize=(10, 5))
            lo = min(min(d_vals), min(n_vals))
            hi = max(max(d_vals), max(n_vals))
            bins = np.linspace(lo, hi, 30)
            ax.hist(n_vals, bins=bins, alpha=0.6,
                    label=f"normal (n={len(n_vals)})", color="#4ea1ff")
            ax.hist(d_vals, bins=bins, alpha=0.6,
                    label=f"defective (n={len(d_vals)})", color="#ff4e4e")
            ax.set_xlabel(metric); ax.set_ylabel("count")
            ax.set_title(f"Distribution of {metric}: defects vs normals")
            ax.legend(); ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(args.save_dir, f"dist_{metric}.png"),
                        dpi=120)
            plt.close(fig)
        print(f"[V2] Plots written to {args.save_dir}/")


if __name__ == "__main__":
    main()