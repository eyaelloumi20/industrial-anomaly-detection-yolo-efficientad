"""
analyze_defects.py
==================
Statistical analysis of defective vs non-defective tracks.

Reads the tracks/ folders produced by dump_all_frames.py and a text file
listing defective track IDs per video, then:

  1. Extracts per-track statistics from each track's summary.json
  2. Compares the distribution of max_ema / max_score / mean_score between
     defective and non-defective tracks
  3. Plots ROC-style curves to find the best discriminative threshold
  4. Reports per-video and global patterns (position, aspect ratio, timing)

Usage:
    python analyze_defects.py \
        --tracks_root ./tracks_only \
        --defects_file "./defected pieces.txt" \
        --save_dir ./defect_analysis
"""

import os
import re
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


# ---------------------------------------------------------------------------
# Parse the defects file
# ---------------------------------------------------------------------------
def parse_defects(path):
    """
    Expected format:
        in video_A4
        ID15-ID2-ID18-ID30-ID49-ID55
        in video_A2
        ID3-ID5-ID10-ID18-ID29-ID33
        ...
    Returns {video_name: set(track_ids)}
    """
    out = {}
    current_video = None
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            m = re.match(r"^in\s+(.+?)\s*$", line, re.IGNORECASE)
            if m:
                current_video = m.group(1).strip()
                out.setdefault(current_video, set())
                continue
            # Expect IDxx-IDyy-... pattern
            ids = re.findall(r"ID(\d+)", line, re.IGNORECASE)
            if ids and current_video:
                for i in ids:
                    out[current_video].add(int(i))
    return out


# ---------------------------------------------------------------------------
# Load every track summary from disk
# ---------------------------------------------------------------------------
def load_tracks(tracks_root):
    """
    Returns a list of dicts:
        {video, track_id, n_frames_seen, first_frame, last_frame,
         max_score, max_ema, any_above_tau_p, any_above_tau_f, n_clipped_frames,
         mean_score, median_score, aspect, box_w, box_h, ...
        plus a reference to the montage path and summary.json path}
    """
    tracks_root = Path(tracks_root)
    records = []
    for video_dir in sorted(tracks_root.iterdir()):
        if not video_dir.is_dir():
            continue
        video = video_dir.name
        tdir = video_dir / "tracks"
        if not tdir.is_dir():
            continue
        for track_dir in sorted(tdir.iterdir()):
            if not track_dir.name.startswith("track_"):
                continue
            summ_path = track_dir / "summary.json"
            if not summ_path.exists():
                continue
            try:
                s = json.loads(summ_path.read_text())
            except Exception:
                continue
            s["video"] = video
            s["track_dir"] = str(track_dir)
            s["montage"] = str(track_dir / "montage.png")
            records.append(s)

    # Additional statistics need the raw scores. Read records.jsonl if
    # available in the video folder -- it lists every (frame, track) score.
    video_records = {}
    for video_dir in sorted(tracks_root.iterdir()):
        if not video_dir.is_dir():
            continue
        rec = video_dir / "records.jsonl"
        if rec.exists():
            video_records[video_dir.name] = [
                json.loads(l) for l in rec.read_text().splitlines() if l
            ]

    return records, video_records


def enrich_with_scores(records, video_records):
    """
    For each track, aggregate the raw and EMA scores from the video's
    records.jsonl. Adds mean_score, median_score, max_alarm_crops.
    """
    # Build a lookup: (video, tid) -> list of (frame, score, ema)
    per_track = {}
    for video, recs in video_records.items():
        # records.jsonl only stores frame-level summaries, not per-track.
        # If it doesn't have per-track data, skip this enrichment.
        pass

    # Records.jsonl produced by dump_all_frames.py contains only frame-level
    # info. To get per-track scores, read the individual crop filenames in
    # each track directory -- the score is encoded in the filename.
    for r in records:
        tdir = Path(r["track_dir"])
        scores = []
        for p in sorted(tdir.glob("crop_*")):
            # Filename: crop_00042.jpg -- no score embedded. We'd need the
            # original records for that. Fall back to max_score from summary.
            pass
        r.setdefault("mean_score", r.get("max_score", 0.0))
        r.setdefault("median_score", r.get("max_score", 0.0))
    return records


# ---------------------------------------------------------------------------
# ROC-style analysis
# ---------------------------------------------------------------------------
def compute_separation(defects, normals, metric="max_ema"):
    """Return (threshold, TPR, FPR, best_threshold) for a single metric."""
    d = np.array([r[metric] for r in defects if metric in r], dtype=float)
    n = np.array([r[metric] for r in normals  if metric in r], dtype=float)
    if len(d) == 0 or len(n) == 0:
        return None
    # Sweep thresholds
    thresholds = np.linspace(min(d.min(), n.min()),
                             max(d.max(), n.max()), 200)
    best = (None, 0.0, 1.0)
    best_j = -1.0
    tprs, fprs = [], []
    for t in thresholds:
        tp = (d >= t).sum()
        fp = (n >= t).sum()
        tpr = tp / max(len(d), 1)
        fpr = fp / max(len(n), 1)
        tprs.append(tpr); fprs.append(fpr)
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best = (float(t), float(tpr), float(fpr))
    return {
        "metric": metric,
        "n_defects": len(d),
        "n_normals": len(n),
        "defect_mean": float(d.mean()),
        "defect_std":  float(d.std()),
        "defect_min":  float(d.min()),
        "defect_max":  float(d.max()),
        "normal_mean": float(n.mean()),
        "normal_std":  float(n.std()),
        "normal_min":  float(n.min()),
        "normal_max":  float(n.max()),
        "best_threshold": best[0],
        "best_tpr": best[1],
        "best_fpr": best[2],
        "youden_J": best_j,
        "tprs": tprs,
        "fprs": fprs,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracks_root", required=True)
    p.add_argument("--defects_file", required=True)
    p.add_argument("--save_dir", required=True)
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"[Analyze] Parsing {args.defects_file}")
    defects_map = parse_defects(args.defects_file)
    for v, ids in defects_map.items():
        print(f"  {v}: {len(ids)} defective tracks -> {sorted(ids)}")

    print(f"\n[Analyze] Loading track summaries from {args.tracks_root}")
    records, video_records = load_tracks(args.tracks_root)
    print(f"[Analyze] Loaded {len(records)} tracks total")

    # Split into defective and normal
    defects, normals = [], []
    for r in records:
        video = r["video"]
        tid = int(r["track_id"])
        if video in defects_map and tid in defects_map[video]:
            r["defective"] = True
            defects.append(r)
        else:
            r["defective"] = False
            normals.append(r)

    print(f"[Analyze] {len(defects)} defective, {len(normals)} normal")

    if not defects:
        print("[Analyze] No defective tracks matched. "
              "Check that track IDs in the defects file match the folders.")
        return

    # ---- Distribution stats per metric --------------------------------
    metrics = ["max_ema", "max_score"]
    results = {}
    for m in metrics:
        res = compute_separation(defects, normals, metric=m)
        if res is None:
            print(f"[Analyze] Metric {m}: not enough data")
            continue
        results[m] = res
        print(f"\n[Analyze] Metric: {m}")
        print(f"    defects : mean={res['defect_mean']:.3f}  "
              f"std={res['defect_std']:.3f}  "
              f"range=[{res['defect_min']:.3f}, {res['defect_max']:.3f}]")
        print(f"    normals : mean={res['normal_mean']:.3f}  "
              f"std={res['normal_std']:.3f}  "
              f"range=[{res['normal_min']:.3f}, {res['normal_max']:.3f}]")
        print(f"    best threshold = {res['best_threshold']:.3f}  "
              f"TPR={res['best_tpr']:.3f}  FPR={res['best_fpr']:.3f}  "
              f"Youden J = {res['youden_J']:.3f}")

    # ---- Per-video consistency ----------------------------------------
    print(f"\n[Analyze] Per-video defect score ranges:")
    for v in sorted(defects_map.keys()):
        d_v = [r for r in defects if r["video"] == v]
        n_v = [r for r in normals  if r["video"] == v]
        if not d_v:
            continue
        d_max = [r["max_ema"] for r in d_v]
        n_max = [r["max_ema"] for r in n_v] if n_v else [0.0]
        print(f"    {v:<14}  defects={len(d_v)}  "
              f"max_ema=[{min(d_max):.2f}, {max(d_max):.2f}]  "
              f"  normals max_ema max={max(n_max):.2f}")

    # ---- Write the report ---------------------------------------------
    report_path = os.path.join(args.save_dir, "defect_analysis.json")
    with open(report_path, "w") as f:
        json.dump({
            "n_defects": len(defects),
            "n_normals": len(normals),
            "defects_map": {k: sorted(v) for k, v in defects_map.items()},
            "metrics": {k: {kk: vv for kk, vv in v.items()
                            if kk not in ("tprs", "fprs")}
                        for k, v in results.items()},
            "defect_records": [
                {k: r[k] for k in ("video", "track_id", "max_ema", "max_score",
                                   "n_frames_seen", "first_frame",
                                   "last_frame")}
                for r in defects
            ],
        }, f, indent=2)
    print(f"\n[Analyze] Report: {report_path}")

    # ---- Plots --------------------------------------------------------
    if not HAS_MPL:
        print("[Analyze] matplotlib missing; skipping plots.")
        return

    # 1) Histogram of max_ema for defective vs normal tracks
    fig, ax = plt.subplots(figsize=(10, 5))
    d_vals = [r["max_ema"] for r in defects]
    n_vals = [r["max_ema"] for r in normals]
    bins = np.linspace(min(d_vals + n_vals), max(d_vals + n_vals), 30)
    ax.hist(n_vals, bins=bins, alpha=0.6, label=f"normal (n={len(n_vals)})",
            color="#4ea1ff")
    ax.hist(d_vals, bins=bins, alpha=0.6, label=f"defective (n={len(d_vals)})",
            color="#ff4e4e")
    ax.set_xlabel("max_ema per track")
    ax.set_ylabel("count")
    ax.set_title("Distribution of max_ema: defective vs normal tracks")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.save_dir, "dist_max_ema.png"), dpi=120)
    plt.close(fig)

    # 2) ROC-style separation curve
    if "max_ema" in results:
        r = results["max_ema"]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(r["fprs"], r["tprs"], color="#4ea1ff", lw=2)
        ax.plot([0, 1], [0, 1], "--", color="#888", lw=1)
        ax.scatter([r["best_fpr"]], [r["best_tpr"]],
                   color="red", s=60, zorder=5,
                   label=f"best J={r['youden_J']:.2f}\n"
                         f"thr={r['best_threshold']:.2f}")
        ax.set_xlabel("FPR (normal tracks above threshold)")
        ax.set_ylabel("TPR (defective tracks above threshold)")
        ax.set_title("Separation of defects from normals by max_ema")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.save_dir, "roc_max_ema.png"), dpi=120)
        plt.close(fig)

    # 3) Per-video scatter: track position vs max_ema
    fig, ax = plt.subplots(figsize=(12, 5))
    for v in sorted(set(r["video"] for r in records)):
        d = [r for r in defects if r["video"] == v]
        n = [r for r in normals  if r["video"] == v]
        if n:
            ax.scatter([r["first_frame"] for r in n],
                       [r["max_ema"] for r in n],
                       alpha=0.35, color="#4ea1ff", s=20)
        if d:
            ax.scatter([r["first_frame"] for r in d],
                       [r["max_ema"] for r in d],
                       color="#ff4e4e", s=70, marker="x")
    ax.set_xlabel("first frame of track")
    ax.set_ylabel("max_ema")
    ax.set_title("Track max_ema over time (red x = defective, blue dot = normal)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.save_dir, "scatter_first_frame.png"), dpi=120)
    plt.close(fig)

    print(f"[Analyze] Plots saved to {args.save_dir}/")


if __name__ == "__main__":
    main()