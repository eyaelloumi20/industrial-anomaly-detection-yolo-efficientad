"""
evaluate_all.py
===============
One-shot track-level evaluation of the anomaly detector with optional
padding-based crop filtering.

Loads per-track score CSVs from one or more model folders (lowres,
baseline), computes a set of per-track features, and for each feature:

  1. Sweeps thresholds and reports the best Youden J point
  2. Prints the full confusion matrix
  3. Lists the TP / FP / FN tracks by name

Features computed per track:
  - median_ema        robust central tendency
  - mean_ema          average
  - max_ema           peak (noisy)
  - p90_ema           near-peak
  - frame_rel_median  median of (ema - frame-median-ema), suppresses
                      "wave" events where all tracks fire together

New: --max_padding_frac
-----------------------
Frames where the crop's black-padding fraction exceeds this value
are EXCLUDED from feature computation. Frames with a lot of padding
come from badly-shaped YOLO boxes (aspect ratio far from 1:1) and
their scores are dominated by the padding artifact, not by the gum.

Recommended value: 0.4  (implies aspect ratio <= 1.67).
Leave unset for no filtering (baseline behaviour).

Usage:
    # No filter (baseline behaviour)
    python evaluate_all.py \
        --tracks_lowres  ./tracks_v3 \
        --tracks_baseline ./tracks_baseline_v3 \
        --defects_file "defected pieces.txt" \
        --save_dir ./eval_v3_no_filter

    # With padding filter
    python evaluate_all.py \
        --tracks_lowres  ./tracks_v3 \
        --tracks_baseline ./tracks_baseline_v3 \
        --defects_file "defected pieces.txt" \
        --max_padding_frac 0.4 \
        --save_dir ./eval_v3_filtered
"""

import os
import re
import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


FEATURES = ["median_ema", "mean_ema", "max_ema", "p90_ema", "frame_rel_median"]


# ===========================================================================
# Ground truth
# ===========================================================================
def parse_defects(path):
    out, current = {}, None
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
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


# ===========================================================================
# Track loading
# ===========================================================================
def load_track_frames(tracks_root):
    """
    Read every track's scores.csv.
    Return: {(video, tid): [(frame, ema, padding_frac), ...]}
    Handles both old scores.csv (no padding_frac column) and new one.
    """
    tracks_root = Path(tracks_root)
    out = {}
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
            rows = []
            with csv_path.open() as f:
                rdr = csv.DictReader(f)
                has_padding_col = "padding_frac" in rdr.fieldnames
                for row in rdr:
                    if row["clipped"] == "1":
                        continue
                    if row["ema"]:
                        pfrac = float(row["padding_frac"]) \
                                if has_padding_col else 0.0
                        rows.append((int(row["frame"]),
                                     float(row["ema"]),
                                     pfrac))
            if rows:
                out[(video_dir.name, tid)] = rows
    return out


def compute_frame_reference(track_frames):
    """For each (video, frame), compute the median of all track EMAs
    at that frame. Return: {(video, frame): frame_median}"""
    by_frame = defaultdict(list)
    for (video, _tid), rows in track_frames.items():
        for fr, ema, _pfr in rows:
            by_frame[(video, fr)].append(ema)
    return {k: float(np.median(v)) for k, v in by_frame.items()}


def compute_features(track_frames, frame_ref, max_padding_frac=None):
    """
    Return { (video, tid) : { feature_name: value, mean_padding_frac } }

    If max_padding_frac is not None, frames whose crop padding fraction
    exceeds it are excluded from all feature computations. Tracks with
    fewer than 5 remaining frames after filtering are skipped entirely.
    """
    out = {}
    for (video, tid), rows in track_frames.items():
        if len(rows) < 5:
            continue
        frames_all = np.array([r[0] for r in rows])
        emas_all   = np.array([r[1] for r in rows], dtype=np.float32)
        pfr_all    = np.array([r[2] for r in rows], dtype=np.float32)

        if max_padding_frac is not None:
            keep = pfr_all <= max_padding_frac
            if keep.sum() < 5:
                continue
            frames = frames_all[keep]
            emas   = emas_all[keep]
            pfr    = pfr_all[keep]
        else:
            frames, emas, pfr = frames_all, emas_all, pfr_all

        rel = np.array([
            ema - frame_ref.get((video, fr), ema)
            for fr, ema in zip(frames, emas)
        ], dtype=np.float32)

        out[(video, tid)] = {
            "video":             video,
            "track_id":          int(tid),
            "n_frames":          int(len(frames)),
            "n_frames_raw":      int(len(rows)),
            "first_frame":       int(frames.min()),
            "last_frame":        int(frames.max()),
            "median_ema":        float(np.median(emas)),
            "mean_ema":          float(emas.mean()),
            "max_ema":           float(emas.max()),
            "p90_ema":           float(np.percentile(emas, 90)),
            "frame_rel_median":  float(np.median(rel)),
            "mean_padding_frac": float(pfr.mean()),
            "max_padding_frac":  float(pfr.max()),
        }
    return out


# ===========================================================================
# Confusion / sweep
# ===========================================================================
def is_defect(video, tid, defects_map):
    return video in defects_map and tid in defects_map[video]


def confusion(track_features, defects_map, feature, threshold):
    tp = fp = tn = fn = 0
    tp_l, fp_l, fn_l = [], [], []
    for key, t in track_features.items():
        positive = t[feature] >= threshold
        defect = is_defect(t["video"], t["track_id"], defects_map)
        if defect and positive:
            tp += 1; tp_l.append(key)
        elif defect:
            fn += 1; fn_l.append(key)
        elif positive:
            fp += 1; fp_l.append(key)
        else:
            tn += 1
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * tpr / max(prec + tpr, 1e-9)
    return {
        "threshold": threshold,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "tpr": tpr, "fpr": fpr,
        "precision": prec, "recall": tpr, "f1": f1,
        "youden_J": tpr - fpr,
        "tp_list": tp_l, "fp_list": fp_l, "fn_list": fn_l,
    }


def sweep(track_features, defects_map, feature, n=200):
    vals = np.array([t[feature] for t in track_features.values()])
    lo, hi = float(vals.min()), float(vals.max())
    ths = np.linspace(lo, hi, n)
    rows = [confusion(track_features, defects_map, feature, t) for t in ths]
    best = max(rows, key=lambda r: r["youden_J"])
    return rows, best


# ===========================================================================
# Reporting
# ===========================================================================
def report_model(name, track_features, defects_map, save_dir):
    print(f"\n{'='*76}")
    print(f"  {name}")
    print(f"{'='*76}")

    n_pos = sum(1 for t in track_features.values()
                if is_defect(t["video"], t["track_id"], defects_map))
    n_neg = len(track_features) - n_pos
    print(f"  tracks: {len(track_features)} total  "
          f"({n_pos} defects, {n_neg} normals)\n")

    results = {}
    for feat in FEATURES:
        rows, best = sweep(track_features, defects_map, feat)
        results[feat] = {
            "best": {k: v for k, v in best.items()
                     if not k.endswith("_list")},
            "sweep_rows": [
                {k: v for k, v in r.items() if not k.endswith("_list")}
                for r in rows
            ],
        }

        print(f"  Feature: {feat}")
        print(f"    best thr = {best['threshold']:>8.4f}   "
              f"TPR={best['tpr']:.3f}  FPR={best['fpr']:.3f}  "
              f"Prec={best['precision']:.3f}  F1={best['f1']:.3f}  "
              f"J={best['youden_J']:.3f}")
        print(f"    TP={best['tp']}  FP={best['fp']}  "
              f"TN={best['tn']}  FN={best['fn']}")

    # Detail block for median_ema
    print(f"\n  Detail for median_ema:")
    best_med = results["median_ema"]["best"]
    detail = confusion(track_features, defects_map,
                       "median_ema", best_med["threshold"])

    print(f"    Missed defects (FN):")
    if not detail["fn_list"]:
        print(f"      none")
    for v, tid in sorted(detail["fn_list"]):
        t = track_features[(v, tid)]
        print(f"      {v:<14} ID{tid:<4} median_ema={t['median_ema']:.3f}  "
              f"max_ema={t['max_ema']:.3f}  "
              f"n={t['n_frames']}  pad={t['mean_padding_frac']:.3f}")

    print(f"    False positives (FP), first 20:")
    fps = sorted(detail["fp_list"],
                 key=lambda k: -track_features[k]["median_ema"])
    for v, tid in fps[:20]:
        t = track_features[(v, tid)]
        print(f"      {v:<14} ID{tid:<4} median_ema={t['median_ema']:.3f}  "
              f"max_ema={t['max_ema']:.3f}  "
              f"n={t['n_frames']}  pad={t['mean_padding_frac']:.3f}")
    if len(fps) > 20:
        print(f"      ... and {len(fps) - 20} more")

    # Save JSON
    with open(os.path.join(save_dir, f"report_{name}.json"), "w") as f:
        json.dump({
            "model": name,
            "n_tracks": len(track_features),
            "n_defects": n_pos,
            "n_normals": n_neg,
            "features": results,
        }, f, indent=2)

    # Save per-track features CSV
    csv_path = os.path.join(save_dir, f"track_features_{name}.csv")
    with open(csv_path, "w", newline="") as f:
        keys = ["video", "track_id", "defective", "n_frames", "n_frames_raw",
                "first_frame", "last_frame"] + FEATURES + \
               ["mean_padding_frac", "max_padding_frac"]
        w = csv.writer(f)
        w.writerow(keys)
        for key, t in sorted(track_features.items()):
            row = [t["video"], t["track_id"],
                   1 if is_defect(t["video"], t["track_id"], defects_map) else 0,
                   t["n_frames"], t.get("n_frames_raw", t["n_frames"]),
                   t["first_frame"], t["last_frame"]]
            row += [t[f] for f in FEATURES]
            row += [t["mean_padding_frac"], t["max_padding_frac"]]
            w.writerow(row)
    print(f"\n  Saved: report_{name}.json")
    print(f"  Saved: track_features_{name}.csv")

    return results


# ===========================================================================
# Plots
# ===========================================================================
def plot_comparison(all_results, save_dir):
    if not HAS_MPL:
        return
    models = list(all_results.keys())
    n_models = len(models)
    n_feats = len(FEATURES)
    width = 0.8 / n_models
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(n_feats)
    for i, m in enumerate(models):
        js = [all_results[m][f]["best"]["youden_J"] for f in FEATURES]
        ax.bar(x + i * width - 0.4 + width / 2, js, width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(FEATURES, rotation=15)
    ax.set_ylabel("Youden J (best)")
    ax.set_title("Per-feature separability across models")
    ax.axhline(0.7, color="#888", ls="--", lw=0.8, label="J = 0.7 (good)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "comparison_youden.png"), dpi=120)
    plt.close(fig)
    print(f"\n  Saved: comparison_youden.png")


def plot_sweep_curves(all_results, save_dir):
    if not HAS_MPL:
        return
    fig, axes = plt.subplots(1, len(FEATURES), figsize=(4 * len(FEATURES), 4))
    for ax, feat in zip(axes, FEATURES):
        for m in all_results:
            rows = all_results[m][feat]["sweep_rows"]
            ths = [r["threshold"] for r in rows]
            tprs = [r["tpr"] for r in rows]
            fprs = [r["fpr"] for r in rows]
            ax.plot(ths, tprs, lw=1.5, label=f"{m} TPR")
            ax.plot(ths, fprs, lw=1.5, ls="--", label=f"{m} FPR")
        ax.set_title(feat)
        ax.set_xlabel("threshold")
        ax.set_ylabel("rate")
        ax.grid(alpha=0.3)
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "sweep_curves.png"), dpi=120)
    plt.close(fig)
    print(f"  Saved: sweep_curves.png")


def plot_distributions(track_features, defects_map, save_dir, model_name):
    if not HAS_MPL:
        return
    n_feats = len(FEATURES)
    fig, axes = plt.subplots(1, n_feats, figsize=(4 * n_feats, 4))
    for ax, feat in zip(axes, FEATURES):
        pos, neg = [], []
        for t in track_features.values():
            v = t[feat]
            if is_defect(t["video"], t["track_id"], defects_map):
                pos.append(v)
            else:
                neg.append(v)
        lo = min(min(pos), min(neg))
        hi = max(max(pos), max(neg))
        bins = np.linspace(lo, hi, 30)
        ax.hist(neg, bins=bins, alpha=0.6,
                label=f"normal (n={len(neg)})", color="#4ea1ff")
        ax.hist(pos, bins=bins, alpha=0.6,
                label=f"defect (n={len(pos)})", color="#ff4e4e")
        ax.set_title(feat)
        ax.set_xlabel("value")
        ax.set_ylabel("count")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"dist_{model_name}.png"), dpi=120)
    plt.close(fig)
    print(f"  Saved: dist_{model_name}.png")


# ===========================================================================
# Ensemble
# ===========================================================================
def ensemble_evaluate(lr_features, bl_features, defects_map, feature):
    common = sorted(set(lr_features.keys()) & set(bl_features.keys()))
    lr_vals = np.array([lr_features[k][feature] for k in common])
    bl_vals = np.array([bl_features[k][feature] for k in common])
    lo = min(lr_vals.min(), bl_vals.min())
    hi = max(lr_vals.max(), bl_vals.max())
    ths = np.linspace(lo, hi, 200)

    def eval_decision(mode, thr_lr, thr_bl):
        tp = fp = tn = fn = 0
        tp_l, fp_l, fn_l = [], [], []
        for k in common:
            lr_hit = lr_features[k][feature] >= thr_lr
            bl_hit = bl_features[k][feature] >= thr_bl
            if mode == "inter":
                pred = lr_hit and bl_hit
            elif mode == "union":
                pred = lr_hit or bl_hit
            else:
                pred = lr_hit
            v, tid = k
            defect = is_defect(v, tid, defects_map)
            if defect and pred:
                tp += 1; tp_l.append(k)
            elif defect:
                fn += 1; fn_l.append(k)
            elif pred:
                fp += 1; fp_l.append(k)
            else:
                tn += 1
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        prec = tp / max(tp + fp, 1)
        f1 = 2 * prec * tpr / max(prec + tpr, 1e-9)
        return {"thr_lr": thr_lr, "thr_bl": thr_bl,
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "tpr": tpr, "fpr": fpr,
                "precision": prec, "f1": f1,
                "youden_J": tpr - fpr,
                "tp_list": tp_l, "fp_list": fp_l, "fn_list": fn_l}

    results = {}
    for mode in ("inter", "union"):
        best = None
        for thr_lr in ths:
            for thr_bl in ths[::4]:
                r = eval_decision(mode, thr_lr, thr_bl)
                if best is None or r["youden_J"] > best["youden_J"]:
                    best = r
        best["mode"] = mode
        results[mode] = best
    return results, common


# ===========================================================================
# Main
# ===========================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracks_lowres", required=True)
    p.add_argument("--tracks_baseline", default=None,
                   help="Optional. If given, run cross-model + ensemble.")
    p.add_argument("--defects_file", required=True)
    p.add_argument("--max_padding_frac", type=float, default=None,
                   help="Exclude per-frame crops with padding fraction above "
                        "this value from score aggregation. Recommended 0.4 "
                        "(implies aspect ratio <= 1.67). Default: no filter.")
    p.add_argument("--save_dir", required=True)
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    defects_map = parse_defects(args.defects_file)
    print(f"{'='*76}")
    print(f"  evaluate_all.py")
    print(f"{'='*76}")
    print(f"  max_padding_frac: {args.max_padding_frac}")
    print(f"  ground truth:")
    for v, ids in defects_map.items():
        print(f"    {v:<14} -> {sorted(ids)}")
    print()

    # ---- Load lowres ----
    print(f"Loading lowres tracks from {args.tracks_lowres}...")
    lr_frames = load_track_frames(args.tracks_lowres)
    lr_ref = compute_frame_reference(lr_frames)
    lr_features = compute_features(lr_frames, lr_ref,
                                   max_padding_frac=args.max_padding_frac)
    print(f"  {len(lr_features)} tracks loaded "
          f"(from {len(lr_frames)} pre-filter)")

    # ---- Load baseline (optional) ----
    bl_features = None
    if args.tracks_baseline and os.path.isdir(args.tracks_baseline):
        print(f"Loading baseline tracks from {args.tracks_baseline}...")
        bl_frames = load_track_frames(args.tracks_baseline)
        bl_ref = compute_frame_reference(bl_frames)
        bl_features = compute_features(bl_frames, bl_ref,
                                       max_padding_frac=args.max_padding_frac)
        print(f"  {len(bl_features)} tracks loaded "
              f"(from {len(bl_frames)} pre-filter)")

    # ---- Per-model evaluation ----
    all_results = {}
    all_results["lowres"] = report_model(
        "lowres", lr_features, defects_map, args.save_dir)

    if bl_features is not None:
        all_results["baseline"] = report_model(
            "baseline", bl_features, defects_map, args.save_dir)

    # ---- Plots ----
    plot_comparison(all_results, args.save_dir)
    plot_sweep_curves(all_results, args.save_dir)
    plot_distributions(lr_features, defects_map, args.save_dir, "lowres")
    if bl_features is not None:
        plot_distributions(bl_features, defects_map, args.save_dir, "baseline")

    # ---- Ensemble ----
    if bl_features is not None:
        print(f"\n{'='*76}")
        print(f"  Ensemble (lowres + baseline)")
        print(f"{'='*76}")
        ens, common = ensemble_evaluate(lr_features, bl_features,
                                        defects_map, "median_ema")
        print(f"  common tracks: {len(common)}")
        for mode in ("union", "inter"):
            r = ens[mode]
            print(f"\n  {mode:<8}  thr_lr={r['thr_lr']:.3f}  "
                  f"thr_bl={r['thr_bl']:.3f}")
            print(f"    TP={r['tp']}  FP={r['fp']}  TN={r['tn']}  FN={r['fn']}")
            print(f"    TPR={r['tpr']:.3f}  FPR={r['fpr']:.3f}  "
                  f"Prec={r['precision']:.3f}  F1={r['f1']:.3f}  "
                  f"J={r['youden_J']:.3f}")

        with open(os.path.join(args.save_dir, "ensemble.json"), "w") as f:
            json.dump({k: {kk: vv for kk, vv in v.items()
                           if not kk.endswith("_list")}
                       for k, v in ens.items()}, f, indent=2)

    # ---- Final summary ----
    print(f"\n{'='*76}")
    print(f"  Summary")
    print(f"{'='*76}")
    for m in all_results:
        print(f"  {m}:")
        for f in FEATURES:
            b = all_results[m][f]["best"]
            print(f"    {f:<18} J={b['youden_J']:.3f}  "
                  f"TPR={b['tpr']:.3f}  FPR={b['fpr']:.3f}  "
                  f"thr={b['threshold']:.3f}")

    print(f"\n  Outputs in: {os.path.abspath(args.save_dir)}")


if __name__ == "__main__":
    main()