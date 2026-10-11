#!/usr/bin/env python3
"""
Calibrate the LowRes192 EMA threshold from the 4 development videos.

Decision rule:
    current per-track EMA >= threshold -> ANOMALY

Track calibration score:
    max EMA observed for that track
"""

from pathlib import Path
import argparse, csv, json
from collections import defaultdict
import numpy as np

VIDEO_MAP = {
    "black": "WhatsApp Video 2026-09-18 at 15.15.353.mp4",
    "metallic": "WhatsApp Video 2026-09-18 at 15.49.53.mp4",
    "white": "WhatsApp Video 2026-09-18 at 15.55.11.mp4",
    "wooden": "WhatsApp Video 2026-09-18 at 15.31.552.mp4",
}

def norm_label(s):
    s = (s or "").strip().lower()
    if s in {"normal", "n", "0"}:
        return 0
    if s in {"anomaly", "anomalous", "a", "1"}:
        return 1
    return None

def metrics(y, scores, thr):
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= thr).astype(int)

    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    acc = (tp + tn) / len(y) if len(y) else 0.0

    return {
        "threshold": float(thr),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "precision": precision, "recall": recall,
        "f1": f1, "fpr": fpr, "accuracy": acc,
    }

def auc_roc(y, scores):
    y = np.asarray(y, dtype=int)
    s = np.asarray(scores, dtype=float)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None

    wins = 0.0
    for p in pos:
        wins += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(wins / (len(pos) * len(neg)))

def candidate_thresholds(scores):
    u = np.unique(np.asarray(scores, dtype=float))
    if len(u) == 1:
        return np.array([u[0]], dtype=float)

    mids = (u[:-1] + u[1:]) / 2.0
    eps = max(1e-9, (u[-1] - u[0]) * 1e-9)
    return np.concatenate(([u[0] - eps], mids, [u[-1] + eps]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", default="./calibration_lowres192_threshold")
    args = ap.parse_args()

    gt_path = Path(args.gt)
    scores_root = Path(args.scores)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Load valid calibration GT tracks.
    # ---------------------------------------------------------------
    gt = {}

    with gt_path.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("role") or "").strip().lower() != "calibration":
                continue

            valid = (r.get("gt_valid_piece") or "").strip().lower()
            if valid not in {"1", "true", "yes"}:
                continue

            label = norm_label(r.get("gt_label"))
            if label is None:
                continue

            bg = (r.get("background") or "").strip().lower()
            video = (r.get("video") or "").strip()
            tid = int(float(r["track_id"]))

            if bg not in VIDEO_MAP:
                continue
            if video != VIDEO_MAP[bg]:
                continue

            gt[(bg, tid)] = {
                "background": bg,
                "video": video,
                "track_id": tid,
                "label": label,
                "label_name": "anomaly" if label else "normal",
            }

    # ---------------------------------------------------------------
    # Load score CSVs; aggregate max EMA per track.
    # ---------------------------------------------------------------
    score_by_key = {}
    n_samples_by_key = defaultdict(int)

    for bg in VIDEO_MAP:
        path = scores_root / bg / "scores.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing score file: {path}")

        tmp = defaultdict(list)

        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                ema = (r.get("ema_score") or "").strip()
                if not ema:
                    continue

                tid = int(float(r["track_id"]))
                tmp[tid].append(float(ema))

        for tid, vals in tmp.items():
            key = (bg, tid)
            score_by_key[key] = max(vals)
            n_samples_by_key[key] = len(vals)

    matched = []
    missing = []

    for key, g in gt.items():
        if key not in score_by_key:
            missing.append(g)
            continue

        matched.append({
            **g,
            "track_score_max_ema": score_by_key[key],
            "n_scored_frames": n_samples_by_key[key],
        })

    # ---------------------------------------------------------------
    # Save matching audit.
    # FIX: include 'label' in fieldnames so DictWriter accepts rows.
    # ---------------------------------------------------------------
    matched_fields = [
        "background",
        "video",
        "track_id",
        "label",
        "label_name",
        "track_score_max_ema",
        "n_scored_frames",
    ]

    with (out / "matched_tracks.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=matched_fields,
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(matched)

    missing_fields = [
        "background",
        "video",
        "track_id",
        "label",
        "label_name",
    ]

    with (out / "missing_tracks.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=missing_fields,
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(missing)

    if not matched:
        raise RuntimeError("No GT tracks matched score tracks.")

    y = np.array([r["label"] for r in matched], dtype=int)
    s = np.array([r["track_score_max_ema"] for r in matched], dtype=float)

    sweep = [metrics(y, s, t) for t in candidate_thresholds(s)]
    sweep.sort(key=lambda m: m["threshold"])

    best = max(
        sweep,
        key=lambda m: (
            m["f1"],
            m["recall"],
            -m["fpr"],
            m["accuracy"],
        ),
    )

    recall_candidates = [m for m in sweep if m["recall"] >= 0.90]
    recall90 = (
        min(
            recall_candidates,
            key=lambda m: (
                m["fpr"],
                -m["f1"],
                -m["precision"],
            ),
        )
        if recall_candidates
        else None
    )

    sweep_fields = [
        "threshold",
        "TP",
        "TN",
        "FP",
        "FN",
        "precision",
        "recall",
        "f1",
        "fpr",
        "accuracy",
    ]

    with (out / "threshold_sweep.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        w = csv.DictWriter(f, fieldnames=sweep_fields)
        w.writeheader()
        w.writerows(sweep)

    per_bg = []

    for bg in VIDEO_MAP:
        rr = [r for r in matched if r["background"] == bg]
        if not rr:
            continue

        yy = [r["label"] for r in rr]
        ss = [r["track_score_max_ema"] for r in rr]
        m = metrics(yy, ss, best["threshold"])
        m["background"] = bg
        m["n"] = len(rr)
        m["normal"] = int(sum(v == 0 for v in yy))
        m["anomaly"] = int(sum(v == 1 for v in yy))
        per_bg.append(m)

    bg_fields = [
        "background",
        "n",
        "normal",
        "anomaly",
        "threshold",
        "TP",
        "TN",
        "FP",
        "FN",
        "precision",
        "recall",
        "f1",
        "fpr",
        "accuracy",
    ]

    with (out / "per_background_best_f1.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        w = csv.DictWriter(f, fieldnames=bg_fields)
        w.writeheader()
        w.writerows(per_bg)

    (out / "lowres192_threshold_best_f1.txt").write_text(
        f"{best['threshold']:.8f}\n",
        encoding="utf-8",
    )

    summary = {
        "model": "LowRes192",
        "threshold_basis": "ema",
        "track_score_for_calibration": "max_ema",
        "ema_alpha": 0.3,
        "detector_conf": 0.10,
        "detector_iou": 0.50,
        "n_gt_calibration_valid": len(gt),
        "n_matched": len(matched),
        "n_missing": len(missing),
        "normal_matched": int(np.sum(y == 0)),
        "anomaly_matched": int(np.sum(y == 1)),
        "auroc_track_max_ema": auc_roc(y, s),
        "best_f1": best,
        "recall_ge_0_90_candidate": recall90,
        "per_background_at_best_f1": per_bg,
    }

    (out / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    auroc = summary["auroc_track_max_ema"]

    print("=" * 74)
    print("LOWRES192 THRESHOLD CALIBRATION")
    print("=" * 74)
    print(f"GT valid calibration tracks : {len(gt)}")
    print(f"Matched scored tracks       : {len(matched)}")
    print(f"Missing scored tracks       : {len(missing)}")
    print(f"Matched normal / anomaly    : {np.sum(y==0)} / {np.sum(y==1)}")
    print(
        f"Track AUROC (max EMA)       : "
        f"{auroc:.4f}" if auroc is not None
        else "Track AUROC (max EMA)       : N/A"
    )

    print()
    print("BEST F1 THRESHOLD")
    print(f"threshold : {best['threshold']:.6f}")
    print(
        f"TP={best['TP']} TN={best['TN']} "
        f"FP={best['FP']} FN={best['FN']} | "
        f"P={best['precision']:.4f} "
        f"R={best['recall']:.4f} "
        f"F1={best['f1']:.4f} "
        f"FPR={best['fpr']:.4f} "
        f"Acc={best['accuracy']:.4f}"
    )

    if recall90:
        print()
        print("RECALL >= 0.90 CANDIDATE")
        print(f"threshold : {recall90['threshold']:.6f}")
        print(
            f"TP={recall90['TP']} TN={recall90['TN']} "
            f"FP={recall90['FP']} FN={recall90['FN']} | "
            f"P={recall90['precision']:.4f} "
            f"R={recall90['recall']:.4f} "
            f"F1={recall90['f1']:.4f} "
            f"FPR={recall90['fpr']:.4f}"
        )

    print()
    print("PER BACKGROUND @ BEST-F1 THRESHOLD")
    for m in per_bg:
        print(
            f"{m['background']:<8s} n={m['n']:>2d} | "
            f"TP={m['TP']:>2d} TN={m['TN']:>2d} "
            f"FP={m['FP']:>2d} FN={m['FN']:>2d} | "
            f"P={m['precision']:.3f} "
            f"R={m['recall']:.3f} "
            f"F1={m['f1']:.3f} "
            f"FPR={m['fpr']:.3f}"
        )

    if missing:
        print()
        print("WARNING: some GT tracks were not score-matched.")
        print("Inspect:", out / "missing_tracks.csv")

    print()
    print("Artifacts:", out.resolve())
    print("DO NOT run the 12 final-test videos yet.")
    print("Send calibration_summary.json + matched_tracks.csv to ChatGPT first.")

if __name__ == "__main__":
    main()
