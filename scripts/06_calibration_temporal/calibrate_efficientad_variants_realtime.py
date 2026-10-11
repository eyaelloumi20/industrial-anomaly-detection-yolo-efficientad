"""
calibrate_efficientad_variants_realtime.py
==========================================
Crop-level calibration that uses THE SAME EfficientAD scorer as
video_streaming_detector_optimized3.py.

Important:
- This calibrates crop-level score threshold tau_p.
- It does NOT fully calibrate tau_track, because production tau_track is applied
  after EMA + track aggregation (max for short tracks / median for long tracks).
  tau_track must be calibrated on labelled calibration VIDEOS/tracks.
- tau_f is written equal to tau_p as a conservative starting value. If you use
  frame alarms in production, calibrate tau_f separately on video frames.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

from compare_pipeline_methods_batched import resize_with_padding
from video_streaming_detector_optimized3 import Scorer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--csv", default="dataset_multi/split_csv/1cls.csv")
    p.add_argument("--dataset_root", default="dataset_multi")
    p.add_argument("--split", default="validation")
    p.add_argument("--category", default="chewinggum")
    p.add_argument("--model_size", default="S", choices=["S", "M"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--ratio", type=float, default=0.1)
    p.add_argument("--pad_ratio", type=float, default=0.02)
    p.add_argument("--denoise", action="store_true")
    p.add_argument("--native_score", action="store_true",
                   help="Calibrate native-map scoring. Must match production.")
    p.add_argument("--compile_efficientad", action="store_true",
                   help="Optional. Calibration correctness does not require compile.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--tau_min", type=float, default=0.05)
    p.add_argument("--tau_max", type=float, default=1.50)
    p.add_argument("--tau_steps", type=int, default=291)
    p.add_argument("--output", required=True)
    return p.parse_args()


def make_scorer_args(a):
    # Scorer only reads these fields.
    return SimpleNamespace(
        ckpt_dir=a.ckpt_dir,
        category=a.category,
        model_size=a.model_size,
        device=a.device,
        fp16=a.fp16,
        ratio=a.ratio,
        native_score=a.native_score,
        compile_efficientad=a.compile_efficientad,
    )


def preprocess_image(path, size, denoise):
    img = Image.open(path).convert("RGB")
    img = resize_with_padding(img, size)

    if denoise:
        arr = np.asarray(img, dtype=np.uint8).copy()
        arr = cv2.GaussianBlur(arr, (3, 3), sigmaX=0.8, sigmaY=0.8)
        arr = cv2.bilateralFilter(arr, d=5, sigmaColor=15, sigmaSpace=15)
        img = Image.fromarray(arr)

    return img


def metrics_at_tau(normal, anomaly, tau):
    tp = int((anomaly >= tau).sum())
    fn = int((anomaly < tau).sum())
    fp = int((normal >= tau).sum())
    tn = int((normal < tau).sum())

    recall = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "tau": float(tau),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "fpr": float(fpr),
        "f1": float(f1),
    }


def main():
    args = parse_args()

    scorer = Scorer(make_scorer_args(args))
    print(
        f"[Calibration] ckpt={args.ckpt_dir} image_size={scorer.image_size} "
        f"dtype={scorer.dtype} native_score={args.native_score} denoise={args.denoise}"
    )

    rows = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("split") != args.split:
                continue
            label = r.get("label", "").strip().lower()
            if label not in ("normal", "anomaly"):
                continue
            path = Path(args.dataset_root) / r["image"]
            rows.append((label, path))

    if not rows:
        raise RuntimeError(f"No '{args.split}' rows found in {args.csv}")

    scores = {"normal": [], "anomaly": []}

    batch_imgs = []
    batch_labels = []

    def flush():
        if not batch_imgs:
            return
        values = scorer(batch_imgs)
        if len(values) != len(batch_labels):
            raise RuntimeError("Score/label alignment error")
        for label, value in zip(batch_labels, values):
            if not math.isfinite(value):
                raise RuntimeError(f"Non-finite score for {label}")
            scores[label].append(float(value))
        batch_imgs.clear()
        batch_labels.clear()

    total = len(rows)
    for i, (label, path) in enumerate(rows, 1):
        if not path.is_file():
            print(f"[WARN] missing: {path}")
            continue

        batch_imgs.append(
            preprocess_image(path, scorer.image_size, args.denoise)
        )
        batch_labels.append(label)

        if len(batch_imgs) >= args.batch_size:
            flush()

        if i % 500 == 0:
            print(f"[Calibration] {i}/{total}")

    flush()

    normal = np.asarray(scores["normal"], dtype=np.float64)
    anomaly = np.asarray(scores["anomaly"], dtype=np.float64)

    if len(normal) == 0 or len(anomaly) == 0:
        raise RuntimeError(
            f"Need both classes. normal={len(normal)} anomaly={len(anomaly)}"
        )

    print()
    print(
        f"Normal  : n={len(normal)} mean={normal.mean():.4f} "
        f"p50={np.median(normal):.4f} p90={np.percentile(normal,90):.4f} "
        f"p99={np.percentile(normal,99):.4f}"
    )
    print(
        f"Anomaly : n={len(anomaly)} mean={anomaly.mean():.4f} "
        f"p10={np.percentile(anomaly,10):.4f} "
        f"p50={np.median(anomaly):.4f}"
    )

    taus = np.linspace(args.tau_min, args.tau_max, args.tau_steps)
    table = [metrics_at_tau(normal, anomaly, tau) for tau in taus]

    # Primary crop threshold criterion: max F1.
    best = max(
        table,
        key=lambda m: (m["f1"], m["recall"], -m["fpr"])
    )

    print()
    print(
        f"BEST CROP tau_p={best['tau']:.4f} | "
        f"P={best['precision']:.4f} R={best['recall']:.4f} "
        f"F1={best['f1']:.4f} FPR={best['fpr']:.4f}"
    )

    # Show useful neighbourhood around optimum.
    near = sorted(table, key=lambda m: abs(m["tau"] - best["tau"]))[:9]
    near = sorted(near, key=lambda m: m["tau"])
    print()
    print(f"{'tau':>8} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5} "
          f"{'Prec':>7} {'Recall':>7} {'FPR':>7} {'F1':>7}")
    for m in near:
        print(
            f"{m['tau']:>8.4f} {m['tp']:>5} {m['fp']:>5} "
            f"{m['fn']:>5} {m['tn']:>5} "
            f"{m['precision']:>7.3f} {m['recall']:>7.3f} "
            f"{m['fpr']:>7.3f} {m['f1']:>7.3f}"
        )

    output = {
        "tau_p": best["tau"],
        # Starting value only. For frame-level alarm behaviour, calibrate on video.
        "tau_f": best["tau"],
        "method": "auto",
        "category": args.category,
        "model_size": args.model_size,
        "ckpt_dir": str(Path(args.ckpt_dir).resolve()),
        "image_size": scorer.image_size,
        "ratio": args.ratio,
        "pad_ratio": args.pad_ratio,
        "denoise": args.denoise,
        "fp16": args.fp16,
        "native_score": args.native_score,
        "split": args.split,
        "n_normal": int(len(normal)),
        "n_anomaly": int(len(anomaly)),
        "crop_metrics": best,
        "tau_track": None,
        "tau_track_note": (
            "NOT calibrated here. Calibrate tau_track on labelled videos after "
            "EMA and the production track aggregation rule."
        ),
        "tau_f_note": (
            "Set equal to crop tau_p as an initial value only. "
            "Calibrate separately on video frames if frame alarms are used."
        ),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2))
    print(f"\nSaved calibration JSON: {out}")
    print(
        "IMPORTANT: production must use the same --fp16/--denoise/"
        "--native_score settings used for this calibration."
    )


if __name__ == "__main__":
    main()
