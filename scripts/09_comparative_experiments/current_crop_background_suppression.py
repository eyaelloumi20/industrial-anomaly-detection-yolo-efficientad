#!/usr/bin/env python3
"""
current_crop_background_suppression.py

Second development experiment.

The first experiment showed:
- stable_square alone reduced AUC;
- stable_square + border blur improved overall AUC;
therefore this experiment isolates the useful factor:

    KEEP the exact original/current crop geometry
    and change only the outer background/context.

Strategies
----------
1) current
   Exact get_crop() from video_streaming_detector_simple_optimized_v2.py.

2) current_border_blur
   Exact current crop + same soft outer-border Gaussian blur used previously.

3) current_border_mean
   Exact current crop + soft outer-border replacement with the crop's central
   mean RGB value. This is much cheaper than Gaussian blur and is intended as
   a real-time-friendly background suppression variant.

Selection
---------
Development only: four calibration videos / 88 valid tracks.
Primary metric: AUROC(max EMA), threshold-free.

Do NOT choose a final operating threshold from this script.
After choosing the preprocessing strategy, rerun the original normal-only
calibration workflow to obtain its final threshold.
"""

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import video_streaming_detector_simple_optimized_v2 as base


CAL_VIDEOS = [
    ("black",    "WhatsApp Video 2026-09-18 at 15.15.353.mp4", "black_cal"),
    ("metallic", "WhatsApp Video 2026-09-18 at 15.49.53.mp4",  "metallic_cal"),
    ("white",    "WhatsApp Video 2026-09-18 at 15.55.11.mp4",  "white_cal"),
    ("wooden",   "WhatsApp Video 2026-09-18 at 15.31.552.mp4", "wooden_cal"),
]

STRATEGIES = [
    "current",
    "current_border_blur",
    "current_border_mean",
]


def norm_label(s):
    s = (s or "").strip().lower()
    if s in {"normal", "n", "0"}:
        return 0
    if s in {"anomaly", "anomalous", "a", "1"}:
        return 1
    return None


def load_gt(path):
    gt = {}
    excluded = defaultdict(int)

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("role") or "").strip().lower() != "calibration":
                continue

            bg = (r.get("background") or "").strip().lower()
            video = (r.get("video") or "").strip()
            label_txt = (r.get("gt_label") or "").strip().lower()
            valid_txt = (r.get("gt_valid_piece") or "").strip().lower()

            if label_txt == "unsure":
                excluded["unsure"] += 1
                continue
            if label_txt == "not_a_piece" or valid_txt not in {"1", "true", "yes"}:
                excluded["not_a_piece"] += 1
                continue

            label = norm_label(label_txt)
            if label is None:
                excluded["unlabeled"] += 1
                continue

            tid = int(float(r["track_id"]))
            gt[(bg, video, tid)] = {
                "background": bg,
                "video": video,
                "track_id": tid,
                "label": label,
                "label_name": "anomaly" if label else "normal",
            }

    return gt, dict(excluded)


def soft_rect_mask(h, w, keep_ratio=0.78, feather_ratio=0.08):
    mask = np.zeros((h, w), dtype=np.float32)

    keep = float(np.clip(keep_ratio, 0.40, 0.98))
    xpad = int(round((1.0 - keep) * 0.5 * w))
    ypad = int(round((1.0 - keep) * 0.5 * h))

    y1 = max(0, ypad)
    y2 = min(h, max(y1 + 1, h - ypad))
    x1 = max(0, xpad)
    x2 = min(w, max(x1 + 1, w - xpad))
    mask[y1:y2, x1:x2] = 1.0

    feather = max(1, int(round(feather_ratio * min(h, w))))
    k = 2 * feather + 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    return np.clip(mask, 0.0, 1.0)[..., None]


_MASK_CACHE = {}


def get_mask(h, w, keep_ratio, feather_ratio):
    key = (h, w, round(float(keep_ratio), 5), round(float(feather_ratio), 5))
    if key not in _MASK_CACHE:
        _MASK_CACHE[key] = soft_rect_mask(
            h, w, keep_ratio=keep_ratio, feather_ratio=feather_ratio
        )
    return _MASK_CACHE[key]


def border_blur(crop_pil, keep_ratio, feather_ratio):
    arr = np.asarray(crop_pil.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]
    mask = get_mask(h, w, keep_ratio, feather_ratio)

    sigma = max(1.0, 0.06 * min(h, w))
    blurred = cv2.GaussianBlur(arr, (0, 0), sigmaX=sigma, sigmaY=sigma)

    out = (
        arr.astype(np.float32) * mask
        + blurred.astype(np.float32) * (1.0 - mask)
    )
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def border_mean(crop_pil, keep_ratio, feather_ratio):
    arr = np.asarray(crop_pil.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]
    mask = get_mask(h, w, keep_ratio, feather_ratio)

    # Estimate neutral replacement color from the inner center, not from the
    # outer background. This keeps each crop's illumination/color scale.
    center_ratio = 0.50
    xpad = int(round((1.0 - center_ratio) * 0.5 * w))
    ypad = int(round((1.0 - center_ratio) * 0.5 * h))
    center = arr[ypad:h-ypad, xpad:w-xpad]
    if center.size == 0:
        center = arr

    fill = center.reshape(-1, 3).mean(axis=0).astype(np.float32)
    fill_img = np.empty_like(arr, dtype=np.float32)
    fill_img[...] = fill

    out = (
        arr.astype(np.float32) * mask
        + fill_img * (1.0 - mask)
    )
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def transform_crop(strategy, crop, args):
    if strategy == "current":
        return crop
    if strategy == "current_border_blur":
        return border_blur(
            crop,
            keep_ratio=args.border_keep_ratio,
            feather_ratio=args.border_feather_ratio,
        )
    if strategy == "current_border_mean":
        return border_mean(
            crop,
            keep_ratio=args.border_keep_ratio,
            feather_ratio=args.border_feather_ratio,
        )
    raise ValueError(strategy)


def score_strategy(
    strategy,
    args,
    scorer,
    detections,
    frame_cache,
    width,
    height,
    n_frames,
):
    ema_state = {}
    rows = []
    t0 = time.perf_counter()

    for start in range(0, n_frames, args.batch_frames):
        end = min(n_frames, start + args.batch_frames)
        crops = []
        metas = []

        for fi in range(start, end):
            dets = detections.get(fi, [])
            clipped = base.classify_boxes(
                dets,
                width,
                height,
                args.clip_margin,
                args.min_box_frac,
            )
            pil_frame = frame_cache.pil_frames[fi]

            for (tid, x1, y1, x2, y2), is_clip in zip(dets, clipped):
                if is_clip:
                    continue

                crop = base.get_crop(
                    pil_frame,
                    (x1, y1, x2, y2),
                    scorer.image_size,
                    args.pad_ratio,
                    args.denoise,
                )
                crop = transform_crop(strategy, crop, args)

                crops.append(crop)
                metas.append((fi, int(tid)))

        scores = []
        for mb in range(0, len(crops), args.gpu_batch_crops):
            scores.extend(scorer(crops[mb:mb + args.gpu_batch_crops]))

        for (fi, tid), raw in zip(metas, scores):
            raw = float(raw)
            prev = ema_state.get(tid, raw)
            ema = args.ema_alpha * raw + (1.0 - args.ema_alpha) * prev
            ema_state[tid] = ema

            rows.append({
                "frame": fi,
                "track_id": tid,
                "raw_score": raw,
                "ema_score": ema,
            })

        if args.log_every > 0 and (end % args.log_every == 0 or end == n_frames):
            elapsed = time.perf_counter() - t0
            print(
                f"  [{strategy}] {end}/{n_frames} frames | "
                f"{end/max(elapsed, 1e-9):.2f} fps | scored={len(rows)}"
            )

    return rows, time.perf_counter() - t0


def save_scores(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["frame", "track_id", "raw_score", "ema_score"],
        )
        w.writeheader()
        w.writerows(rows)


def per_track(rows):
    d = defaultdict(list)
    for r in rows:
        d[int(r["track_id"])].append(float(r["ema_score"]))

    out = {}
    for tid, vals in d.items():
        a = np.asarray(vals, dtype=float)
        out[tid] = {
            "n": len(a),
            "max_ema": float(np.max(a)),
            "p90_ema": float(np.quantile(a, 0.90)),
            "median_ema": float(np.median(a)),
            "mean_ema": float(np.mean(a)),
        }
    return out


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


def cls_metrics(y, pred):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    fpr = fp / (fp + tn) if fp + tn else 0.0
    acc = (tp + tn) / len(y) if len(y) else 0.0
    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "precision": precision, "recall": recall,
        "f1": f1, "fpr": fpr, "accuracy": acc,
    }


def best_f1(y, scores):
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    candidates = np.unique(scores)
    best = None

    for t in candidates:
        m = cls_metrics(y, (scores >= t).astype(int))
        row = {"threshold": float(t), **m}
        if (
            best is None
            or row["f1"] > best["f1"] + 1e-12
            or (
                abs(row["f1"] - best["f1"]) <= 1e-12
                and row["fpr"] < best["fpr"]
            )
        ):
            best = row

    return best


def evaluate(strategy, video_rows, gt):
    records = []
    missing = []

    stats_cache = {
        k: per_track(v)
        for k, v in video_rows.items()
    }

    for (bg, video, tid), g in gt.items():
        s = stats_cache.get((bg, video), {}).get(tid)
        if s is None:
            missing.append(g)
            continue
        records.append({**g, **s})

    y = np.asarray([r["label"] for r in records], dtype=int)
    s = np.asarray([r["max_ema"] for r in records], dtype=float)

    overall = {
        "strategy": strategy,
        "n": len(records),
        "normal": int(np.sum(y == 0)),
        "anomaly": int(np.sum(y == 1)),
        "missing": len(missing),
        "auroc_max_ema": auc_roc(y, s),
        "best_f1": best_f1(y, s),
    }

    per_bg = []
    for bg in ["black", "metallic", "white", "wooden"]:
        rr = [r for r in records if r["background"] == bg]
        yy = np.asarray([r["label"] for r in rr], dtype=int)
        ss = np.asarray([r["max_ema"] for r in rr], dtype=float)

        per_bg.append({
            "background": bg,
            "n": len(rr),
            "normal": int(np.sum(yy == 0)),
            "anomaly": int(np.sum(yy == 1)),
            "auroc_max_ema": auc_roc(yy, ss),
            "best_f1": best_f1(yy, ss),
        })

    return overall, per_bg, records, missing


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt", default="./gt_reserved_newyolo/tracks_to_label.csv")
    p.add_argument("--videos_root", default="./videos_background")
    p.add_argument(
        "--yolo_det_weights",
        default="./runs/detect/runs/detect/yolo_runs/chewinggum_yolov8n_multibg/weights/best.pt",
    )
    p.add_argument("--ckpt_dir", default="./ckpt_lowres_192")
    p.add_argument("--method", default="lowres")
    p.add_argument("--category", default="chewinggum")
    p.add_argument("--model_size", default="S")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--fp16", action="store_true", default=True)

    p.add_argument("--conf", type=float, default=0.10)
    p.add_argument("--iou", type=float, default=0.50)
    p.add_argument("--ratio", type=float, default=0.1)

    p.add_argument("--ema_alpha", type=float, default=0.3)
    p.add_argument("--pad_ratio", type=float, default=0.02)
    p.add_argument("--denoise", action="store_true")
    p.add_argument("--clip_margin", type=int, default=5)
    p.add_argument("--min_box_frac", type=float, default=0.5)

    p.add_argument("--border_keep_ratio", type=float, default=0.78)
    p.add_argument("--border_feather_ratio", type=float, default=0.08)

    p.add_argument("--batch_frames", type=int, default=16)
    p.add_argument("--gpu_batch_crops", type=int, default=8)
    p.add_argument("--log_every", type=int, default=120)

    p.add_argument(
        "--out",
        default="./current_crop_background_suppression",
    )
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    gt, excluded = load_gt(args.gt)

    print("=" * 92)
    print("CURRENT-CROP BACKGROUND SUPPRESSION EXPERIMENT")
    print("=" * 92)
    print(f"Valid calibration GT tracks : {len(gt)}")
    print(f"Excluded                    : {excluded}")
    print(f"EMA alpha                   : {args.ema_alpha}")
    print(f"Border keep ratio           : {args.border_keep_ratio}")
    print(f"Border feather ratio        : {args.border_feather_ratio}")
    print()
    print("All strategies preserve the EXACT current crop geometry.")
    print("Primary selection metric: AUROC(max EMA).")
    print()

    scorer = base.EfficientADScorer(args)
    dummy = Image.new("RGB", (scorer.image_size, scorer.image_size))
    _ = scorer([dummy] * min(args.gpu_batch_crops, 8))

    all_rows = {s: {} for s in STRATEGIES}
    timing = defaultdict(dict)
    detector_info = {}

    for bg, video_name, tag in CAL_VIDEOS:
        video_path = Path(args.videos_root) / bg / video_name
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        print()
        print("=" * 92)
        print(f"{bg.upper()} | {video_name}")
        print("=" * 92)

        t0 = time.perf_counter()
        (
            detections,
            width,
            height,
            src_fps,
            n_frames,
            frame_cache,
        ) = base.run_yolo_once(
            str(video_path),
            args.yolo_det_weights,
            args.conf,
            args.iou,
            device=args.device,
            max_frames=None,
            detect_unknown=False,
            otsu_max_area_frac=0.25,
            cache_frames=True,
        )
        yolo_sec = time.perf_counter() - t0

        detector_info[tag] = {
            "background": bg,
            "video": video_name,
            "frames": n_frames,
            "detections": int(sum(len(v) for v in detections.values())),
            "yolo_seconds": yolo_sec,
            "yolo_fps": n_frames / max(yolo_sec, 1e-9),
        }

        for strategy in STRATEGIES:
            print(f"\n--- {strategy} ---")
            rows, sec = score_strategy(
                strategy,
                args,
                scorer,
                detections,
                frame_cache,
                width,
                height,
                n_frames,
            )
            all_rows[strategy][(bg, video_name)] = rows
            timing[strategy][tag] = {
                "seconds": sec,
                "fps": n_frames / max(sec, 1e-9),
                "n_scored": len(rows),
            }

            d = out / strategy / bg / tag
            d.mkdir(parents=True, exist_ok=True)
            save_scores(d / "scores.csv", rows)

    summary = {
        "experiment": {
            "development_only": True,
            "strategies": STRATEGIES,
            "primary_metric": "AUROC(max EMA)",
            "settings": {
                "ema_alpha": args.ema_alpha,
                "border_keep_ratio": args.border_keep_ratio,
                "border_feather_ratio": args.border_feather_ratio,
                "conf": args.conf,
                "iou": args.iou,
            },
        },
        "ground_truth": {
            "valid": len(gt),
            "excluded": excluded,
        },
        "detector": detector_info,
        "strategies": {},
    }

    summary_rows = []
    bg_rows = []
    track_rows = []

    print()
    print("=" * 92)
    print("RESULTS — ORIGINAL CURRENT CROP GEOMETRY")
    print("=" * 92)

    for strategy in STRATEGIES:
        overall, per_bg, records, missing = evaluate(
            strategy, all_rows[strategy], gt
        )
        overall["timing"] = timing[strategy]
        summary["strategies"][strategy] = {
            "overall": overall,
            "per_background": per_bg,
            "missing": missing,
        }

        b = overall["best_f1"]
        print(
            f"{strategy:<24s} "
            f"n={overall['n']:>3d} missing={overall['missing']:>2d} | "
            f"AUC={overall['auroc_max_ema']:.4f} | "
            f"best-dev-F1={b['f1']:.4f} @ {b['threshold']:.6f} | "
            f"P={b['precision']:.3f} R={b['recall']:.3f} "
            f"FPR={b['fpr']:.3f}"
        )

        summary_rows.append({
            "strategy": strategy,
            "n": overall["n"],
            "missing": overall["missing"],
            "normal": overall["normal"],
            "anomaly": overall["anomaly"],
            "auroc_max_ema": overall["auroc_max_ema"],
            "best_dev_threshold": b["threshold"],
            "best_dev_TP": b["TP"],
            "best_dev_TN": b["TN"],
            "best_dev_FP": b["FP"],
            "best_dev_FN": b["FN"],
            "best_dev_precision": b["precision"],
            "best_dev_recall": b["recall"],
            "best_dev_f1": b["f1"],
            "best_dev_fpr": b["fpr"],
            "best_dev_accuracy": b["accuracy"],
        })

        for x in per_bg:
            bb = x["best_f1"]
            bg_rows.append({
                "strategy": strategy,
                "background": x["background"],
                "n": x["n"],
                "normal": x["normal"],
                "anomaly": x["anomaly"],
                "auroc_max_ema": x["auroc_max_ema"],
                "best_dev_threshold": bb["threshold"],
                "best_dev_f1": bb["f1"],
                "best_dev_precision": bb["precision"],
                "best_dev_recall": bb["recall"],
                "best_dev_fpr": bb["fpr"],
            })

        for r in records:
            track_rows.append({"strategy": strategy, **r})

    ranking = sorted(
        summary_rows,
        key=lambda x: x["auroc_max_ema"],
        reverse=True,
    )

    print()
    print("AUC ranking:")
    for i, r in enumerate(ranking, 1):
        print(f"  {i}. {r['strategy']}: {r['auroc_max_ema']:.4f}")

    print()
    print("PER BACKGROUND AUC")
    for bg in ["black", "metallic", "white", "wooden"]:
        print(f"{bg}:")
        for r in bg_rows:
            if r["background"] == bg:
                print(f"  {r['strategy']:<24s} {r['auroc_max_ema']:.4f}")

    with (out / "strategy_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        fields = list(summary_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary_rows)

    with (out / "per_background_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        fields = list(bg_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(bg_rows)

    with (out / "track_statistics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        fields = [
            "strategy","background","video","track_id","label","label_name",
            "n","max_ema","p90_ema","median_ema","mean_ema",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(track_rows)

    (out / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("Do NOT freeze the best-development threshold.")
    print("If a background-suppression strategy wins, rerun the original")
    print("normal-only calibration using that exact preprocessing.")
    print()
    print(f"Artifacts: {out.resolve()}")


if __name__ == "__main__":
    main()
