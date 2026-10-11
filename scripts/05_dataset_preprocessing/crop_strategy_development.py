#!/usr/bin/env python3
"""
crop_strategy_development.py
============================

Development-only experiment for the four NEW-background calibration videos.

Purpose
-------
Compare three EfficientAD input strategies while freezing:
  - NEW multi-background YOLO
  - ByteTrack settings
  - LowRes192 EfficientAD weights
  - EMA alpha
  - anomaly model itself

Strategies
----------
A) current
   Exact crop preprocessing used by video_streaming_detector_simple_optimized_v2.py.

B) stable_square
   EMA-smoothed YOLO box center/size -> canonical square crop.

C) stable_square_border_blur
   Same canonical square crop as B, but only the OUTER BORDER is blurred.
   This suppresses background texture while preserving the centered object.
   It is intentionally NOT called segmentation.

Important experimental rule
---------------------------
This script uses ONLY the four development/calibration videos and their existing
manual labels. It does NOT use the 12 final-test videos.

Selection should be based primarily on AUROC (threshold-free).
The "best F1 threshold" is development-only diagnostic information and must NOT
be used as the final operating threshold. Once a crop strategy is selected, run
the original normal-only calibration workflow again for that selected strategy.
"""

import argparse
import csv
import json
import math
import os
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
    "stable_square",
    "stable_square_border_blur",
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


def clamp_box(box, width, height):
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(width - 1), float(x1)))
    y1 = max(0.0, min(float(height - 1), float(y1)))
    x2 = max(x1 + 1.0, min(float(width), float(x2)))
    y2 = max(y1 + 1.0, min(float(height), float(y2)))
    return (x1, y1, x2, y2)


def update_smoothed_square(state, tid, raw_box, alpha, scale, width, height):
    x1, y1, x2, y2 = map(float, raw_box)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = max(2.0, x2 - x1)
    h = max(2.0, y2 - y1)

    cur = np.array([cx, cy, w, h], dtype=np.float64)

    if tid in state:
        cur = alpha * cur + (1.0 - alpha) * state[tid]

    state[tid] = cur

    cx, cy, w, h = cur.tolist()
    side = max(w, h) * float(scale)
    side = max(4.0, side)

    box = (
        cx - 0.5 * side,
        cy - 0.5 * side,
        cx + 0.5 * side,
        cy + 0.5 * side,
    )
    return clamp_box(box, width, height)


def crop_square(pil_img, box, out_size):
    x1, y1, x2, y2 = box
    crop = pil_img.crop((
        int(math.floor(x1)),
        int(math.floor(y1)),
        int(math.ceil(x2)),
        int(math.ceil(y2)),
    ))
    if crop.mode != "RGB":
        crop = crop.convert("RGB")
    return crop.resize((out_size, out_size), Image.Resampling.BILINEAR)


def blur_outer_border(crop_pil, keep_ratio=0.78, feather_ratio=0.08):
    """
    Preserve the centered region; blur only the outer context.

    This is a conservative background-suppression test, not segmentation.
    """
    arr = np.asarray(crop_pil, dtype=np.uint8)
    h, w = arr.shape[:2]

    sigma = max(1.0, 0.06 * min(h, w))
    blurred = cv2.GaussianBlur(arr, (0, 0), sigmaX=sigma, sigmaY=sigma)

    mask = np.zeros((h, w), dtype=np.float32)

    keep = float(np.clip(keep_ratio, 0.40, 0.95))
    xpad = int(round((1.0 - keep) * 0.5 * w))
    ypad = int(round((1.0 - keep) * 0.5 * h))

    mask[ypad:max(ypad + 1, h - ypad),
         xpad:max(xpad + 1, w - xpad)] = 1.0

    feather = max(1, int(round(feather_ratio * min(h, w))))
    k = 2 * feather + 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    mask = np.clip(mask, 0.0, 1.0)[..., None]

    out = arr.astype(np.float32) * mask + blurred.astype(np.float32) * (1.0 - mask)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def make_crop(strategy, pil_frame, raw_box, square_box, scorer, args):
    if strategy == "current":
        return base.get_crop(
            pil_frame,
            raw_box,
            scorer.image_size,
            args.pad_ratio,
            args.denoise,
        )

    crop = crop_square(pil_frame, square_box, scorer.image_size)

    if strategy == "stable_square_border_blur":
        crop = blur_outer_border(
            crop,
            keep_ratio=args.border_keep_ratio,
            feather_ratio=args.border_feather_ratio,
        )

    return crop


def score_video_strategy(
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
    smooth_state = {}
    rows = []
    t0_all = time.perf_counter()

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
                raw_box = (x1, y1, x2, y2)

                # Update smoothing state even if this particular observation is clipped.
                square_box = update_smoothed_square(
                    smooth_state,
                    int(tid),
                    raw_box,
                    args.box_smooth_alpha,
                    args.square_scale,
                    width,
                    height,
                )

                if is_clip:
                    continue

                crop = make_crop(
                    strategy,
                    pil_frame,
                    raw_box,
                    square_box,
                    scorer,
                    args,
                )
                crops.append(crop)
                metas.append((fi, int(tid), raw_box, square_box))

        scores = []
        for mb in range(0, len(crops), args.gpu_batch_crops):
            scores.extend(scorer(crops[mb:mb + args.gpu_batch_crops]))

        for meta, raw in zip(metas, scores):
            fi, tid, raw_box, square_box = meta
            raw = float(raw)

            prev = ema_state.get(tid, raw)
            ema = args.ema_alpha * raw + (1.0 - args.ema_alpha) * prev
            ema_state[tid] = ema

            rows.append({
                "frame": fi,
                "track_id": tid,
                "raw_score": raw,
                "ema_score": ema,
                "raw_x1": raw_box[0],
                "raw_y1": raw_box[1],
                "raw_x2": raw_box[2],
                "raw_y2": raw_box[3],
                "crop_x1": square_box[0] if strategy != "current" else raw_box[0],
                "crop_y1": square_box[1] if strategy != "current" else raw_box[1],
                "crop_x2": square_box[2] if strategy != "current" else raw_box[2],
                "crop_y2": square_box[3] if strategy != "current" else raw_box[3],
            })

        if args.log_every > 0 and (end % args.log_every == 0 or end == n_frames):
            elapsed = time.perf_counter() - t0_all
            print(
                f"  [{strategy}] {end}/{n_frames} frames | "
                f"{end/max(elapsed,1e-9):.2f} fps | scored={len(rows)}"
            )

    return rows, time.perf_counter() - t0_all


def save_rows(path, rows):
    fields = [
        "frame","track_id","raw_score","ema_score",
        "raw_x1","raw_y1","raw_x2","raw_y2",
        "crop_x1","crop_y1","crop_x2","crop_y2",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def track_stats(rows):
    seq = defaultdict(list)
    for r in rows:
        seq[int(r["track_id"])].append(float(r["ema_score"]))

    out = {}
    for tid, vals in seq.items():
        a = np.asarray(vals, dtype=np.float64)
        out[tid] = {
            "n": len(a),
            "max_ema": float(np.max(a)),
            "p90_ema": float(np.quantile(a, 0.90)),
            "median_ema": float(np.median(a)),
            "mean_ema": float(np.mean(a)),
        }
    return out


def auc_roc(y, s):
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None

    wins = 0.0
    for p in pos:
        wins += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(wins / (len(pos) * len(neg)))


def classification_metrics(y, pred):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)

    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))

    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2*p*r/(p+r) if p+r else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    acc = (tp + tn) / len(y) if len(y) else 0.0

    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "precision": p, "recall": r, "f1": f1,
        "fpr": fpr, "accuracy": acc,
    }


def best_f1_threshold(y, scores):
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)

    candidates = np.unique(scores)
    # Include just above max for completeness.
    candidates = np.concatenate([candidates, [np.nextafter(scores.max(), np.inf)]])

    best = None
    for t in candidates:
        pred = (scores >= t).astype(int)
        m = classification_metrics(y, pred)
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


def evaluate_strategy(strategy, per_video_rows, gt):
    records = []
    missing = []

    for (bg, video, tid), g in gt.items():
        stats = track_stats(per_video_rows.get((bg, video), []))
        s = stats.get(tid)

        if s is None:
            missing.append(g)
            continue

        records.append({
            **g,
            **s,
        })

    y = np.asarray([r["label"] for r in records], dtype=int)
    max_scores = np.asarray([r["max_ema"] for r in records], dtype=float)

    overall = {
        "strategy": strategy,
        "n": len(records),
        "normal": int(np.sum(y == 0)),
        "anomaly": int(np.sum(y == 1)),
        "missing": len(missing),
        "auroc_max_ema": auc_roc(y, max_scores) if len(records) else None,
        "best_f1": best_f1_threshold(y, max_scores) if len(records) else None,
    }

    per_bg = []
    for bg in ["black", "metallic", "white", "wooden"]:
        rr = [r for r in records if r["background"] == bg]
        if not rr:
            continue
        yy = np.asarray([r["label"] for r in rr], dtype=int)
        ss = np.asarray([r["max_ema"] for r in rr], dtype=float)
        per_bg.append({
            "background": bg,
            "n": len(rr),
            "normal": int(np.sum(yy == 0)),
            "anomaly": int(np.sum(yy == 1)),
            "auroc_max_ema": auc_roc(yy, ss),
            "best_f1": best_f1_threshold(yy, ss),
        })

    return overall, per_bg, records, missing


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--gt",
        default="./gt_reserved_newyolo/tracks_to_label.csv",
    )
    p.add_argument(
        "--videos_root",
        default="./videos_background",
    )
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

    p.add_argument(
        "--box_smooth_alpha",
        type=float,
        default=0.35,
        help="EMA alpha for cx, cy, width, height in strategies B/C.",
    )
    p.add_argument(
        "--square_scale",
        type=float,
        default=1.06,
        help="Square side = max(smoothed width,height) * this value.",
    )
    p.add_argument(
        "--border_keep_ratio",
        type=float,
        default=0.78,
        help="Central fraction kept sharp in strategy C.",
    )
    p.add_argument(
        "--border_feather_ratio",
        type=float,
        default=0.08,
    )

    p.add_argument("--batch_frames", type=int, default=16)
    p.add_argument("--gpu_batch_crops", type=int, default=8)
    p.add_argument("--log_every", type=int, default=120)

    p.add_argument(
        "--out",
        default="./crop_strategy_development",
    )

    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    gt, excluded = load_gt(args.gt)

    print("=" * 88)
    print("CROP-STRATEGY DEVELOPMENT EXPERIMENT")
    print("=" * 88)
    print(f"Valid calibration GT tracks : {len(gt)}")
    print(f"Excluded                    : {excluded}")
    print(f"YOLO                        : {args.yolo_det_weights}")
    print(f"EfficientAD                 : {args.ckpt_dir}")
    print(f"EMA alpha                   : {args.ema_alpha}")
    print(f"Box smoothing alpha         : {args.box_smooth_alpha}")
    print(f"Square scale                : {args.square_scale}")
    print()
    print("Primary selection metric: AUROC(max EMA) on the same development tracks.")
    print("Best-F1 thresholds below are diagnostic only.")
    print()

    # Load EfficientAD once.
    scorer = base.EfficientADScorer(args)

    dummy = Image.new("RGB", (scorer.image_size, scorer.image_size))
    _ = scorer([dummy] * min(args.gpu_batch_crops, 8))

    per_strategy_video_rows = {s: {} for s in STRATEGIES}
    timing = defaultdict(dict)
    detector_summary = {}

    # Run each calibration video: YOLO ONCE, then all crop strategies on same detections.
    for bg, video_name, tag in CAL_VIDEOS:
        video_path = Path(args.videos_root) / bg / video_name
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        print()
        print("=" * 88)
        print(f"{bg.upper()} | {video_name}")
        print("=" * 88)

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

        detector_summary[tag] = {
            "background": bg,
            "video": video_name,
            "frames": n_frames,
            "yolo_seconds": yolo_sec,
            "yolo_fps": n_frames / max(yolo_sec, 1e-9),
            "detections": int(sum(len(v) for v in detections.values())),
        }

        for strategy in STRATEGIES:
            print(f"\n--- Strategy: {strategy} ---")
            rows, sec = score_video_strategy(
                strategy,
                args,
                scorer,
                detections,
                frame_cache,
                width,
                height,
                n_frames,
            )
            per_strategy_video_rows[strategy][(bg, video_name)] = rows
            timing[strategy][tag] = {
                "seconds": sec,
                "fps": n_frames / max(sec, 1e-9),
                "n_scored": len(rows),
            }

            strategy_dir = out / strategy / bg / tag
            strategy_dir.mkdir(parents=True, exist_ok=True)
            save_rows(strategy_dir / "scores.csv", rows)

    # Evaluation.
    all_summary = {
        "experiment": {
            "purpose": "development-only crop preprocessing comparison",
            "strategies": STRATEGIES,
            "primary_metric": "AUROC using per-track max EMA",
            "final_test_used_for_selection": False,
            "settings": {
                "ema_alpha": args.ema_alpha,
                "box_smooth_alpha": args.box_smooth_alpha,
                "square_scale": args.square_scale,
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
        "detector": detector_summary,
        "strategies": {},
    }

    track_rows_all = []
    summary_rows = []
    per_bg_rows = []

    print()
    print("=" * 88)
    print("RESULTS — SAME DEVELOPMENT TRACKS")
    print("=" * 88)

    for strategy in STRATEGIES:
        overall, per_bg, records, missing = evaluate_strategy(
            strategy,
            per_strategy_video_rows[strategy],
            gt,
        )

        overall["timing"] = timing[strategy]
        all_summary["strategies"][strategy] = {
            "overall": overall,
            "per_background": per_bg,
            "missing": missing,
        }

        b = overall["best_f1"]
        auc = overall["auroc_max_ema"]
        print(
            f"{strategy:<27s} "
            f"n={overall['n']:>3d} missing={overall['missing']:>2d} | "
            f"AUC={auc:.4f} | "
            f"best-dev-F1={b['f1']:.4f} @ thr={b['threshold']:.6f} | "
            f"P={b['precision']:.3f} R={b['recall']:.3f} FPR={b['fpr']:.3f}"
        )

        summary_rows.append({
            "strategy": strategy,
            "n": overall["n"],
            "missing": overall["missing"],
            "normal": overall["normal"],
            "anomaly": overall["anomaly"],
            "auroc_max_ema": auc,
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
            per_bg_rows.append({
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
            track_rows_all.append({
                "strategy": strategy,
                **r,
            })

    # Ranking is development-only and is based on threshold-free AUC.
    ranking = sorted(
        summary_rows,
        key=lambda r: (
            -1 if r["auroc_max_ema"] is None else r["auroc_max_ema"]
        ),
        reverse=True,
    )

    print()
    print("AUC ranking (development only):")
    for i, r in enumerate(ranking, 1):
        print(f"  {i}. {r['strategy']}: {r['auroc_max_ema']:.4f}")

    print()
    print("IMPORTANT: Do not freeze the best-dev-F1 threshold.")
    print("After choosing a crop strategy, rerun the ORIGINAL normal-only calibration")
    print("workflow for that strategy, then evaluate once on the held-out test set.")

    # Save summary tables.
    with (out / "strategy_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(summary_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary_rows)

    with (out / "per_background_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(per_bg_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(per_bg_rows)

    with (out / "track_statistics_all_strategies.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "strategy","background","video","track_id","label","label_name",
            "n","max_ema","p90_ema","median_ema","mean_ema",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(track_rows_all)

    (out / "crop_strategy_summary.json").write_text(
        json.dumps(all_summary, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"Artifacts: {out.resolve()}")


if __name__ == "__main__":
    main()
