#!/usr/bin/env python3
"""
mean_suppression_grid.py

Development-only grid search for a generic background-suppression setting.

Keeps EXACT current crop geometry and varies only:
  - central keep ratio
  - mean-fill blend strength

Primary selection:
  1) macro-average AUROC across black/metallic/white/wooden
  2) tie-breaker: worst-background AUROC

Why macro AUC?
Pooled AUC can be distorted by background-dependent score offsets. For a generic
solution we want each physical background to separate normal/anomaly reasonably.

No final-test videos are used here.
No final threshold is selected here.
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

KEEP_RATIOS = [0.84, 0.88, 0.92]
STRENGTHS = [0.40, 0.70, 1.00]


def norm_label(s):
    s = (s or "").strip().lower()
    if s in {"normal", "n", "0"}:
        return 0
    if s in {"anomaly", "anomalous", "a", "1"}:
        return 1
    return None


def load_gt(path):
    gt = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("role") or "").strip().lower() != "calibration":
                continue
            bg = (r.get("background") or "").strip().lower()
            video = (r.get("video") or "").strip()
            label_txt = (r.get("gt_label") or "").strip().lower()
            valid_txt = (r.get("gt_valid_piece") or "").strip().lower()
            if label_txt in {"unsure", "not_a_piece"}:
                continue
            if valid_txt not in {"1", "true", "yes"}:
                continue
            label = norm_label(label_txt)
            if label is None:
                continue
            tid = int(float(r["track_id"]))
            gt[(bg, video, tid)] = {
                "background": bg,
                "video": video,
                "track_id": tid,
                "label": label,
            }
    return gt


_MASK_CACHE = {}


def soft_rect_mask(h, w, keep_ratio, feather_ratio):
    key = (h, w, round(keep_ratio, 5), round(feather_ratio, 5))
    if key in _MASK_CACHE:
        return _MASK_CACHE[key]

    mask = np.zeros((h, w), dtype=np.float32)
    xpad = int(round((1.0 - keep_ratio) * 0.5 * w))
    ypad = int(round((1.0 - keep_ratio) * 0.5 * h))

    x1, x2 = xpad, max(xpad + 1, w - xpad)
    y1, y2 = ypad, max(ypad + 1, h - ypad)
    mask[y1:y2, x1:x2] = 1.0

    feather = max(1, int(round(feather_ratio * min(h, w))))
    k = 2 * feather + 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    mask = np.clip(mask, 0.0, 1.0)[..., None]

    _MASK_CACHE[key] = mask
    return mask


def mean_suppress(crop_pil, keep_ratio, feather_ratio, strength):
    arr = np.asarray(crop_pil.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]
    mask = soft_rect_mask(h, w, keep_ratio, feather_ratio)

    # Same central-mean concept as the successful previous experiment.
    center_ratio = 0.50
    xpad = int(round((1.0 - center_ratio) * 0.5 * w))
    ypad = int(round((1.0 - center_ratio) * 0.5 * h))
    center = arr[ypad:h-ypad, xpad:w-xpad]
    if center.size == 0:
        center = arr

    fill = center.reshape(-1, 3).mean(axis=0).astype(np.float32)
    fill_img = np.empty_like(arr, dtype=np.float32)
    fill_img[...] = fill

    arrf = arr.astype(np.float32)

    # Partial suppression in the outer region.
    outer = (1.0 - strength) * arrf + strength * fill_img
    out = arrf * mask + outer * (1.0 - mask)

    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def variants():
    out = [("current", None, None)]
    for keep in KEEP_RATIOS:
        for strength in STRENGTHS:
            name = f"mean_k{int(round(keep*100)):02d}_s{int(round(strength*100)):02d}"
            out.append((name, keep, strength))
    return out


def score_variant(name, keep, strength, args, scorer, detections, frame_cache, width, height, n_frames):
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
                dets, width, height, args.clip_margin, args.min_box_frac
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

                if name != "current":
                    crop = mean_suppress(
                        crop,
                        keep_ratio=keep,
                        feather_ratio=args.feather_ratio,
                        strength=strength,
                    )

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

    sec = time.perf_counter() - t0
    return rows, sec


def track_scores(rows):
    d = defaultdict(list)
    for r in rows:
        d[int(r["track_id"])].append(float(r["ema_score"]))
    return {tid: max(vals) for tid, vals in d.items() if vals}


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


def evaluate(all_video_rows, gt):
    records = []
    cache = {k: track_scores(v) for k, v in all_video_rows.items()}

    for (bg, video, tid), g in gt.items():
        s = cache.get((bg, video), {}).get(tid)
        if s is None:
            continue
        records.append({
            "background": bg,
            "video": video,
            "track_id": tid,
            "label": g["label"],
            "score": s,
        })

    y = np.asarray([r["label"] for r in records], dtype=int)
    s = np.asarray([r["score"] for r in records], dtype=float)
    pooled = auc_roc(y, s)

    bg_auc = {}
    for bg in ["black", "metallic", "white", "wooden"]:
        rr = [r for r in records if r["background"] == bg]
        yy = np.asarray([r["label"] for r in rr], dtype=int)
        ss = np.asarray([r["score"] for r in rr], dtype=float)
        bg_auc[bg] = auc_roc(yy, ss)

    vals = [bg_auc[b] for b in ["black", "metallic", "white", "wooden"]]
    macro = float(np.mean(vals))
    worst = float(np.min(vals))

    return {
        "n": len(records),
        "pooled_auc": pooled,
        "macro_auc": macro,
        "worst_background_auc": worst,
        "black_auc": bg_auc["black"],
        "metallic_auc": bg_auc["metallic"],
        "white_auc": bg_auc["white"],
        "wooden_auc": bg_auc["wooden"],
    }, records


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
    p.add_argument("--feather_ratio", type=float, default=0.08)
    p.add_argument("--batch_frames", type=int, default=16)
    p.add_argument("--gpu_batch_crops", type=int, default=8)
    p.add_argument("--out", default="./mean_suppression_grid")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    gt = load_gt(args.gt)
    varspec = variants()

    print("=" * 96)
    print("GENERIC MEAN-SUPPRESSION GRID — DEVELOPMENT ONLY")
    print("=" * 96)
    print(f"GT tracks       : {len(gt)}")
    print(f"Keep ratios     : {KEEP_RATIOS}")
    print(f"Blend strengths : {STRENGTHS}")
    print("Selection       : highest macro-background AUC, then highest worst-background AUC")
    print()

    scorer = base.EfficientADScorer(args)
    dummy = Image.new("RGB", (scorer.image_size, scorer.image_size))
    _ = scorer([dummy] * min(8, args.gpu_batch_crops))

    all_variant_rows = {name: {} for name, _, _ in varspec}
    runtime = defaultdict(dict)

    for bg, video_name, tag in CAL_VIDEOS:
        video_path = Path(args.videos_root) / bg / video_name
        print()
        print("=" * 96)
        print(f"{bg.upper()} | {video_name}")
        print("=" * 96)

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

        for name, keep, strength in varspec:
            print(f"  scoring {name} ...", end="", flush=True)
            rows, sec = score_variant(
                name, keep, strength, args, scorer,
                detections, frame_cache, width, height, n_frames
            )
            all_variant_rows[name][(bg, video_name)] = rows
            runtime[name][tag] = {
                "seconds": sec,
                "fps": n_frames / max(sec, 1e-9),
            }
            print(f" {n_frames/max(sec,1e-9):.2f} fps")

    results = []
    all_tracks = []

    for name, keep, strength in varspec:
        m, records = evaluate(all_variant_rows[name], gt)
        fps_vals = [x["fps"] for x in runtime[name].values()]
        row = {
            "variant": name,
            "keep_ratio": "" if keep is None else keep,
            "strength": "" if strength is None else strength,
            **m,
            "mean_scoring_fps": float(np.mean(fps_vals)),
        }
        results.append(row)

        for r in records:
            all_tracks.append({"variant": name, **r})

    results.sort(
        key=lambda r: (r["macro_auc"], r["worst_background_auc"]),
        reverse=True,
    )

    print()
    print("=" * 96)
    print("RESULTS")
    print("=" * 96)
    print(
        f"{'variant':<18} {'pool':>7} {'macro':>7} {'worst':>7} "
        f"{'black':>7} {'metal':>7} {'white':>7} {'wood':>7} {'fps':>7}"
    )
    for r in results:
        print(
            f"{r['variant']:<18} "
            f"{r['pooled_auc']:>7.4f} "
            f"{r['macro_auc']:>7.4f} "
            f"{r['worst_background_auc']:>7.4f} "
            f"{r['black_auc']:>7.4f} "
            f"{r['metallic_auc']:>7.4f} "
            f"{r['white_auc']:>7.4f} "
            f"{r['wooden_auc']:>7.4f} "
            f"{r['mean_scoring_fps']:>7.2f}"
        )

    winner = results[0]
    print()
    print("DEVELOPMENT WINNER")
    print(json.dumps(winner, indent=2))
    print()
    print("Do NOT use this to choose a final anomaly threshold.")
    print("Next: freeze only the preprocessing variant, rerun original normal-only calibration,")
    print("then evaluate on a fresh hold-out set if possible.")

    with (out / "grid_results.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(results[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)

    with (out / "track_scores_all_variants.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["variant", "background", "video", "track_id", "label", "score"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_tracks)

    (out / "grid_results.json").write_text(
        json.dumps({
            "selection_rule": "macro AUC, tie-break by worst-background AUC",
            "keep_ratios": KEEP_RATIOS,
            "strengths": STRENGTHS,
            "results": results,
            "winner": winner,
        }, indent=2),
        encoding="utf-8",
    )

    print(f"Artifacts: {out.resolve()}")


if __name__ == "__main__":
    main()
