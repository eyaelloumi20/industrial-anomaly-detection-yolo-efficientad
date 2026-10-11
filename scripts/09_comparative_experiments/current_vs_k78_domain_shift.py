#!/usr/bin/env python3
"""
current_vs_k78_domain_shift.py

Goal
----
Quantify score-distribution shift between:

A) current preprocessing (original v2 crop behavior)
B) K78 mean-border preprocessing (v3):
   keep=0.78, feather=0.08, strength=1.0

Domains
-------
Original normal calibration:
  - videoN1.mp4
  - video_N.mp4
  - video_N2.mp4

New-background development videos:
  - black
  - metallic
  - white
  - wooden

The comparison freezes:
  - NEW multi-background YOLO
  - LowRes192
  - EMA alpha = 0.3
  - same detections / tracking per video

Outputs
-------
- score_distribution_summary.csv
- domain_shift_summary.csv
- per_video_summary.csv
- hist_current_vs_k78.png
- boxplot_domain_scores.png
- results.json

This experiment is diagnostic. It does not tune the final threshold.
"""

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import video_streaming_detector_simple_optimized_v3 as base


ORIGINAL_NORMAL = [
    ("orig_normal", "videoN1", "./videos_N/videoN1.mp4"),
    ("orig_normal", "video_N", "./videos_N/video_N.mp4"),
    ("orig_normal", "video_N2", "./videos_N/video_N2.mp4"),
]

NEW_DEV = [
    ("black",    "black_cal",    "./videos_background/black/WhatsApp Video 2026-09-18 at 15.15.353.mp4"),
    ("metallic", "metallic_cal", "./videos_background/metallic/WhatsApp Video 2026-09-18 at 15.49.53.mp4"),
    ("white",    "white_cal",    "./videos_background/white/WhatsApp Video 2026-09-18 at 15.55.11.mp4"),
    ("wooden",   "wooden_cal",   "./videos_background/wooden/WhatsApp Video 2026-09-18 at 15.31.552.mp4"),
]


def score_video(
    mode,
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

                if mode == "k78":
                    crop = base.apply_mean_border_suppression(
                        crop,
                        keep_ratio=0.78,
                        feather_ratio=0.08,
                        strength=1.0,
                    )

                crops.append(crop)
                metas.append((fi, int(tid)))

        scores = []
        for mb in range(0, len(crops), args.gpu_batch_crops):
            scores.extend(
                scorer(
                    crops[mb:mb + args.gpu_batch_crops]
                )
            )

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

    return rows, time.perf_counter() - t0


def summarize(vals):
    a = np.asarray(vals, dtype=float)
    return {
        "n": int(len(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "p05": float(np.quantile(a, 0.05)),
        "p25": float(np.quantile(a, 0.25)),
        "median": float(np.median(a)),
        "p75": float(np.quantile(a, 0.75)),
        "p95": float(np.quantile(a, 0.95)),
        "p99": float(np.quantile(a, 0.99)),
        "max": float(np.max(a)),
    }


def track_maxes(rows):
    d = defaultdict(list)
    for r in rows:
        d[int(r["track_id"])].append(float(r["ema_score"]))
    return [max(v) for v in d.values() if v]


def standardized_mean_shift(ref, target):
    ref = np.asarray(ref, dtype=float)
    target = np.asarray(target, dtype=float)
    pooled = math.sqrt(
        (np.var(ref) + np.var(target)) / 2.0
    )
    if pooled <= 1e-12:
        return 0.0
    return float(
        (np.mean(target) - np.mean(ref)) / pooled
    )


def overlap_coefficient(a, b, bins=120):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    lo = min(float(a.min()), float(b.min()))
    hi = max(float(a.max()), float(b.max()))
    if hi <= lo:
        return 1.0

    ha, edges = np.histogram(
        a, bins=bins, range=(lo, hi), density=True
    )
    hb, _ = np.histogram(
        b, bins=edges, density=True
    )
    widths = np.diff(edges)
    return float(
        np.sum(
            np.minimum(ha, hb) * widths
        )
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--yolo_det_weights",
        default="./runs/detect/runs/detect/yolo_runs/chewinggum_yolov8n_multibg/weights/best.pt",
    )
    p.add_argument(
        "--ckpt_dir",
        default="./ckpt_lowres_192",
    )
    p.add_argument("--method", default="lowres")
    p.add_argument("--category", default="chewinggum")
    p.add_argument("--model_size", default="S")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--fp16", action="store_true", default=True)

    p.add_argument("--conf", type=float, default=0.10)
    p.add_argument("--iou", type=float, default=0.50)
    p.add_argument("--ratio", type=float, default=0.1)

    p.add_argument("--pad_ratio", type=float, default=0.02)
    p.add_argument("--denoise", action="store_true")
    p.add_argument("--ema_alpha", type=float, default=0.3)
    p.add_argument("--clip_margin", type=int, default=5)
    p.add_argument("--min_box_frac", type=float, default=0.5)

    p.add_argument("--batch_frames", type=int, default=16)
    p.add_argument("--gpu_batch_crops", type=int, default=8)

    p.add_argument(
        "--out",
        default="./current_vs_k78_domain_shift",
    )

    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    scorer = base.EfficientADScorer(args)
    dummy = Image.new(
        "RGB",
        (scorer.image_size, scorer.image_size),
    )
    _ = scorer(
        [dummy] * min(8, args.gpu_batch_crops)
    )

    videos = ORIGINAL_NORMAL + NEW_DEV

    all_scores = {
        "current": {},
        "k78": {},
    }

    per_video_rows = []

    print("=" * 96)
    print("CURRENT vs K78 DOMAIN-SHIFT DIAGNOSTIC")
    print("=" * 96)
    print("Same YOLO / tracking / LowRes192 / EMA for both preprocessings.")
    print()

    for domain, tag, path_str in videos:
        video_path = Path(path_str)

        if not video_path.exists():
            raise FileNotFoundError(video_path)

        print()
        print("=" * 96)
        print(f"{domain.upper()} | {video_path.name}")
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

        for mode in ["current", "k78"]:
            print(f"  scoring {mode} ...", end="", flush=True)

            rows, sec = score_video(
                mode,
                args,
                scorer,
                detections,
                frame_cache,
                width,
                height,
                n_frames,
            )

            all_scores[mode][tag] = {
                "domain": domain,
                "video": str(video_path),
                "rows": rows,
            }

            ema_vals = [
                float(r["ema_score"])
                for r in rows
            ]

            s = summarize(ema_vals)

            per_video_rows.append({
                "mode": mode,
                "domain": domain,
                "tag": tag,
                "video": str(video_path),
                "fps": n_frames / max(sec, 1e-9),
                **s,
            })

            print(
                f" {n_frames/max(sec,1e-9):.2f} fps | "
                f"mean={s['mean']:.4f} p99={s['p99']:.4f}"
            )

    # Pool original calibration scores per preprocessing.
    ref_current = []
    ref_k78 = []

    for tag in ["videoN1", "video_N", "video_N2"]:
        ref_current.extend(
            float(r["ema_score"])
            for r in all_scores["current"][tag]["rows"]
        )
        ref_k78.extend(
            float(r["ema_score"])
            for r in all_scores["k78"][tag]["rows"]
        )

    tau_current = float(
        np.quantile(
            np.asarray(ref_current),
            0.99,
        )
    )
    tau_k78 = float(
        np.quantile(
            np.asarray(ref_k78),
            0.99,
        )
    )

    domain_rows = []

    for mode, ref_vals, tau in [
        ("current", ref_current, tau_current),
        ("k78", ref_k78, tau_k78),
    ]:
        for tag in [
            "black_cal",
            "metallic_cal",
            "white_cal",
            "wooden_cal",
        ]:
            item = all_scores[mode][tag]
            vals = [
                float(r["ema_score"])
                for r in item["rows"]
            ]
            tmax = track_maxes(item["rows"])

            row = {
                "mode": mode,
                "domain": item["domain"],
                "calibrated_tau": tau,
                "n_scores": len(vals),
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "p95": float(np.quantile(vals, 0.95)),
                "p99": float(np.quantile(vals, 0.99)),
                "max": float(np.max(vals)),
                "fraction_frames_above_tau": float(
                    np.mean(
                        np.asarray(vals) >= tau
                    )
                ),
                "fraction_tracks_max_above_tau": float(
                    np.mean(
                        np.asarray(tmax) >= tau
                    )
                ) if tmax else 0.0,
                "standardized_mean_shift_vs_orig_normal": standardized_mean_shift(
                    ref_vals,
                    vals,
                ),
                "hist_overlap_vs_orig_normal": overlap_coefficient(
                    ref_vals,
                    vals,
                ),
            }
            domain_rows.append(row)

    print()
    print("=" * 96)
    print("CALIBRATED THRESHOLDS FROM ORIGINAL NORMAL DOMAIN")
    print("=" * 96)
    print(f"current tau99 : {tau_current:.6f}")
    print(f"k78 tau99     : {tau_k78:.6f}")

    print()
    print("=" * 96)
    print("DOMAIN SHIFT SUMMARY")
    print("=" * 96)

    for mode in ["current", "k78"]:
        print()
        print(mode.upper())
        for r in domain_rows:
            if r["mode"] != mode:
                continue

            print(
                f"{r['domain']:<9s} "
                f"mean={r['mean']:.4f} "
                f"p99={r['p99']:.4f} "
                f"frames>=tau={100*r['fraction_frames_above_tau']:.2f}% "
                f"tracks>=tau={100*r['fraction_tracks_max_above_tau']:.2f}% "
                f"shift={r['standardized_mean_shift_vs_orig_normal']:+.2f} "
                f"overlap={r['hist_overlap_vs_orig_normal']:.3f}"
            )

    # CSV outputs.
    with (out / "per_video_summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        fields = list(
            per_video_rows[0].keys()
        )
        w = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        w.writeheader()
        w.writerows(per_video_rows)

    with (out / "domain_shift_summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        fields = list(
            domain_rows[0].keys()
        )
        w = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        w.writeheader()
        w.writerows(domain_rows)

    # Compact summary table.
    summary_rows = []
    for mode, ref_vals, tau in [
        ("current", ref_current, tau_current),
        ("k78", ref_k78, tau_k78),
    ]:
        s = summarize(ref_vals)
        summary_rows.append({
            "mode": mode,
            "domain": "orig_normal_pooled",
            "calibrated_tau99": tau,
            **s,
        })

        for tag in [
            "black_cal",
            "metallic_cal",
            "white_cal",
            "wooden_cal",
        ]:
            vals = [
                float(r["ema_score"])
                for r in all_scores[mode][tag]["rows"]
            ]
            s = summarize(vals)
            summary_rows.append({
                "mode": mode,
                "domain": all_scores[mode][tag]["domain"],
                "calibrated_tau99": tau,
                **s,
            })

    with (out / "score_distribution_summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        fields = list(
            summary_rows[0].keys()
        )
        w = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        w.writeheader()
        w.writerows(summary_rows)

    # Plot 1: histograms by mode.
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 9),
    )

    for ax, mode, ref_vals, tau in [
        (
            axes[0],
            "current",
            ref_current,
            tau_current,
        ),
        (
            axes[1],
            "k78",
            ref_k78,
            tau_k78,
        ),
    ]:
        ax.hist(
            ref_vals,
            bins=80,
            alpha=0.45,
            density=True,
            label="orig_normal",
        )

        for tag in [
            "black_cal",
            "metallic_cal",
            "white_cal",
            "wooden_cal",
        ]:
            vals = [
                float(r["ema_score"])
                for r in all_scores[mode][tag]["rows"]
            ]
            ax.hist(
                vals,
                bins=80,
                alpha=0.28,
                density=True,
                label=all_scores[mode][tag]["domain"],
            )

        ax.axvline(
            tau,
            linestyle="--",
            linewidth=1.5,
            label=f"tau99={tau:.3f}",
        )
        ax.set_title(
            f"{mode}: EMA score distributions"
        )
        ax.set_xlabel("EMA anomaly score")
        ax.set_ylabel("Density")
        ax.legend()

    fig.tight_layout()
    fig.savefig(
        out / "hist_current_vs_k78.png",
        dpi=180,
    )
    plt.close(fig)

    # Plot 2: boxplot.
    labels = []
    data = []

    for mode in ["current", "k78"]:
        ref = (
            ref_current
            if mode == "current"
            else ref_k78
        )
        labels.append(
            f"{mode}\norig"
        )
        data.append(ref)

        for tag in [
            "black_cal",
            "metallic_cal",
            "white_cal",
            "wooden_cal",
        ]:
            labels.append(
                f"{mode}\n{all_scores[mode][tag]['domain']}"
            )
            data.append([
                float(r["ema_score"])
                for r in all_scores[mode][tag]["rows"]
            ])

    fig = plt.figure(
        figsize=(14, 6),
    )
    plt.boxplot(
        data,
        labels=labels,
        showfliers=False,
    )
    plt.ylabel("EMA anomaly score")
    plt.title(
        "Score distribution by preprocessing and domain"
    )
    plt.xticks(rotation=30)
    plt.tight_layout()
    fig.savefig(
        out / "boxplot_domain_scores.png",
        dpi=180,
    )
    plt.close(fig)

    results = {
        "thresholds": {
            "current_tau99": tau_current,
            "k78_tau99": tau_k78,
        },
        "domain_shift": domain_rows,
        "per_video": per_video_rows,
    }

    (out / "results.json").write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(f"Artifacts: {out.resolve()}")


if __name__ == "__main__":
    main()
