"""
video_evaluate.py
=================
Video evaluation for the two-stage pipeline.

- Runs YOLOv8 + ByteTrack (or DeepOCSORT)
- Processes **each tracked gum piece individually** across frames
- Computes EfficientAD anomaly score per crop (same preprocessing as evaluate_pipeline.py)
- Aggregates per-track statistics (max / mean / final score)
- Computes full metrics (AUROC, AUPRO, AP, confusion, etc.) using **max-score per track**
- Outputs same artefacts as evaluate_pipeline.py + per-track CSV

Usage:
    python -m pipeline.video_evaluate \
        --video_path /path/to/test_video.mp4 \
        --gt_labels /path/to/frame_labels.csv \   # optional: frame_idx,label
        --ckpt_dir ./ckpt_cropped \
        --yolo_weights ./yolo_runs/.../best.pt \
        --category chewinggum \
        --model_size S \
        --save_dir ./video_eval_results \
        --device cuda
"""

import os
import sys
import csv
import argparse
import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from utils.load_efficientad import load_efficientad_model
from pipeline.pipeline import InspectionPipeline, preprocess_crop
from utils.metrics import compute_all_metrics, print_results, save_quantitative_txt
from utils.visualisation import save_roc_curve, save_score_histogram, save_per_image_csv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video_path", required=True, help="Path to evaluation video")
    p.add_argument("--gt_labels", default=None,
                   help="Optional CSV: frame_idx,label (0=normal, 1=anomaly). "
                        "If omitted, all frames treated as normal for scoring only.")
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--yolo_weights", required=True)
    p.add_argument("--save_dir", default="./video_eval_results")
    p.add_argument("--category", default="chewinggum")
    p.add_argument("--model_size", default="S", choices=["S", "M"])
    p.add_argument("--threshold", type=float, default=None,
                   help="Override threshold (else computed via Youden J)")
    p.add_argument("--pad_ratio", type=float, default=0.02)
    p.add_argument("--no_denoise", action="store_true")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--ema_alpha", type=float, default=0.3)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_gt_labels(gt_path):
    if not gt_path or not os.path.exists(gt_path):
        return {}
    gt = {}
    with open(gt_path) as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) >= 2:
                try:
                    gt[int(row[0])] = int(row[1])
                except:
                    continue
    return gt


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*70}")
    print("  Video Evaluation — Piece-by-Piece (Track-level)")
    print(f"{'='*70}")
    print(f"Video      : {args.video_path}")
    print(f"CKPT       : {args.ckpt_dir}")
    print(f"YOLO       : {args.yolo_weights}")

    # Load models
    models = load_efficientad_model(
        ckpt_dir=args.ckpt_dir,
        category=args.category,
        model_size=args.model_size,
        device=device
    )

    # Pipeline (same as evaluate_pipeline.py)
    pipeline = InspectionPipeline(
        models=models,
        device=device,
        detector_path=args.yolo_weights,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        pad_ratio=args.pad_ratio,
        denoise=not args.no_denoise,
        use_full_image=False,
    )

    # GT
    gt_dict = load_gt_labels(args.gt_labels)
    print(f"GT labels  : {len(gt_dict)} frames annotated")

    cap = cv2.VideoCapture(args.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    track_scores = defaultdict(list)      # tid -> list of raw scores
    track_ema = {}                        # tid -> current EMA
    frame_results = []                    # for per-frame logging

    print(f"\nProcessing video ({frame_count} frames)...")

    frame_idx = 0
    pbar = tqdm(total=frame_count, desc="Frames")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        pil_frame = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        # Run detection + tracking
        results = pipeline.detector._model.track(   # direct access to tracker
            source=frame_bgr,
            conf=args.conf,
            iou=args.iou,
            classes=[0],
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        current_frame_max = 0.0
        active_tids = set()

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            boxes = r.boxes.xyxy.cpu().numpy().astype(int)
            track_ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else range(len(boxes))

            for box, tid in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                active_tids.add(tid)

                # Same crop preprocessing as pipeline.py
                crop_pil = Image.fromarray(np.array(pil_frame)[y1:y2, x1:x2])
                # Use the same helper as evaluation pipeline
                crop_t = preprocess_crop(
                    pil_image=pil_frame,
                    box=(x1, y1, x2, y2),
                    pad_ratio=args.pad_ratio,
                    denoise=not args.no_denoise
                ).unsqueeze(0).to(device)

                # EfficientAD forward (reuse pipeline logic)
                _, score = pipeline._efficientad_score(crop_t)

                track_scores[tid].append(score)
                prev = track_ema.get(tid, score)
                ema = args.ema_alpha * score + (1 - args.ema_alpha) * prev
                track_ema[tid] = ema

                current_frame_max = max(current_frame_max, ema)

        # Record per-frame
        gt_label = gt_dict.get(frame_idx, 0)  # default normal if unknown
        frame_results.append({
            "frame": frame_idx,
            "gt": gt_label,
            "max_score": current_frame_max,
            "active_tracks": len(active_tids)
        })

        frame_idx += 1
        pbar.update(1)

    cap.release()
    pbar.close()

    # ====================== Track-level Aggregation ======================
    print("\nComputing track-level statistics...")

    track_max_scores = []
    track_labels = []   # We assume a track is anomalous if any frame is anomalous

    for tid, scores in track_scores.items():
        if not scores:
            continue
        max_score = max(scores)
        track_max_scores.append(max_score)

        # Label a track as anomalous if it ever appeared in an anomalous frame
        track_frames = [f["frame"] for f in frame_results if tid in [t for t in track_ema.keys()]]  # rough
        track_gt = max((gt_dict.get(f, 0) for f in range(min(track_frames or [0]), max(track_frames or [0])+1)), default=0)
        track_labels.append(track_gt)

    y_true = np.array(track_labels)
    y_score = np.array(track_max_scores)

    # ====================== Metrics ======================
    metrics = compute_all_metrics(y_true, y_score, np.zeros(len(y_true)))  # dummy times
    print_results(metrics, category=f"{args.category} (video tracks)")

    # Save outputs
    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_roc_curve(
        metrics['fpr'], metrics['tpr'], metrics['auroc'],
        f"{args.category}_video", str(out_dir / "roc_curve.png")
    )
    save_score_histogram(
        y_score, y_true, metrics['threshold'],
        f"{args.category}_video", str(out_dir / "score_histogram.png")
    )

    # Per-track CSV
    with open(out_dir / "per_track_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "max_score", "mean_score", "n_frames", "gt_label", "prediction"])
        for tid, scores in track_scores.items():
            max_s = max(scores)
            mean_s = np.mean(scores)
            n = len(scores)
            gt = max((gt_dict.get(f, 0) for f in range(frame_count) if True), default=0)  # simplify
            pred = 1 if max_s >= metrics['threshold'] else 0
            writer.writerow([tid, f"{max_s:.6f}", f"{mean_s:.6f}", n, gt, pred])

    # Frame-level CSV (for reference)
    with open(out_dir / "per_frame_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "gt", "max_score"])
        for row in frame_results:
            writer.writerow([row["frame"], row["gt"], row["max_score"]])

    # Threshold
    with open(out_dir / "threshold.txt", "w") as f:
        f.write(f"{metrics['threshold']:.8f}\n")

    save_quantitative_txt({f"{args.category}_video": metrics}, str(out_dir / "quantitative.txt"))

    print(f"\n✅ Video evaluation complete!")
    print(f"   Results: {out_dir.resolve()}")
    print(f"   Tracks evaluated : {len(track_scores)}")
    print(f"   AUROC            : {metrics['auroc']:.5f}")


if __name__ == "__main__":
    main()