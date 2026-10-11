"""
single_photo_inference.py
=========================
Evaluate the full two-stage pipeline (YOLO detection → cropped chewing-gum pieces → EfficientAD anomaly scoring)
on **ONE single production-line photo**.

What it does:
  1. Runs YOLOv8 to detect every chewing-gum piece
  2. For each detection:
       - Crops with 5% context padding
       - Resizes to 256×256 + Gaussian + Bilateral denoising (exactly like build_crop_dataset.py)
       - Saves the clean cropped photo
       - Runs EfficientAD on the crop
       - Generates a hot heatmap overlay on the crop
       - Saves the overlay
  3. Prints detection boxes + per-crop anomaly scores + final prediction

Usage (example):
    python single_photo_inference.py \
        --photo_path      /home/ahmed/framework/M89bP.jpg\
        --ckpt_dir        ./ckpt_cropped \
        --yolo_weights    /home/ahmed/framework/runs/detect/yolo_runs/chewinggum_finetune2/weights/best.pt\
        --out_dir         ./single_photo_results8 \
        --device          cuda

Output folder will contain:
    cropped_0.png          ← clean 256×256 crop
    overlay_0.png          ← EfficientAD heatmap blended on the crop
    ...
    full_image_annotated.png  ← original photo with all boxes + scores drawn
"""

import os
import time
import argparse
import csv
import numpy as np
from PIL import Image
import cv2
import torch
from torchvision import transforms
import matplotlib

matplotlib.use("Agg")  # no GUI

# ── Import your framework modules ───────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
import sys
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from utils.detection import Detector
from utils.load_efficientad import load_efficientad_model
from utils.efficientad_inference import run_inference, anomaly_map_to_heatmap
from pipeline.pipeline import preprocess_crop

IMAGE_SIZE = 256


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--photo_path',   required=True, help='Path to the single production-line photo')
    p.add_argument('--ckpt_dir',     required=True, help='EfficientAD checkpoint folder (with chewinggum_*.pth)')
    p.add_argument('--yolo_weights', required=True, help='Fine-tuned YOLOv8 .pt')
    p.add_argument('--out_dir',      default='./single_photo_results')
    p.add_argument('--category',     default='chewinggum')
    p.add_argument('--model_size',   default='S', choices=['S', 'M'])
    p.add_argument('--device',       default='cuda')
    p.add_argument('--pad_ratio',    type=float, default=0.02)
    p.add_argument('--threshold',    type=float, default=0.5, help='Anomaly threshold (you can override later)')
    return p.parse_args()


# ── Helpers: identical to build_crop_dataset.py ──────────────────────────────

def resize_with_padding(crop: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    """
    Resize keeping aspect ratio, then pad with black to fill size×size.
    This ensures the full object is always visible without any cropping or stretching.
    Mirrors resize_with_padding() in build_crop_dataset.py exactly.
    """
    original_w, original_h = crop.size
    scale  = size / max(original_w, original_h)
    new_w  = int(original_w * scale)
    new_h  = int(original_h * scale)

    resized = crop.resize((new_w, new_h), Image.LANCZOS)

    # Create black canvas and paste centered
    canvas = Image.new('RGB', (size, size), (0, 0, 0))
    pad_x  = (size - new_w) // 2
    pad_y  = (size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas


def get_processed_crop_pil(pil_img, box, pad_ratio=0.02, denoise=True):
    """
    Crop tightly on the detected ROI, refine with Otsu to remove background,
    rotate 90° if the crop is portrait (height > width) so it fills the canvas
    better after resize, resize preserving aspect ratio with black padding to
    256×256, optionally apply Gaussian + bilateral denoising.
    Mirrors crop_and_denoise() in build_crop_dataset.py exactly (+ rotation).
    Returns a PIL Image (RGB).
    """
    W, H = pil_img.size
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1

    px = int(bw * pad_ratio)
    py = int(bh * pad_ratio)
    x1 = max(0, x1 - px);  y1 = max(0, y1 - py)
    x2 = min(W, x2 + px);  y2 = min(H, y2 + py)

    crop = pil_img.crop((x1, y1, x2, y2)).convert('RGB')

    # ── Otsu refinement: remove dark background inside the YOLO box ──────────
    arr_gray = np.array(crop.convert('L'), dtype=np.uint8)
    _, mask  = cv2.threshold(arr_gray, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(mask)
    if coords is not None:
        rx, ry, rw, rh = cv2.boundingRect(coords)
        coverage = (rw * rh) / (arr_gray.shape[1] * arr_gray.shape[0])
        if coverage >= 0.30:
            margin = 2
            rx = max(0, rx - margin)
            ry = max(0, ry - margin)
            rw = min(crop.width  - rx, rw + 2 * margin)
            rh = min(crop.height - ry, rh + 2 * margin)
            crop = crop.crop((rx, ry, rx + rw, ry + rh))

    # ── Rotate vertical crops to landscape before resizing ───────────────────
    # When the ROI is taller than wide (portrait), rotating 90° ensures
    # resize_with_padding scales on the longer axis, minimising black padding.
    if crop.height > crop.width:
        crop = crop.rotate(90, expand=True)

    # ── Resize keeping aspect ratio + black padding to 256×256 ───────────────
    crop = resize_with_padding(crop, IMAGE_SIZE)

    if denoise:
        arr = np.array(crop, dtype=np.uint8)
        arr = cv2.GaussianBlur(arr, (3, 3), sigmaX=0.8, sigmaY=0.8)
        arr = cv2.bilateralFilter(arr, d=5, sigmaColor=15, sigmaSpace=15)
        crop = Image.fromarray(arr)

    return crop


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device

    print(f"[SinglePhoto] Loading EfficientAD ({args.model_size}) from {args.ckpt_dir} ...")
    models = load_efficientad_model(
        ckpt_dir=args.ckpt_dir,
        category=args.category,
        model_size=args.model_size,
        device=device
    )

    print(f"[SinglePhoto] Loading YOLO detector from {args.yolo_weights} ...")
    detector = Detector(
        model_path=args.yolo_weights,
        conf_threshold=0.25,
        iou_threshold=0.45,
        device=device
    )

    pil_img = Image.open(args.photo_path).convert('RGB')
    print(f"[SinglePhoto] Input image: {pil_img.size} → {args.photo_path}")

    # ── Stage 1: YOLO Detection ───────────────────────────────────
    t_yolo_start = time.perf_counter()
    boxes = detector.detect(pil_img)
    t_yolo_ms = (time.perf_counter() - t_yolo_start) * 1000
    print(f"[SinglePhoto] Detected {len(boxes)} chewing-gum piece(s) "
          f"[YOLO: {t_yolo_ms:.1f} ms]")

    # ── Stage 2: Per-crop processing ──────────────────────────────
    to_tensor = transforms.ToTensor()
    t_total_start = time.perf_counter()

    crop_results = []   # for annotation
    csv_rows = []

    for i, box in enumerate(boxes):
        t_crop_start = time.perf_counter()

        crop_pil = get_processed_crop_pil(pil_img, box,
                                          pad_ratio=args.pad_ratio,
                                          denoise=True)

        # Save clean crop
        crop_path = os.path.join(args.out_dir, f"cropped_{i:03d}.png")
        crop_pil.save(crop_path)

        # EfficientAD
        crop_t = to_tensor(crop_pil).unsqueeze(0).to(device)
        anomaly_map_t, score_t = run_inference(crop_t, models, ratio=0.1)
        anomaly_map_np = anomaly_map_t[0, 0].cpu().numpy()
        score = float(score_t[0])

        pred = "ANOMALY" if score >= args.threshold else "NORMAL"
        t_crop_ms = (time.perf_counter() - t_crop_start) * 1000

        crop_results.append((box, score, pred))

        print(f"  Crop {i:03d} | Box {box} | Score {score:.4f} → {pred} "
              f"[crop time: {t_crop_ms:.1f} ms]")

        # Save overlay
        overlay_pil = anomaly_map_to_heatmap(anomaly_map_np, crop_pil)
        overlay_path = os.path.join(args.out_dir, f"overlay_{i:03d}.png")
        overlay_pil.save(overlay_path)

        # CSV row
        x1, y1, x2, y2 = box
        csv_rows.append({
            "crop_id": i,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "width": x2-x1, "height": y2-y1,
            "score": round(score, 6),
            "prediction": pred,
            "crop_time_ms": round(t_crop_ms, 2),
            "crop_path": f"cropped_{i:03d}.png",
            "overlay_path": f"overlay_{i:03d}.png"
        })

    # Total timing
    t_pipeline_total_ms = (time.perf_counter() - t_total_start) * 1000
    total_time_ms = t_yolo_ms + t_pipeline_total_ms

    print(f"\n{'='*80}")
    print("  TIMING SUMMARY")
    print(f"{'='*80}")
    print(f"  YOLO detection          : {t_yolo_ms:8.1f} ms")
    print(f"  EfficientAD all crops   : {t_pipeline_total_ms:8.1f} ms "
          f"({len(boxes)} crops)")
    print(f"  TOTAL pipeline time     : {total_time_ms:8.1f} ms")
    print(f"{'='*80}")

    # ── Save annotated full image ─────────────────────────────────────
    annotated = np.array(pil_img.convert('RGB'))
    for i, (box, score, pred) in enumerate(crop_results):
        x1, y1, x2, y2 = box
        color = (0, 0, 255) if pred == "ANOMALY" else (0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        cv2.putText(annotated, f"ID{i} {score:.3f}",
                    (x1, max(y1 - 15, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    annotated_pil = Image.fromarray(annotated)
    annotated_pil.save(os.path.join(args.out_dir, "full_image_annotated.png"))
    print(f"  full_image_annotated.png → saved with boxes + scores")

    # ── Save CSV ─────────────────────────────────────────────────────
    csv_path = os.path.join(args.out_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "crop_id","x1","y1","x2","y2","width","height",
            "score","prediction","crop_time_ms","crop_path","overlay_path"
        ])
        writer.writeheader()
        writer.writerows(csv_rows)

    # Add TOTAL row
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["TOTAL","","","","","","","",
                         f"YOLO: {t_yolo_ms:.1f}ms + EAD: {t_pipeline_total_ms:.1f}ms",
                         round(total_time_ms, 2),"",""])

    print(f"\n[SinglePhoto] Done! All files saved to: {args.out_dir}")
    print("   • full_image_annotated.png  ← Full photo with boxes & scores")
    print("   • results.csv               ← All crops + times + TOTAL")
    print("   • cropped_*.png + overlay_*.png")
