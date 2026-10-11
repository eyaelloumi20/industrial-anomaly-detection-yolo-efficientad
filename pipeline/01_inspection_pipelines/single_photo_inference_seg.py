"""
single_photo_inference.py
=========================
Evaluate the full two-stage pipeline (YOLO **segmentation** detection →
cropped chewing-gum pieces → EfficientAD anomaly scoring) on ONE single
production-line photo.

What it does:
  1. Runs a YOLOv8-seg model to detect every chewing-gum piece and obtain
     its pixel-level segmentation mask.
  2. For each detection:
       - Crops the tight bounding box with 5% context padding.
       - Applies the segmentation mask (background → black) so EfficientAD
         only sees the actual gum surface.
       - Resizes to 256×256 + Gaussian + Bilateral denoising (same as
         build_crop_dataset.py).
       - Saves the clean masked crop.
       - Runs EfficientAD on the crop.
       - Generates a hot heatmap overlay on the crop.
       - Saves the overlay.
  3. Prints detection boxes + per-crop anomaly scores + final prediction.

Usage (example):
    python single_photo_inference.py \
        --photo_path      /home/ahmed/framework/M89bP.jpg \
        --ckpt_dir        ./ckpt_cropped \
        --yolo_weights    /home/ahmed/framework/runs/segment/chewinggum_seg/weights/best.pt \
        --out_dir         ./single_photo_results_seg \
        --device          cuda

Output folder will contain:
    cropped_000.png            ← clean 256×256 masked crop
    overlay_000.png            ← EfficientAD heatmap blended on the crop
    ...
    full_image_annotated.png   ← original photo with segmentation masks,
                                  boxes and scores drawn
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

from utils.load_efficientad import load_efficientad_model
from utils.efficientad_inference import run_inference, anomaly_map_to_heatmap

IMAGE_SIZE = 256


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--photo_path",   required=True,
                   help="Path to the single production-line photo")
    p.add_argument("--ckpt_dir",     required=True,
                   help="EfficientAD checkpoint folder (with chewinggum_*.pth)")
    p.add_argument("--yolo_weights", required=True,
                   help="Fine-tuned YOLOv8-seg .pt weights")
    p.add_argument("--out_dir",      default="./single_photo_results_seg")
    p.add_argument("--category",     default="chewinggum")
    p.add_argument("--model_size",   default="S", choices=["S", "M"])
    p.add_argument("--device",       default="cuda")
    p.add_argument("--conf",         type=float, default=0.25)
    p.add_argument("--iou",          type=float, default=0.45)
    p.add_argument("--pad_ratio",    type=float, default=0.02)
    p.add_argument("--threshold",    type=float, default=0.5,
                   help="Anomaly score threshold")
    return p.parse_args()


# ── Segmentation detector wrapper ───────────────────────────────────────────

class SegmentationDetector:
    """
    Wraps a YOLOv8-seg model.  Returns a list of dicts, each containing:
        'box'  : (x1, y1, x2, y2) int pixel coords
        'mask' : H×W uint8 binary mask (255 = foreground) at original image size
        'conf' : float confidence
    """

    def __init__(self, model_path: str, conf: float = 0.25,
                 iou: float = 0.45, device: str = "cuda"):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.model.to(device)
        self.conf = conf
        self.iou  = iou

    def detect(self, pil_img: Image.Image) -> list[dict]:
        """Run segmentation inference; return list of detection dicts."""
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        H, W  = frame.shape[:2]

        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            classes=[0],
            verbose=False,
        )

        detections = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue

            boxes = r.boxes.xyxy.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()

            # masks are returned as (N, H, W) float32 in [0,1] – already
            # resized to the original image resolution by ultralytics ≥ 8.0.
            if r.masks is not None:
                masks_np = (r.masks.data.cpu().numpy() * 255).astype(np.uint8)
                # Ultralytics may return masks at a down-scaled resolution;
                # resize each one back to the original image size.
                resized_masks = []
                for m in masks_np:
                    if m.shape != (H, W):
                        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                    resized_masks.append(m)
            else:
                # Fallback: synthesise a rectangular mask from the bounding box
                resized_masks = []
                for box in boxes:
                    m = np.zeros((H, W), dtype=np.uint8)
                    m[box[1]:box[3], box[0]:box[2]] = 255
                    resized_masks.append(m)

            for box, mask, conf in zip(boxes, resized_masks, confs):
                detections.append({
                    "box":  tuple(box.tolist()),   # (x1, y1, x2, y2)
                    "mask": mask,                  # H×W uint8
                    "conf": float(conf),
                })

        return detections


# ── Crop helpers ─────────────────────────────────────────────────────────────

def resize_with_padding(crop: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    """
    Resize keeping aspect ratio, then pad with black to fill size×size.
    Mirrors resize_with_padding() in build_crop_dataset.py exactly.
    """
    ow, oh = crop.size
    scale  = size / max(ow, oh)
    nw, nh = int(ow * scale), int(oh * scale)
    resized = crop.resize((nw, nh), Image.LANCZOS)
    canvas  = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def get_processed_crop_pil(
    pil_img: Image.Image,
    box: tuple,
    mask_full: np.ndarray,       # H×W uint8 binary mask at original image size
    pad_ratio: float = 0.02,
    denoise: bool = True,
) -> Image.Image:
    """
    1. Expand the bounding box by pad_ratio.
    2. Apply the segmentation mask (background → black) on the full image.
    3. Crop the masked image to the padded bounding box.
    4. Rotate portrait crops to landscape.
    5. Resize with black padding to IMAGE_SIZE×IMAGE_SIZE.
    6. Optional Gaussian + bilateral denoising.
    Returns a PIL Image (RGB).
    """
    W, H = pil_img.size
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1

    # ── Pad bounding box ─────────────────────────────────────────────────────
    px = int(bw * pad_ratio)
    py = int(bh * pad_ratio)
    x1 = max(0, x1 - px);  y1 = max(0, y1 - py)
    x2 = min(W, x2 + px);  y2 = min(H, y2 + py)

    # ── Apply segmentation mask ──────────────────────────────────────────────
    img_arr  = np.array(pil_img.convert("RGB"), dtype=np.uint8)
    # Black-out everything outside the mask
    binary   = (mask_full > 127).astype(np.uint8)          # 0 / 1
    masked   = img_arr * binary[:, :, np.newaxis]           # background → 0

    # ── Crop to padded box ───────────────────────────────────────────────────
    crop_arr = masked[y1:y2, x1:x2]
    crop     = Image.fromarray(crop_arr)

    # ── Rotate portrait crops to landscape ───────────────────────────────────
    if crop.height > crop.width:
        crop = crop.rotate(90, expand=True)

    # ── Resize with black padding → IMAGE_SIZE × IMAGE_SIZE ─────────────────
    crop = resize_with_padding(crop, IMAGE_SIZE)

    # ── Denoising ────────────────────────────────────────────────────────────
    if denoise:
        arr = np.array(crop, dtype=np.uint8)
        arr = cv2.GaussianBlur(arr, (3, 3), sigmaX=0.8, sigmaY=0.8)
        arr = cv2.bilateralFilter(arr, d=5, sigmaColor=15, sigmaSpace=15)
        crop = Image.fromarray(arr)

    return crop


# ── Annotation helper ─────────────────────────────────────────────────────────

def draw_seg_annotation(
    annotated: np.ndarray,
    detection: dict,
    idx: int,
    score: float,
    pred: str,
) -> np.ndarray:
    """
    Draw a semi-transparent filled segmentation mask and a thin border,
    then label with crop ID and score.
    """
    color_bgr = (0, 0, 255) if pred == "ANOMALY" else (0, 255, 0)
    x1, y1, x2, y2 = detection["box"]
    mask = detection["mask"]   # H×W uint8

    # Semi-transparent mask fill
    overlay = annotated.copy()
    fill = np.zeros_like(annotated, dtype=np.uint8)
    fill[mask > 127] = color_bgr
    cv2.addWeighted(fill, 0.35, overlay, 0.65, 0, overlay)
    # Draw mask contour
    contours, _ = cv2.findContours(
        (mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, color_bgr, 2)

    # Label
    label = f"ID{idx} {score:.3f}"
    cv2.putText(
        overlay, label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color_bgr, 2, cv2.LINE_AA,
    )
    return overlay


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device

    print(f"[SinglePhoto] Loading EfficientAD ({args.model_size}) "
          f"from {args.ckpt_dir} ...")
    models = load_efficientad_model(
        ckpt_dir   = args.ckpt_dir,
        category   = args.category,
        model_size = args.model_size,
        device     = device,
    )

    print(f"[SinglePhoto] Loading YOLO-seg detector from {args.yolo_weights} ...")
    detector = SegmentationDetector(
        model_path = args.yolo_weights,
        conf       = args.conf,
        iou        = args.iou,
        device     = device,
    )

    pil_img = Image.open(args.photo_path).convert("RGB")
    print(f"[SinglePhoto] Input image: {pil_img.size} → {args.photo_path}")

    # ── Stage 1: YOLO-seg Detection ──────────────────────────────────────────
    t_yolo_start = time.perf_counter()
    detections   = detector.detect(pil_img)
    t_yolo_ms    = (time.perf_counter() - t_yolo_start) * 1000
    print(f"[SinglePhoto] Detected {len(detections)} chewing-gum piece(s) "
          f"[YOLO-seg: {t_yolo_ms:.1f} ms]")

    # ── Stage 2: Per-crop processing ─────────────────────────────────────────
    to_tensor = transforms.ToTensor()
    t_total_start = time.perf_counter()

    crop_results = []   # (detection, score, pred)
    csv_rows     = []

    for i, det in enumerate(detections):
        t_crop_start = time.perf_counter()

        crop_pil = get_processed_crop_pil(
            pil_img,
            det["box"],
            det["mask"],
            pad_ratio = args.pad_ratio,
            denoise   = True,
        )

        # Save clean masked crop
        crop_path = os.path.join(args.out_dir, f"cropped_{i:03d}.png")
        crop_pil.save(crop_path)

        # EfficientAD inference
        crop_t = to_tensor(crop_pil).unsqueeze(0).to(device)
        anomaly_map_t, score_t = run_inference(crop_t, models, ratio=0.1)
        anomaly_map_np = anomaly_map_t[0, 0].cpu().numpy()
        score = float(score_t[0])

        pred        = "ANOMALY" if score >= args.threshold else "NORMAL"
        t_crop_ms   = (time.perf_counter() - t_crop_start) * 1000

        crop_results.append((det, score, pred))

        print(f"  Crop {i:03d} | Box {det['box']} | conf {det['conf']:.2f} | "
              f"Score {score:.4f} → {pred}  [crop time: {t_crop_ms:.1f} ms]")

        # Save heatmap overlay
        overlay_pil  = anomaly_map_to_heatmap(anomaly_map_np, crop_pil)
        overlay_path = os.path.join(args.out_dir, f"overlay_{i:03d}.png")
        overlay_pil.save(overlay_path)

        x1, y1, x2, y2 = det["box"]
        csv_rows.append({
            "crop_id":     i,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "width":       x2 - x1,
            "height":      y2 - y1,
            "conf":        round(det["conf"], 4),
            "score":       round(score, 6),
            "prediction":  pred,
            "crop_time_ms": round(t_crop_ms, 2),
            "crop_path":   f"cropped_{i:03d}.png",
            "overlay_path": f"overlay_{i:03d}.png",
        })

    t_pipeline_total_ms = (time.perf_counter() - t_total_start) * 1000
    total_time_ms       = t_yolo_ms + t_pipeline_total_ms

    print(f"\n{'='*80}")
    print("  TIMING SUMMARY")
    print(f"{'='*80}")
    print(f"  YOLO-seg detection      : {t_yolo_ms:8.1f} ms")
    print(f"  EfficientAD all crops   : {t_pipeline_total_ms:8.1f} ms "
          f"({len(detections)} crops)")
    print(f"  TOTAL pipeline time     : {total_time_ms:8.1f} ms")
    print(f"{'='*80}")

    # ── Save annotated full image ─────────────────────────────────────────────
    annotated = np.array(pil_img.convert("RGB"))
    for i, (det, score, pred) in enumerate(crop_results):
        annotated = draw_seg_annotation(annotated, det, i, score, pred)

    Image.fromarray(annotated).save(
        os.path.join(args.out_dir, "full_image_annotated.png")
    )
    print("  full_image_annotated.png → saved with seg masks + scores")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.out_dir, "results.csv")
    fieldnames = [
        "crop_id", "x1", "y1", "x2", "y2", "width", "height",
        "conf", "score", "prediction", "crop_time_ms",
        "crop_path", "overlay_path",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow([
            "TOTAL", "", "", "", "", "", "", "",
            f"YOLO: {t_yolo_ms:.1f}ms + EAD: {t_pipeline_total_ms:.1f}ms",
            round(total_time_ms, 2), "", "", "",
        ])

    print(f"\n[SinglePhoto] Done! All files saved to: {args.out_dir}")
    print("   • full_image_annotated.png  ← Full photo with seg masks & scores")
    print("   • results.csv               ← All crops + times + TOTAL")
    print("   • cropped_*.png + overlay_*.png")


if __name__ == "__main__":
    main()
