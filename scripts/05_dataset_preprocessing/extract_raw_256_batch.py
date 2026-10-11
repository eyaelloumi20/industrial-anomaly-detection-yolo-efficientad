#!/usr/bin/env python3
"""
extract_raw_256_batch.py
------------------------
Apply the RAW mask extraction method (fillPoly direct) on every image
in an input folder and save crops at 256×256 (padding-aware, no distortion).

Usage:
    python extract_raw_256_batch.py \
        --input  ./images \
        --seg-model chewinggum_seg_v22.pt \
        --output ./output_raw_256 \
        [--conf 0.25] [--iou 0.45] [--device cpu]

Output structure:
    <output>/
        <image_stem>/
            det000_<class>_c0.XX_raw_256.png
            det001_<class>_c0.XX_raw_256.png
            ...
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────
# Supported image extensions
# ─────────────────────────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ─────────────────────────────────────────────────────────────
# Resize with padding (letterbox) — preserves ratio, transparent bg
# ─────────────────────────────────────────────────────────────

def resize_with_padding(img_rgba: np.ndarray, target_size: int) -> np.ndarray:
    h, w = img_rgba.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_size, target_size, 4), dtype=np.uint8)

    scale = target_size / max(h, w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(img_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_size, target_size, 4), dtype=np.uint8)
    pad_top  = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return canvas


# ─────────────────────────────────────────────────────────────
# Method RAW — direct fillPoly, no post-processing
# ─────────────────────────────────────────────────────────────

def method_raw(frame_bgr: np.ndarray, polygon: np.ndarray) -> np.ndarray | None:
    pts = polygon.astype(np.int32)
    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()

    H, W = frame_bgr.shape[:2]
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(W - 1, x_max), min(H - 1, y_max)

    crop_bgr = frame_bgr[y_min:y_max + 1, x_min:x_max + 1]
    if crop_bgr.size == 0:
        return None

    pts_s = pts - np.array([x_min, y_min])
    h_c   = y_max - y_min + 1
    w_c   = x_max - x_min + 1
    mask  = np.zeros((h_c, w_c), dtype=np.uint8)
    cv2.fillPoly(mask, [pts_s.reshape(-1, 1, 2)], 255)

    rgba = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask
    return rgba


# ─────────────────────────────────────────────────────────────
# Process a single image
# ─────────────────────────────────────────────────────────────

def process_image(img_path: Path, seg_model: YOLO,
                  out_dir: Path, conf: float, iou: float, device) -> int:
    """
    Run segmentation on one image, extract raw crops at 256×256.
    Returns the number of crops saved.
    """
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"  [WARN] Cannot read image: {img_path.name} — skipped")
        return 0

    # Sub-folder named after the image stem
    img_out_dir = out_dir / img_path.stem
    img_out_dir.mkdir(parents=True, exist_ok=True)

    seg_results = seg_model.predict(
        source=frame, conf=conf, iou=iou, device=device, verbose=False
    )
    seg_res = seg_results[0]

    if seg_res.masks is None or len(seg_res.masks.xy) == 0:
        print(f"  [WARN] No detection in {img_path.name}")
        return 0

    seg_polygons = seg_res.masks.xy
    seg_cls      = seg_res.boxes.cls.cpu().numpy().astype(int)
    seg_confs    = seg_res.boxes.conf.cpu().numpy()
    names        = seg_res.names

    saved = 0
    for idx, (polygon, cls_id, conf_val) in enumerate(
        zip(seg_polygons, seg_cls, seg_confs)
    ):
        if len(polygon) < 3:
            continue

        class_name = names[cls_id]
        rgba = method_raw(frame, polygon)
        if rgba is None:
            print(f"    [det{idx:03d}] empty crop — skipped")
            continue

        resized = resize_with_padding(rgba, 256)
        fname   = f"{img_path.stem}_det{idx:03d}_{class_name}_c{conf_val:.2f}_raw_256.png"
        cv2.imwrite(str(img_out_dir / fname), resized)
        saved += 1

    return saved


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Batch RAW 256×256 crop extraction with YOLO segmentation"
    )
    p.add_argument("--input",     required=True,
                   help="Folder containing input images")
    p.add_argument("--seg-model", required=True,
                   help="YOLO segmentation model (.pt)")
    p.add_argument("--output",    default="./output_raw_256",
                   help="Output folder (default: ./output_raw_256)")
    p.add_argument("--conf",      type=float, default=0.25,
                   help="Confidence threshold (default: 0.25)")
    p.add_argument("--iou",       type=float, default=0.45,
                   help="IoU threshold (default: 0.45)")
    p.add_argument("--device",    default="",
                   help="Device: '' auto, 'cpu', '0' GPU0")
    return p.parse_args()


def main():
    args = parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"[ERROR] Input folder not found: {input_dir}")
        sys.exit(1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect images
    images = sorted([
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ])
    if not images:
        print(f"[ERROR] No images found in {input_dir}")
        sys.exit(1)

    print(f"[INFO] Found {len(images)} image(s) in {input_dir}")
    print(f"[INFO] Loading seg model: {args.seg_model}")
    seg_model = YOLO(args.seg_model)

    device = args.device if args.device else None

    total_images = 0
    total_crops  = 0

    print("=" * 60)
    for img_path in images:
        print(f"[{total_images + 1}/{len(images)}] {img_path.name}")
        n = process_image(img_path, seg_model, out_dir,
                          conf=args.conf, iou=args.iou, device=device)
        print(f"  → {n} crop(s) saved")
        total_images += 1
        total_crops  += n

    print("=" * 60)
    print(f"[DONE] {total_images} image(s) processed, "
          f"{total_crops} crop(s) saved → {out_dir.resolve()}")


if __name__ == "__main__":
    main()