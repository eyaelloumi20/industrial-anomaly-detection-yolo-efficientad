"""
extract_gum_pieces.py
---------------------
Extracts individual chewinggum pieces from a video using a YOLOv8 segmentation model.
Saves each piece as a transparent PNG (RGBA) with contrast + sharpness enhancement.

Usage:
    python extract_gum_pieces.py --model /home/ahmed/two-stage-efficientad-inspection/sam_yolo_pipeline/runs/segment/runs/segment/chewinggum_seg_v1/weights/best.pt --video video_ch4.mp4

Requirements:
    pip install ultralytics opencv-python-headless numpy Pillow tqdm
"""

import argparse
import os
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from ultralytics import YOLO

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def _box_iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1);  iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2);  iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0


def enhance_piece(rgba_crop: Image.Image, contrast: float, sharpness: float) -> Image.Image:
    """
    Enhance contrast + sharpness on RGB channels only.
    Transparent background is fully preserved (returns RGBA).
    """
    r, g, b, a = rgba_crop.split()
    rgb = Image.merge("RGB", (r, g, b))
    if contrast != 1.0:
        rgb = ImageEnhance.Contrast(rgb).enhance(contrast)
    if sharpness != 1.0:
        rgb = ImageEnhance.Sharpness(rgb).enhance(sharpness)
    r2, g2, b2 = rgb.split()
    return Image.merge("RGBA", (r2, g2, b2, a))


def extract_gum_pieces(
    model_path:   str,
    video_path:   str,
    output_dir:   str   = "extracted_gum_pieces",
    conf:         float = 0.25,
    iou:          float = 0.45,
    frame_skip:   int   = 1,
    min_area:     int   = 500,
    padding:      int   = 10,       # slightly more padding for cleaner look
    dedup_iou:    float = 0.85,
    dedup_window: int   = 10,
    run_tag:      str   = "",
    contrast:     float = 1.8,      # 1.0 = no change, 1.8 = strong boost
    sharpness:    float = 2.0,      # 1.0 = no change, 2.0 = sharper
    save_transparent: bool = False, # also save the raw RGBA PNG if needed
):
    os.makedirs(output_dir, exist_ok=True)

    model = YOLO(model_path)
    cap   = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video : {total_frames} frames @ {fps:.1f} fps")
    print(f"Output: {output_dir}/")

    recent_boxes: dict[int, deque] = {}
    frame_idx     = 0
    piece_idx     = 0
    skipped_dedup = 0

    frames_to_process = (total_frames + frame_skip - 1) // frame_skip
    pbar = tqdm(total=frames_to_process, unit="frame") if HAS_TQDM else None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        results = model(frame, conf=0.7, iou=iou, verbose=False)
        result  = results[0]

        if result.masks is None:
            frame_idx += 1
            if pbar: pbar.update(1)
            continue

        h, w = frame.shape[:2]
        frame_rgba = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")

        confs     = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        names     = result.names
        polygons  = result.masks.xy  # polygon contours at full frame resolution

        n_det = len(polygons)

        for det_idx in range(n_det):
            poly       = polygons[det_idx]
            conf_score = float(confs[det_idx])
            cls_id     = int(class_ids[det_idx])
            cls_name   = names.get(cls_id, f"class{cls_id}")

            if len(poly) < 3:
                continue

            # Rasterise polygon at full resolution
            pts         = poly.astype(np.int32).reshape((-1, 1, 2))
            binary_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(binary_mask, [pts], 1)

            area = int(binary_mask.sum())
            if area < min_area:
                continue

            # Tight bounding box + padding
            ys, xs = np.where(binary_mask)
            x_min  = max(int(xs.min()) - padding, 0)
            y_min  = max(int(ys.min()) - padding, 0)
            x_max  = min(int(xs.max()) + padding, w - 1)
            y_max  = min(int(ys.max()) + padding, h - 1)
            box    = (x_min, y_min, x_max, y_max)

            # Deduplication
            if cls_id not in recent_boxes:
                recent_boxes[cls_id] = deque(maxlen=dedup_window)
            is_duplicate = any(_box_iou(box, prev) >= dedup_iou
                               for prev in recent_boxes[cls_id])
            recent_boxes[cls_id].append(box)
            if is_duplicate:
                skipped_dedup += 1
                continue

            # Build RGBA crop
            alpha_channel = Image.fromarray((binary_mask * 255).astype(np.uint8), mode="L")
            r, g, b, _    = frame_rgba.split()
            piece_rgba    = Image.merge("RGBA", (r, g, b, alpha_channel))
            cropped_rgba  = piece_rgba.crop((x_min, y_min, x_max + 1, y_max + 1))

            # Enhance: contrast + sharpness (transparency preserved)
            final_img = enhance_piece(cropped_rgba, contrast=contrast, sharpness=sharpness)

            tag_part = f"_{run_tag}" if run_tag else ""
            base_name = (
                f"frame{frame_idx:06d}"
                f"_det{det_idx:02d}"
                f"_{cls_name}"
                f"_conf{conf_score:.2f}"
                f"_area{area}"
                f"{tag_part}"
            )

            # Resize to fixed square (EfficientAD expects uniform input size)
            final_img = final_img.resize((256, 256), Image.LANCZOS)

            # Save enhanced RGBA (transparent background)
            final_img.save(os.path.join(output_dir, base_name + ".png"), format="PNG")

            # Optionally also save raw transparent version (also resized)
            if save_transparent:
                cropped_rgba.resize((256, 256), Image.LANCZOS).save(
                    os.path.join(output_dir, base_name + "_transparent.png"), format="PNG"
                )

            piece_idx += 1

        if pbar:
            pbar.set_postfix(saved=piece_idx, skipped=skipped_dedup)
            pbar.update(1)
        elif frame_idx % 100 == 0:
            print(f"  Frame {frame_idx}/{total_frames} — saved {piece_idx}, "
                  f"dedup-skipped {skipped_dedup}")

        frame_idx += 1

    cap.release()
    if pbar:
        pbar.close()

    print(f"\nDone!")
    print(f"  Extracted : {piece_idx} unique gum pieces → '{output_dir}/'")
    print(f"  Dedup skip: {skipped_dedup} duplicate detections suppressed")


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract chewinggum pieces from video using YOLOv8-seg"
    )
    p.add_argument("--model",             required=True)
    p.add_argument("--video",             required=True)
    p.add_argument("--output_dir",        default="extracted_gum_pieces")
    p.add_argument("--conf",              type=float, default=0.25)
    p.add_argument("--iou",               type=float, default=0.45)
    p.add_argument("--frame_skip",        type=int,   default=1)
    p.add_argument("--min_area",          type=int,   default=500)
    p.add_argument("--padding",           type=int,   default=10)
    p.add_argument("--dedup_iou",         type=float, default=0.85)
    p.add_argument("--dedup_window",      type=int,   default=10)
    p.add_argument("--run_tag",           type=str,   default="")
    p.add_argument("--contrast",          type=float, default=1.8,
                   help="Contrast enhancement factor (1.0=none, 1.8=strong). Default: 1.8")
    p.add_argument("--sharpness",         type=float, default=2.0,
                   help="Sharpness enhancement factor (1.0=none, 2.0=strong). Default: 2.0")
    p.add_argument("--save_transparent",  action="store_true",
                   help="Also save a raw transparent PNG alongside the enhanced version")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_gum_pieces(
        model_path        = args.model,
        video_path        = args.video,
        output_dir        = args.output_dir,
        conf              = args.conf,
        iou               = args.iou,
        frame_skip        = args.frame_skip,
        min_area          = args.min_area,
        padding           = args.padding,
        dedup_iou         = args.dedup_iou,
        dedup_window      = args.dedup_window,
        run_tag           = args.run_tag,
        contrast          = args.contrast,
        sharpness         = args.sharpness,
        save_transparent  = args.save_transparent,
    )