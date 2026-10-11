"""
generate_seg_labels.py
-----------------------
Auto-generate YOLO segmentation labels from your existing detection dataset
using YOLOv8 det model + SAM2.

Output structure (completely separate from your original dataset):
  /home/ahmed/.../project_ch_split_seg/
      images/
          train/  → symlinked or copied from original
          valid/
          test/
      labels/
          train/  → new polygon label .txt files
          valid/
          test/
      chewinggum_seg.yaml

Usage:
    python generate_seg_labels.py \
        --model best.pt \
        --dataset_root /path/to/project_ch_split \
        --output_root  /path/to/project_ch_split_seg \
        --conf 0.68

Requirements:
    pip install ultralytics opencv-python-headless numpy
"""

import argparse
import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO, SAM


# ── Helpers ───────────────────────────────────────────────────────────────────

def mask_to_polygon(binary_mask: np.ndarray, epsilon_factor: float = 0.005):
    """Convert binary mask → simplified normalized polygon, or None."""
    h, w = binary_mask.shape
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    epsilon  = epsilon_factor * cv2.arcLength(contour, True)
    approx   = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) < 3:
        return None

    points = approx.reshape(-1, 2).astype(float)
    points[:, 0] /= w
    points[:, 1] /= h
    points = np.clip(points, 0.0, 1.0)
    return points.flatten().tolist()


def symlink_images(src_dir: Path, dst_dir: Path):
    """Create symlinks in dst_dir pointing to images in src_dir."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    count = 0
    for img in sorted(src_dir.iterdir()):
        if img.suffix.lower() in image_extensions:
            link = dst_dir / img.name
            if not link.exists():
                link.symlink_to(img.resolve())
            count += 1
    return count


def write_yaml(output_root: Path, nc: int, names: list):
    """Write chewinggum_seg.yaml into the new dataset root."""
    yaml_path = output_root / "chewinggum_seg.yaml"
    lines = [
        f"path: {output_root.resolve()}",
        "train: images/train",
        "val:   images/valid",
        "test:  images/test",
        f"nc: {nc}",
        "names:",
    ]
    for i, name in enumerate(names):
        lines.append(f"  {i}: {name}")
    yaml_path.write_text("\n".join(lines) + "\n")
    return yaml_path


# ── Core processing ───────────────────────────────────────────────────────────

def process_split(
    det_model,
    sam_model_obj,
    images_src: Path,
    images_dst: Path,
    labels_dst: Path,
    conf: float,
    iou: float,
    min_area: int,
):
    # Symlink images so YOLO can find them
    n_imgs = symlink_images(images_src, images_dst)
    labels_dst.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted([
        p for p in images_src.iterdir()
        if p.suffix.lower() in image_extensions
    ])

    if not image_paths:
        print(f"  ⚠ No images found in {images_src}")
        return 0

    processed, skipped = 0, 0

    for i, img_path in enumerate(image_paths):
        frame = cv2.imread(str(img_path))
        if frame is None:
            skipped += 1
            continue

        h, w = frame.shape[:2]
        out_label = labels_dst / (img_path.stem + ".txt")

        # ── Detect ────────────────────────────────────────────────────────
        det_res = det_model(frame, conf=conf, iou=iou, verbose=False)
        boxes   = det_res[0].boxes

        if boxes is None or len(boxes) == 0:
            out_label.write_text("")
            skipped += 1
            continue

        xyxy      = boxes.xyxy.cpu().numpy().astype(int)
        class_ids = boxes.cls.cpu().numpy().astype(int)

        # ── SAM2 masks ────────────────────────────────────────────────────
        sam_res = sam_model_obj(frame, bboxes=xyxy, verbose=False)
        if sam_res[0].masks is None:
            out_label.write_text("")
            skipped += 1
            continue

        masks_data = sam_res[0].masks.data.cpu().numpy()

        # ── Masks → YOLO polygon labels ───────────────────────────────────
        label_lines = []
        for j in range(len(masks_data)):
            raw_mask = masks_data[j]
            if raw_mask.shape != (h, w):
                raw_mask = cv2.resize(
                    raw_mask.astype(np.float32), (w, h),
                    interpolation=cv2.INTER_LINEAR
                )
            binary = (raw_mask > 0.5).astype(np.uint8)
            if binary.sum() < min_area:
                continue
            polygon = mask_to_polygon(binary)
            if polygon is None:
                continue
            class_id  = int(class_ids[j]) if j < len(class_ids) else 0
            coords_str = " ".join(f"{v:.6f}" for v in polygon)
            label_lines.append(f"{class_id} {coords_str}")

        out_label.write_text("\n".join(label_lines))
        processed += 1

        if (i + 1) % 100 == 0:
            print(f"    [{i+1}/{len(image_paths)}] {processed} labeled, {skipped} skipped")

    print(f"  ✅ {processed} labeled  |  {skipped} empty/skipped  |  images symlinked: {n_imgs}")
    return processed


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_labels(
    model_path:   str,
    dataset_root: str,
    output_root:  str,
    conf:         float = 0.68,
    iou:          float = 0.45,
    min_area:     int   = 200,
    sam_model:    str   = "sam2.1_b.pt",
    splits:       list  = None,
    nc:           int   = 1,
    names:        list  = None,
):
    if splits is None:
        splits = ["train", "valid", "test"]
    if names is None:
        names = ["chewinggum"]

    dataset_root = Path(dataset_root)
    output_root  = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading detection model : {model_path}")
    det_model = YOLO(model_path)

    print(f"Loading SAM2            : {sam_model}  (auto-downloads if needed)")
    sam_model_obj = SAM(sam_model)

    total = 0
    for split in splits:
        images_src = dataset_root / "images" / split
        images_dst = output_root  / "images" / split
        labels_dst = output_root  / "labels" / split

        if not images_src.exists():
            print(f"\n[SKIP] {images_src} does not exist")
            continue

        print(f"\n── Split: {split} ──────────────────────────────────────────")
        total += process_split(
            det_model, sam_model_obj,
            images_src, images_dst, labels_dst,
            conf, iou, min_area,
        )

    # Write yaml
    yaml_path = write_yaml(output_root, nc, names)

    print(f"""
╔══════════════════════════════════════════════════════════╗
║  Label generation complete!                              ║
║                                                          ║
║  New seg dataset : {str(output_root):<38}║
║  Total labeled   : {total:<38}║
║  YAML            : {str(yaml_path):<38}║
╚══════════════════════════════════════════════════════════╝

Next — train your seg model:

  yolo task=segment mode=train \\
       model=yolov8n-seg.pt \\
       data={yaml_path} \\
       epochs=50 imgsz=640 batch=16 device=0 \\
       patience=15 \\
       project=./yolo_runs2 \\
       name=chewinggum_seg
""")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Auto-generate YOLO seg labels (det model + SAM2) into a NEW folder"
    )
    parser.add_argument("--model",        required=True,
                        help="Path to YOLOv8 detection best.pt")
    parser.add_argument("--dataset_root", required=True,
                        help="Original dataset root (contains images/train, images/valid …)")
    parser.add_argument("--output_root",  required=True,
                        help="NEW folder to write seg dataset into (images + labels + yaml)")
    parser.add_argument("--conf",         type=float, default=0.68)
    parser.add_argument("--iou",          type=float, default=0.45)
    parser.add_argument("--min_area",     type=int,   default=200,
                        help="Min mask area in pixels² to keep (default: 200)")
    parser.add_argument("--sam_model",    default="sam2.1_b.pt",
                        choices=["sam2.1_t.pt", "sam2.1_b.pt", "sam2.1_l.pt"],
                        help="SAM2 variant: t=tiny/fast, b=balanced, l=large/accurate")
    parser.add_argument("--splits",       nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--nc",           type=int,  default=1)
    parser.add_argument("--names",        nargs="+", default=["chewinggum"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_labels(
        model_path   = args.model,
        dataset_root = args.dataset_root,
        output_root  = args.output_root,
        conf         = args.conf,
        iou          = args.iou,
        min_area     = args.min_area,
        sam_model    = args.sam_model,
        splits       = args.splits,
        nc           = args.nc,
        names        = args.names,
    )
