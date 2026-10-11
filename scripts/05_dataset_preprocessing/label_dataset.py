"""
label_dataset.py  (VS Code / local version)
--------------------------------------------
Automatically creates YOLO segmentation labels for all images in your
images/ folder using YOLOv8's segmentation model.

Steps:
  1. Load images from images/
  2. Run YOLOv8 segmentation inference on each image
  3. Write YOLO polygon label files to labels/
  4. Create dataset.yaml
  5. Zip the labels folder
"""

import os
import shutil
from pathlib import Path

# ========================== INSTALL CHECK ==========================
try:
    from ultralytics import YOLO
except ImportError:
    import subprocess, sys
    print("Installing ultralytics...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
    from ultralytics import YOLO

# ========================== CONFIGURATION ==========================
BASE_DIR    = r"/home/ahmed/Downloads/gum_dataset/gum_dataset"   # ← Update this path
CLASS_NAME  = "chewinggum"

# YOLOv8 segmentation model to use for inference.
# Options (auto-downloaded on first run):
#   "yolov8n-seg.pt"  ← nano,  fastest, least accurate
#   "yolov8s-seg.pt"  ← small
#   "yolov8m-seg.pt"  ← medium
#   "yolov8l-seg.pt"  ← large
#   "yolov8x-seg.pt"  ← extra-large, most accurate
# OR point to your own fine-tuned model:
#   MODEL_PATH = r"C:\path\to\your_model.pt"
MODEL_PATH  = "yolov8n-seg.pt"

# Confidence threshold — detections below this are ignored
CONF_THRESH = 0.25

# Class index to keep from the model's output.
# If using a pretrained COCO model, set to None to keep ALL classes.
# Set to an integer (e.g. 0) to keep only that class.
KEEP_CLASS  = None   # e.g. 0 for "person" in COCO

# ========================== SETUP DIRS ==========================
IMAGES_DIR = os.path.join(BASE_DIR, "images")
LABELS_DIR = os.path.join(BASE_DIR, "labels")

os.makedirs(LABELS_DIR, exist_ok=True)

if not os.path.isdir(IMAGES_DIR):
    raise FileNotFoundError(f"Images directory not found: {IMAGES_DIR}\nPlease update BASE_DIR.")

# ========================== LOAD MODEL ==========================
print(f"Loading model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print("Model loaded.\n")

# ========================== PROCESS IMAGES ==========================
image_paths = sorted(
    p for p in Path(IMAGES_DIR).glob("*.*")
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
)

if not image_paths:
    raise FileNotFoundError(f"No images found in: {IMAGES_DIR}")

print(f"Found {len(image_paths)} image(s). Starting labeling...\n")

labeled_count  = 0
skipped_count  = 0

for img_path in image_paths:
    results = model(str(img_path), conf=CONF_THRESH, verbose=False)
    result  = results[0]

    label_lines = []

    if result.masks is not None:
        for i, (mask_xy, cls_id) in enumerate(
            zip(result.masks.xy, result.boxes.cls.int().tolist())
        ):
            # Filter by class if requested
            if KEEP_CLASS is not None and cls_id != KEEP_CLASS:
                continue

            # Normalize polygon coordinates to [0, 1]
            h, w = result.orig_shape
            polygon = mask_xy  # shape (N, 2), absolute pixel coords

            if len(polygon) < 3:
                continue  # skip degenerate polygons

            # Build YOLO segmentation line: class_id x1 y1 x2 y2 ...
            coords = " ".join(
                f"{x / w:.6f} {y / h:.6f}"
                for x, y in polygon
            )
            label_lines.append(f"0 {coords}")   # always write class 0

    # Write label file (empty file = no detections, which is valid YOLO)
    label_path = Path(LABELS_DIR) / (img_path.stem + ".txt")
    with open(label_path, "w") as f:
        f.write("\n".join(label_lines))

    if label_lines:
        print(f"  ✅ {img_path.name}  →  {len(label_lines)} object(s)")
        labeled_count += 1
    else:
        print(f"  ⚠️  {img_path.name}  →  no detections")
        skipped_count += 1

# ========================== SUMMARY ==========================
print(f"\n{'='*50}")
print(f"Total images     : {len(image_paths)}")
print(f"With labels      : {labeled_count}")
print(f"No detections    : {skipped_count}")

# ========================== CREATE dataset.yaml ==========================
dataset_yaml = f"""\
path: {BASE_DIR}

train: images
val: images

names:
  0: {CLASS_NAME}
"""

yaml_path = os.path.join(BASE_DIR, "dataset.yaml")
with open(yaml_path, "w") as f:
    f.write(dataset_yaml)

print(f"\n✅ dataset.yaml created at: {yaml_path}")
print(open(yaml_path).read())

# ========================== ZIP LABELS ==========================
zip_output_path = os.path.join(BASE_DIR, "labels")
shutil.make_archive(zip_output_path, "zip", LABELS_DIR)
final_zip = f"{zip_output_path}.zip"

print(f"✅ Labels zipped: {final_zip}")
print(f"\nDone! Your labels are in:\n  {LABELS_DIR}\n  {final_zip}")