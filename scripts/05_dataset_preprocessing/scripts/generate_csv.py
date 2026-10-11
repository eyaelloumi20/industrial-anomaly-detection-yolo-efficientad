"""
generate_csv.py  –  CSV generator for EfficientAD retraining
=============================================================

TWO-SPLIT layout
----------------
  train      – normal images only (70%)     → EfficientAD student/AE training
  test       – normal images only (15%)     → quantile calibration
  validation – normal (15%) + ALL anomaly   → AUROC monitoring + pipeline eval

Expected layout:

    <DATASET_ROOT>/
    ├── split_csv/
    │   └── 1cls.csv
    └── <CATEGORY>/
        └── Data/
            ├── Images/
            │   ├── Normal/
            │   └── Anomaly/     (optional – can be empty or missing)
            └── Masks/
                └── Anomaly/     (optional – masks matched by filename)

Usage:
    Edit DATASET_ROOT and CATEGORY below, then run:
        python scripts.generate_csv.py
"""

import csv
import random
from pathlib import Path

# ── USER SETTINGS ─────────────────────────────────────────────────────────────
DATASET_ROOT = Path("/home/ahmed/two-stage-efficientad-inspection/dataset_new")
CATEGORY     = "chewinggum"
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_RATIO = 0.70   # of normal images → train
TEST_RATIO  = 0.15   # of normal images → test (quantile calibration)
                     # remaining 0.15   → validation (alongside ALL anomalies)
SEED        = 42
IMG_EXTS    = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

# ── Paths ─────────────────────────────────────────────────────────────────────
category_dir = DATASET_ROOT / CATEGORY
normal_dir   = category_dir / "Data" / "Images" / "Normal"
anomaly_dir  = category_dir / "Data" / "Images" / "Anomaly"
mask_dir     = category_dir / "Data" / "Masks"  / "Anomaly"
out_csv      = DATASET_ROOT / "split_csv" / "1cls.csv"

# ── Sanity checks ─────────────────────────────────────────────────────────────
assert category_dir.is_dir(), f"Category folder not found: {category_dir}"
assert normal_dir.is_dir(),   f"Normal images folder not found: {normal_dir}"
assert TRAIN_RATIO + TEST_RATIO < 1.0, \
    "TRAIN_RATIO + TEST_RATIO must be < 1.0 to leave room for validation"

# ── Collect images ────────────────────────────────────────────────────────────
img_normal  = sorted(p for p in normal_dir.glob("*")
                     if p.suffix.lower() in IMG_EXTS)
img_anomaly = sorted(p for p in anomaly_dir.glob("*")
                     if p.suffix.lower() in IMG_EXTS) \
              if anomaly_dir.is_dir() else []

print(f"Dataset root   : {DATASET_ROOT}")
print(f"Category       : {CATEGORY}")
print(f"Normal images  : {len(img_normal)}")
print(f"Anomaly images : {len(img_anomaly)}")

if not img_normal:
    raise RuntimeError(f"No images found in {normal_dir}")

# ── Split normal images: train / test / val ───────────────────────────────────
random.seed(SEED)
shuffled = img_normal.copy()
random.shuffle(shuffled)

n_total = len(shuffled)
n_train = int(n_total * TRAIN_RATIO)
n_test  = int(n_total * TEST_RATIO)

train_normal = shuffled[:n_train]
test_normal  = shuffled[n_train : n_train + n_test]
val_normal   = shuffled[n_train + n_test:]   # remaining ~15%

print(f"\nNormal split   : train={len(train_normal)}  "
      f"test={len(test_normal)}  val(normal)={len(val_normal)}")
print(f"Anomaly        : all {len(img_anomaly)} → validation only")

# ── Build CSV rows ────────────────────────────────────────────────────────────
rows = []

# train — normal only (EfficientAD trains exclusively on these)
for img in train_normal:
    rows.append([CATEGORY, "train", "normal",
                 img.relative_to(DATASET_ROOT).as_posix(), ""])

# test — normal only (quantile calibration, no anomalies ever here)
for img in test_normal:
    rows.append([CATEGORY, "test", "normal",
                 img.relative_to(DATASET_ROOT).as_posix(), ""])

# validation — normal remainder + ALL anomalies (AUROC computed here)
for img in val_normal:
    rows.append([CATEGORY, "validation", "normal",
                 img.relative_to(DATASET_ROOT).as_posix(), ""])

for img in img_anomaly:
    mask     = mask_dir / img.name
    mask_rel = mask.relative_to(DATASET_ROOT).as_posix() if mask.exists() else ""
    rows.append([CATEGORY, "validation", "anomaly",
                 img.relative_to(DATASET_ROOT).as_posix(), mask_rel])

# ── Write CSV ─────────────────────────────────────────────────────────────────
out_csv.parent.mkdir(parents=True, exist_ok=True)
with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["category", "split", "label", "image_path", "mask_path"])
    writer.writerows(rows)

print(f"\nCSV written → {out_csv}")
print(f"  train      (normal only) : {len(train_normal)}  ← EfficientAD training")
print(f"  test       (normal only) : {len(test_normal)}   ← quantile calibration")
print(f"  validation (normal)      : {len(val_normal)}")
print(f"  validation (anomaly)     : {len(img_anomaly)}   ← AUROC monitoring")
print(f"  total rows               : {len(rows)}")

if len(img_anomaly) == 0:
    print(f"\n  NOTE: No anomaly images found in {anomaly_dir}")
    print(f"        AUROC will be 0.0 during training (expected).")
    print(f"        Add defective gum images to that folder and rerun")
    print(f"        this script when you have them.")