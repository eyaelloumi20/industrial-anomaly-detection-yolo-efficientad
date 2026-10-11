#!/usr/bin/env bash
set -euo pipefail

cd /home/ahmed/two-stage-efficientad-inspection

SRC_DATASET="./dataset_multi"
AUG_DATASET="./dataset_aug_fast"

# Frozen teacher from the original trained model
TEACHER="./ckpt_lowres_192/best_teacher.pth"

# Two independent experimental checkpoints
CKPT_BASELINE="./ckpt_baseline_256_aug_fast"
CKPT_LOWRES="./ckpt_lowres_192_aug_fast"

LOG_BASELINE="./baseline256_aug_fast_train.log"
LOG_LOWRES="./lowres192_aug_fast_train.log"

ITERATIONS=70000
AUG_PER_IMAGE=1
SEED=2026

echo "======================================================================"
echo "FAST AUGMENTATION EXPERIMENT — BASELINE 256 + LOWRES192"
echo "======================================================================"
echo "Source dataset    : $SRC_DATASET"
echo "Augmented dataset : $AUG_DATASET"
echo "Baseline ckpt     : $CKPT_BASELINE"
echo "LowRes192 ckpt    : $CKPT_LOWRES"
echo "Iterations/model  : $ITERATIONS"
echo "Aug/original      : $AUG_PER_IMAGE"
echo
echo "Only NORMAL TRAIN images are augmented."
echo "TEST and VALIDATION remain unchanged."
echo "Original dataset/checkpoints are NOT modified."
echo "======================================================================"

# ----------------------------------------------------------------------
# 0) Checks
# ----------------------------------------------------------------------
test -f "$SRC_DATASET/split_csv/1cls.csv" || {
  echo "ERROR: $SRC_DATASET/split_csv/1cls.csv not found"
  exit 1
}

test -f "$TEACHER" || {
  echo "ERROR: teacher not found: $TEACHER"
  exit 1
}

test -f "./retrain_lowres.py" || {
  echo "ERROR: retrain_lowres.py not found"
  exit 1
}

# ----------------------------------------------------------------------
# 1) Build augmented dataset ONCE
# ----------------------------------------------------------------------
echo
echo "[1/5] Building augmented dataset..."

rm -rf "$AUG_DATASET"

python - <<'PY'
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
from collections import Counter
import csv, os, random, shutil
import numpy as np

SRC = Path("dataset_multi")
DST = Path("dataset_aug_fast")
CSV_SRC = SRC / "split_csv" / "1cls.csv"
CSV_DST = DST / "split_csv" / "1cls.csv"

AUG_PER_IMAGE = 1
SEED = 2026
rng = random.Random(SEED)
np_rng = np.random.default_rng(SEED)

def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)

print("Cloning original dataset...")
shutil.copytree(SRC, DST, copy_function=link_or_copy)

with CSV_SRC.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    fieldnames = reader.fieldnames

if not rows or not fieldnames:
    raise RuntimeError("Dataset CSV is empty.")

path_col = None
for candidate in ("image_path", "image", "path", "filepath", "file"):
    if candidate in fieldnames:
        path_col = candidate
        break
if path_col is None:
    raise RuntimeError(f"Cannot find image-path column. Columns: {fieldnames}")

def is_normal(v):
    return str(v).strip().lower() in {"0", "normal", "good", "ok"}

train_rows = [
    r for r in rows
    if str(r.get("split", "")).strip().lower() == "train"
    and is_normal(r.get("label", ""))
]

print(f"Original CSV rows     : {len(rows)}")
print(f"Normal training images: {len(train_rows)}")

AUG_DIR = DST / "chewinggum" / "Data" / "Images" / "Normal" / "aug_fast"
AUG_DIR.mkdir(parents=True, exist_ok=True)

def gamma_adjust(img, gamma):
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.clip(arr, 0, 1) ** gamma
    return Image.fromarray(
        np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    )

def add_sensor_noise(img, sigma):
    arr = np.asarray(img).astype(np.float32)
    noise = np_rng.normal(0.0, sigma, arr.shape)
    return Image.fromarray(
        np.clip(arr + noise, 0, 255).astype(np.uint8)
    )

new_rows = []
created = 0

for idx, row in enumerate(train_rows):
    rel = Path(row[path_col])
    src_img = SRC / rel

    if not src_img.exists():
        raise FileNotFoundError(src_img)

    with Image.open(src_img) as im0:
        base = im0.convert("RGB")

        for k in range(AUG_PER_IMAGE):
            im = base.copy()

            # Acquisition-oriented augmentation:
            # phone exposure / room lighting / mild blur / sensor variation.
            im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.75, 1.25))
            im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.82, 1.18))
            im = ImageEnhance.Color(im).enhance(rng.uniform(0.90, 1.10))
            im = gamma_adjust(im, rng.uniform(0.85, 1.15))

            if rng.random() < 0.35:
                im = im.filter(
                    ImageFilter.GaussianBlur(radius=rng.uniform(0.25, 0.80))
                )

            if rng.random() < 0.35:
                im = add_sensor_noise(im, rng.uniform(1.0, 4.0))

            # Keep raw storage uniform at 256x256.
            im = im.resize((256, 256), Image.Resampling.LANCZOS)

            out_name = f"aug_{idx:06d}_{k}_{src_img.stem}.png"
            out_path = AUG_DIR / out_name
            im.save(out_path, compress_level=2)

            nr = dict(row)
            nr[path_col] = str(out_path.relative_to(DST))
            nr["split"] = "train"
            new_rows.append(nr)

            created += 1
            if created % 1000 == 0:
                print(f"  created {created}")

all_rows = rows + new_rows

with CSV_DST.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)

print()
print("======================================================================")
print("AUGMENTED DATASET READY")
print("======================================================================")
print(f"Added augmented normal images: {created}")
print(f"Final CSV rows              : {len(all_rows)}")

for split in ("train", "test", "validation"):
    c = Counter()
    for r in all_rows:
        if str(r.get("split", "")).strip().lower() == split:
            c[str(r.get("label", "")).strip().lower()] += 1
    print(f"{split:12s}: {sum(c.values()):5d}  {dict(c)}")

print(f"CSV: {CSV_DST}")
PY

# ----------------------------------------------------------------------
# 2) Check image sizes
# ----------------------------------------------------------------------
echo
echo "[2/5] Checking stored image dimensions..."

python - <<'PY'
from pathlib import Path
from PIL import Image
from collections import Counter

root = Path("dataset_aug_fast/chewinggum/Data/Images")
sizes = Counter()

for p in root.rglob("*"):
    if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        try:
            with Image.open(p) as im:
                sizes[im.size] += 1
        except Exception:
            pass

print("IMAGE SIZES")
for size, n in sorted(sizes.items()):
    print(size, n)
PY

# ----------------------------------------------------------------------
# 3) Train BASELINE EfficientAD at 256x256
# ----------------------------------------------------------------------
echo
echo "[3/5] Training BASELINE EfficientAD at 256x256..."
echo "Checkpoint: $CKPT_BASELINE"

rm -rf "$CKPT_BASELINE"

python retrain_lowres.py \
  --dataset_path "$AUG_DATASET" \
  --ckpt_dir "$CKPT_BASELINE" \
  --teacher_path "$TEACHER" \
  --category chewinggum \
  --model_size S \
  --image_size 256 \
  --iterations "$ITERATIONS" \
  --batch_size 1 \
  --lr 1e-4 \
  --print_freq 500 \
  --device cuda:0 \
  --no_imagenet 2>&1 | tee "$LOG_BASELINE"

cp -f "$TEACHER" "$CKPT_BASELINE/best_teacher.pth"

echo
echo "Baseline 256 training finished."
echo

# ----------------------------------------------------------------------
# 4) Train LOWRES192 on exactly the same augmented dataset
# ----------------------------------------------------------------------
echo
echo "[4/5] Training LOWRES192 EfficientAD at 192x192..."
echo "Checkpoint: $CKPT_LOWRES"

rm -rf "$CKPT_LOWRES"

python retrain_lowres.py \
  --dataset_path "$AUG_DATASET" \
  --ckpt_dir "$CKPT_LOWRES" \
  --teacher_path "$TEACHER" \
  --category chewinggum \
  --model_size S \
  --image_size 192 \
  --iterations "$ITERATIONS" \
  --batch_size 1 \
  --lr 1e-4 \
  --print_freq 500 \
  --device cuda:0 \
  --no_imagenet 2>&1 | tee "$LOG_LOWRES"

cp -f "$TEACHER" "$CKPT_LOWRES/best_teacher.pth"

# ----------------------------------------------------------------------
# 5) Summary
# ----------------------------------------------------------------------
echo
echo "[5/5] Finished."
echo
echo "======================================================================"
echo "AUGMENTED TRAINING COMPLETE"
echo "======================================================================"
echo "Dataset:"
echo "  $AUG_DATASET"
echo
echo "BASELINE 256:"
echo "  checkpoint = $CKPT_BASELINE"
echo "  log        = $LOG_BASELINE"
echo
echo "LOWRES192:"
echo "  checkpoint = $CKPT_LOWRES"
echo "  log        = $LOG_LOWRES"
echo
echo "IMPORTANT:"
echo "Do NOT replace the original checkpoints yet."
echo "First calibrate and evaluate both augmented models on the SAME videos."
echo
echo "Best-validation checkpoints should already be exposed as:"
echo "  chewinggum_student.pth"
echo "  chewinggum_autoencoder.pth"
echo "  chewinggum_quantiles.npy"
echo "======================================================================"
