#!/usr/bin/env bash
set -euo pipefail

cd /home/ahmed/two-stage-efficientad-inspection

SRC_DATASET="./dataset_multi"
AUG_DATASET="./dataset_aug_fast"

# Frozen teacher from the original trained model
TEACHER="./ckpt_lowres_192/best_teacher.pth"

# Two independent experimental checkpoints
CKPT_BASELINE="./ckpt_baseline_256_aug_70k"
CKPT_LOWRES="./ckpt_lowres_192_aug_70k"

LOG_BASELINE="./baseline256_aug_70k_train.log"
LOG_LOWRES="./lowres192_aug_70k_train.log"

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
# 1) Reuse existing augmented dataset
# ----------------------------------------------------------------------
echo
echo "[1/5] Reusing existing augmented dataset..."

test -f "$AUG_DATASET/split_csv/1cls.csv" || {
  echo "ERROR: augmented CSV missing: $AUG_DATASET/split_csv/1cls.csv"
  exit 1
}

AUG_COUNT=$(find "$AUG_DATASET" -type f -name 'aug_*.png' | wc -l)

echo "Augmented dataset : $AUG_DATASET"
echo "Augmented images  : $AUG_COUNT"

if [ "$AUG_COUNT" -ne 6751 ]; then
    echo "ERROR: expected 6751 augmented images, found $AUG_COUNT"
    exit 1
fi

python - <<'PYVERIFY'
import pandas as pd

df = pd.read_csv("dataset_aug_fast/split_csv/1cls.csv")

print("CSV rows:", len(df))
print(df["split"].value_counts().to_string())

assert len(df) == 18230, f"Expected 18230 rows, got {len(df)}"
assert (df["split"] == "train").sum() == 13502
assert (df["split"] == "test").sum() == 1951
assert (df["split"] == "validation").sum() == 2777

print("Augmented dataset verification: OK")
PYVERIFY

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
