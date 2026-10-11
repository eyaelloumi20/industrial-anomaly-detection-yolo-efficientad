#!/usr/bin/env bash
set -euo pipefail

cd /home/ahmed/two-stage-efficientad-inspection

DATASET="./dataset_multi_bg_fast"
CKPT="./ckpt_lowres_192_multibg_fast"
TEACHER="./ckpt_lowres_192/best_teacher.pth"
LOG="./lowres192_multibg_fast_train.log"

if [[ ! -f "$DATASET/split_csv/1cls.csv" ]]; then
  echo "ERROR: dataset not built yet."
  echo "Run: python build_multibg_fast_dataset.py"
  exit 1
fi

if [[ ! -f "$TEACHER" ]]; then
  echo "ERROR: teacher not found: $TEACHER"
  exit 1
fi

echo "======================================================================"
echo "FAST LOWRES192 MULTI-BACKGROUND TRAINING"
echo "======================================================================"
echo "Dataset    : $DATASET"
echo "Checkpoint : $CKPT"
echo "Iterations : 20000"
echo "Teacher    : $TEACHER"
echo
echo "This is the FAST experiment. Existing ckpt_lowres_192 is untouched."
echo

rm -rf "$CKPT"

python retrain_lowres.py \
  --dataset_path "$DATASET" \
  --ckpt_dir "$CKPT" \
  --teacher_path "$TEACHER" \
  --category chewinggum \
  --model_size S \
  --image_size 192 \
  --iterations 20000 \
  --batch_size 1 \
  --lr 1e-4 \
  --print_freq 500 \
  --device cuda:0 \
  --no_imagenet 2>&1 | tee "$LOG"

echo
echo "======================================================================"
echo "TRAINING FINISHED"
echo "======================================================================"
echo "Checkpoint: $CKPT"
echo "Log       : $LOG"
echo
find "$CKPT" -maxdepth 1 -type f -printf "%f\n" | sort
