#!/usr/bin/env bash
# Train all EfficientAD optimization variants for comparison.
# Sequential — assumes one GPU. Total runtime ≈ 2.5-3 hours.

set -u
cd /home/ahmed/two-stage-efficientad-inspection

DATASET="$(pwd)/dataset_multi"
TEACHER="$(pwd)/ckpt_cropped_multi/best_teacher.pth"
LOGDIR="$(pwd)/variant_logs"
mkdir -p "$LOGDIR"

if [ ! -f "$TEACHER" ]; then
    echo "ERROR: teacher not found at $TEACHER"
    exit 1
fi
if [ ! -f "$DATASET/split_csv/1cls.csv" ]; then
    echo "ERROR: dataset CSV not found at $DATASET/split_csv/1cls.csv"
    exit 1
fi

run() {
    local name="$1"; shift
    local log="$LOGDIR/${name}.log"
    echo ""
    echo "======================================================================"
    echo "  STARTING: $name"
    echo "  LOG:      $log"
    echo "======================================================================"
    t0=$(date +%s)
    if "$@" > "$log" 2>&1; then
        t1=$(date +%s)
        echo "  DONE in $(( (t1 - t0) / 60 )) min"
    else
        echo "  FAILED (exit code $?)  -- see $log"
    fi
}

# ---- 1. Low-res 192 -----------------------------------------------------
run lowres_192 \
    python retrain_lowres.py \
        --dataset_path "$DATASET" \
        --ckpt_dir     ./ckpt_lowres_192 \
        --teacher_path "$TEACHER" \
        --category     chewinggum \
        --model_size   S \
        --image_size   192 \
        --iterations   70000 \
        --batch_size   1 \
        --lr           1e-4 \
        --print_freq   1000 \
        --device       cuda:0 \
        --no_imagenet

# ---- 2. Low-res 224 -----------------------------------------------------
run lowres_224 \
    python retrain_lowres.py \
        --dataset_path "$DATASET" \
        --ckpt_dir     ./ckpt_lowres_224 \
        --teacher_path "$TEACHER" \
        --category     chewinggum \
        --model_size   S \
        --image_size   224 \
        --iterations   70000 \
        --batch_size   1 \
        --lr           1e-4 \
        --print_freq   1000 \
        --device       cuda:0 \
        --no_imagenet

# ---- 3. Thin width_mult=0.5 ---------------------------------------------
run thin_050 \
    python retrain_thin.py \
        --dataset_path "$DATASET" \
        --ckpt_dir     ./ckpt_thin_050 \
        --teacher_path "$TEACHER" \
        --category     chewinggum \
        --model_size   S \
        --width_mult   0.5 \
        --iterations   70000 \
        --batch_size   1 \
        --lr           1e-4 \
        --print_freq   1000 \
        --device       cuda:0 \
        --no_imagenet

# ---- 4. Thin width_mult=0.75 --------------------------------------------
run thin_075 \
    python retrain_thin.py \
        --dataset_path "$DATASET" \
        --ckpt_dir     ./ckpt_thin_075 \
        --teacher_path "$TEACHER" \
        --category     chewinggum \
        --model_size   S \
        --width_mult   0.75 \
        --iterations   70000 \
        --batch_size   1 \
        --lr           1e-4 \
        --print_freq   1000 \
        --device       cuda:0 \
        --no_imagenet

# ---- 5. Pruned 0.3 (fine-tune from baseline) ----------------------------
run pruned_030 \
    python prune_finetune.py \
        --dataset_path        "$DATASET" \
        --ckpt_dir            ./ckpt_cropped_multi \
        --out_dir             ./ckpt_pruned_030 \
        --teacher_path        "$TEACHER" \
        --category            chewinggum \
        --model_size          S \
        --prune_ratio         0.3 \
        --finetune_iterations 10000 \
        --batch_size          1 \
        --lr                  3e-5 \
        --print_freq          500 \
        --device              cuda:0

# ---- 6. QAT int8 (skip if not deploying on CPU) -------------------------
# This converts to int8 which runs CPU-only. If you're deploying on GPU,
# skip this — QAT gives you nothing on GPU. Uncomment to enable:
# run qat_int8 \
#     python train_qat.py \
#         --dataset_path   "$DATASET" \
#         --ckpt_dir       ./ckpt_cropped_multi \
#         --out_dir        ./ckpt_qat \
#         --teacher_path   "$TEACHER" \
#         --category       chewinggum \
#         --model_size     S \
#         --backend        fbgemm \
#         --qat_iterations 3000 \
#         --batch_size     1 \
#         --lr             1e-5 \
#         --print_freq     500 \
#         --device         cuda:0

echo ""
echo "======================================================================"
echo "  ALL TRAINING COMPLETE"
echo "  Logs in $LOGDIR/"
echo "======================================================================"
ls -d ckpt_lowres_192 ckpt_lowres_224 ckpt_thin_050 ckpt_thin_075 ckpt_pruned_030 2>/dev/null