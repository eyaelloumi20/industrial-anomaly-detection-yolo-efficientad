#!/usr/bin/env bash
set -euo pipefail

cd /home/ahmed/two-stage-efficientad-inspection

PY=python
DET=video_streaming_detector_simple_optimized_v2.py

YOLO="./runs/detect/runs/detect/yolo_runs/chewinggum_yolov8n_multibg/weights/best.pt"
CKPT="./ckpt_lowres_192"
VIDEOS="./videos_background"
OUT="./final_unseen_lowres192_newyolo_recalibrated"

# Original calibration workflow + new YOLO
THR=0.83485001

COMMON=(
  --yolo_det_weights "$YOLO"
  --method lowres
  --ckpt_dir "$CKPT"
  --threshold "$THR"
  --category chewinggum
  --model_size S
  --device cuda:0
  --fp16
  --threshold_basis ema
  --ema_alpha 0.3
  --conf 0.10
  --iou 0.50
  --clip_margin 5
  --min_box_frac 0.5
  --batch_frames 16
  --batch_crops 96
  --gpu_batch_crops 8
  --crop_workers 12
  --no_detect_unknown
  --no_video
)

mkdir -p "$OUT"

run_one () {
  BG="$1"
  NAME="$2"
  TAG="$3"

  echo
  echo "============================================================"
  echo "$TAG"
  echo "============================================================"

  $PY "$DET" \
    --video "$VIDEOS/$BG/$NAME" \
    "${COMMON[@]}" \
    --save_dir "$OUT/$BG/$TAG"
}

echo "============================================================"
echo "NEW YOLO + LOWRES192 FINAL TEST"
echo "Recalibrated threshold: $THR"
echo "============================================================"

# BLACK
run_one black "WhatsApp Video 2026-09-18 at 15.14.22.mp4" "black_01"
run_one black "WhatsApp Video 2026-09-18 at 15.15.35.mp4" "black_02"
run_one black "WhatsApp Video 2026-09-18 at 15.15.352.mp4" "black_03"

# METALLIC
run_one metallic "WhatsApp Video 2026-09-18 at 15.47.23.mp4" "metallic_01"
run_one metallic "WhatsApp Video 2026-09-18 at 15.49.36.mp4" "metallic_02"

# WHITE
run_one white "WhatsApp Video 2026-09-18 at 15.55.45.mp4" "white_01"
run_one white "WhatsApp Video 2026-09-18 at 15.55.46.mp4" "white_02"
run_one white "WhatsApp Video 2026-09-18 at 15.55.461.mp4" "white_03"

# WOODEN
run_one wooden "WhatsApp Video 2026-09-18 at 15.31.55.mp4" "wooden_01"
run_one wooden "WhatsApp Video 2026-09-18 at 15.31.551.mp4" "wooden_02"
run_one wooden "WhatsApp Video 2026-09-18 at 15.31.56.mp4" "wooden_03"
run_one wooden "WhatsApp Video 2026-09-18 at 15.31.561.mp4" "wooden_04"

echo
echo "============================================================"
echo "ALL 12 VIDEOS FINISHED"
echo "============================================================"
echo
echo "Now evaluate:"
echo
echo "python evaluate_final_unseen_lowres192.py \\"
echo "  --gt ./gt_reserved_newyolo/tracks_to_label.csv \\"
echo "  --scores ./final_unseen_lowres192_newyolo_recalibrated \\"
echo "  --threshold $THR \\"
echo "  --out ./final_unseen_lowres192_newyolo_recalibrated_evaluation"
