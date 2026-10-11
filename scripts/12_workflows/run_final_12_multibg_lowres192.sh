#!/usr/bin/env bash
set -euo pipefail

cd /home/ahmed/two-stage-efficientad-inspection

YOLO="./runs/detect/runs/detect/yolo_runs/chewinggum_yolov8n_multibg/weights/best.pt"
DETECTOR="./video_streaming_detector_simple_optimized_v3.py"
NEW_CKPT="./ckpt_lowres_192_multibg_fast"
GT="./gt_reserved_newyolo/tracks_to_label.csv"

NEW_TAU="0.3219970100"

OUT="./final_unseen_lowres192_multibg"
EVAL_OUT="./final_unseen_lowres192_multibg_evaluation"
LOG="./final_unseen_lowres192_multibg_run.log"

mkdir -p "$OUT"
rm -f "$LOG"

echo "======================================================================" | tee -a "$LOG"
echo "FINAL 12-VIDEO TEST — MULTI-BACKGROUND LOWRES192" | tee -a "$LOG"
echo "======================================================================" | tee -a "$LOG"
echo "Checkpoint : $NEW_CKPT" | tee -a "$LOG"
echo "Threshold  : $NEW_TAU" | tee -a "$LOG"
echo "YOLO       : $YOLO" | tee -a "$LOG"
echo "Crop       : current/original" | tee -a "$LOG"
echo "EMA alpha  : 0.3" | tee -a "$LOG"
echo | tee -a "$LOG"

run_one () {
  local BG="$1"
  local TAG="$2"
  local VIDEO="$3"
  local SAVE="$OUT/$BG/$TAG"

  mkdir -p "$SAVE"

  echo | tee -a "$LOG"
  echo "============================================================" | tee -a "$LOG"
  echo "$BG / $TAG" | tee -a "$LOG"
  echo "$VIDEO" | tee -a "$LOG"
  echo "============================================================" | tee -a "$LOG"

  python "$DETECTOR" \
    --video "$VIDEO" \
    --yolo_det_weights "$YOLO" \
    --method lowres \
    --ckpt_dir "$NEW_CKPT" \
    --threshold "$NEW_TAU" \
    --category chewinggum \
    --model_size S \
    --device cuda:0 \
    --fp16 \
    --threshold_basis ema \
    --ema_alpha 0.3 \
    --crop_preprocess current \
    --conf 0.10 \
    --iou 0.50 \
    --pad_ratio 0.02 \
    --clip_margin 5 \
    --min_box_frac 0.5 \
    --batch_frames 16 \
    --batch_crops 96 \
    --gpu_batch_crops 8 \
    --crop_workers 12 \
    --no_detect_unknown \
    --no_video \
    --save_dir "$SAVE" 2>&1 | tee -a "$LOG"
}

run_one "black" "black_01" "./videos_background/black/WhatsApp Video 2026-09-18 at 15.14.22.mp4"
run_one "black" "black_02" "./videos_background/black/WhatsApp Video 2026-09-18 at 15.15.35.mp4"
run_one "black" "black_03" "./videos_background/black/WhatsApp Video 2026-09-18 at 15.15.352.mp4"

run_one "metallic" "metallic_01" "./videos_background/metallic/WhatsApp Video 2026-09-18 at 15.47.23.mp4"
run_one "metallic" "metallic_02" "./videos_background/metallic/WhatsApp Video 2026-09-18 at 15.49.36.mp4"

run_one "white" "white_01" "./videos_background/white/WhatsApp Video 2026-09-18 at 15.55.45.mp4"
run_one "white" "white_02" "./videos_background/white/WhatsApp Video 2026-09-18 at 15.55.46.mp4"
run_one "white" "white_03" "./videos_background/white/WhatsApp Video 2026-09-18 at 15.55.461.mp4"

run_one "wooden" "wooden_01" "./videos_background/wooden/WhatsApp Video 2026-09-18 at 15.31.55.mp4"
run_one "wooden" "wooden_02" "./videos_background/wooden/WhatsApp Video 2026-09-18 at 15.31.551.mp4"
run_one "wooden" "wooden_03" "./videos_background/wooden/WhatsApp Video 2026-09-18 at 15.31.56.mp4"
run_one "wooden" "wooden_04" "./videos_background/wooden/WhatsApp Video 2026-09-18 at 15.31.561.mp4"

echo | tee -a "$LOG"
echo "======================================================================" | tee -a "$LOG"
echo "ALL 12 VIDEOS FINISHED" | tee -a "$LOG"
echo "======================================================================" | tee -a "$LOG"

echo | tee -a "$LOG"
echo "Running final track-level evaluation..." | tee -a "$LOG"

rm -rf "$EVAL_OUT"

python evaluate_final_unseen_lowres192.py \
  --gt "$GT" \
  --scores "$OUT" \
  --threshold "$NEW_TAU" \
  --out "$EVAL_OUT" 2>&1 | tee "$EVAL_OUT.console.txt"

echo | tee -a "$LOG"
echo "======================================================================" | tee -a "$LOG"
echo "FINAL TEST COMPLETE" | tee -a "$LOG"
echo "======================================================================" | tee -a "$LOG"
echo "Scores     : $OUT" | tee -a "$LOG"
echo "Evaluation : $EVAL_OUT" | tee -a "$LOG"
echo "Console    : $EVAL_OUT.console.txt" | tee -a "$LOG"
echo
echo "Send me the OVERALL and PER BACKGROUND blocks."
