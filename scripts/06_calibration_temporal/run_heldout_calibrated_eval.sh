#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/home/ahmed/two-stage-efficientad-inspection"
cd "$ROOT"

DETECTOR="video_streaming_detector_simple_optimized_v2.py"
YOLO="./runs/detect/yolo_runs/chewinggum_yolov8n_v2/weights/best.pt"

run_one () {
    local NAME="$1"
    local METHOD="$2"
    local CKPT="$3"
    local THR="$4"
    local VIDEO="$5"
    local TAG="$6"

    python "$DETECTOR" \
      --video "$VIDEO" \
      --yolo_det_weights "$YOLO" \
      --method "$METHOD" \
      --ckpt_dir "$CKPT" \
      --thresholds_json "$THR" \
      --category chewinggum \
      --model_size S \
      --device cuda:0 \
      --fp16 \
      --threshold_basis ema \
      --ema_alpha 0.3 \
      --clip_margin 5 \
      --min_box_frac 0.5 \
      --batch_frames 16 \
      --batch_crops 96 \
      --gpu_batch_crops 8 \
      --crop_workers 12 \
      --no_detect_unknown \
      --no_video \
      --save_dir "./heldout_eval/$NAME/$TAG"
}

for SPEC in \
  "baseline|baseline|./ckpt_cropped_multi|./video_calibration_new/video_thresholds_baseline_ema.json" \
  "lowres192|lowres|./ckpt_lowres_192|./video_calibration_new/video_thresholds_lowres192_ema.json" \
  "lowres224|lowres|./ckpt_lowres_224|./video_calibration_new/video_thresholds_lowres224_ema.json" \
  "thin075|thin|./ckpt_thin_075|./video_calibration_new/video_thresholds_thin075_ema.json" \
  "thin050|thin|./ckpt_thin_050|./video_calibration_new/video_thresholds_thin050_ema.json" \
  "pruned030|pruned|./ckpt_pruned_030|./video_calibration_new/video_thresholds_pruned030_ema.json"
do
    IFS='|' read -r NAME METHOD CKPT THR <<< "$SPEC"

    echo "======================================================================"
    echo "$NAME : video_A"
    echo "======================================================================"
    run_one "$NAME" "$METHOD" "$CKPT" "$THR" "./video_test/video_A.mp4" "video_A"

    echo "======================================================================"
    echo "$NAME : video_A3"
    echo "======================================================================"
    run_one "$NAME" "$METHOD" "$CKPT" "$THR" "./video_test/video_A3.mp4" "video_A3"

    echo "======================================================================"
    echo "$NAME : video_A4"
    echo "======================================================================"
    run_one "$NAME" "$METHOD" "$CKPT" "$THR" "./video_test/video_A4.mp4" "video_A4"
done

echo
echo "Held-out score files saved under:"
echo "  $ROOT/heldout_eval/"
