#!/usr/bin/env bash
set -e
VIDEO_DIR="/home/ahmed/two-stage-efficientad-inspection/video_test"
YOLO="/home/ahmed/two-stage-efficientad-inspection/runs/detect/yolo_runs/chewinggum_yolov8n_v2/weights/best.pt"
CKPT_A="/home/ahmed/framework/ckpt_cropped_large1"
JSON_A="./video_calibration_new/video_thresholds_baseline.json"
CKPT_C="./ckpt_192"
JSON_C="./video_calibration/video_thresholds_lowres.json"

for MODE in F G; do
  for V in video_A video_A1 video_A2 video_A3 video_A4; do
    echo ""
    echo "=== mode=$MODE  video=$V  ==="
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python video_streaming_detector.py \
      --mode $MODE \
      --video "${VIDEO_DIR}/${V}.mp4" \
      --yolo_det_weights "${YOLO}" \
      --ckpt_A "${CKPT_A}" --thresholds_A "${JSON_A}" \
      --ckpt_C "${CKPT_C}" --thresholds_C "${JSON_C}" \
      --category chewinggum \
      --model_size S \
      --inactive_frames 30 \
      --min_samples 10 \
      --batch_frames 16 --batch_crops 96 --crop_workers 12 \
      --fp16 --device cuda:0 \
      --no_open_browser \
      --save_dir "./options/${MODE}_min10/${V}"
  done
done