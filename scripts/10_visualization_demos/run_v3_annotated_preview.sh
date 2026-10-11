#!/usr/bin/env bash
set -euo pipefail

cd /home/ahmed/two-stage-efficientad-inspection

# Visual preview of the integrated v3 pipeline.
# IMPORTANT: threshold is intentionally set very high so the preview does NOT
# pretend to make a calibrated anomaly decision before recalibration.
# You can still inspect boxes, tracks, crops/scores, and video rendering.

python video_streaming_detector_simple_optimized_v3.py \
  --video "./videos_background/white/WhatsApp Video 2026-09-18 at 15.55.11.mp4" \
  --yolo_det_weights "./runs/detect/runs/detect/yolo_runs/chewinggum_yolov8n_multibg/weights/best.pt" \
  --method lowres \
  --ckpt_dir ./ckpt_lowres_192 \
  --threshold 999.0 \
  --category chewinggum \
  --model_size S \
  --device cuda:0 \
  --fp16 \
  --threshold_basis ema \
  --ema_alpha 0.3 \
  --crop_preprocess mean_border \
  --mean_border_keep_ratio 0.78 \
  --mean_border_feather_ratio 0.08 \
  --mean_border_strength 1.0 \
  --conf 0.10 \
  --iou 0.50 \
  --clip_margin 5 \
  --min_box_frac 0.5 \
  --batch_frames 16 \
  --batch_crops 96 \
  --gpu_batch_crops 8 \
  --crop_workers 12 \
  --no_detect_unknown \
  --save_dir ./annotated_preview_v3

echo
echo "Preview finished."
echo "Check: ./annotated_preview_v3"
