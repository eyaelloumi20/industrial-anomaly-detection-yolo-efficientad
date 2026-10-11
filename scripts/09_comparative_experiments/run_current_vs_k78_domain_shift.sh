#!/usr/bin/env bash
set -euo pipefail

cd /home/ahmed/two-stage-efficientad-inspection

python current_vs_k78_domain_shift.py \
  --yolo_det_weights ./runs/detect/runs/detect/yolo_runs/chewinggum_yolov8n_multibg/weights/best.pt \
  --ckpt_dir ./ckpt_lowres_192 \
  --method lowres \
  --category chewinggum \
  --model_size S \
  --device cuda:0 \
  --fp16 \
  --conf 0.10 \
  --iou 0.50 \
  --ema_alpha 0.3 \
  --pad_ratio 0.02 \
  --clip_margin 5 \
  --min_box_frac 0.5 \
  --batch_frames 16 \
  --gpu_batch_crops 8 \
  --out ./current_vs_k78_domain_shift
