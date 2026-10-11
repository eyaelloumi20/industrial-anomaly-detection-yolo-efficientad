#!/usr/bin/env bash
set -euo pipefail

cd /home/ahmed/two-stage-efficientad-inspection

python compare_old_new_yolo_fair.py \
  --gt ./gt_reserved_newyolo/tracks_to_label.csv \
  --old-scores ./final_unseen_lowres192_oldyolo \
  --new-scores ./final_unseen_lowres192_newyolo_recalibrated \
  --old-threshold 0.833292 \
  --new-threshold 0.83485001 \
  --out ./old_vs_new_yolo_fair_comparison
