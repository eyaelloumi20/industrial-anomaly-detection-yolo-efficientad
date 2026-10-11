#!/usr/bin/env bash
set -euo pipefail

cd /home/ahmed/two-stage-efficientad-inspection

PY=python
DET=video_streaming_detector_simple_optimized_v2.py

YOLO="./runs/detect/runs/detect/yolo_runs/chewinggum_yolov8n_multibg/weights/best.pt"
CKPT="./ckpt_lowres_192"
VIDEOS="./videos_background"
OUT="./calibration_lowres192"

# IMPORTANT:
# conf=0.10 and iou=0.50 are intentionally kept identical to the
# ground-truth proposal generation, so ByteTrack IDs remain comparable.
#
# threshold=0.50 is only a placeholder. Raw/EMA scores do NOT depend on it.
# The real threshold is selected afterwards by calibrate_lowres192_threshold.py.

COMMON=(
  --yolo_det_weights "$YOLO"
  --method lowres
  --ckpt_dir "$CKPT"
  --threshold 0.50
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

echo "============================================================"
echo "LOWRES192 CALIBRATION — 4 DEVELOPMENT VIDEOS ONLY"
echo "============================================================"

echo "[1/4] BLACK"
$PY "$DET" \
  --video "$VIDEOS/black/WhatsApp Video 2026-09-18 at 15.15.353.mp4" \
  "${COMMON[@]}" \
  --save_dir "$OUT/black"

echo "[2/4] METALLIC"
$PY "$DET" \
  --video "$VIDEOS/metallic/WhatsApp Video 2026-09-18 at 15.49.53.mp4" \
  "${COMMON[@]}" \
  --save_dir "$OUT/metallic"

echo "[3/4] WHITE"
$PY "$DET" \
  --video "$VIDEOS/white/WhatsApp Video 2026-09-18 at 15.55.11.mp4" \
  "${COMMON[@]}" \
  --save_dir "$OUT/white"

echo "[4/4] WOODEN"
$PY "$DET" \
  --video "$VIDEOS/wooden/WhatsApp Video 2026-09-18 at 15.31.552.mp4" \
  "${COMMON[@]}" \
  --save_dir "$OUT/wooden"

echo
echo "DONE."
echo "Score files:"
find "$OUT" -name scores.csv -print
echo
echo "Next:"
echo "python calibrate_lowres192_threshold.py \\"
echo "  --gt ./gt_reserved_newyolo/tracks_to_label.csv \\"
echo "  --scores ./calibration_lowres192 \\"
echo "  --out ./calibration_lowres192_threshold"
