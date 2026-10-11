#!/usr/bin/env bash
# run_all_options.sh
# Runs the streaming detector on all 5 defect videos with 4 configurations.
# Each config's output goes to its own directory tree.
set -e

VIDEO_DIR="/home/ahmed/two-stage-efficientad-inspection/video_test"
YOLO="/home/ahmed/two-stage-efficientad-inspection/runs/detect/yolo_runs/chewinggum_yolov8n_v2/weights/best.pt"
NEW_CKPT="/home/ahmed/framework/ckpt_cropped_large1"
NEW_JSON="./video_calibration_new/video_thresholds_baseline.json"
OLD_CKPT="./ckpt_192"
OLD_JSON="./video_calibration/video_thresholds_lowres.json"

VIDEOS="video_A video_A1 video_A2 video_A3 video_A4"

# Common flags
COMMON_FLAGS="--category chewinggum --model_size S \
  --persist_K 3 --clear_M 5 --gamma 0.5 \
  --clip_margin 5 --min_box_frac 0.5 \
  --batch_frames 16 --batch_crops 96 --crop_workers 12 \
  --no_video --fp16"

run_config () {
  local NAME=$1; shift
  echo ""
  echo "============================================================"
  echo "  Running $NAME"
  echo "============================================================"
  for V in $VIDEOS; do
    echo "  -> $V"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python video_streaming_detector.py \
      --video "${VIDEO_DIR}/${V}.mp4" \
      --yolo_det_weights "${YOLO}" \
      ${COMMON_FLAGS} \
      "$@" \
      --save_dir "./options/${NAME}/${V}"
  done
}

# A - new ckpt, median
run_config A_new_median \
  --method baseline --ckpt_dir "${NEW_CKPT}" --thresholds_json "${NEW_JSON}" \
  --score_mode median --tau_track 1.335 --device cuda:0

# B - new ckpt, mean
run_config B_new_mean \
  --method baseline --ckpt_dir "${NEW_CKPT}" --thresholds_json "${NEW_JSON}" \
  --score_mode mean --tau_track 1.215 --device cuda:0

# C - old lowres, median
run_config C_old_median \
  --method lowres --ckpt_dir "${OLD_CKPT}" --thresholds_json "${OLD_JSON}" \
  --score_mode median --tau_track 0.883 --device cuda:0

# D - old lowres, frame_rel
run_config D_old_framerel \
  --method lowres --ckpt_dir "${OLD_CKPT}" --thresholds_json "${OLD_JSON}" \
  --score_mode frame_rel --tau_track 0.373 --device cuda:0

echo ""
echo "============================================================"
echo "  All 4 configs done. Computing ensembles..."
echo "============================================================"
python compare_options.py