#!/usr/bin/env bash
# run_all_videos.sh
# =================
# Run video_streaming_detector.py on all 5 defect videos with the
# current Rule C settings, then print a TPR/FPR summary table.

set -e

# ---- Configure ----
VIDEO_DIR="/home/ahmed/two-stage-efficientad-inspection/video_test"
YOLO="/home/ahmed/two-stage-efficientad-inspection/runs/detect/yolo_runs/chewinggum_yolov8n_v2/weights/best.pt"
CKPT="./ckpt_192"
THRESH_JSON="./video_calibration/video_thresholds_lowres.json"
OUT_BASE="./results_ruleC"

# Extra flag to pass to the detector: "--use_max_only" or leave empty
EXTRA_FLAGS=""

# ---- Run detector on each video ----
for V in video_A video_A1 video_A2 video_A3 video_A4; do
    echo ""
    echo "=================================================="
    echo "  Running ${V}"
    echo "=================================================="
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python video_streaming_detector1.py \
        --video "${VIDEO_DIR}/${V}.mp4" \
        --yolo_det_weights "${YOLO}" \
        --method lowres --ckpt_dir "${CKPT}" \
        --thresholds_json "${THRESH_JSON}" \
        --category chewinggum --model_size S \
        --tau_track 1.29 --ema_window 15 \
        --persist_K 3 --clear_M 5 --gamma 0.5 \
        --clip_margin 5 --min_box_frac 0.5 \
        --batch_frames 16 --batch_crops 96 --crop_workers 12 \
        --no_video --fp16 --device cuda:0 \
        ${EXTRA_FLAGS} \
        --save_dir "${OUT_BASE}_${V}"
done

# ---- Print TPR summary ----
python - <<'PYEOF'
import json
from pathlib import Path

truths = {
    "video_A":   {1, 4, 12, 18, 34, 36},
    "video_A1":  {8, 11, 12, 15, 23, 30},
    "video_A2":  {3, 5, 10, 18, 29, 33},
    "video_A3":  {6, 8, 18, 27, 34, 50},
    "video_A4":  {2, 15, 18, 30, 49, 55},
}

total_tp = 0
total_pos = 0
total_fp = 0
total_fired = 0

print(f"\n{'video':<12} {'TP':>5} {'FP':>5} {'FN':>5}   Details")
print("-" * 90)

for name, truth in truths.items():
    p = Path(f"./results_ruleC_{name}/records_lowres.jsonl")
    if not p.exists():
        print(f"{name:<12}  MISSING")
        continue
    recs = [json.loads(l) for l in p.open()]
    fired = set()
    for r in recs:
        for f in r.get("fired_tracks", []):
            fired.add(f["track_id"])
    tp = truth & fired
    fn = truth - fired
    fp = fired - truth

    total_tp += len(tp); total_pos += len(truth)
    total_fp += len(fp); total_fired += len(fired)

    details = ""
    if fn:
        details = f"FN={sorted(fn)}"
    print(f"{name:<12} {len(tp):>5} {len(fp):>5} {len(fn):>5}   {details}")

print()
print(f"TOTAL TPR = {total_tp}/{total_pos} = "
      f"{total_tp/max(total_pos,1):.3f}")
print(f"TOTAL FPR = {total_fp}/{total_fired} = "
      f"{total_fp/max(total_fired,1):.3f}")
print(f"TOTAL J   = {total_tp/max(total_pos,1) - total_fp/max(total_fired,1):.3f}")
PYEOF