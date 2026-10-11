#!/usr/bin/env bash
# run_all_step4_videos.sh
# =========================
# Step 4A helper (NOT one of the read-only diagnostic scripts -- just a
# loop around your EXISTING compare_pipeline_methods_batched.py, run once
# per video listed in manifest.csv). Nothing about scoring/YOLO/ByteTrack/
# calibration logic is touched or reimplemented here.
#
# What it does:
#   1. Reads every video_filename out of dataset_step4/manifest.csv
#   2. Runs compare_pipeline_methods_batched.py on it (baseline method only,
#      no annotated video output -- just the scores CSV, to keep this fast)
#   3. Writes runs.csv mapping video_filename -> its scores_baseline.csv,
#      ready to feed straight into build_step4_dataset.py
#
# EDIT THESE THREE PATHS FIRST:
YOLO_WEIGHTS="/home/ahmed/two-stage-efficientad-inspection/runs/detect/yolo_runs/chewinggum_yolov8n_v2/weights/best.pt"
BASELINE_DIR="/home/ahmed/framework/ckpt_cropped_large"
DATASET_ROOT="./dataset_step4"

MANIFEST="${DATASET_ROOT}/manifest.csv"
RESULTS_ROOT="./results_step4"
RUNS_CSV="./runs.csv"

set -e

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: manifest not found at $MANIFEST"
    exit 1
fi

mkdir -p "$RESULTS_ROOT"
echo "video_filename,scores_csv,method" > "$RUNS_CSV"

# Skip the header line, read the rest. video_filename is the first CSV
# column; this simple cut works because none of your filenames contain a
# literal comma (they contain spaces and parentheses, which are fine).
tail -n +2 "$MANIFEST" | while IFS=',' read -r video_filename rest; do
    [ -z "$video_filename" ] && continue

    video_path="${DATASET_ROOT}/${video_filename}"
    if [ ! -f "$video_path" ]; then
        echo "  [SKIP] '$video_filename' not found at '$video_path'"
        continue
    fi

    # sanitize filename into a safe folder name (spaces/parens -> underscore)
    safe_name=$(echo "$video_filename" \
        | sed 's/\.mp4$//' \
        | tr ' ()' '___' \
        | tr -s '_')
    save_dir="${RESULTS_ROOT}/${safe_name}"

    echo ""
    echo "=================================================================="
    echo "  Scoring: $video_filename"
    echo "  -> $save_dir"
    echo "=================================================================="

    python compare_pipeline_methods_batched.py \
        --video "$video_path" \
        --yolo_det_weights "$YOLO_WEIGHTS" \
        --baseline_dir "$BASELINE_DIR" \
        --skip_methods lowres thin qat pruned \
        --category chewinggum --model_size S \
        --save_dir "$save_dir" \
        --no_video

    scores_csv="${save_dir}/baseline/scores_baseline.csv"
    if [ -f "$scores_csv" ]; then
        # CSV-escape the filename if it contains a comma (none of yours do,
        # but this keeps the generated runs.csv robust either way)
        echo "\"${video_filename}\",\"${scores_csv}\",baseline" >> "$RUNS_CSV"
        echo "  [OK] wrote $scores_csv"
    else
        echo "  [WARN] expected scores file not found: $scores_csv"
    fi
done

echo ""
echo "=================================================================="
echo "  Done. runs.csv written to $RUNS_CSV"
echo "=================================================================="
cat "$RUNS_CSV"
