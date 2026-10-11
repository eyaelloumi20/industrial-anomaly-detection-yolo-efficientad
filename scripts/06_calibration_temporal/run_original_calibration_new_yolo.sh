#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# ORIGINAL EfficientAD CALIBRATION — SAME WORKFLOW, NEW YOLO
# ============================================================
#
# This wrapper keeps the original calibration methodology intact:
#   - same calibrate_ema_thresholds.py
#   - same 3 normal calibration videos:
#       videos_N/videoN1.mp4
#       videos_N/video_N.mp4
#       videos_N/video_N2.mp4
#   - same detector script:
#       video_streaming_detector_simple_optimized_v2.py
#   - same EMA alpha = 0.3
#   - same FP16 / batch settings
#   - same 99th-percentile normal-EMA threshold selection
#   - same model variants handled by calibrate_ema_thresholds.py
#
# ONLY change:
#   OLD YOLO -> NEW multi-background YOLO
#
# The script preserves your previous calibration outputs by moving them
# aside temporarily and restores them after the new-YOLO calibration.
# New-YOLO outputs are saved separately.
# ============================================================

ROOT="/home/ahmed/two-stage-efficientad-inspection"
cd "$ROOT"

CAL_SCRIPT="$ROOT/calibrate_ema_thresholds.py"

OLD_YOLO_ABS="$ROOT/runs/detect/yolo_runs/chewinggum_yolov8n_v2/weights/best.pt"
NEW_YOLO_ABS="$ROOT/runs/detect/runs/detect/yolo_runs/chewinggum_yolov8n_multibg/weights/best.pt"

OLD_YOLO_REL="./runs/detect/yolo_runs/chewinggum_yolov8n_v2/weights/best.pt"
NEW_YOLO_REL="./runs/detect/runs/detect/yolo_runs/chewinggum_yolov8n_multibg/weights/best.pt"

CAL_DIR="$ROOT/calibration_ema_variants"
THR_DIR="$ROOT/video_calibration_new"

NEW_CAL_DIR="$ROOT/calibration_ema_variants_newyolo"
NEW_THR_DIR="$ROOT/video_calibration_new_newyolo"

STAMP="$(date +%Y%m%d_%H%M%S)"
SCRIPT_BACKUP="$ROOT/calibrate_ema_thresholds.py.backup_${STAMP}"
CAL_BACKUP=""
THR_BACKUP=""

echo "============================================================"
echo "ORIGINAL CALIBRATION WORKFLOW + NEW YOLO"
echo "============================================================"
echo "Project     : $ROOT"
echo "Calibration : $CAL_SCRIPT"
echo "New YOLO    : $NEW_YOLO_ABS"
echo

# ---------------------------
# Sanity checks
# ---------------------------
for f in \
  "$CAL_SCRIPT" \
  "$ROOT/video_streaming_detector_simple_optimized_v2.py" \
  "$NEW_YOLO_ABS" \
  "$ROOT/videos_N/videoN1.mp4" \
  "$ROOT/videos_N/video_N.mp4" \
  "$ROOT/videos_N/video_N2.mp4"
do
  if [[ ! -e "$f" ]]; then
    echo "ERROR: missing required file:"
    echo "  $f"
    exit 1
  fi
done

if [[ -e "$NEW_CAL_DIR" || -e "$NEW_THR_DIR" ]]; then
  echo "ERROR: new-YOLO output directory already exists:"
  [[ -e "$NEW_CAL_DIR" ]] && echo "  $NEW_CAL_DIR"
  [[ -e "$NEW_THR_DIR" ]] && echo "  $NEW_THR_DIR"
  echo
  echo "Rename/remove it first so previous results are not overwritten."
  exit 1
fi

# ---------------------------
# Backup original script
# ---------------------------
cp -a "$CAL_SCRIPT" "$SCRIPT_BACKUP"

restore_everything () {
  set +e

  # Restore calibration script.
  if [[ -f "$SCRIPT_BACKUP" ]]; then
    cp -a "$SCRIPT_BACKUP" "$CAL_SCRIPT"
  fi

  # If a new run produced the normal output folders, preserve them.
  if [[ -d "$CAL_DIR" ]]; then
    if [[ ! -e "$NEW_CAL_DIR" ]]; then
      mv "$CAL_DIR" "$NEW_CAL_DIR"
    fi
  fi

  if [[ -d "$THR_DIR" ]]; then
    if [[ ! -e "$NEW_THR_DIR" ]]; then
      mv "$THR_DIR" "$NEW_THR_DIR"
    fi
  fi

  # Restore old outputs.
  if [[ -n "$CAL_BACKUP" && -d "$CAL_BACKUP" ]]; then
    mv "$CAL_BACKUP" "$CAL_DIR"
  fi

  if [[ -n "$THR_BACKUP" && -d "$THR_BACKUP" ]]; then
    mv "$THR_BACKUP" "$THR_DIR"
  fi

  echo
  echo "============================================================"
  echo "RESTORE COMPLETE"
  echo "============================================================"
  echo "Original calibration script restored."
  echo "Original calibration outputs restored."
  [[ -d "$NEW_CAL_DIR" ]] && echo "New-YOLO score outputs : $NEW_CAL_DIR"
  [[ -d "$NEW_THR_DIR" ]] && echo "New-YOLO thresholds    : $NEW_THR_DIR"
}

trap restore_everything EXIT

# ---------------------------
# Temporarily move old outputs
# ---------------------------
if [[ -d "$CAL_DIR" ]]; then
  CAL_BACKUP="${CAL_DIR}.backup_${STAMP}"
  echo "Backing up existing calibration scores:"
  echo "  $CAL_DIR"
  echo "  -> $CAL_BACKUP"
  mv "$CAL_DIR" "$CAL_BACKUP"
fi

if [[ -d "$THR_DIR" ]]; then
  THR_BACKUP="${THR_DIR}.backup_${STAMP}"
  echo "Backing up existing threshold files:"
  echo "  $THR_DIR"
  echo "  -> $THR_BACKUP"
  mv "$THR_DIR" "$THR_BACKUP"
fi

# ---------------------------
# Patch ONLY the YOLO path
# ---------------------------
python - <<'PY'
from pathlib import Path

p = Path("/home/ahmed/two-stage-efficientad-inspection/calibrate_ema_thresholds.py")
txt = p.read_text(encoding="utf-8")

replacements = [
    (
        "/home/ahmed/two-stage-efficientad-inspection/runs/detect/yolo_runs/chewinggum_yolov8n_v2/weights/best.pt",
        "/home/ahmed/two-stage-efficientad-inspection/runs/detect/runs/detect/yolo_runs/chewinggum_yolov8n_multibg/weights/best.pt",
    ),
    (
        "./runs/detect/yolo_runs/chewinggum_yolov8n_v2/weights/best.pt",
        "./runs/detect/runs/detect/yolo_runs/chewinggum_yolov8n_multibg/weights/best.pt",
    ),
    (
        "runs/detect/yolo_runs/chewinggum_yolov8n_v2/weights/best.pt",
        "runs/detect/runs/detect/yolo_runs/chewinggum_yolov8n_multibg/weights/best.pt",
    ),
]

count = 0
for old, new in replacements:
    if old in txt:
        n = txt.count(old)
        txt = txt.replace(old, new)
        count += n

if count == 0:
    raise SystemExit(
        "ERROR: Could not find the previous YOLO path inside "
        "calibrate_ema_thresholds.py. Nothing was changed."
    )

p.write_text(txt, encoding="utf-8")
print(f"Patched YOLO path occurrence(s): {count}")
PY

echo
echo "Verifying patched YOLO path..."
grep -n "chewinggum_yolov8n" "$CAL_SCRIPT" || true
echo

# ---------------------------
# Run the ORIGINAL calibration script
# ---------------------------
echo "============================================================"
echo "RUNNING ORIGINAL calibrate_ema_thresholds.py"
echo "============================================================"
python "$CAL_SCRIPT"

echo
echo "============================================================"
echo "CALIBRATION FINISHED SUCCESSFULLY"
echo "============================================================"

# The EXIT trap now:
# 1) stores new outputs as *_newyolo
# 2) restores old outputs
# 3) restores original calibration script
