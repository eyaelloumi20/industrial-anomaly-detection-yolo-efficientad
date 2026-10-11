#!/usr/bin/env bash
set -euo pipefail

cd /home/ahmed/two-stage-efficientad-inspection

# ============================================================
# AUGMENTED LOWRES192 — COMPLETE FINAL TEST
# 1) Compute p99 EMA threshold from the 3 normal calibration videos
# 2) Run the same 12 reserved background videos
# 3) Evaluate against the existing manual GT
# ============================================================

CKPT="./ckpt_lowres_192_aug_70k"
CAL="./aug70k_lowres192_calibration"
THR_JSON="./aug70k_lowres192_threshold.json"

YOLO="./runs/detect/runs/detect/yolo_runs/chewinggum_yolov8n_multibg/weights/best.pt"
DETECTOR="./video_streaming_detector_simple_optimized_v2.py"

GT="./gt_reserved_newyolo/tracks_to_label.csv"
EVALUATOR="./evaluate_final_unseen_lowres192.py"

OUT="./final_unseen_lowres192_aug70k"
EVAL_OUT="./final_unseen_lowres192_aug70k_evaluation"
FULL_LOG="./final_unseen_lowres192_aug70k_full.log"

echo "======================================================================"
echo "AUGMENTED LOWRES192 — COMPLETE CALIBRATION + 12-VIDEO TEST"
echo "======================================================================"
echo "Checkpoint : $CKPT"
echo "Calibration: $CAL"
echo "YOLO       : $YOLO"
echo "GT         : $GT"
echo "======================================================================"

# ------------------------------------------------------------
# 0) Required-file checks
# ------------------------------------------------------------
for F in \
  "$CKPT/best_teacher.pth" \
  "$CKPT/chewinggum_student.pth" \
  "$CKPT/chewinggum_autoencoder.pth" \
  "$CKPT/chewinggum_quantiles.npy" \
  "$CAL/videoN1/scores.csv" \
  "$CAL/video_N/scores.csv" \
  "$CAL/video_N2/scores.csv" \
  "$YOLO" \
  "$DETECTOR" \
  "$GT" \
  "$EVALUATOR"
do
  if [[ ! -f "$F" ]]; then
    echo "ERROR: missing required file:"
    echo "  $F"
    exit 1
  fi
done

# ------------------------------------------------------------
# 1) Compute p99 threshold from ALL EMA scores
# ------------------------------------------------------------
echo
echo "[1/3] Computing normal-only p99 EMA threshold..."

python - <<'PY'
from pathlib import Path
import csv
import json
import numpy as np

root = Path("./aug70k_lowres192_calibration")
files = [
    root / "videoN1" / "scores.csv",
    root / "video_N" / "scores.csv",
    root / "video_N2" / "scores.csv",
]

all_vals = []
per_video = {}

for p in files:
    vals = []
    with p.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            s = row.get("ema_score", "")
            if s not in ("", None):
                vals.append(float(s))

    if not vals:
        raise RuntimeError(f"No EMA values in {p}")

    a = np.asarray(vals, dtype=float)
    all_vals.extend(vals)

    per_video[p.parent.name] = {
        "n": int(len(a)),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "p95": float(np.quantile(a, 0.95)),
        "p99": float(np.quantile(a, 0.99)),
        "max": float(a.max()),
    }

a = np.asarray(all_vals, dtype=float)

if len(a) != 18810:
    print(f"WARNING: expected 18810 EMA scores, found {len(a)}")

tau = float(np.quantile(a, 0.99))

result = {
    "model": "ckpt_lowres_192_aug_70k",
    "basis": "ema",
    "ema_alpha": 0.3,
    "calibration": "normal-only global p99",
    "n_scores": int(len(a)),
    "mean": float(a.mean()),
    "std": float(a.std()),
    "min": float(a.min()),
    "max": float(a.max()),
    "tau_ema": tau,
    "per_video": per_video,
}

Path("./aug70k_lowres192_threshold.json").write_text(
    json.dumps(result, indent=2),
    encoding="utf-8",
)

Path("./aug70k_lowres192_threshold.txt").write_text(
    f"{tau:.10f}\n",
    encoding="utf-8",
)

print("=" * 68)
print("AUGMENTED LOWRES192 NORMAL-ONLY CALIBRATION")
print("=" * 68)
print(f"n_scores : {len(a)}")
print(f"mean     : {a.mean():.10f}")
print(f"std      : {a.std():.10f}")
print(f"min      : {a.min():.10f}")
print(f"max      : {a.max():.10f}")
print(f"tau p99  : {tau:.10f}")
print()
for name, d in per_video.items():
    print(
        f"{name:8s} n={d['n']:5d} "
        f"mean={d['mean']:.6f} "
        f"p99={d['p99']:.6f} "
        f"max={d['max']:.6f}"
    )
print("=" * 68)
PY

THRESHOLD="$(tr -d '[:space:]' < ./aug70k_lowres192_threshold.txt)"

if [[ -z "$THRESHOLD" ]]; then
  echo "ERROR: threshold could not be read."
  exit 1
fi

echo
echo "Frozen augmented threshold = $THRESHOLD"

# ------------------------------------------------------------
# 2) Run the SAME 12 reserved videos
# ------------------------------------------------------------
echo
echo "[2/3] Running the 12 reserved videos..."
echo "Output: $OUT"

rm -rf "$OUT" "$EVAL_OUT"
mkdir -p "$OUT"

run_video () {
    local BG="$1"
    local TAG="$2"
    local VIDEO="$3"

    if [[ ! -f "$VIDEO" ]]; then
        echo
        echo "ERROR: video not found:"
        echo "  $VIDEO"
        exit 1
    fi

    echo
    echo "======================================================================"
    echo "$BG / $TAG"
    echo "$VIDEO"
    echo "======================================================================"

    python "$DETECTOR" \
      --video "$VIDEO" \
      --yolo_det_weights "$YOLO" \
      --method lowres \
      --ckpt_dir "$CKPT" \
      --category chewinggum \
      --model_size S \
      --device cuda:0 \
      --fp16 \
      --conf 0.25 \
      --iou 0.45 \
      --threshold "$THRESHOLD" \
      --threshold_basis ema \
      --ema_alpha 0.3 \
      --batch_frames 16 \
      --batch_crops 96 \
      --gpu_batch_crops 8 \
      --crop_workers 12 \
      --clip_margin 5 \
      --min_box_frac 0.5 \
      --no_detect_unknown \
      --no_video \
      --save_dir "$OUT/$BG/$TAG"
}

# Black — 3
run_video "black" "black_01" \
  "./videos_background/black/WhatsApp Video 2026-09-18 at 15.14.22.mp4"

run_video "black" "black_02" \
  "./videos_background/black/WhatsApp Video 2026-09-18 at 15.15.35.mp4"

run_video "black" "black_03" \
  "./videos_background/black/WhatsApp Video 2026-09-18 at 15.15.352.mp4"

# Metallic — 2
run_video "metallic" "metallic_01" \
  "./videos_background/metallic/WhatsApp Video 2026-09-18 at 15.47.23.mp4"

run_video "metallic" "metallic_02" \
  "./videos_background/metallic/WhatsApp Video 2026-09-18 at 15.49.36.mp4"

# White — 3
run_video "white" "white_01" \
  "./videos_background/white/WhatsApp Video 2026-09-18 at 15.55.45.mp4"

run_video "white" "white_02" \
  "./videos_background/white/WhatsApp Video 2026-09-18 at 15.55.46.mp4"

run_video "white" "white_03" \
  "./videos_background/white/WhatsApp Video 2026-09-18 at 15.55.461.mp4"

# Wooden — 4
run_video "wooden" "wooden_01" \
  "./videos_background/wooden/WhatsApp Video 2026-09-18 at 15.31.55.mp4"

run_video "wooden" "wooden_02" \
  "./videos_background/wooden/WhatsApp Video 2026-09-18 at 15.31.551.mp4"

run_video "wooden" "wooden_03" \
  "./videos_background/wooden/WhatsApp Video 2026-09-18 at 15.31.56.mp4"

run_video "wooden" "wooden_04" \
  "./videos_background/wooden/WhatsApp Video 2026-09-18 at 15.31.561.mp4"

echo
echo "12 videos completed."

COUNT=$(find "$OUT" -type f -name scores.csv | wc -l)
echo "scores.csv files found: $COUNT"

if [[ "$COUNT" -ne 12 ]]; then
    echo "ERROR: expected 12 score files, found $COUNT"
    exit 1
fi

# ------------------------------------------------------------
# 3) Existing GT evaluator
# ------------------------------------------------------------
echo
echo "[3/3] Evaluating against manual GT..."

python "$EVALUATOR" \
  --gt "$GT" \
  --scores "$OUT" \
  --threshold "$THRESHOLD" \
  --out "$EVAL_OUT"

# ------------------------------------------------------------
# Final concise comparison
# ------------------------------------------------------------
echo
echo "======================================================================"
echo "TEST COMPLETE"
echo "======================================================================"
echo "Augmented LowRes192 threshold : $THRESHOLD"
echo "Scores                         : $OUT"
echo "Evaluation                     : $EVAL_OUT"
echo
echo "ORIGINAL LOWRES192 REFERENCE"
echo "TP=14  TN=160  FP=79  FN=26"
echo "Precision=0.1505"
echo "Recall   =0.3500"
echo "F1       =0.2105"
echo "FPR      =0.3305"
echo "Accuracy =0.6237"
echo "AUROC    =0.55094"
echo
echo "Compare the evaluator output immediately above with this reference."
echo "======================================================================"
