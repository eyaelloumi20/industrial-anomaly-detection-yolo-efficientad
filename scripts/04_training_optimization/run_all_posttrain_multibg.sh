#!/usr/bin/env bash
set -euo pipefail

cd /home/ahmed/two-stage-efficientad-inspection

OLD_CKPT="./ckpt_lowres_192"
NEW_CKPT="./ckpt_lowres_192_multibg_fast"
YOLO="./runs/detect/runs/detect/yolo_runs/chewinggum_yolov8n_multibg/weights/best.pt"
DETECTOR="./video_streaming_detector_simple_optimized_v3.py"

OUT="./multibg_lowres192_posttrain"
CAL_OUT="$OUT/calibration"
DEV_OUT="$OUT/dev_backgrounds"

mkdir -p "$CAL_OUT" "$DEV_OUT"

echo "======================================================================"
echo "POST-TRAIN LOWRES192 MULTI-BACKGROUND PIPELINE"
echo "======================================================================"

# ----------------------------------------------------------------------
# 1) Complete checkpoint
# ----------------------------------------------------------------------
echo
echo "[1/6] Copying frozen teacher into the new checkpoint..."
cp -f "$OLD_CKPT/best_teacher.pth" "$NEW_CKPT/best_teacher.pth"

for f in \
  "$NEW_CKPT/best_teacher.pth" \
  "$NEW_CKPT/chewinggum_student.pth" \
  "$NEW_CKPT/chewinggum_autoencoder.pth" \
  "$NEW_CKPT/chewinggum_quantiles.npy"
do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing $f"
    exit 1
  fi
done

echo "Checkpoint OK:"
ls -lh \
  "$NEW_CKPT/best_teacher.pth" \
  "$NEW_CKPT/chewinggum_student.pth" \
  "$NEW_CKPT/chewinggum_autoencoder.pth" \
  "$NEW_CKPT/chewinggum_quantiles.npy"

# ----------------------------------------------------------------------
# 2) Calibrate new checkpoint on the 3 original normal videos
# ----------------------------------------------------------------------
echo
echo "[2/6] Running normal-only EMA calibration on the new checkpoint..."

declare -a CAL_VIDEOS=(
  "./videos_N/videoN1.mp4"
  "./videos_N/video_N.mp4"
  "./videos_N/video_N2.mp4"
)

for VIDEO in "${CAL_VIDEOS[@]}"; do
  TAG="$(basename "$VIDEO" .mp4)"
  SAVE="$CAL_OUT/$TAG"
  mkdir -p "$SAVE"

  python "$DETECTOR" \
    --video "$VIDEO" \
    --yolo_det_weights "$YOLO" \
    --method lowres \
    --ckpt_dir "$NEW_CKPT" \
    --threshold 0.45 \
    --category chewinggum \
    --model_size S \
    --device cuda:0 \
    --fp16 \
    --threshold_basis ema \
    --ema_alpha 0.3 \
    --crop_preprocess current \
    --conf 0.10 \
    --iou 0.50 \
    --pad_ratio 0.02 \
    --clip_margin 5 \
    --min_box_frac 0.5 \
    --batch_frames 16 \
    --batch_crops 96 \
    --gpu_batch_crops 8 \
    --crop_workers 12 \
    --no_detect_unknown \
    --no_video \
    --save_dir "$SAVE"
done

# Compute p99 threshold exactly from all valid EMA scores.
python - <<'PY'
from pathlib import Path
import csv, json, numpy as np

root = Path("multibg_lowres192_posttrain/calibration")
tags = ["videoN1", "video_N", "video_N2"]

all_ema = []
per_video = {}

for tag in tags:
    p = root / tag / "scores.csv"
    vals = []
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            x = (r.get("ema_score") or "").strip()
            if x:
                vals.append(float(x))
    if not vals:
        raise SystemExit(f"No EMA scores in {p}")
    all_ema.extend(vals)
    a = np.asarray(vals, dtype=float)
    per_video[tag] = {
        "n": int(len(a)),
        "mean": float(a.mean()),
        "p95": float(np.quantile(a, .95)),
        "p99": float(np.quantile(a, .99)),
        "max": float(a.max()),
    }

a = np.asarray(all_ema, dtype=float)
tau = float(np.quantile(a, .99))

result = {
    "model": "LowRes192 multi-background",
    "calibration_type": "normal_only_quantile",
    "basis": "ema",
    "ema_alpha": 0.3,
    "quantile": 0.99,
    "tau_ema": tau,
    "n_scores": int(len(a)),
    "mean": float(a.mean()),
    "std": float(a.std()),
    "min": float(a.min()),
    "max": float(a.max()),
    "per_video": per_video,
}

out = Path("multibg_lowres192_posttrain/new_threshold.json")
out.write_text(json.dumps(result, indent=2), encoding="utf-8")

Path("multibg_lowres192_posttrain/NEW_TAU.txt").write_text(
    f"{tau:.10f}\n", encoding="utf-8"
)

print("="*72)
print("NEW MULTI-BACKGROUND LOWRES192 CALIBRATION")
print("="*72)
print(f"tau_ema  : {tau:.6f}")
print(f"n_scores : {len(a)}")
print(f"mean     : {a.mean():.6f}")
print(f"std      : {a.std():.6f}")
print(f"min      : {a.min():.6f}")
print(f"max      : {a.max():.6f}")
print(f"saved    : {out}")
PY

NEW_TAU="$(cat "$OUT/NEW_TAU.txt")"
echo "New calibrated threshold: $NEW_TAU"

# ----------------------------------------------------------------------
# 3) Run the new checkpoint on the four development backgrounds
# ----------------------------------------------------------------------
echo
echo "[3/6] Running the new model on black / metallic / white / wooden..."

declare -A DEV_VIDEOS
DEV_VIDEOS[black]="./videos_background/black/WhatsApp Video 2026-09-18 at 15.15.353.mp4"
DEV_VIDEOS[metallic]="./videos_background/metallic/WhatsApp Video 2026-09-18 at 15.49.53.mp4"
DEV_VIDEOS[white]="./videos_background/white/WhatsApp Video 2026-09-18 at 15.55.11.mp4"
DEV_VIDEOS[wooden]="./videos_background/wooden/WhatsApp Video 2026-09-18 at 15.31.552.mp4"

for BG in black metallic white wooden; do
  VIDEO="${DEV_VIDEOS[$BG]}"
  SAVE="$DEV_OUT/$BG"
  mkdir -p "$SAVE"

  python "$DETECTOR" \
    --video "$VIDEO" \
    --yolo_det_weights "$YOLO" \
    --method lowres \
    --ckpt_dir "$NEW_CKPT" \
    --threshold "$NEW_TAU" \
    --category chewinggum \
    --model_size S \
    --device cuda:0 \
    --fp16 \
    --threshold_basis ema \
    --ema_alpha 0.3 \
    --crop_preprocess current \
    --conf 0.10 \
    --iou 0.50 \
    --pad_ratio 0.02 \
    --clip_margin 5 \
    --min_box_frac 0.5 \
    --batch_frames 16 \
    --batch_crops 96 \
    --gpu_batch_crops 8 \
    --crop_workers 12 \
    --no_detect_unknown \
    --no_video \
    --save_dir "$SAVE"
done

# ----------------------------------------------------------------------
# 4) Summarize score distributions and compare with old threshold
# ----------------------------------------------------------------------
echo
echo "[4/6] Summarizing background score behavior..."

python - <<'PY'
from pathlib import Path
import csv, json, numpy as np

root = Path("multibg_lowres192_posttrain/dev_backgrounds")
tau = float(Path("multibg_lowres192_posttrain/NEW_TAU.txt").read_text().strip())

rows = []
for bg in ["black", "metallic", "white", "wooden"]:
    p = root / bg / "scores.csv"
    vals = []
    track = {}
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            e = (r.get("ema_score") or "").strip()
            if not e:
                continue
            x = float(e)
            vals.append(x)
            try:
                tid = int(float(r["track_id"]))
                track[tid] = max(track.get(tid, -1e30), x)
            except Exception:
                pass

    a = np.asarray(vals, dtype=float)
    t = np.asarray(list(track.values()), dtype=float)
    row = {
        "background": bg,
        "n_scores": int(len(a)),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p95": float(np.quantile(a,.95)),
        "p99": float(np.quantile(a,.99)),
        "max": float(a.max()),
        "frame_fraction_above_tau": float(np.mean(a >= tau)),
        "track_fraction_above_tau": float(np.mean(t >= tau)) if len(t) else 0.0,
    }
    rows.append(row)

print("\nNEW MODEL — BACKGROUND SUMMARY")
print("-"*100)
print(f"{'BG':<10}{'mean':>10}{'p95':>10}{'p99':>10}{'max':>10}{'frames>=tau':>16}{'tracks>=tau':>16}")
for r in rows:
    print(
        f"{r['background']:<10}"
        f"{r['mean']:>10.4f}"
        f"{r['p95']:>10.4f}"
        f"{r['p99']:>10.4f}"
        f"{r['max']:>10.4f}"
        f"{100*r['frame_fraction_above_tau']:>15.2f}%"
        f"{100*r['track_fraction_above_tau']:>15.2f}%"
    )

Path("multibg_lowres192_posttrain/dev_background_summary.json").write_text(
    json.dumps(rows, indent=2), encoding="utf-8"
)
PY

# ----------------------------------------------------------------------
# 5) Create one annotated visual example using the NEW model
# ----------------------------------------------------------------------
echo
echo "[5/6] Creating annotated white-background example..."

python "$DETECTOR" \
  --video "./videos_background/white/WhatsApp Video 2026-09-18 at 15.55.11.mp4" \
  --yolo_det_weights "$YOLO" \
  --method lowres \
  --ckpt_dir "$NEW_CKPT" \
  --threshold "$NEW_TAU" \
  --category chewinggum \
  --model_size S \
  --device cuda:0 \
  --fp16 \
  --threshold_basis ema \
  --ema_alpha 0.3 \
  --crop_preprocess current \
  --conf 0.10 \
  --iou 0.50 \
  --pad_ratio 0.02 \
  --clip_margin 5 \
  --min_box_frac 0.5 \
  --batch_frames 16 \
  --batch_crops 96 \
  --gpu_batch_crops 8 \
  --crop_workers 12 \
  --no_detect_unknown \
  --save_dir "$OUT/annotated_white"

# ----------------------------------------------------------------------
# 6) Final report
# ----------------------------------------------------------------------
echo
echo "[6/6] Final summary"
echo "======================================================================"
echo "POST-TRAIN PIPELINE COMPLETE"
echo "======================================================================"
echo "New checkpoint : $NEW_CKPT"
echo "New tau_ema    : $NEW_TAU"
echo "Calibration    : $OUT/new_threshold.json"
echo "BG summary     : $OUT/dev_background_summary.json"
echo "Annotated video: $OUT/annotated_white/annotated.mp4"
echo
echo "Next scientific step:"
echo "  Compare OLD vs NEW LowRes192 on the same labelled held-out pieces,"
echo "  then fill the report table with Precision/Recall/F1/FPR/Accuracy/AUROC."
echo
echo "Open annotated example with:"
echo "  xdg-open $OUT/annotated_white/annotated.mp4"
