#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# TEST CLIPPED METHOD ON ONE VIDEO + PICK ONE REAL FRAME
# CONTAINING ALL 3 EXAMPLES FOR FIGURE 22:
#
#   GREEN  = complete / well-contained detection
#   RED    = border-clipped detection
#   ORANGE = unusually small detection
#
# Usage:
#   ./test_clipped_pick_single_frame.sh "/path/to/video.mp4"
#
# Optional:
#   ./test_clipped_pick_single_frame.sh "/path/to/video.mp4" ./fig22_single_frame
# ============================================================

cd /home/ahmed/two-stage-efficientad-inspection

VIDEO="${1:-}"
OUT="${2:-./fig22_single_frame}"

if [[ -z "$VIDEO" ]]; then
  echo "Usage: $0 /path/to/video.mp4 [output_dir]"
  exit 1
fi

if [[ ! -f "$VIDEO" ]]; then
  echo "ERROR: video not found: $VIDEO"
  exit 1
fi

YOLO="./runs/detect/runs/detect/yolo_runs/chewinggum_yolov8n_multibg/weights/best.pt"
CKPT="./ckpt_lowres_192"
DETECTOR="./video_streaming_detector_simple_optimized_v2.py"

THRESHOLD="0.834850"
EMA_ALPHA="0.3"

rm -rf "$OUT"
mkdir -p "$OUT/pipeline" "$OUT/figure22"

echo "======================================================================"
echo "CLIPPED METHOD TEST + SINGLE-FRAME FIGURE 22 EXTRACTION"
echo "======================================================================"
echo "Video      : $VIDEO"
echo "Output     : $OUT"
echo "YOLO       : $YOLO"
echo "Checkpoint : $CKPT"
echo "======================================================================"

# ------------------------------------------------------------
# 1) Run the retained pipeline on the complete video
# ------------------------------------------------------------
echo
echo "[1/2] Running clipped/current-crop inspection pipeline..."

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
  --ema_alpha "$EMA_ALPHA" \
  --batch_frames 16 \
  --batch_crops 96 \
  --gpu_batch_crops 8 \
  --crop_workers 12 \
  --clip_margin 5 \
  --min_box_frac 0.5 \
  --save_dir "$OUT/pipeline"

# ------------------------------------------------------------
# 2) Search for ONE frame containing all 3 examples
# ------------------------------------------------------------
echo
echo "[2/2] Searching for ONE frame containing all three examples..."

python - "$VIDEO" "$YOLO" "$OUT/figure22" <<'PY'
import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

video_path = sys.argv[1]
weights = sys.argv[2]
out_dir = Path(sys.argv[3])
out_dir.mkdir(parents=True, exist_ok=True)

model = YOLO(weights)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {video_path}")

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Slightly wider border zone helps find visually clear clipped detections.
BORDER_PX = max(18, int(0.018 * min(W, H)))

frames = []
all_areas = []

print(f"Video: {W}x{H}, {FPS:.2f} FPS, ~{N} frames")
print("Pass 1: collecting detections and box-size statistics...")

frame_idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break

    result = model.predict(
        source=frame,
        conf=0.25,
        iou=0.45,
        verbose=False,
        device=0
    )[0]

    dets = []
    if result.boxes is not None and len(result.boxes):
        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        confs = result.boxes.conf.detach().cpu().numpy()

        for box, conf in zip(xyxy, confs):
            x1, y1, x2, y2 = map(float, box)
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            area = bw * bh

            edge_dist = min(x1, y1, W-x2, H-y2)
            border = edge_dist <= BORDER_PX

            d = {
                "box": (x1, y1, x2, y2),
                "conf": float(conf),
                "area": float(area),
                "edge_dist": float(edge_dist),
                "border": bool(border),
            }
            dets.append(d)
            all_areas.append(area)

    if dets:
        frames.append((frame_idx, dets))

    frame_idx += 1

cap.release()

if not all_areas:
    raise RuntimeError("No YOLO detections found.")

areas = np.asarray(all_areas, dtype=float)
median_area = float(np.median(areas))
q25 = float(np.quantile(areas, 0.25))
q75 = float(np.quantile(areas, 0.75))

# Definitions used ONLY to select a visually illustrative report frame.
def classify(d):
    # Border/clipped candidate.
    if d["border"] and d["conf"] >= 0.40:
        return "border"

    # Small candidate, compared with global detection-size distribution.
    if (not d["border"]
        and d["area"] <= 0.48 * median_area
        and d["conf"] >= 0.35):
        return "small"

    # Complete / normal-size candidate.
    if (not d["border"]
        and q25 <= d["area"] <= q75
        and d["conf"] >= 0.65):
        return "complete"

    return None

# Score each frame. Strong preference for three distinct detections.
candidates = []

for idx, dets in frames:
    by_type = {"complete": [], "border": [], "small": []}

    for d in dets:
        t = classify(d)
        if t:
            by_type[t].append(d)

    if all(by_type[t] for t in by_type):
        # Best example of each category.
        complete = max(by_type["complete"], key=lambda d: d["conf"])
        border = max(
            by_type["border"],
            key=lambda d: d["conf"] - 0.01 * max(d["edge_dist"], 0)
        )
        small = min(by_type["small"], key=lambda d: d["area"])

        chosen = [complete, border, small]

        # Avoid boxes heavily overlapping one another.
        def iou(a, b):
            ax1, ay1, ax2, ay2 = a["box"]
            bx1, by1, bx2, by2 = b["box"]
            ix1, iy1 = max(ax1,bx1), max(ay1,by1)
            ix2, iy2 = min(ax2,bx2), min(ay2,by2)
            iw, ih = max(0,ix2-ix1), max(0,iy2-iy1)
            inter = iw*ih
            union = a["area"] + b["area"] - inter
            return inter/union if union > 0 else 0

        max_overlap = max(
            iou(chosen[0], chosen[1]),
            iou(chosen[0], chosen[2]),
            iou(chosen[1], chosen[2])
        )

        # Higher = better.
        score = (
            complete["conf"]
            + border["conf"]
            + small["conf"]
            + min(1.0, abs(border["edge_dist"]) / BORDER_PX) * 0.1
            - 2.0 * max_overlap
        )

        candidates.append(
            (score, idx, complete, border, small, max_overlap)
        )

if not candidates:
    print()
    print("No single frame met all three strict criteria.")
    print("Trying a relaxed search...")

    candidates = []

    for idx, dets in frames:
        complete_pool = [
            d for d in dets
            if not d["border"] and d["conf"] >= 0.50
        ]
        border_pool = [
            d for d in dets
            if d["border"] and d["conf"] >= 0.30
        ]
        small_pool = [
            d for d in dets
            if not d["border"] and d["area"] <= 0.65 * median_area
        ]

        if not (complete_pool and border_pool and small_pool):
            continue

        complete = max(
            complete_pool,
            key=lambda d: d["conf"] - abs(d["area"]-median_area)/median_area*0.15
        )
        border = min(
            border_pool,
            key=lambda d: d["edge_dist"]
        )
        small = min(
            small_pool,
            key=lambda d: d["area"]
        )

        # Require three different boxes.
        if complete is small:
            alt = sorted(small_pool, key=lambda d: d["area"])
            small = next((x for x in alt if x is not complete), None)
            if small is None:
                continue

        score = complete["conf"] + border["conf"] + small["conf"]
        candidates.append((score, idx, complete, border, small, 0.0))

if not candidates:
    raise RuntimeError(
        "This video does not appear to contain one frame with all three "
        "examples. Try another video."
    )

candidates.sort(key=lambda x: x[0], reverse=True)
best = candidates[0]
score, best_idx, complete, border, small, overlap = best

print()
print("Best frame found:")
print(f"  frame : {best_idx}")
print(f"  time  : {best_idx/FPS:.2f} s")
print(f"  score : {score:.3f}")

for name, d in [
    ("GREEN complete", complete),
    ("RED border-clipped", border),
    ("ORANGE small", small),
]:
    print(
        f"  {name:20s} conf={d['conf']:.3f} "
        f"area={d['area']:.0f} edge_dist={d['edge_dist']:.1f} "
        f"box={tuple(round(v,1) for v in d['box'])}"
    )

# Read exact frame.
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, best_idx)
ok, frame = cap.read()
cap.release()

if not ok:
    raise RuntimeError(f"Could not retrieve frame {best_idx}")

raw = frame.copy()
annot = frame.copy()

# Colors in BGR.
COLORS = {
    "complete": (0, 190, 0),       # green
    "border":   (0, 0, 255),       # red
    "small":    (0, 140, 255),     # orange
}

selected = [
    ("complete", "Complete detection", complete),
    ("border", "Border-clipped detection", border),
    ("small", "Small detection", small),
]

# Draw boxes with label backgrounds that do not overlap the objects.
for kind, label, d in selected:
    x1, y1, x2, y2 = [int(round(v)) for v in d["box"]]

    # Clamp drawing coordinates so red border stays visible.
    x1d = max(4, x1)
    y1d = max(4, y1)
    x2d = min(W-5, x2)
    y2d = min(H-5, y2)

    color = COLORS[kind]
    cv2.rectangle(annot, (x1d,y1d), (x2d,y2d), color, 5)

    text = f"{label} ({d['conf']:.2f})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.95
    thick = 2
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)

    # Put label above if possible, otherwise below.
    tx = max(8, min(x1d, W-tw-16))
    if y1d - th - 18 >= 0:
        ty = y1d - 10
    else:
        ty = min(H-th-8, y2d + th + 14)

    cv2.rectangle(
        annot,
        (tx-5, ty-th-7),
        (tx+tw+5, ty+base+5),
        (255,255,255),
        -1
    )
    cv2.putText(
        annot, text, (tx,ty),
        font, scale, color, thick, cv2.LINE_AA
    )

# Save raw and annotated frame.
raw_path = out_dir / f"FIG22_frame_{best_idx:05d}_RAW.jpg"
ann_path = out_dir / f"FIG22_frame_{best_idx:05d}_ANNOTATED.jpg"

cv2.imwrite(str(raw_path), raw, [cv2.IMWRITE_JPEG_QUALITY, 98])
cv2.imwrite(str(ann_path), annot, [cv2.IMWRITE_JPEG_QUALITY, 98])

# Also save one clean crop centered around all 3 boxes, with generous context.
boxes = [complete["box"], border["box"], small["box"]]
ux1 = min(b[0] for b in boxes)
uy1 = min(b[1] for b in boxes)
ux2 = max(b[2] for b in boxes)
uy2 = max(b[3] for b in boxes)

pad_x = max(100, int(0.15*(ux2-ux1)))
pad_y = max(100, int(0.15*(uy2-uy1)))

cx1 = max(0, int(ux1)-pad_x)
cy1 = max(0, int(uy1)-pad_y)
cx2 = min(W, int(ux2)+pad_x)
cy2 = min(H, int(uy2)+pad_y)

crop = annot[cy1:cy2, cx1:cx2]
crop_path = out_dir / f"FIG22_frame_{best_idx:05d}_REPORT_CROP.jpg"
cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 98])

print()
print("Saved:")
print("  RAW       :", raw_path)
print("  ANNOTATED :", ann_path)
print("  REPORT    :", crop_path)
print()
print("Use REPORT_CROP first for Figure 22.")
PY

echo
echo "======================================================================"
echo "DONE"
echo "======================================================================"
echo "Open:"
echo "  $OUT/figure22/"
echo
echo "Preferred report image:"
echo "  FIG22_*_REPORT_CROP.jpg"
echo "======================================================================"
