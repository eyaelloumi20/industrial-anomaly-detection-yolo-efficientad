#!/usr/bin/env bash
set -euo pipefail

cd /home/ahmed/two-stage-efficientad-inspection

OUT="${1:-./fig22_best_single_frame_search}"
YOLO="./runs/detect/runs/detect/yolo_runs/chewinggum_yolov8n_multibg/weights/best.pt"

rm -rf "$OUT"
mkdir -p "$OUT"

python - "$YOLO" "$OUT" <<'PY'
import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

weights = sys.argv[1]
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

videos = [
    ("black_01", "./videos_background/black/WhatsApp Video 2026-09-18 at 15.14.22.mp4"),
    ("black_02", "./videos_background/black/WhatsApp Video 2026-09-18 at 15.15.35.mp4"),
    ("black_03", "./videos_background/black/WhatsApp Video 2026-09-18 at 15.15.352.mp4"),

    ("metallic_01", "./videos_background/metallic/WhatsApp Video 2026-09-18 at 15.47.23.mp4"),
    ("metallic_02", "./videos_background/metallic/WhatsApp Video 2026-09-18 at 15.49.36.mp4"),

    ("white_01", "./videos_background/white/WhatsApp Video 2026-09-18 at 15.55.45.mp4"),
    ("white_02", "./videos_background/white/WhatsApp Video 2026-09-18 at 15.55.46.mp4"),
    ("white_03", "./videos_background/white/WhatsApp Video 2026-09-18 at 15.55.461.mp4"),

    ("wooden_01", "./videos_background/wooden/WhatsApp Video 2026-09-18 at 15.31.55.mp4"),
    ("wooden_02", "./videos_background/wooden/WhatsApp Video 2026-09-18 at 15.31.551.mp4"),
    ("wooden_03", "./videos_background/wooden/WhatsApp Video 2026-09-18 at 15.31.56.mp4"),
    ("wooden_04", "./videos_background/wooden/WhatsApp Video 2026-09-18 at 15.31.561.mp4"),
]

model = YOLO(weights)

def iou(a, b):
    ax1, ay1, ax2, ay2 = a["box"]
    bx1, by1, bx2, by2 = b["box"]
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0,ix2-ix1), max(0,iy2-iy1)
    inter = iw*ih
    union = a["area"] + b["area"] - inter
    return inter/union if union > 0 else 0.0

global_candidates = []

for tag, video_path in videos:
    p = Path(video_path)
    if not p.exists():
        print(f"[SKIP] Missing: {video_path}")
        continue

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[SKIP] Cannot open: {video_path}")
        continue

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    BORDER_PX = max(18, int(0.018 * min(W, H)))

    print(f"\n=== {tag} ===")
    print(f"{video_path}")
    print(f"{W}x{H} @ {FPS:.2f} fps, ~{N} frames")

    # First pass: gather frame detections and global area stats for this video.
    frames = []
    all_areas = []
    fi = 0
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
                x1,y1,x2,y2 = map(float, box)
                bw = max(1.0, x2-x1)
                bh = max(1.0, y2-y1)
                area = bw*bh
                edge_dist = min(x1, y1, W-x2, H-y2)
                dets.append({
                    "box": (x1,y1,x2,y2),
                    "conf": float(conf),
                    "area": area,
                    "edge_dist": edge_dist,
                    "border": edge_dist <= BORDER_PX,
                })
                all_areas.append(area)

        if dets:
            frames.append((fi, dets))
        fi += 1

    cap.release()

    if not all_areas:
        print("No detections.")
        continue

    areas = np.asarray(all_areas, dtype=float)
    median_area = float(np.median(areas))
    q25 = float(np.quantile(areas, 0.25))
    q75 = float(np.quantile(areas, 0.75))

    best_for_video = None

    for fi, dets in frames:
        complete_pool = [
            d for d in dets
            if (not d["border"] and q25 <= d["area"] <= q75 and d["conf"] >= 0.60)
        ]
        border_pool = [
            d for d in dets
            if d["border"] and d["conf"] >= 0.35
        ]
        small_pool = [
            d for d in dets
            if (not d["border"] and d["area"] <= 0.55*median_area and d["conf"] >= 0.30)
        ]

        if not (complete_pool and border_pool and small_pool):
            continue

        # Try combinations and favor separated boxes.
        for complete in sorted(complete_pool, key=lambda d: -d["conf"])[:4]:
            for border in sorted(border_pool, key=lambda d: d["edge_dist"])[:4]:
                for small in sorted(small_pool, key=lambda d: d["area"])[:4]:
                    # distinct detections
                    if complete is border or complete is small or border is small:
                        continue

                    overlaps = [iou(complete,border), iou(complete,small), iou(border,small)]
                    max_ov = max(overlaps)

                    # Reject visually messy overlaps.
                    if max_ov > 0.12:
                        continue

                    # Prefer good confidence, very edge-near border case, and small box.
                    small_ratio = small["area"] / max(median_area, 1)
                    edge_bonus = max(0.0, (BORDER_PX - border["edge_dist"]) / BORDER_PX)
                    score = (
                        complete["conf"]
                        + border["conf"]
                        + small["conf"]
                        + 0.25*edge_bonus
                        + 0.20*(1.0-small_ratio)
                        - 2.0*max_ov
                    )

                    item = {
                        "score": score,
                        "tag": tag,
                        "video": video_path,
                        "frame": fi,
                        "fps": FPS,
                        "W": W,
                        "H": H,
                        "complete": complete,
                        "border": border,
                        "small": small,
                        "median_area": median_area,
                    }

                    if best_for_video is None or score > best_for_video["score"]:
                        best_for_video = item

    if best_for_video:
        print(
            f"Candidate found: frame={best_for_video['frame']} "
            f"time={best_for_video['frame']/FPS:.2f}s "
            f"score={best_for_video['score']:.3f}"
        )
        global_candidates.append(best_for_video)
    else:
        print("No frame with all three separated examples.")

if not global_candidates:
    raise RuntimeError(
        "None of the searched videos contains one usable frame with "
        "complete + border-clipped + small detections. Try additional videos."
    )

best = max(global_candidates, key=lambda x: x["score"])

print("\n" + "="*72)
print("BEST FIGURE 22 FRAME")
print("="*72)
print("video :", best["video"])
print("tag   :", best["tag"])
print("frame :", best["frame"])
print("time  :", f"{best['frame']/best['fps']:.2f} s")
print("score :", f"{best['score']:.3f}")
print("="*72)

# Retrieve best frame
cap = cv2.VideoCapture(best["video"])
cap.set(cv2.CAP_PROP_POS_FRAMES, best["frame"])
ok, frame = cap.read()
cap.release()
if not ok:
    raise RuntimeError("Could not retrieve selected frame.")

raw = frame.copy()
annot = frame.copy()
W, H = best["W"], best["H"]

COLORS = {
    "complete": (0, 190, 0),
    "border": (0, 0, 255),
    "small": (0, 140, 255),
}
LABELS = {
    "complete": "Complete",
    "border": "Clipped",
    "small": "Small",
}

for kind in ("complete", "border", "small"):
    d = best[kind]
    x1,y1,x2,y2 = [int(round(v)) for v in d["box"]]
    x1d, y1d = max(4,x1), max(4,y1)
    x2d, y2d = min(W-5,x2), min(H-5,y2)
    color = COLORS[kind]

    cv2.rectangle(annot, (x1d,y1d), (x2d,y2d), color, 4)

    text = LABELS[kind]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.75
    thick = 2
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)

    tx = max(8, min(x1d, W-tw-16))
    if y1d - th - 14 >= 0:
        ty = y1d - 8
    else:
        ty = min(H-th-8, y2d + th + 12)

    cv2.rectangle(
        annot,
        (tx-4,ty-th-6),
        (tx+tw+4,ty+base+4),
        (255,255,255),
        -1
    )
    cv2.putText(
        annot, text, (tx,ty),
        font, scale, color, thick, cv2.LINE_AA
    )

raw_path = out_dir / "FIG22_best_RAW.jpg"
ann_path = out_dir / "FIG22_best_ANNOTATED.jpg"
cv2.imwrite(str(raw_path), raw, [cv2.IMWRITE_JPEG_QUALITY, 98])
cv2.imwrite(str(ann_path), annot, [cv2.IMWRITE_JPEG_QUALITY, 98])

# Crop around all three examples.
boxes = [best[k]["box"] for k in ("complete","border","small")]
ux1=min(b[0] for b in boxes); uy1=min(b[1] for b in boxes)
ux2=max(b[2] for b in boxes); uy2=max(b[3] for b in boxes)

pad_x=max(100,int(0.18*(ux2-ux1)))
pad_y=max(90,int(0.18*(uy2-uy1)))

cx1=max(0,int(ux1)-pad_x); cy1=max(0,int(uy1)-pad_y)
cx2=min(W,int(ux2)+pad_x); cy2=min(H,int(uy2)+pad_y)

crop = annot[cy1:cy2,cx1:cx2]
crop_path = out_dir / "FIG22_best_REPORT_CROP.jpg"
cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 98])

(out_dir / "selection.txt").write_text(
    f"video={best['video']}\n"
    f"tag={best['tag']}\n"
    f"frame={best['frame']}\n"
    f"time_seconds={best['frame']/best['fps']:.4f}\n"
    f"score={best['score']:.6f}\n",
    encoding="utf-8"
)

print("\nSaved:")
print(" ", raw_path)
print(" ", ann_path)
print(" ", crop_path)
print("\nUse FIG22_best_REPORT_CROP.jpg first.")
PY

echo
echo "======================================================================"
echo "SEARCH FINISHED"
echo "======================================================================"
echo "Open:"
echo "  $OUT/FIG22_best_REPORT_CROP.jpg"
echo
echo "Selection details:"
echo "  $OUT/selection.txt"
echo "======================================================================"
