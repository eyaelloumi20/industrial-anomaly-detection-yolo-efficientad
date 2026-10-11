#!/usr/bin/env bash
set -euo pipefail

cd /home/ahmed/two-stage-efficientad-inspection

VIDEO="${1:-./videos_background/black/WhatsApp Video 2026-09-18 at 15.15.351.mp4}"
CSV="${2:-./ema_alpha_comparison_fixed/selected_track_with_all_ema.csv}"
OUT="${3:-./ema_final_qualitative}"
TRACK_ID="${4:-16}"

rm -rf "$OUT"
mkdir -p "$OUT"

python - "$VIDEO" "$CSV" "$OUT" "$TRACK_ID" <<'PY'
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg
import cv2

video_path = Path(sys.argv[1])
csv_path = Path(sys.argv[2])
out_dir = Path(sys.argv[3])
track_id = float(sys.argv[4])

if not video_path.exists():
    raise FileNotFoundError(video_path)
if not csv_path.exists():
    raise FileNotFoundError(csv_path)

df = pd.read_csv(csv_path)

# Identify columns robustly
def first_existing(cols, names):
    for n in names:
        if n in cols:
            return n
    return None

cols = list(df.columns)
frame_col = first_existing(cols, ["frame_idx","frame","frame_id","frame_number"])
track_col = first_existing(cols, ["track_id","track","id"])
raw_col = first_existing(cols, ["raw_score","score","anom_score","current_score"])
ema_col = first_existing(cols, ["ema_0.3","ema_0_3","ema03","ema_score","ema"])

if frame_col is None or raw_col is None or ema_col is None:
    raise RuntimeError(f"Required columns missing. Columns: {cols}")

# If track column is absent because CSV is already selected-track-only, just use all rows.
if track_col is not None:
    tdf = df[pd.to_numeric(df[track_col], errors="coerce") == track_id].copy()
    if tdf.empty:
        # fallback to all rows if already filtered to one track
        tdf = df.copy()
else:
    tdf = df.copy()

tdf[frame_col] = pd.to_numeric(tdf[frame_col], errors="coerce")
tdf[raw_col] = pd.to_numeric(tdf[raw_col], errors="coerce")
tdf[ema_col] = pd.to_numeric(tdf[ema_col], errors="coerce")
tdf = tdf[tdf[frame_col].notna()].sort_values(frame_col).reset_index(drop=True)
tdf[frame_col] = tdf[frame_col].astype(int)

# Try to use bbox columns from CSV if present.
x1c = first_existing(cols, ["x1","xmin","left"])
y1c = first_existing(cols, ["y1","ymin","top"])
x2c = first_existing(cols, ["x2","xmax","right"])
y2c = first_existing(cols, ["y2","ymax","bottom"])
has_bbox = all(c is not None for c in [x1c,y1c,x2c,y2c])

# Select representative valid-score frames around the same regions you used previously.
valid = tdf[tdf[raw_col].notna() & tdf[ema_col].notna()].copy()
if len(valid) < 4:
    raise RuntimeError("Not enough valid points to build qualitative figure.")

# Prefer the earlier chosen frames if available, otherwise derive them.
preferred = [92, 121, 226, 232]
chosen_rows = []

for pf in preferred:
    cand = valid.iloc[(valid[frame_col] - pf).abs().argsort()[:1]]
    if not cand.empty:
        chosen_rows.append(cand.iloc[0])

# ensure uniqueness
uniq = []
seen = set()
for r in chosen_rows:
    fi = int(r[frame_col])
    if fi not in seen:
        seen.add(fi)
        uniq.append(r)

# fallback if needed
if len(uniq) < 4:
    raw_vals = valid[raw_col].to_numpy(dtype=float)
    ema_vals = valid[ema_col].to_numpy(dtype=float)
    idxs = [
        int(np.argmax(raw_vals)),
        int(np.argmin(raw_vals)),
        int(len(valid)*0.65),
        int(np.argmax(ema_vals)),
    ]
    for i in idxs:
        r = valid.iloc[i]
        fi = int(r[frame_col])
        if fi not in seen:
            seen.add(fi)
            uniq.append(r)
        if len(uniq) == 4:
            break

chosen_rows = uniq[:4]

# Open video once and save annotated frames.
cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_files = []

# If bbox is absent from selected_track CSV, attempt to read sibling original scores.csv
# to recover bbox for same track/frame.
orig_scores_candidates = [
    Path("./fig22_single_frame/pipeline/scores.csv"),
    csv_path.parent.parent / "fig22_single_frame" / "pipeline" / "scores.csv",
]
orig_df = None
for p in orig_scores_candidates:
    if p.exists():
        try:
            temp = pd.read_csv(p)
            if track_col in temp.columns and frame_col in temp.columns:
                orig_df = temp
                break
        except Exception:
            pass

for j,row in enumerate(chosen_rows, start=1):
    fi = int(row[frame_col])
    raw = float(row[raw_col])
    ema = float(row[ema_col])

    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ok, frame = cap.read()
    if not ok:
        continue

    bbox = None

    if has_bbox:
        try:
            vals = [row[x1c], row[y1c], row[x2c], row[y2c]]
            if all(pd.notna(v) for v in vals):
                bbox = tuple(map(float, vals))
        except Exception:
            bbox = None

    if bbox is None and orig_df is not None:
        # robust match against original scores file
        odf = orig_df.copy()
        odf[frame_col] = pd.to_numeric(odf[frame_col], errors="coerce")
        if track_col in odf.columns:
            odf[track_col] = pd.to_numeric(odf[track_col], errors="coerce")
            hit = odf[
                (odf[frame_col] == fi) &
                (odf[track_col] == track_id)
            ]
        else:
            hit = odf[odf[frame_col] == fi]

        if not hit.empty:
            hr = hit.iloc[0]
            x1o = first_existing(list(odf.columns), ["x1","xmin","left"])
            y1o = first_existing(list(odf.columns), ["y1","ymin","top"])
            x2o = first_existing(list(odf.columns), ["x2","xmax","right"])
            y2o = first_existing(list(odf.columns), ["y2","ymax","bottom"])
            if all(c is not None for c in [x1o,y1o,x2o,y2o]):
                vals = [hr[x1o], hr[y1o], hr[x2o], hr[y2o]]
                if all(pd.notna(v) for v in vals):
                    bbox = tuple(map(float, vals))

    # Draw track box if available.
    if bbox is not None:
        x1,y1,x2,y2 = [int(round(v)) for v in bbox]
        x1 = max(3,x1); y1=max(3,y1); x2=min(W-4,x2); y2=min(H-4,y2)

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,180,255),4)

        label = f"Track {int(track_id)}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.75
        thick = 2
        (tw,th),base = cv2.getTextSize(label,font,scale,thick)
        tx = max(6,min(x1,W-tw-12))
        ty = max(th+8,y1-8)

        cv2.rectangle(frame,(tx-4,ty-th-5),(tx+tw+4,ty+base+4),(255,255,255),-1)
        cv2.putText(frame,label,(tx,ty),font,scale,(0,120,220),thick,cv2.LINE_AA)

        # Crop around tracked object with context
        pad_x = max(100, int(1.5*(x2-x1)))
        pad_y = max(90, int(1.5*(y2-y1)))
        cx=(x1+x2)//2; cy=(y1+y2)//2
        rx1=max(0,cx-pad_x); ry1=max(0,cy-pad_y)
        rx2=min(W,cx+pad_x); ry2=min(H,cy+pad_y)
        shown = frame[ry1:ry2,rx1:rx2]
    else:
        # fallback: full frame
        shown = frame

    outp = out_dir / f"qual_track{int(track_id)}_{j:02d}_frame_{fi:05d}.jpg"
    cv2.imwrite(str(outp), shown, [cv2.IMWRITE_JPEG_QUALITY,98])
    frame_files.append((outp, fi, raw, ema))

cap.release()

# Build publication-ready composite.
frames_arr = tdf[frame_col].to_numpy(dtype=int)
raw_arr = tdf[raw_col].to_numpy(dtype=float)
ema_arr = tdf[ema_col].to_numpy(dtype=float)

fig = plt.figure(figsize=(15,9))
gs = GridSpec(
    2, 4,
    height_ratios=[1.0,1.25],
    hspace=0.26,
    wspace=0.08
)

for j,(fp,fi,raw,ema) in enumerate(frame_files[:4]):
    ax = fig.add_subplot(gs[0,j])
    ax.imshow(mpimg.imread(fp))
    ax.axis("off")
    ax.set_title(
        f"Frame {fi}\nRaw = {raw:.3f}   |   EMA = {ema:.3f}",
        fontsize=11
    )

axg = fig.add_subplot(gs[1,:])
axg.plot(frames_arr, raw_arr, label="Raw score", linewidth=1.35, alpha=0.55)
axg.plot(frames_arr, ema_arr, label=r"EMA ($\alpha=0.3$)", linewidth=2.7)

for _,fi,raw,ema in frame_files[:4]:
    axg.axvline(fi, linestyle="--", alpha=0.45, linewidth=1.3)

axg.set_xlabel("Frame index")
axg.set_ylabel("Anomaly score")
axg.set_title("Qualitative effect of EMA smoothing on the tracked object")
axg.grid(True, alpha=0.22)
axg.legend()

fig.tight_layout()
final_path = out_dir / "ema_qualitative_track16_final.png"
fig.savefig(final_path, dpi=280, bbox_inches="tight")
plt.close(fig)

print("="*72)
print("FINAL EMA QUALITATIVE FIGURE GENERATED")
print("="*72)
print("Output:", final_path)
print("Frames:")
for item in frame_files:
    print(" ", item[0])
print("="*72)
PY

echo
echo "Done."
echo "Open:"
echo "  $OUT/ema_qualitative_track16_final.png"
