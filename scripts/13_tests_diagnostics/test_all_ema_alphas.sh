#!/usr/bin/env bash
set -euo pipefail

cd /home/ahmed/two-stage-efficientad-inspection

VIDEO="${1:-./videos_background/black/WhatsApp Video 2026-09-18 at 15.15.351.mp4}"
CSV="${2:-./fig22_single_frame/pipeline/scores.csv}"
OUT="${3:-./ema_alpha_comparison}"

rm -rf "$OUT"
mkdir -p "$OUT"

python - "$VIDEO" "$CSV" "$OUT" <<'PY'
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

video_path = Path(sys.argv[1])
csv_path = Path(sys.argv[2])
out_dir = Path(sys.argv[3])

if not video_path.exists():
    raise FileNotFoundError(video_path)
if not csv_path.exists():
    raise FileNotFoundError(csv_path)

df = pd.read_csv(csv_path)

def first_existing(cols):
    for c in cols:
        if c in df.columns:
            return c
    return None

frame_col = first_existing(["frame_idx","frame","frame_id","frame_number"])
track_col = first_existing(["track_id","track","id"])
raw_col = first_existing(["raw_score","score","anom_score","current_score"])
ema_col = first_existing(["ema_score","ema","score_ema"])

if frame_col is None or raw_col is None:
    raise RuntimeError(f"Needed columns not found. CSV columns: {list(df.columns)}")

if track_col is None:
    df["__track__"] = 0
    track_col = "__track__"

# Choose a track that is both long enough and informative.
stats = df.groupby(track_col).agg(
    n=(frame_col,"count"),
    max_raw=(raw_col,"max"),
    mean_raw=(raw_col,"mean")
).reset_index()

stats = stats[stats["n"] >= 8]
if len(stats) == 0:
    stats = df.groupby(track_col).agg(
        n=(frame_col,"count"),
        max_raw=(raw_col,"max"),
        mean_raw=(raw_col,"mean")
    ).reset_index()

stats["score"] = stats["max_raw"] + 0.002*stats["n"]
chosen_track = stats.sort_values("score", ascending=False).iloc[0][track_col]

tdf = df[df[track_col] == chosen_track].copy().sort_values(frame_col).reset_index(drop=True)

raw = tdf[raw_col].astype(float).to_numpy()
frames = tdf[frame_col].astype(int).to_numpy()

def ema(vals, alpha):
    out=[]
    prev=None
    for v in vals:
        prev = float(v) if prev is None else alpha*float(v)+(1-alpha)*prev
        out.append(prev)
    return np.array(out, dtype=float)

alphas = [0.1,0.3,0.5,0.7]
emas = {a: ema(raw,a) for a in alphas}

# --- metrics to compare smoothing behavior ---
rows=[]
for a in alphas:
    e = emas[a]
    # smoothness: mean absolute frame-to-frame change
    mad = float(np.mean(np.abs(np.diff(e)))) if len(e) > 1 else 0.0
    # tracking fidelity to raw score
    mae = float(np.mean(np.abs(raw-e)))
    # peak delay relative to raw peak
    raw_peak_i = int(np.argmax(raw))
    ema_peak_i = int(np.argmax(e))
    delay_frames = int(ema_peak_i - raw_peak_i)
    # correlation
    corr = float(np.corrcoef(raw,e)[0,1]) if len(raw) > 1 else 1.0
    rows.append({
        "alpha": a,
        "mean_abs_step": mad,
        "mae_vs_raw": mae,
        "peak_delay_frames": delay_frames,
        "corr_vs_raw": corr,
        "max_ema": float(np.max(e)),
        "mean_ema": float(np.mean(e)),
    })

metrics = pd.DataFrame(rows)
metrics.to_csv(out_dir/"alpha_comparison_metrics.csv", index=False)

# --- plot 1: raw + all EMA curves ---
plt.figure(figsize=(12,6))
plt.plot(frames, raw, label="Raw score", linewidth=1.5)
for a in alphas:
    plt.plot(frames, emas[a], label=f"EMA α={a}", linewidth=2)
plt.xlabel("Frame index")
plt.ylabel("Anomaly score")
plt.title(f"EMA comparison on track {chosen_track}")
plt.grid(True, alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(out_dir/"ema_alpha_comparison_all.png", dpi=240)
plt.close()

# --- plot 2: separate panels ---
for a in alphas:
    plt.figure(figsize=(10,4.5))
    plt.plot(frames, raw, label="Raw score", linewidth=1.4)
    plt.plot(frames, emas[a], label=f"EMA α={a}", linewidth=2.2)
    plt.xlabel("Frame index")
    plt.ylabel("Anomaly score")
    plt.title(f"Raw vs EMA (α={a}) — track {chosen_track}")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir/f"ema_alpha_{str(a).replace('.','p')}.png", dpi=220)
    plt.close()

# --- choose representative frames based on α=0.3 curve ---
e03 = emas[0.3]

# pick start, rising, peak raw, peak EMA, end
idxs = [
    0,
    max(0, len(frames)//3),
    int(np.argmax(raw)),
    int(np.argmax(e03)),
    len(frames)-1
]
idxs = list(dict.fromkeys(idxs))  # preserve order, remove duplicates

cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

for j,i in enumerate(idxs, start=1):
    fi = int(frames[i])
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ok, frame = cap.read()
    if not ok:
        continue

    label = (
        f"frame={fi}  raw={raw[i]:.3f}  "
        f"a0.1={emas[0.1][i]:.3f}  "
        f"a0.3={emas[0.3][i]:.3f}  "
        f"a0.5={emas[0.5][i]:.3f}  "
        f"a0.7={emas[0.7][i]:.3f}"
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thick = 2
    (tw,th),base = cv2.getTextSize(label,font,scale,thick)
    cv2.rectangle(frame,(8,8),(min(frame.shape[1]-8,16+tw),24+th+base),(255,255,255),-1)
    cv2.putText(frame,label,(12,20+th//2),font,scale,(0,0,0),thick,cv2.LINE_AA)

    cv2.imwrite(str(out_dir/f"frame_{j:02d}_{fi:05d}.jpg"),frame,[cv2.IMWRITE_JPEG_QUALITY,97])

cap.release()

# --- summary ---
best_smooth = metrics.sort_values("mean_abs_step").iloc[0]
best_follow = metrics.sort_values("mae_vs_raw").iloc[0]

with open(out_dir/"summary.txt","w",encoding="utf-8") as f:
    f.write(f"Chosen track: {chosen_track}\n")
    f.write(f"Number of samples: {len(tdf)}\n\n")
    f.write(metrics.to_string(index=False))
    f.write("\n\n")
    f.write(f"Smoothest curve by mean_abs_step: alpha={best_smooth['alpha']}\n")
    f.write(f"Closest to raw score by MAE: alpha={best_follow['alpha']}\n")

print("="*72)
print("EMA ALPHA COMPARISON COMPLETE")
print("="*72)
print("Chosen track:", chosen_track)
print("Samples     :", len(tdf))
print()
print(metrics.to_string(index=False))
print()
print("Outputs:")
print(" ", out_dir/"ema_alpha_comparison_all.png")
for a in alphas:
    print(" ", out_dir/f"ema_alpha_{str(a).replace('.','p')}.png")
print(" ", out_dir/"alpha_comparison_metrics.csv")
print(" ", out_dir/"summary.txt")
print(" ", out_dir/"frame_*.jpg")
print("="*72)
PY
