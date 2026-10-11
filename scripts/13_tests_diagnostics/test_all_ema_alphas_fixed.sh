#!/usr/bin/env bash
set -euo pipefail

cd /home/ahmed/two-stage-efficientad-inspection

VIDEO="${1:-./videos_background/black/WhatsApp Video 2026-09-18 at 15.15.351.mp4}"
CSV="${2:-./fig22_single_frame/pipeline/scores.csv}"
OUT="${3:-./ema_alpha_comparison_fixed}"

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

def first_existing(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None

cols = list(df.columns)
frame_col = first_existing(cols, ["frame_idx","frame","frame_id","frame_number"])
track_col = first_existing(cols, ["track_id","track","id"])
raw_col = first_existing(cols, ["raw_score","score","anom_score","current_score"])
ema_existing_col = first_existing(cols, ["ema_score","ema","score_ema"])

if frame_col is None or raw_col is None:
    raise RuntimeError(f"Could not identify required columns. CSV columns: {cols}")

if track_col is None:
    df["__track__"] = 0
    track_col = "__track__"

df[raw_col] = pd.to_numeric(df[raw_col], errors="coerce")
df[frame_col] = pd.to_numeric(df[frame_col], errors="coerce")
df = df[df[frame_col].notna()].copy()
df[frame_col] = df[frame_col].astype(int)

# ------------------------------------------------------------------
# Pick a useful track based on VALID scored rows, not all rows.
# ------------------------------------------------------------------
track_stats = []
for tid, g in df.groupby(track_col):
    gv = g[g[raw_col].notna()].copy()
    if len(gv) < 20:
        continue
    vals = gv[raw_col].to_numpy(dtype=float)
    track_stats.append({
        "track_id": tid,
        "valid_n": len(gv),
        "span": int(gv[frame_col].max() - gv[frame_col].min()),
        "max_raw": float(np.max(vals)),
        "min_raw": float(np.min(vals)),
        "std_raw": float(np.std(vals)),
        "range_raw": float(np.max(vals)-np.min(vals)),
    })

if not track_stats:
    raise RuntimeError("No track has at least 20 valid anomaly scores.")

stats = pd.DataFrame(track_stats)

# Prefer tracks with useful variation and enough temporal length.
stats["selection_score"] = (
    2.0 * stats["range_raw"] +
    0.8 * stats["std_raw"] +
    0.002 * stats["valid_n"] +
    0.0005 * stats["span"]
)

chosen_track = stats.sort_values("selection_score", ascending=False).iloc[0]["track_id"]

tdf = df[df[track_col] == chosen_track].copy().sort_values(frame_col).reset_index(drop=True)
raw = tdf[raw_col].to_numpy(dtype=float)
frames = tdf[frame_col].to_numpy(dtype=int)

# ------------------------------------------------------------------
# Runtime-faithful EMA:
# - before first valid score: NaN
# - valid score: update EMA
# - missing score later: hold previous EMA, exactly like no new evidence
# ------------------------------------------------------------------
def ema_hold_missing(vals, alpha):
    out = np.full(len(vals), np.nan, dtype=float)
    prev = np.nan
    for i, v in enumerate(vals):
        if np.isfinite(v):
            if not np.isfinite(prev):
                prev = float(v)
            else:
                prev = alpha * float(v) + (1.0-alpha) * prev
        out[i] = prev
    return out

alphas = [0.1, 0.3, 0.5, 0.7]
emas = {a: ema_hold_missing(raw, a) for a in alphas}

# ------------------------------------------------------------------
# Quantitative comparison only where raw is valid.
# ------------------------------------------------------------------
rows = []
valid_raw = np.isfinite(raw)

for a in alphas:
    e = emas[a]
    valid = valid_raw & np.isfinite(e)
    rv = raw[valid]
    ev = e[valid]

    if len(rv) < 2:
        raise RuntimeError(f"Too few valid values for alpha={a}")

    # Smoothness measured over EMA samples in chronological order.
    finite_e = e[np.isfinite(e)]
    mean_abs_step = float(np.mean(np.abs(np.diff(finite_e)))) if len(finite_e) > 1 else np.nan
    mae_vs_raw = float(np.mean(np.abs(rv-ev)))
    corr = float(np.corrcoef(rv,ev)[0,1]) if np.std(rv) > 0 and np.std(ev) > 0 else np.nan

    # Peak delay in FRAME units, not row units.
    raw_valid_indices = np.where(valid_raw)[0]
    raw_peak_i = raw_valid_indices[int(np.nanargmax(raw[raw_valid_indices]))]
    ema_valid_indices = np.where(np.isfinite(e))[0]
    ema_peak_i = ema_valid_indices[int(np.nanargmax(e[ema_valid_indices]))]
    peak_delay_frames = int(frames[ema_peak_i] - frames[raw_peak_i])

    rows.append({
        "alpha": a,
        "mean_abs_step": mean_abs_step,
        "mae_vs_raw": mae_vs_raw,
        "peak_delay_frames": peak_delay_frames,
        "corr_vs_raw": corr,
        "max_ema": float(np.nanmax(e)),
        "mean_ema": float(np.nanmean(e)),
        "valid_raw_scores": int(valid_raw.sum()),
    })

metrics = pd.DataFrame(rows)
metrics.to_csv(out_dir/"alpha_comparison_metrics.csv", index=False)

# ------------------------------------------------------------------
# Main comparison plot
# ------------------------------------------------------------------
plt.figure(figsize=(12,6))
plt.plot(frames, raw, label="Raw score", linewidth=1.2, alpha=0.55)

for a in alphas:
    lw = 2.8 if a == 0.3 else 1.9
    plt.plot(frames, emas[a], label=f"EMA α={a}", linewidth=lw)

plt.xlabel("Frame index")
plt.ylabel("Anomaly score")
plt.title(f"Effect of EMA smoothing — track {chosen_track}")
plt.grid(True, alpha=0.22)
plt.legend()
plt.tight_layout()
plt.savefig(out_dir/"ema_alpha_comparison_all_fixed.png", dpi=260)
plt.close()

# Separate plots.
for a in alphas:
    plt.figure(figsize=(10,4.8))
    plt.plot(frames, raw, label="Raw score", linewidth=1.3, alpha=0.65)
    plt.plot(frames, emas[a], label=f"EMA α={a}", linewidth=2.3)
    plt.xlabel("Frame index")
    plt.ylabel("Anomaly score")
    plt.title(f"Raw score vs EMA (α={a}) — track {chosen_track}")
    plt.grid(True, alpha=0.22)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir/f"ema_alpha_{str(a).replace('.','p')}_fixed.png", dpi=230)
    plt.close()

# ------------------------------------------------------------------
# Qualitative figure using real frames and the actual α=0.3 curve.
# Pick four valid points: high raw spike, after spike, middle, later fluctuation.
# ------------------------------------------------------------------
e03 = emas[0.3]
valid_idx = np.where(valid_raw & np.isfinite(e03))[0]

raw_peak_idx = valid_idx[int(np.argmax(raw[valid_idx]))]

# Pick a later low point after peak if possible.
after = valid_idx[valid_idx > raw_peak_idx]
if len(after):
    after_low_idx = after[int(np.argmin(raw[after[:min(len(after),40)]]))]
else:
    after_low_idx = valid_idx[len(valid_idx)//3]

# Pick max raw in second half to show another fluctuation.
second_half = valid_idx[valid_idx >= valid_idx[len(valid_idx)//2]]
late_peak_idx = second_half[int(np.argmax(raw[second_half]))] if len(second_half) else valid_idx[-1]

# One stable later frame.
stable_idx = valid_idx[min(len(valid_idx)-1, int(len(valid_idx)*0.75))]

chosen_indices = []
for i in [raw_peak_idx, after_low_idx, stable_idx, late_peak_idx]:
    if int(i) not in chosen_indices:
        chosen_indices.append(int(i))
chosen_indices = chosen_indices[:4]

cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

frame_paths = []
for j, i in enumerate(chosen_indices, start=1):
    fi = int(frames[i])
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ok, frame = cap.read()
    if not ok:
        continue

    outp = out_dir / f"qual_frame_{j:02d}_{fi:05d}.jpg"
    cv2.imwrite(str(outp), frame, [cv2.IMWRITE_JPEG_QUALITY, 98])
    frame_paths.append((outp, i))

cap.release()

# Build one final qualitative composite with frames on top + graph below.
if frame_paths:
    from matplotlib.gridspec import GridSpec
    import matplotlib.image as mpimg

    fig = plt.figure(figsize=(15,9))
    gs = GridSpec(2, len(frame_paths), height_ratios=[1,1.25], hspace=0.22, wspace=0.08)

    for j,(fp,i) in enumerate(frame_paths):
        ax = fig.add_subplot(gs[0,j])
        ax.imshow(mpimg.imread(fp))
        ax.axis("off")
        ax.set_title(
            f"Frame {frames[i]}\nRaw={raw[i]:.3f} | EMA(0.3)={e03[i]:.3f}",
            fontsize=11
        )

    axg = fig.add_subplot(gs[1,:])
    axg.plot(frames, raw, label="Raw score", linewidth=1.3, alpha=0.55)
    axg.plot(frames, e03, label="EMA α=0.3", linewidth=2.6)

    for fp,i in frame_paths:
        axg.axvline(frames[i], linestyle="--", alpha=0.45)

    axg.set_xlabel("Frame index")
    axg.set_ylabel("Anomaly score")
    axg.set_title("Real-frame qualitative view of EMA smoothing")
    axg.grid(True, alpha=0.22)
    axg.legend()

    fig.tight_layout()
    fig.savefig(out_dir/"ema_qualitative_real_frames_alpha03.png", dpi=260)
    plt.close(fig)

# ------------------------------------------------------------------
# Save cleaned track data for auditability.
# ------------------------------------------------------------------
out_track = tdf[[frame_col, track_col, raw_col]].copy()
for a in alphas:
    out_track[f"ema_{a}"] = emas[a]
out_track.to_csv(out_dir/"selected_track_with_all_ema.csv", index=False)

# Summary
smoothest = metrics.loc[metrics["mean_abs_step"].idxmin()]
closest = metrics.loc[metrics["mae_vs_raw"].idxmin()]

with open(out_dir/"summary_fixed.txt","w",encoding="utf-8") as f:
    f.write(f"Chosen track: {chosen_track}\n")
    f.write(f"Rows in track: {len(tdf)}\n")
    f.write(f"Valid raw scores: {int(valid_raw.sum())}\n")
    f.write(f"Missing raw scores: {int((~valid_raw).sum())}\n\n")
    f.write(metrics.to_string(index=False))
    f.write("\n\n")
    f.write(f"Smoothest EMA by mean_abs_step: alpha={smoothest['alpha']}\n")
    f.write(f"Closest to raw by MAE: alpha={closest['alpha']}\n")

print("="*80)
print("EMA ALPHA COMPARISON — FIXED")
print("="*80)
print(f"Chosen track      : {chosen_track}")
print(f"Rows              : {len(tdf)}")
print(f"Valid raw scores  : {int(valid_raw.sum())}")
print(f"Missing raw scores: {int((~valid_raw).sum())}")
print()
print(metrics.to_string(index=False))
print()
print("Key outputs:")
print(" ", out_dir/"ema_alpha_comparison_all_fixed.png")
print(" ", out_dir/"ema_qualitative_real_frames_alpha03.png")
print(" ", out_dir/"alpha_comparison_metrics.csv")
print(" ", out_dir/"summary_fixed.txt")
print("="*80)
PY
