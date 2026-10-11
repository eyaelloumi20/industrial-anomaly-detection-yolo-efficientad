#!/usr/bin/env python3
import argparse
from pathlib import Path
import cv2
import pandas as pd
import matplotlib.pyplot as plt


def find_first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def compute_ema(values, alpha=0.3):
    ema = []
    prev = None
    for v in values:
        if prev is None:
            prev = v
        else:
            prev = alpha * v + (1 - alpha) * prev
        ema.append(prev)
    return ema


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_frame(video_path, frame_idx, out_path, crop_box=None, margin=20):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        print(f"[WARN] Could not read frame {frame_idx}")
        return False

    if crop_box is not None:
        x1, y1, x2, y2 = crop_box
        h, w = frame.shape[:2]
        x1 = max(0, int(x1) - margin)
        y1 = max(0, int(y1) - margin)
        x2 = min(w, int(x2) + margin)
        y2 = min(h, int(y2) + margin)
        frame = frame[y1:y2, x1:x2]

    cv2.imwrite(str(out_path), frame)
    return True


def main():
    parser = argparse.ArgumentParser(description="Draw EMA graph and extract representative frames from a video.")
    parser.add_argument("--video", required=True, help="Path to source video")
    parser.add_argument("--csv", required=True, help="Path to scores.csv")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--alpha", type=float, default=0.3, help="EMA alpha if EMA not already in CSV")
    parser.add_argument("--threshold", type=float, default=None, help="Optional anomaly threshold")
    parser.add_argument("--track_id", type=int, default=None, help="Optional specific track ID")
    parser.add_argument("--margin", type=int, default=20, help="Crop margin if bbox exists")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    df = pd.read_csv(args.csv)

    # Detect columns robustly
    track_col = find_first_existing(df, ["track_id", "track", "id"])
    frame_col = find_first_existing(df, ["frame_idx", "frame", "frame_id", "frame_number"])
    score_col = find_first_existing(df, ["score", "raw_score", "anom_score", "current_score"])
    ema_col = find_first_existing(df, ["ema", "ema_score", "score_ema"])

    x1_col = find_first_existing(df, ["x1", "xmin", "left"])
    y1_col = find_first_existing(df, ["y1", "ymin", "top"])
    x2_col = find_first_existing(df, ["x2", "xmax", "right"])
    y2_col = find_first_existing(df, ["y2", "ymax", "bottom"])

    if frame_col is None:
        raise RuntimeError(f"Could not find frame column in CSV. Columns are: {list(df.columns)}")
    if score_col is None:
        raise RuntimeError(f"Could not find score column in CSV. Columns are: {list(df.columns)}")

    if track_col is None:
        # fake single-track case
        track_col = "__track_id__"
        df[track_col] = 0

    # Choose track
    if args.track_id is not None:
        tdf = df[df[track_col] == args.track_id].copy()
        if len(tdf) == 0:
            raise RuntimeError(f"Track {args.track_id} not found.")
        chosen_track = args.track_id
    else:
        # choose the most informative track:
        # highest max score, then longest duration
        stats = df.groupby(track_col).agg(
            max_score=(score_col, "max"),
            n_frames=(frame_col, "count")
        ).reset_index()
        stats = stats.sort_values(["max_score", "n_frames"], ascending=[False, False])
        chosen_track = stats.iloc[0][track_col]
        tdf = df[df[track_col] == chosen_track].copy()

    tdf = tdf.sort_values(frame_col).reset_index(drop=True)

    # EMA
    if ema_col is None:
        tdf["ema_score"] = compute_ema(tdf[score_col].tolist(), alpha=args.alpha)
        ema_col = "ema_score"

    # Plot graph
    plt.figure(figsize=(11, 5))
    plt.plot(tdf[frame_col], tdf[score_col], label="Raw anomaly score", linewidth=1.8)
    plt.plot(tdf[frame_col], tdf[ema_col], label=f"EMA score (alpha={args.alpha})", linewidth=2.4)

    if args.threshold is not None:
        plt.axhline(args.threshold, linestyle="--", label=f"Threshold = {args.threshold:.4f}")

    plt.xlabel("Frame index")
    plt.ylabel("Score")
    plt.title(f"Raw score and EMA evolution for track {chosen_track}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    graph_path = out_dir / "ema_graph.png"
    plt.savefig(graph_path, dpi=220)
    plt.close()

    # Representative frames
    # 1) first frame
    first_row = tdf.iloc[0]

    # 2) peak raw score
    peak_row = tdf.loc[tdf[score_col].idxmax()]

    # 3) threshold crossing or max EMA
    crossing_row = None
    if args.threshold is not None:
        crossed = tdf[tdf[ema_col] >= args.threshold]
        if len(crossed) > 0:
            crossing_row = crossed.iloc[0]

    if crossing_row is None:
        crossing_row = tdf.loc[tdf[ema_col].idxmax()]

    # 4) stable end / last frame
    last_row = tdf.iloc[-1]

    selections = [
        ("frame_start.png", first_row, "Start of track"),
        ("frame_peak_raw.png", peak_row, "Peak raw score"),
        ("frame_ema_event.png", crossing_row, "EMA-trigger frame"),
        ("frame_end.png", last_row, "End of track"),
    ]

    summary_lines = []
    summary_lines.append(f"Chosen track: {chosen_track}")
    summary_lines.append(f"Rows in chosen track: {len(tdf)}")
    summary_lines.append(f"Graph: {graph_path}")
    summary_lines.append("")

    has_bbox = all(c is not None for c in [x1_col, y1_col, x2_col, y2_col])

    for fname, row, desc in selections:
        frame_idx = int(row[frame_col])
        crop_box = None
        if has_bbox:
            crop_box = (row[x1_col], row[y1_col], row[x2_col], row[y2_col])

        full_out = out_dir / fname
        ok = save_frame(args.video, frame_idx, full_out, crop_box=crop_box, margin=args.margin)
        if ok:
            summary_lines.append(
                f"{desc}: frame={frame_idx}, raw={row[score_col]:.4f}, ema={row[ema_col]:.4f}, file={fname}"
            )

    # Save track CSV
    tdf.to_csv(out_dir / "selected_track_data.csv", index=False)

    # Save summary
    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print("=" * 70)
    print("EMA VISUALS GENERATED")
    print("=" * 70)
    print(f"Chosen track : {chosen_track}")
    print(f"Graph        : {graph_path}")
    print(f"Data         : {out_dir / 'selected_track_data.csv'}")
    print(f"Summary      : {out_dir / 'summary.txt'}")
    print("Frames:")
    for fname, _, _ in selections:
        print(f"  - {out_dir / fname}")


if __name__ == "__main__":
    main()
