"""
get_video_dims.py
==================
Step 4A utility (diagnosis phase, read-only).

Reads video metadata (width, height, fps, frame_count, duration) for every
video referenced in manifest.csv, or every video file found under a
directory tree. Writes a single CSV that build_step4_dataset.py joins
against the per-detection score CSVs.

This script does NOT touch YOLO, ByteTrack, EfficientAD, scoring,
thresholding, calibration, or model weights. It only opens each video with
cv2.VideoCapture to read its container metadata.

Usage
-----
Resolve every video_filename listed in manifest.csv, searching for each one
under --video_root (recursively):

    python get_video_dims.py \
        --manifest   ./dataset_step4/manifest.csv \
        --video_root ./dataset_step4 \
        --out        ./dataset_step4/video_dims.csv

Or just scan a directory tree for video files (no manifest needed):

    python get_video_dims.py \
        --video_root ./dataset_step4 \
        --out        ./dataset_step4/video_dims.csv

Output columns
--------------
video_filename, video_path, width, height, fps, frame_count, duration_s,
found (bool -- False if a manifest video_filename could not be located)
"""

import os
import csv
import argparse
from typing import Dict, List, Optional

import cv2

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract width/height/fps/frame_count for Step 4 videos.")
    p.add_argument("--manifest", default=None,
                   help="manifest.csv with a 'video_filename' column. "
                        "If given, dims are extracted ONLY for these files "
                        "(searched recursively under --video_root).")
    p.add_argument("--video_root", required=True,
                   help="Root directory to search for video files.")
    p.add_argument("--out", required=True,
                   help="Output CSV path, e.g. ./dataset_step4/video_dims.csv")
    return p.parse_args()


def index_video_root(video_root: str) -> Dict[str, str]:
    """Map {basename: full_path} for every video file under video_root.
    If a basename appears more than once, the first match found wins and a
    warning is printed -- keep filenames unique across the dataset."""
    index: Dict[str, str] = {}
    for root, _dirs, files in os.walk(video_root):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in VIDEO_EXTS:
                if fn in index:
                    print(f"  [WARN] duplicate video filename '{fn}' found at "
                          f"'{os.path.join(root, fn)}' -- keeping first match "
                          f"'{index[fn]}'. Rename one of them to avoid ambiguity.")
                    continue
                index[fn] = os.path.join(root, fn)
    return index


def read_manifest_filenames(manifest_path: str) -> List[str]:
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        if "video_filename" not in (reader.fieldnames or []):
            raise ValueError(
                f"manifest.csv at '{manifest_path}' has no 'video_filename' "
                f"column. Found columns: {reader.fieldnames}")
        return [row["video_filename"].strip() for row in reader
                if row.get("video_filename", "").strip()]


def read_dims(video_path: str) -> Optional[dict]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration_s = (frame_count / fps) if fps > 0 else 0.0
    return dict(width=width, height=height, fps=fps,
               frame_count=frame_count, duration_s=duration_s)


def main():
    args = parse_args()

    print(f"[get_video_dims] Indexing video files under '{args.video_root}' ...")
    index = index_video_root(args.video_root)
    print(f"[get_video_dims] Found {len(index)} video file(s) on disk.")

    if args.manifest:
        target_filenames = read_manifest_filenames(args.manifest)
        print(f"[get_video_dims] manifest.csv lists {len(target_filenames)} "
              f"video(s) -- resolving each against the files found on disk.")
    else:
        target_filenames = sorted(index.keys())
        print(f"[get_video_dims] No --manifest given -- scanning ALL "
              f"{len(target_filenames)} video file(s) found on disk.")

    rows = []
    n_missing = 0
    n_unreadable = 0
    for fn in target_filenames:
        path = index.get(fn)
        if path is None:
            print(f"  [MISSING] '{fn}' listed but not found under "
                  f"'{args.video_root}'")
            rows.append(dict(video_filename=fn, video_path="",
                             width="", height="", fps="", frame_count="",
                             duration_s="", found=False))
            n_missing += 1
            continue

        dims = read_dims(path)
        if dims is None:
            print(f"  [UNREADABLE] '{fn}' at '{path}' -- cv2 could not open it")
            rows.append(dict(video_filename=fn, video_path=path,
                             width="", height="", fps="", frame_count="",
                             duration_s="", found=False))
            n_unreadable += 1
            continue

        print(f"  [OK] {fn:<30s} {dims['width']}x{dims['height']}  "
              f"{dims['fps']:.2f}fps  {dims['frame_count']} frames  "
              f"({dims['duration_s']:.1f}s)")
        rows.append(dict(video_filename=fn, video_path=path,
                         width=dims["width"], height=dims["height"],
                         fps=dims["fps"], frame_count=dims["frame_count"],
                         duration_s=round(dims["duration_s"], 3), found=True))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "video_filename", "video_path", "width", "height", "fps",
            "frame_count", "duration_s", "found"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[get_video_dims] Wrote {len(rows)} row(s) -> {args.out}")
    if n_missing or n_unreadable:
        print(f"[get_video_dims] WARNING: {n_missing} missing, "
              f"{n_unreadable} unreadable -- fix these before running "
              f"build_step4_dataset.py, or those videos' detections will be "
              f"dropped at the join step.")


if __name__ == "__main__":
    main()
