#!/usr/bin/env python3
"""
Export a visual review of all manually-labelled NORMAL tracks
before using them for EfficientAD multi-background training.

Nothing is trained and no dataset is modified.

For each normal track:
- extracts 5 representative ORIGINAL/current crops
- builds one horizontal strip image per track
- builds one overview contact sheet per background
- creates normal_tracks_review.csv with blank APPROVE column

After visual review, fill APPROVE with:
  yes  -> keep for training
  no   -> exclude from training

Optional notes can be added in the NOTES column.
"""

from pathlib import Path
import csv
import re
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/home/ahmed/two-stage-efficientad-inspection")

MATCHED = ROOT / "calibration_lowres192_threshold/matched_tracks.csv"
GT = ROOT / "gt_reserved_newyolo/tracks_to_label.csv"

OUT = ROOT / "normal_track_review"

IMAGE_SIZE = 192
PAD_RATIO = 0.02
DENOISE = False
N_REP = 5

VIDEO_INFO = {
    "black": {
        "video": ROOT / "videos_background/black/WhatsApp Video 2026-09-18 at 15.15.353.mp4",
        "scores": ROOT / "calibration_lowres192/black/scores.csv",
        "tokens": ["black", "15.15.353"],
    },
    "metallic": {
        "video": ROOT / "videos_background/metallic/WhatsApp Video 2026-09-18 at 15.49.53.mp4",
        "scores": ROOT / "calibration_lowres192/metallic/scores.csv",
        "tokens": ["metallic", "15.49.53"],
    },
    "white": {
        "video": ROOT / "videos_background/white/WhatsApp Video 2026-09-18 at 15.55.11.mp4",
        "scores": ROOT / "calibration_lowres192/white/scores.csv",
        "tokens": ["white", "15.55.11"],
    },
    "wooden": {
        "video": ROOT / "videos_background/wooden/WhatsApp Video 2026-09-18 at 15.31.552.mp4",
        "scores": ROOT / "calibration_lowres192/wooden/scores.csv",
        "tokens": ["wooden", "15.31.552"],
    },
}

def norm_name(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def find_col(fieldnames, candidates):
    m = {norm_name(x): x for x in fieldnames}
    for c in candidates:
        if norm_name(c) in m:
            return m[norm_name(c)]
    return None

def detect_bg_from_row(row):
    text = " ".join(str(v) for v in row.values()).lower()
    for bg, info in VIDEO_INFO.items():
        for token in info["tokens"]:
            if token.lower() in text:
                return bg
    return None

def load_normal_tracks():
    src = MATCHED if MATCHED.exists() else GT
    if not src.exists():
        raise SystemExit(
            f"ERROR: no label file found.\nExpected:\n  {MATCHED}\n  or\n  {GT}"
        )

    with src.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise SystemExit(f"ERROR: empty label file: {src}")

    fields = rows[0].keys()
    label_col = find_col(fields, ["label", "gt_label", "manual_label", "class"])
    tid_col = find_col(fields, [
        "track_id", "gt_track_id", "new_track_id",
        "reference_track_id", "ref_track_id"
    ])
    bg_col = find_col(fields, ["background", "bg", "domain", "group", "score_group"])

    if label_col is None or tid_col is None:
        raise SystemExit(
            f"ERROR: could not identify label/track columns in {src}\n"
            f"Columns: {list(fields)}"
        )

    out = defaultdict(set)

    for r in rows:
        lab = str(r.get(label_col, "")).strip().lower()
        if lab not in {"normal", "0", "good", "ok"}:
            continue

        bg = None
        if bg_col:
            raw = str(r.get(bg_col, "")).lower()
            for known in VIDEO_INFO:
                if known in raw:
                    bg = known
                    break
        if bg is None:
            bg = detect_bg_from_row(r)
        if bg is None:
            continue

        try:
            tid = int(float(str(r.get(tid_col, "")).strip()))
        except Exception:
            continue

        out[bg].add(tid)

    print(f"Using labels: {src}")
    for bg in VIDEO_INFO:
        print(f"  {bg:<9s}: {len(out[bg])} NORMAL tracks")

    return out

def evenly_spaced(rows, n):
    if len(rows) <= n:
        return rows
    idxs = np.linspace(0, len(rows)-1, n).round().astype(int)
    return [rows[int(i)] for i in idxs]

def read_track_boxes(csv_path, allowed):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise SystemExit(f"ERROR: empty score CSV: {csv_path}")

    needed = {"frame", "track_id", "x1", "y1", "x2", "y2"}
    missing = needed - set(rows[0].keys())
    if missing:
        raise SystemExit(f"ERROR: {csv_path} missing columns {sorted(missing)}")

    by_track = defaultdict(list)
    for r in rows:
        try:
            tid = int(float(r["track_id"]))
            if tid not in allowed:
                continue

            if str(r.get("verdict", "")).upper() == "CLIPPED":
                continue

            by_track[tid].append({
                "frame": int(float(r["frame"])),
                "x1": float(r["x1"]),
                "y1": float(r["y1"]),
                "x2": float(r["x2"]),
                "y2": float(r["y2"]),
            })
        except Exception:
            continue

    for tid in by_track:
        by_track[tid].sort(key=lambda x: x["frame"])
    return by_track

def add_label(im, text, h=28):
    im = im.convert("RGB")
    canvas = Image.new("RGB", (im.width, im.height + h), "white")
    canvas.paste(im, (0, h))
    d = ImageDraw.Draw(canvas)
    d.text((6, 6), text, fill="black")
    return canvas

def make_track_strip(crops, bg, tid):
    labeled = []
    for i, (frame_no, crop) in enumerate(crops, start=1):
        labeled.append(add_label(crop, f"{bg} | T{tid} | f{frame_no}"))

    total_w = sum(x.width for x in labeled)
    max_h = max(x.height for x in labeled)
    strip = Image.new("RGB", (total_w, max_h), "white")

    x = 0
    for im in labeled:
        strip.paste(im, (x, 0))
        x += im.width
    return strip

def build_overview(track_images, bg):
    if not track_images:
        return None

    thumb_w = 420
    resized = []
    for tid, img in track_images:
        ratio = thumb_w / img.width
        new_h = max(1, int(img.height * ratio))
        r = img.resize((thumb_w, new_h))
        r = add_label(r, f"{bg.upper()} — NORMAL TRACK {tid}", h=30)
        resized.append((tid, r))

    cols = 2
    rows = (len(resized) + cols - 1) // cols
    cell_w = thumb_w
    cell_h = max(im.height for _, im in resized)

    sheet = Image.new("RGB", (cols*cell_w, rows*cell_h), "white")

    for idx, (_, im) in enumerate(resized):
        x = (idx % cols) * cell_w
        y = (idx // cols) * cell_h
        sheet.paste(im, (x, y))

    return sheet

def extract_bg(bg, info, normal_ids):
    from compare_pipeline_methods_batched import get_crop

    video = info["video"]
    scores = info["scores"]

    if not video.exists():
        raise SystemExit(f"ERROR: missing video: {video}")
    if not scores.exists():
        raise SystemExit(f"ERROR: missing scores: {scores}")

    by_track = read_track_boxes(scores, normal_ids)

    requests = []
    for tid in sorted(normal_ids):
        if tid not in by_track or not by_track[tid]:
            print(f"WARNING: {bg} track {tid} not found in score CSV")
            continue
        reps = evenly_spaced(by_track[tid], N_REP)
        for r in reps:
            requests.append((r["frame"], tid, r))

    req_by_frame = defaultdict(list)
    for frame_no, tid, r in requests:
        req_by_frame[frame_no].append((tid, r))

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"ERROR: cannot open video: {video}")

    crops_by_track = defaultdict(list)
    target_frames = set(req_by_frame)
    max_frame = max(target_frames) if target_frames else -1
    frame_idx = 0

    while frame_idx <= max_frame:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx in target_frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)

            for tid, r in req_by_frame[frame_idx]:
                crop = get_crop(
                    pil,
                    (r["x1"], r["y1"], r["x2"], r["y2"]),
                    IMAGE_SIZE,
                    PAD_RATIO,
                    DENOISE,
                )
                crops_by_track[tid].append((frame_idx, crop))

        frame_idx += 1

    cap.release()

    bg_dir = OUT / bg
    bg_dir.mkdir(parents=True, exist_ok=True)

    track_images = []
    review_rows = []

    for tid in sorted(normal_ids):
        crops = crops_by_track.get(tid, [])
        if not crops:
            continue

        strip = make_track_strip(crops, bg, tid)
        out_path = bg_dir / f"track_{tid:04d}.jpg"
        strip.save(out_path, quality=95)

        track_images.append((tid, strip))
        review_rows.append({
            "background": bg,
            "track_id": tid,
            "manual_label": "normal",
            "preview": str(out_path.relative_to(ROOT)),
            "APPROVE": "",
            "NOTES": "",
        })

    overview = build_overview(track_images, bg)
    if overview is not None:
        overview.save(OUT / f"overview_{bg}.jpg", quality=95)

    print(f"{bg:<9s}: exported {len(track_images)} track previews")
    return review_rows

def main():
    if OUT.exists():
        import shutil
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    print("=" * 88)
    print("NORMAL TRACK VISUAL REVIEW EXPORT")
    print("=" * 88)
    print("No training. No dataset modification.\n")

    normals = load_normal_tracks()
    all_rows = []

    for bg, info in VIDEO_INFO.items():
        all_rows.extend(extract_bg(bg, info, normals[bg]))

    review_csv = OUT / "normal_tracks_review.csv"
    fields = [
        "background",
        "track_id",
        "manual_label",
        "preview",
        "APPROVE",
        "NOTES",
    ]

    with review_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    print("\n" + "=" * 88)
    print("REVIEW PACKAGE READY")
    print("=" * 88)
    print(f"Folder : {OUT}")
    print(f"CSV    : {review_csv}")
    print()
    print("Open the overview images first:")
    for bg in VIDEO_INFO:
        print(f"  {OUT / f'overview_{bg}.jpg'}")
    print()
    print("Then fill APPROVE=yes/no in normal_tracks_review.csv.")
    print("Do NOT start training until you finish this review.")

if __name__ == "__main__":
    main()
