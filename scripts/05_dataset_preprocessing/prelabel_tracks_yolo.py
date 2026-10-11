#!/usr/bin/env python3
"""
YOLO + ByteTrack assisted ground-truth preparation.

This does NOT invent anomaly labels.
It uses the trained YOLO detector only to:
  1) detect chewing-gum pieces,
  2) assign track IDs with ByteTrack,
  3) save representative crops/context images,
  4) export annotated videos for manual verification.

The anomaly ground truth is assigned manually afterward with label_tracks.py.
"""

from pathlib import Path
from collections import defaultdict
import argparse, csv, math, shutil
import cv2
import numpy as np
from ultralytics import YOLO

VIDEO_EXTS = {".mp4",".avi",".mov",".mkv",".m4v",".webm"}

def safe(s):
    import re
    return re.sub(r"[^A-Za-z0-9._()-]+","_",s).strip("_")

def clipped(x1,y1,x2,y2,w,h,margin=3):
    return x1 <= margin or y1 <= margin or x2 >= w-margin or y2 >= h-margin

def expand_box(x1,y1,x2,y2,w,h,pad=0.12):
    bw, bh = x2-x1, y2-y1
    px, py = bw*pad, bh*pad
    return (
        max(0,int(x1-px)), max(0,int(y1-py)),
        min(w,int(x2+px)), min(h,int(y2+py))
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True,
                    help="Root folder containing black/metallic/white/wooden folders")
    ap.add_argument("--weights", required=True, help="YOLO best.pt")
    ap.add_argument("--out", default="gt_yolo_assisted")
    ap.add_argument("--conf", type=float, default=0.15,
                    help="Low detection threshold for GT proposals; manual audit is still required")
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--min-track-frames", type=int, default=2)
    args = ap.parse_args()

    video_root = Path(args.videos)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out/"tracks").mkdir(parents=True)
    (out/"annotated_videos").mkdir(parents=True)

    model = YOLO(args.weights)

    track_rows = []
    all_videos = sorted(
        p for p in video_root.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )

    print(f"Found {len(all_videos)} videos")

    for vi, video in enumerate(all_videos,1):
        background = video.parent.name
        vstem = safe(video.stem)
        print(f"\n[{vi}/{len(all_videos)}] {background}/{video.name}")

        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            print("  ERROR: cannot open")
            continue

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(out/"annotated_videos"/f"{background}__{vstem}.mp4"),
            fourcc, fps, (W,H)
        )

        # Data accumulated per ByteTrack ID.
        tracks = defaultdict(lambda: {
            "first_frame": None,
            "last_frame": None,
            "seen_frames": 0,
            "conf_sum": 0.0,
            "max_conf": 0.0,
            "best_quality": -1.0,
            "best_crop": None,
            "best_context": None,
            "best_frame": None,
            "best_box": None,
            "ever_clipped": False,
        })

        frame_idx = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            results = model.track(
                frame,
                persist=True,
                tracker=args.tracker,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )
            r = results[0]
            annotated = frame.copy()

            if r.boxes is not None and len(r.boxes):
                xyxy = r.boxes.xyxy.detach().cpu().numpy()
                confs = r.boxes.conf.detach().cpu().numpy()
                ids = (
                    r.boxes.id.detach().cpu().numpy().astype(int)
                    if r.boxes.id is not None else np.arange(len(xyxy), dtype=int)
                )

                for box, conf, tid in zip(xyxy, confs, ids):
                    x1,y1,x2,y2 = map(float,box)
                    xi1,yi1,xi2,yi2 = map(int,[x1,y1,x2,y2])
                    bw, bh = max(1,xi2-xi1), max(1,yi2-yi1)
                    area = bw*bh
                    is_clip = clipped(x1,y1,x2,y2,W,H)

                    t = tracks[int(tid)]
                    t["first_frame"] = frame_idx if t["first_frame"] is None else t["first_frame"]
                    t["last_frame"] = frame_idx
                    t["seen_frames"] += 1
                    t["conf_sum"] += float(conf)
                    t["max_conf"] = max(t["max_conf"],float(conf))
                    t["ever_clipped"] = t["ever_clipped"] or is_clip

                    # Prefer a large, confident, non-border observation.
                    quality = float(conf) * math.sqrt(area)
                    if is_clip:
                        quality *= 0.20

                    if quality > t["best_quality"]:
                        ex1,ey1,ex2,ey2 = expand_box(x1,y1,x2,y2,W,H,0.15)
                        crop = frame[ey1:ey2,ex1:ex2].copy()
                        context = frame.copy()
                        cv2.rectangle(context,(xi1,yi1),(xi2,yi2),(0,255,255),3)
                        cv2.putText(
                            context, f"Track {tid}", (max(0,xi1),max(30,yi1-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0,255,255),2,cv2.LINE_AA
                        )
                        t["best_quality"] = quality
                        t["best_crop"] = crop
                        t["best_context"] = context
                        t["best_frame"] = frame_idx
                        t["best_box"] = (xi1,yi1,xi2,yi2)

                    color = (0,255,0) if not is_clip else (0,165,255)
                    cv2.rectangle(annotated,(xi1,yi1),(xi2,yi2),color,2)
                    cv2.putText(
                        annotated, f"ID {tid} {conf:.2f}",
                        (max(0,xi1),max(25,yi1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.65,color,2,cv2.LINE_AA
                    )

            writer.write(annotated)

        cap.release()
        writer.release()

        for tid, t in sorted(tracks.items()):
            if t["seen_frames"] < args.min_track_frames or t["best_crop"] is None:
                continue

            prefix = f"{background}__{vstem}__track_{tid:04d}"
            crop_file = out/"tracks"/f"{prefix}__crop.jpg"
            context_file = out/"tracks"/f"{prefix}__context.jpg"
            cv2.imwrite(str(crop_file), t["best_crop"], [int(cv2.IMWRITE_JPEG_QUALITY),95])
            cv2.imwrite(str(context_file), t["best_context"], [int(cv2.IMWRITE_JPEG_QUALITY),92])

            x1,y1,x2,y2 = t["best_box"]
            track_rows.append({
                "background": background,
                "video": video.name,
                "video_relpath": str(video.relative_to(video_root)),
                "track_id": tid,
                "first_frame": t["first_frame"],
                "last_frame": t["last_frame"],
                "seen_frames": t["seen_frames"],
                "mean_yolo_conf": round(t["conf_sum"]/t["seen_frames"],6),
                "max_yolo_conf": round(t["max_conf"],6),
                "representative_frame": t["best_frame"],
                "x1":x1,"y1":y1,"x2":x2,"y2":y2,
                "ever_clipped": int(t["ever_clipped"]),
                "crop_file": str(crop_file.relative_to(out)),
                "context_file": str(context_file.relative_to(out)),
                # filled by human review:
                "gt_label": "",
                "gt_valid_piece": "",
                "gt_notes": "",
            })

        print(f"  tracks kept: {sum(1 for r in track_rows if r['video']==video.name)}")

    fields = list(track_rows[0].keys()) if track_rows else []
    with open(out/"tracks_to_label.csv","w",encoding="utf-8",newline="") as f:
        w = csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(track_rows)

    print("\nDONE")
    print(f"Tracks to label: {len(track_rows)}")
    print(f"CSV: {out/'tracks_to_label.csv'}")
    print(f"Annotated videos: {out/'annotated_videos'}")
    print("\nIMPORTANT: review the annotated videos for missed YOLO pieces before calling this ground truth.")

if __name__ == "__main__":
    main()
