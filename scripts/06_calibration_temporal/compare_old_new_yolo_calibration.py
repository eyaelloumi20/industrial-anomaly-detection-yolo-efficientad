#!/usr/bin/env python3
"""
Compare the OLD and NEW YOLO detectors on ONLY the 4 calibration videos.

This is a development check, not the final test.
The 12 final-test videos are not opened by this script.

Outputs:
  yolo_calibration_compare/
    summary.csv
    per_video.csv
    annotated/
      <background>__<video>__old.mp4
      <background>__<video>__new.mp4
      <background>__<video>__side_by_side.mp4
    sampled_frames/
      ...
"""

from pathlib import Path
import argparse, csv, math, shutil
import cv2
import numpy as np
from ultralytics import YOLO

CALIBRATION_VIDEOS = [
    ("black",    "WhatsApp Video 2026-09-18 at 15.15.353.mp4"),
    ("metallic", "WhatsApp Video 2026-09-18 at 15.49.53.mp4"),
    ("white",    "WhatsApp Video 2026-09-18 at 15.55.11.mp4"),
    ("wooden",   "WhatsApp Video 2026-09-18 at 15.31.552.mp4"),
]

OLD_WEIGHTS = (
    "/home/ahmed/two-stage-efficientad-inspection/"
    "runs/detect/yolo_runs/chewinggum_yolov8n_v2/weights/best.pt"
)

NEW_WEIGHTS = (
    "/home/ahmed/two-stage-efficientad-inspection/"
    "runs/detect/runs/detect/yolo_runs/"
    "chewinggum_yolov8n_multibg/weights/best.pt"
)

def draw_boxes(frame, boxes, title, threshold):
    out = frame.copy()
    h, w = out.shape[:2]
    for (x1,y1,x2,y2,conf) in boxes:
        if conf < threshold:
            continue
        x1,y1,x2,y2 = map(int,[x1,y1,x2,y2])
        cv2.rectangle(out,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(
            out, f"{conf:.2f}", (x1,max(22,y1-5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0,255,0), 2, cv2.LINE_AA
        )
    cv2.rectangle(out,(0,0),(w,42),(0,0,0),-1)
    cv2.putText(
        out, title, (12,29),
        cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255,255,255), 2, cv2.LINE_AA
    )
    return out

def fit_height(img, h):
    scale = h / img.shape[0]
    w = max(1,int(img.shape[1]*scale))
    return cv2.resize(img,(w,h),interpolation=cv2.INTER_AREA)

def run_video(weights, video_path, out_video, sample_dir, tag,
              infer_conf=0.05, display_conf=0.25, imgsz=640, device="0"):

    # New model object per video guarantees tracker/predictor state is not reused.
    model = YOLO(weights)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (W,H)
    )

    sample_indices = set(
        np.linspace(0,max(total-1,0),6).round().astype(int).tolist()
    )

    frame_idx = -1
    total_boxes_015 = 0
    total_boxes_025 = 0
    total_boxes_050 = 0
    frames_any_025 = 0
    frames_zero_025 = 0
    confidences = []
    per_frame_counts = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        r = model.predict(
            frame,
            conf=infer_conf,
            iou=0.50,
            imgsz=imgsz,
            device=device,
            verbose=False
        )[0]

        boxes = []
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.detach().cpu().numpy()
            confs = r.boxes.conf.detach().cpu().numpy()
            for box, conf in zip(xyxy,confs):
                boxes.append((*box.tolist(),float(conf)))
                confidences.append(float(conf))

        n015 = sum(b[4] >= 0.15 for b in boxes)
        n025 = sum(b[4] >= 0.25 for b in boxes)
        n050 = sum(b[4] >= 0.50 for b in boxes)

        total_boxes_015 += n015
        total_boxes_025 += n025
        total_boxes_050 += n050
        frames_any_025 += int(n025 > 0)
        frames_zero_025 += int(n025 == 0)
        per_frame_counts.append(n025)

        annotated = draw_boxes(
            frame, boxes,
            f"{tag.upper()} YOLO | conf >= {display_conf:.2f}",
            display_conf
        )
        writer.write(annotated)

        if frame_idx in sample_indices:
            cv2.imwrite(
                str(sample_dir/f"{tag}__f{frame_idx:06d}.jpg"),
                annotated,
                [int(cv2.IMWRITE_JPEG_QUALITY),92]
            )

    cap.release()
    writer.release()

    nframes = max(frame_idx+1,1)
    conf_arr = np.asarray(confidences,float) if confidences else np.asarray([0.0])
    counts = np.asarray(per_frame_counts,float) if per_frame_counts else np.asarray([0.0])

    return {
        "frames": nframes,
        "boxes_conf015": total_boxes_015,
        "boxes_conf025": total_boxes_025,
        "boxes_conf050": total_boxes_050,
        "avg_boxes_per_frame_conf025": float(total_boxes_025/nframes),
        "frames_with_detection_conf025_pct": float(100*frames_any_025/nframes),
        "frames_zero_detection_conf025_pct": float(100*frames_zero_025/nframes),
        "mean_conf_all_conf005plus": float(conf_arr.mean()),
        "median_conf_all_conf005plus": float(np.median(conf_arr)),
        "p25_conf_all_conf005plus": float(np.percentile(conf_arr,25)),
        "frame_count_std_conf025": float(counts.std()),
    }

def side_by_side(old_path, new_path, out_path):
    a = cv2.VideoCapture(str(old_path))
    b = cv2.VideoCapture(str(new_path))
    fps = float(a.get(cv2.CAP_PROP_FPS) or 25.0)

    ok1, f1 = a.read()
    ok2, f2 = b.read()
    if not(ok1 and ok2):
        a.release(); b.release()
        return

    H = min(f1.shape[0],f2.shape[0],720)
    aa = fit_height(f1,H)
    bb = fit_height(f2,H)
    W = aa.shape[1] + bb.shape[1]

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,(W,H)
    )
    writer.write(np.hstack([aa,bb]))

    while True:
        ok1,f1 = a.read()
        ok2,f2 = b.read()
        if not(ok1 and ok2):
            break
        aa = fit_height(f1,H)
        bb = fit_height(f2,H)
        writer.write(np.hstack([aa,bb]))

    a.release(); b.release(); writer.release()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="./videos_background")
    ap.add_argument("--old", default=OLD_WEIGHTS)
    ap.add_argument("--new", default=NEW_WEIGHTS)
    ap.add_argument("--out", default="./yolo_calibration_compare")
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    ann = out/"annotated"
    samp = out/"sampled_frames"
    ann.mkdir(parents=True)
    samp.mkdir(parents=True)

    rows = []

    for bg, name in CALIBRATION_VIDEOS:
        video = Path(args.videos)/bg/name
        if not video.exists():
            raise FileNotFoundError(video)

        print(f"\n=== {bg.upper()} | {name} ===")

        stem = video.stem.replace(" ","_")
        old_video = ann/f"{bg}__{stem}__old.mp4"
        new_video = ann/f"{bg}__{stem}__new.mp4"
        side_video = ann/f"{bg}__{stem}__side_by_side.mp4"

        old_samp = samp/f"{bg}__{stem}"
        new_samp = old_samp
        old_samp.mkdir(exist_ok=True)

        print("  old model...")
        old_m = run_video(
            args.old, video, old_video, old_samp, "old",
            imgsz=args.imgsz, device=args.device
        )

        print("  new model...")
        new_m = run_video(
            args.new, video, new_video, new_samp, "new",
            imgsz=args.imgsz, device=args.device
        )

        side_by_side(old_video,new_video,side_video)

        for model_name, m in [("old",old_m),("new",new_m)]:
            rows.append({
                "background": bg,
                "video": name,
                "model": model_name,
                **m
            })

        print(
            f"  OLD mean conf={old_m['mean_conf_all_conf005plus']:.3f}, "
            f"zero frames@.25={old_m['frames_zero_detection_conf025_pct']:.1f}%"
        )
        print(
            f"  NEW mean conf={new_m['mean_conf_all_conf005plus']:.3f}, "
            f"zero frames@.25={new_m['frames_zero_detection_conf025_pct']:.1f}%"
        )

    fields = list(rows[0].keys())
    with open(out/"per_video.csv","w",encoding="utf-8",newline="") as f:
        w = csv.DictWriter(f,fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    # Aggregate by background/model.
    grouped = {}
    for r in rows:
        key=(r["background"],r["model"])
        grouped.setdefault(key,[]).append(r)

    summary=[]
    metric_names=[
        "boxes_conf015","boxes_conf025","boxes_conf050",
        "avg_boxes_per_frame_conf025",
        "frames_with_detection_conf025_pct",
        "frames_zero_detection_conf025_pct",
        "mean_conf_all_conf005plus",
        "median_conf_all_conf005plus",
        "p25_conf_all_conf005plus",
        "frame_count_std_conf025",
    ]

    for (bg,model_name), rr in grouped.items():
        row={"background":bg,"model":model_name}
        for m in metric_names:
            row[m]=float(np.mean([x[m] for x in rr]))
        summary.append(row)

    with open(out/"summary.csv","w",encoding="utf-8",newline="") as f:
        fields=list(summary[0].keys())
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader(); w.writerows(summary)

    (out/"README.txt").write_text(
        "This comparison uses ONLY the four calibration/development videos.\n"
        "Do not interpret the detection-count statistics as true precision/recall;\n"
        "ground truth is not available yet. Use the side-by-side videos to check\n"
        "whether the new detector recovers pieces missed by the old detector.\n"
        "The 12 final-test videos remain untouched.\n",
        encoding="utf-8"
    )

    print("\nDONE")
    print("Results:",out.resolve())
    print("Upload the whole yolo_calibration_compare folder/ZIP to ChatGPT.")

if __name__ == "__main__":
    main()
