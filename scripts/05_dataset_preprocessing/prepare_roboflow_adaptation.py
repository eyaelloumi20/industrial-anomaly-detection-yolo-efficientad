#!/usr/bin/env python3
"""
Prepare 240 YOLOv8 images for fast correction in Roboflow.

- 8 adaptation videos (2 per background)
- 30 spaced frames/video
- 24 train + 6 valid frames per video
- current local YOLOv8n pre-annotations
- no Roboflow AI Label Assist required

The 16 reserved videos are NEVER touched by this script.
"""

from pathlib import Path
import argparse, csv, shutil, zipfile, re
import cv2
import numpy as np
from ultralytics import YOLO

SELECTED = [
    ("black", "WhatsApp Video 2026-09-18 at 15.14.37.mp4"),
    ("black", "WhatsApp Video 2026-09-18 at 15.15.351.mp4"),
    ("metallic", "WhatsApp Video 2026-09-18 at 15.49.25.mp4"),
    ("metallic", "WhatsApp Video 2026-09-18 at 15.49.45.mp4"),
    ("white", "WhatsApp Video 2026-09-18 at 15.55.05.mp4"),
    ("white", "WhatsApp Video 2026-09-18 at 15.55.462.mp4"),
    ("wooden", "WhatsApp Video 2026-09-18 at 15.30.42.mp4"),
    ("wooden", "WhatsApp Video 2026-09-18 at 15.31.01.mp4"),
]

def safe(s):
    return re.sub(r"[^A-Za-z0-9._-]+","_",s).strip("_")

def yolo_line(cls, x1,y1,x2,y2,w,h):
    x1=max(0,min(float(w),x1)); x2=max(0,min(float(w),x2))
    y1=max(0,min(float(h),y1)); y2=max(0,min(float(h),y2))
    bw=max(0.0,x2-x1); bh=max(0.0,y2-y1)
    if bw < 2 or bh < 2:
        return None
    xc=(x1+x2)/2/w; yc=(y1+y2)/2/h
    return f"{cls} {xc:.6f} {yc:.6f} {bw/w:.6f} {bh/h:.6f}"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--videos", default="./videos_background")
    ap.add_argument(
        "--weights",
        default="/home/ahmed/two-stage-efficientad-inspection/runs/detect/yolo_runs/chewinggum_yolov8n_v2/weights/best.pt"
    )
    ap.add_argument("--out", default="./roboflow_yolo_adaptation")
    ap.add_argument("--frames-per-video", type=int, default=30)
    ap.add_argument("--valid-per-video", type=int, default=6)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    args=ap.parse_args()

    vroot=Path(args.videos)
    out=Path(args.out)
    if out.exists():
        shutil.rmtree(out)

    for split in ("train","valid"):
        (out/split/"images").mkdir(parents=True,exist_ok=True)
        (out/split/"labels").mkdir(parents=True,exist_ok=True)
    (out/"previews").mkdir(parents=True,exist_ok=True)

    model=YOLO(args.weights)
    manifest=[]
    totals={"train":0,"valid":0,"boxes":0,"empty":0}

    for bg, name in SELECTED:
        video=vroot/bg/name
        if not video.exists():
            raise FileNotFoundError(f"Missing: {video}")

        cap=cv2.VideoCapture(str(video))
        n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n <= 0:
            cap.release()
            raise RuntimeError(f"No frames: {video}")

        # Avoid the extreme first/last 5% where partial objects are common.
        idxs=np.unique(np.linspace(int(n*0.05), max(int(n*0.95)-1,0),
                                   args.frames_per_video).round().astype(int))
        # If rounding reduced the count, fill with additional unique indices.
        if len(idxs) < args.frames_per_video:
            pool=np.arange(int(n*0.05), max(int(n*0.95),int(n*0.05)+1))
            for x in pool:
                if x not in idxs:
                    idxs=np.append(idxs,x)
                    if len(idxs)>=args.frames_per_video:
                        break
            idxs=np.sort(idxs[:args.frames_per_video])

        # Every fifth sampled frame goes to validation => 6 of 30.
        valid_positions=set(
            np.linspace(0,len(idxs)-1,args.valid_per_video).round().astype(int).tolist()
        )

        print(f"\n{bg}: {name} | frames={n} | sampled={len(idxs)}")

        for pos, frame_idx in enumerate(idxs):
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(frame_idx))
            ok, frame=cap.read()
            if not ok:
                print("  warning: cannot read",frame_idx)
                continue

            H,W=frame.shape[:2]
            split="valid" if pos in valid_positions else "train"

            pred=model.predict(
                frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                device=args.device, verbose=False
            )[0]

            stem=f"{bg}__{safe(Path(name).stem)}__f{int(frame_idx):06d}"
            img_path=out/split/"images"/f"{stem}.jpg"
            lbl_path=out/split/"labels"/f"{stem}.txt"

            cv2.imwrite(str(img_path),frame,[int(cv2.IMWRITE_JPEG_QUALITY),95])

            lines=[]
            preview=frame.copy()
            if pred.boxes is not None:
                for b in pred.boxes:
                    xy=b.xyxy[0].detach().cpu().numpy().tolist()
                    conf=float(b.conf[0].detach().cpu())
                    cls=int(b.cls[0].detach().cpu())
                    line=yolo_line(cls,*xy,W,H)
                    if line:
                        lines.append(line)
                        x1,y1,x2,y2=map(int,xy)
                        cv2.rectangle(preview,(x1,y1),(x2,y2),(0,255,0),2)
                        cv2.putText(preview,f"{conf:.2f}",(x1,max(20,y1-5)),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,0),2,cv2.LINE_AA)

            lbl_path.write_text("\n".join(lines)+("\n" if lines else ""),encoding="utf-8")

            totals[split]+=1
            totals["boxes"]+=len(lines)
            totals["empty"]+=int(len(lines)==0)

            # Save preview only for the first 2 frames of each video and empty-prediction frames.
            if pos < 2 or len(lines)==0:
                cv2.imwrite(str(out/"previews"/f"{stem}.jpg"),preview,
                            [int(cv2.IMWRITE_JPEG_QUALITY),90])

            manifest.append({
                "background":bg,
                "video":name,
                "frame_index":int(frame_idx),
                "split":split,
                "detections_prelabel":len(lines),
                "image":str(img_path.relative_to(out)),
                "label":str(lbl_path.relative_to(out)),
            })

        cap.release()

    with open(out/"manifest.csv","w",encoding="utf-8",newline="") as f:
        fields=list(manifest[0].keys())
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader(); w.writerows(manifest)

    (out/"data.yaml").write_text(
        "path: .\ntrain: train/images\nval: valid/images\n\n"
        "names:\n  0: chewing-gum\n",
        encoding="utf-8"
    )

    (out/"IMPORTANT.txt").write_text(
        "These are PRE-ANNOTATIONS, not final ground truth.\n"
        "In Roboflow, inspect EVERY image:\n"
        "1) add every missed chewing-gum box,\n"
        "2) delete false boxes,\n"
        "3) resize inaccurate boxes,\n"
        "4) keep exactly one class: chewing-gum.\n"
        "Do not use the 16 reserved videos for this adaptation dataset.\n",
        encoding="utf-8"
    )

    zip_path=out.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob("*"):
            if p.is_file() and p.parent.name != "previews":
                z.write(p,p.relative_to(out.parent))

    print("\nDONE")
    print("Train images:",totals["train"])
    print("Valid images:",totals["valid"])
    print("Pre-labeled boxes:",totals["boxes"])
    print("Frames with zero YOLO boxes:",totals["empty"])
    print("Dataset:",out.resolve())
    print("ZIP:",zip_path.resolve())

if __name__=="__main__":
    main()
