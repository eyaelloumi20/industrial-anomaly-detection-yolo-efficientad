#!/usr/bin/env python3
from pathlib import Path
from collections import defaultdict
import argparse, csv, math, re, shutil
import cv2
import numpy as np
from ultralytics import YOLO

VIDEOS = [
    ("calibration","black","WhatsApp Video 2026-09-18 at 15.15.353.mp4"),
    ("calibration","metallic","WhatsApp Video 2026-09-18 at 15.49.53.mp4"),
    ("calibration","white","WhatsApp Video 2026-09-18 at 15.55.11.mp4"),
    ("calibration","wooden","WhatsApp Video 2026-09-18 at 15.31.552.mp4"),

    ("test","black","WhatsApp Video 2026-09-18 at 15.14.22.mp4"),
    ("test","black","WhatsApp Video 2026-09-18 at 15.15.35.mp4"),
    ("test","black","WhatsApp Video 2026-09-18 at 15.15.352.mp4"),

    ("test","metallic","WhatsApp Video 2026-09-18 at 15.47.23.mp4"),
    ("test","metallic","WhatsApp Video 2026-09-18 at 15.49.36.mp4"),

    ("test","white","WhatsApp Video 2026-09-18 at 15.55.45.mp4"),
    ("test","white","WhatsApp Video 2026-09-18 at 15.55.46.mp4"),
    ("test","white","WhatsApp Video 2026-09-18 at 15.55.461.mp4"),

    ("test","wooden","WhatsApp Video 2026-09-18 at 15.31.55.mp4"),
    ("test","wooden","WhatsApp Video 2026-09-18 at 15.31.551.mp4"),
    ("test","wooden","WhatsApp Video 2026-09-18 at 15.31.56.mp4"),
    ("test","wooden","WhatsApp Video 2026-09-18 at 15.31.561.mp4"),
]

DEFAULT_WEIGHTS = (
    "/home/ahmed/two-stage-efficientad-inspection/"
    "runs/detect/runs/detect/yolo_runs/"
    "chewinggum_yolov8n_multibg/weights/best.pt"
)

def safe(s):
    return re.sub(r"[^A-Za-z0-9._()-]+","_",s).strip("_")

def clipped(x1,y1,x2,y2,w,h,margin=3):
    return x1 <= margin or y1 <= margin or x2 >= w-margin or y2 >= h-margin

def expand_box(x1,y1,x2,y2,w,h,pad=0.15):
    bw, bh = x2-x1, y2-y1
    return (
        max(0,int(x1-bw*pad)), max(0,int(y1-bh*pad)),
        min(w,int(x2+bw*pad)), min(h,int(y2+bh*pad)),
    )

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--videos",default="./videos_background")
    ap.add_argument("--weights",default=DEFAULT_WEIGHTS)
    ap.add_argument("--out",default="./gt_reserved_newyolo")
    ap.add_argument("--conf",type=float,default=0.10)
    ap.add_argument("--iou",type=float,default=0.50)
    ap.add_argument("--imgsz",type=int,default=640)
    ap.add_argument("--device",default="0")
    ap.add_argument("--tracker",default="bytetrack.yaml")
    ap.add_argument("--min-track-frames",type=int,default=2)
    args=ap.parse_args()

    out=Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out/"tracks").mkdir(parents=True)
    (out/"annotated_videos").mkdir(parents=True)

    rows=[]
    untracked=[]

    for vi,(role,bg,name) in enumerate(VIDEOS,1):
        video=Path(args.videos)/bg/name
        if not video.exists():
            raise FileNotFoundError(video)

        print(f"\n[{vi}/{len(VIDEOS)}] {role.upper()} | {bg} | {name}")

        # IMPORTANT: fresh model per video => fresh tracker state.
        model=YOLO(args.weights)

        cap=cv2.VideoCapture(str(video))
        fps=float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer=cv2.VideoWriter(
            str(out/"annotated_videos"/f"{role}__{bg}__{safe(video.stem)}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"), fps, (W,H)
        )

        tracks=defaultdict(lambda:{
            "first":None,"last":None,"seen":0,"conf_sum":0.0,"max_conf":0.0,
            "best_q":-1.0,"best_crop":None,"best_context":None,
            "best_frame":None,"best_box":None,"ever_clipped":False
        })

        fi=-1
        while True:
            ok,frame=cap.read()
            if not ok:
                break
            fi+=1

            r=model.track(
                frame,persist=True,tracker=args.tracker,
                conf=args.conf,iou=args.iou,imgsz=args.imgsz,
                device=args.device,verbose=False
            )[0]

            ann=frame.copy()

            if r.boxes is not None and len(r.boxes):
                xyxy=r.boxes.xyxy.detach().cpu().numpy()
                confs=r.boxes.conf.detach().cpu().numpy()
                ids = None if r.boxes.id is None else r.boxes.id.detach().cpu().numpy()

                for j,(box,conf) in enumerate(zip(xyxy,confs)):
                    x1,y1,x2,y2=map(float,box)
                    xi1,yi1,xi2,yi2=map(int,[x1,y1,x2,y2])

                    if ids is None or np.isnan(ids[j]):
                        untracked.append({
                            "role":role,"background":bg,"video":name,
                            "frame":fi,"conf":float(conf),
                            "x1":xi1,"y1":yi1,"x2":xi2,"y2":yi2
                        })
                        cv2.rectangle(ann,(xi1,yi1),(xi2,yi2),(0,0,255),2)
                        cv2.putText(ann,f"UNTRACKED {conf:.2f}",(xi1,max(25,yi1-4)),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,255),2,cv2.LINE_AA)
                        continue

                    tid=int(ids[j])
                    bw=max(1,xi2-xi1); bh=max(1,yi2-yi1)
                    area=bw*bh
                    is_clip=clipped(x1,y1,x2,y2,W,H)

                    t=tracks[tid]
                    t["first"]=fi if t["first"] is None else t["first"]
                    t["last"]=fi
                    t["seen"]+=1
                    t["conf_sum"]+=float(conf)
                    t["max_conf"]=max(t["max_conf"],float(conf))
                    t["ever_clipped"] = t["ever_clipped"] or is_clip

                    q=float(conf)*math.sqrt(area)
                    if is_clip: q*=0.2

                    if q>t["best_q"]:
                        ex1,ey1,ex2,ey2=expand_box(x1,y1,x2,y2,W,H)
                        crop=frame[ey1:ey2,ex1:ex2].copy()
                        ctx=frame.copy()
                        cv2.rectangle(ctx,(xi1,yi1),(xi2,yi2),(0,255,255),3)
                        cv2.putText(ctx,f"Track {tid}",(xi1,max(30,yi1-6)),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.85,(0,255,255),2,cv2.LINE_AA)
                        t["best_q"]=q
                        t["best_crop"]=crop
                        t["best_context"]=ctx
                        t["best_frame"]=fi
                        t["best_box"]=(xi1,yi1,xi2,yi2)

                    color=(0,255,0) if not is_clip else (0,165,255)
                    cv2.rectangle(ann,(xi1,yi1),(xi2,yi2),color,2)
                    cv2.putText(ann,f"ID {tid} {conf:.2f}",(xi1,max(25,yi1-4)),
                                cv2.FONT_HERSHEY_SIMPLEX,0.58,color,2,cv2.LINE_AA)

            writer.write(ann)

        cap.release(); writer.release()

        kept=0
        for tid,t in sorted(tracks.items()):
            if t["seen"]<args.min_track_frames or t["best_crop"] is None:
                continue
            kept+=1
            prefix=f"{role}__{bg}__{safe(video.stem)}__track_{tid:04d}"
            crop_file=out/"tracks"/f"{prefix}__crop.jpg"
            ctx_file=out/"tracks"/f"{prefix}__context.jpg"
            cv2.imwrite(str(crop_file),t["best_crop"],[int(cv2.IMWRITE_JPEG_QUALITY),95])
            cv2.imwrite(str(ctx_file),t["best_context"],[int(cv2.IMWRITE_JPEG_QUALITY),92])
            x1,y1,x2,y2=t["best_box"]

            rows.append({
                "role":role,"background":bg,"video":name,"track_id":tid,
                "first_frame":t["first"],"last_frame":t["last"],
                "seen_frames":t["seen"],
                "mean_yolo_conf":round(t["conf_sum"]/t["seen"],6),
                "max_yolo_conf":round(t["max_conf"],6),
                "representative_frame":t["best_frame"],
                "x1":x1,"y1":y1,"x2":x2,"y2":y2,
                "ever_clipped":int(t["ever_clipped"]),
                "crop_file":str(crop_file.relative_to(out)),
                "context_file":str(ctx_file.relative_to(out)),
                "gt_label":"","gt_valid_piece":"","gt_notes":""
            })

        print(f"  tracks kept: {kept}")

    if rows:
        with open(out/"tracks_to_label.csv","w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    if untracked:
        with open(out/"untracked_detections.csv","w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(untracked[0].keys()))
            w.writeheader(); w.writerows(untracked)
    else:
        (out/"untracked_detections.csv").write_text(
            "role,background,video,frame,conf,x1,y1,x2,y2\n",
            encoding="utf-8"
        )

    print("\nDONE")
    print("Tracks to label:",len(rows))
    print("Untracked detections:",len(untracked))
    print("Output:",out.resolve())

if __name__=="__main__":
    main()
