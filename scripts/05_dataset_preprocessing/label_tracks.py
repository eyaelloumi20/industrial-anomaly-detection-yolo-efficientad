#!/usr/bin/env python3
"""
Interactive manual labeling of YOLO/ByteTrack proposals.

Keys
----
n : NORMAL
a : ANOMALY
u : UNSURE
x : NOT_A_PIECE / false YOLO proposal
s : skip / leave unlabeled
b : go back one row
q : save and quit

The window shows the full-frame context and the representative crop.
"""

from pathlib import Path
import argparse, csv
import cv2
import numpy as np

def fit(img, max_w, max_h):
    if img is None:
        return np.zeros((max_h,max_w,3),np.uint8)
    s = min(max_w/img.shape[1], max_h/img.shape[0])
    w,h = max(1,int(img.shape[1]*s)), max(1,int(img.shape[0]*s))
    return cv2.resize(img,(w,h),interpolation=cv2.INTER_AREA)

def canvas(context,crop,row,i,n):
    W,H=1500,850
    out=np.full((H,W,3),245,np.uint8)

    c1=fit(context,950,650)
    c2=fit(crop,480,650)

    y1=100+(650-c1.shape[0])//2
    y2=100+(650-c2.shape[0])//2
    out[y1:y1+c1.shape[0],25:25+c1.shape[1]]=c1
    out[y2:y2+c2.shape[0],995:995+c2.shape[1]]=c2

    lines=[
        f"[{i+1}/{n}]  {row['background']} | {row['video']} | Track {row['track_id']}",
        "n=NORMAL   a=ANOMALY   u=UNSURE   x=NOT A PIECE   s=SKIP   b=BACK   q=QUIT",
        f"Current GT: {row.get('gt_label','') or 'UNLABELED'}   "
        f"Frames: {row['first_frame']}-{row['last_frame']}   "
        f"Seen: {row['seen_frames']}   YOLO max conf: {row['max_yolo_conf']}",
    ]
    for k,text in enumerate(lines):
        cv2.putText(out,text,(25,35+k*30),cv2.FONT_HERSHEY_SIMPLEX,
                    0.72 if k==0 else 0.58,(0,0,0),2 if k==0 else 1,cv2.LINE_AA)
    return out

def save(path,rows,fields):
    with open(path,"w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dir",default="gt_yolo_assisted")
    ap.add_argument("--csv",default="tracks_to_label.csv")
    args=ap.parse_args()

    root=Path(args.dir)
    path=root/args.csv
    with open(path,"r",encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))
    fields=list(rows[0].keys())

    i=0
    while i<len(rows):
        row=rows[i]
        context=cv2.imread(str(root/row["context_file"]))
        crop=cv2.imread(str(root/row["crop_file"]))
        view=canvas(context,crop,row,i,len(rows))
        cv2.imshow("YOLO-assisted ground truth",view)
        key=cv2.waitKey(0)&0xFF

        if key==ord("n"):
            row["gt_label"]="normal"; row["gt_valid_piece"]="1"; i+=1
        elif key==ord("a"):
            row["gt_label"]="anomaly"; row["gt_valid_piece"]="1"; i+=1
        elif key==ord("u"):
            row["gt_label"]="unsure"; row["gt_valid_piece"]="1"; i+=1
        elif key==ord("x"):
            row["gt_label"]="not_a_piece"; row["gt_valid_piece"]="0"; i+=1
        elif key==ord("s"):
            i+=1
        elif key==ord("b"):
            i=max(0,i-1)
        elif key==ord("q") or key==27:
            break

        if i%10==0:
            save(path,rows,fields)

    save(path,rows,fields)
    cv2.destroyAllWindows()

    valid=[r for r in rows if r.get("gt_valid_piece")=="1"]
    normal=sum(r.get("gt_label")=="normal" for r in valid)
    anomaly=sum(r.get("gt_label")=="anomaly" for r in valid)
    unsure=sum(r.get("gt_label")=="unsure" for r in valid)
    unlabeled=sum(not r.get("gt_label") for r in rows)

    print("Saved:",path)
    print("Normal :",normal)
    print("Anomaly:",anomaly)
    print("Unsure :",unsure)
    print("Unlabeled:",unlabeled)
    print("\nNow review the annotated videos for YOLO misses / duplicate fragmented tracks.")

if __name__=="__main__":
    main()
