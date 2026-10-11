#!/usr/bin/env python3
from pathlib import Path
import argparse,csv
import cv2, numpy as np

def fit(img,max_w,max_h):
    s=min(max_w/img.shape[1],max_h/img.shape[0])
    return cv2.resize(img,(max(1,int(img.shape[1]*s)),max(1,int(img.shape[0]*s))),
                      interpolation=cv2.INTER_AREA)

def save(path,rows,fields):
    with open(path,"w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dir",default="./gt_reserved_newyolo")
    args=ap.parse_args()
    root=Path(args.dir)
    path=root/"tracks_to_label.csv"

    with open(path,"r",encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))
    fields=list(rows[0].keys())

    i=0
    while i<len(rows):
        r=rows[i]
        ctx=cv2.imread(str(root/r["context_file"]))
        crop=cv2.imread(str(root/r["crop_file"]))
        if ctx is None or crop is None:
            i+=1; continue

        canvas=np.full((850,1500,3),245,np.uint8)
        a=fit(ctx,950,650); b=fit(crop,480,650)
        canvas[100:100+a.shape[0],25:25+a.shape[1]]=a
        canvas[100:100+b.shape[0],995:995+b.shape[1]]=b

        texts=[
            f"[{i+1}/{len(rows)}] {r['role'].upper()} | {r['background']} | Track {r['track_id']}",
            r["video"],
            "n=NORMAL  a=ANOMALY  u=UNSURE  x=NOT_A_PIECE  s=SKIP  b=BACK  q=QUIT",
            f"Current: {r.get('gt_label','') or 'UNLABELED'}"
        ]
        for k,t in enumerate(texts):
            cv2.putText(canvas,t,(25,30+k*27),cv2.FONT_HERSHEY_SIMPLEX,
                        0.68 if k<2 else 0.55,(0,0,0),2 if k==0 else 1,cv2.LINE_AA)

        cv2.imshow("Ground truth labeling",canvas)
        key=cv2.waitKey(0)&0xFF

        if key==ord("n"):
            r["gt_label"]="normal"; r["gt_valid_piece"]="1"; i+=1
        elif key==ord("a"):
            r["gt_label"]="anomaly"; r["gt_valid_piece"]="1"; i+=1
        elif key==ord("u"):
            r["gt_label"]="unsure"; r["gt_valid_piece"]="1"; i+=1
        elif key==ord("x"):
            r["gt_label"]="not_a_piece"; r["gt_valid_piece"]="0"; i+=1
        elif key==ord("s"): i+=1
        elif key==ord("b"): i=max(0,i-1)
        elif key in (ord("q"),27): break

        if i%10==0: save(path,rows,fields)

    save(path,rows,fields)
    cv2.destroyAllWindows()
    print("Saved:",path)

if __name__=="__main__":
    main()
