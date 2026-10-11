"""
cross_ensemble.py
=================
Compare new ckpt (median_ema) vs old lowres (frame_rel_median) as a
two-feature ensemble. A track is flagged if BOTH models fire on it.
Requires both models to have run on the same videos with the same
track IDs (deterministic ByteTrack).
"""
import csv, json, argparse
from pathlib import Path
import numpy as np

def load_csv(path):
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out[(r["video"], int(r["track_id"]))] = {
                "median_ema": float(r["median_ema"]),
                "mean_ema":   float(r["mean_ema"]),
                "frame_rel_median": float(r["frame_rel_median"]),
                "defective": int(r["defective"]),
            }
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--A_csv", required=True, help="new ckpt CSV")
    ap.add_argument("--B_csv", required=True, help="old lowres CSV")
    ap.add_argument("--A_feature", default="median_ema")
    ap.add_argument("--B_feature", default="frame_rel_median")
    ap.add_argument("--mode", choices=["inter","union"], default="inter")
    args = ap.parse_args()

    A = load_csv(args.A_csv)
    B = load_csv(args.B_csv)
    common = sorted(set(A) & set(B))

    a_vals = np.array([A[k][args.A_feature] for k in common])
    b_vals = np.array([B[k][args.B_feature] for k in common])

    ths_a = np.linspace(a_vals.min(), a_vals.max(), 200)
    ths_b = np.linspace(b_vals.min(), b_vals.max(), 200)

    best = None
    for ta in ths_a:
        for tb in ths_b[::4]:
            tp=fp=tn=fn=0
            for k in common:
                a_hit = A[k][args.A_feature] >= ta
                b_hit = B[k][args.B_feature] >= tb
                pred = (a_hit and b_hit) if args.mode=="inter" else (a_hit or b_hit)
                defect = A[k]["defective"]
                if defect and pred: tp+=1
                elif defect:        fn+=1
                elif pred:          fp+=1
                else:               tn+=1
            tpr = tp/max(tp+fn,1)
            fpr = fp/max(fp+tn,1)
            j = tpr-fpr
            if best is None or j > best["J"]:
                best = {"thr_A":float(ta),"thr_B":float(tb),
                        "TP":tp,"FP":fp,"TN":tn,"FN":fn,
                        "TPR":tpr,"FPR":fpr,"J":j}

    print(f"\n{args.mode} ensemble of {args.A_feature} vs {args.B_feature}")
    print(f"  thr_A={best['thr_A']:.3f} thr_B={best['thr_B']:.3f}")
    print(f"  TP={best['TP']} FP={best['FP']} TN={best['TN']} FN={best['FN']}")
    print(f"  TPR={best['TPR']:.3f}  FPR={best['FPR']:.3f}  J={best['J']:.3f}")

if __name__ == "__main__":
    main()