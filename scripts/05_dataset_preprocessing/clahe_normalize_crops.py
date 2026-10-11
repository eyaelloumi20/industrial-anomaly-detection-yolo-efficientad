"""
clahe_normalize_crops.py
========================
Apply CLAHE to every crop in a VisA-format dataset folder.
CLAHE is applied on the L channel of LAB — colors are preserved,
only local contrast is normalized.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def clahe_normalize(pil_img, clip=2.0, grid=(8, 8)):
    arr = np.array(pil_img.convert("RGB"))
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source dataset root")
    ap.add_argument("--dst", required=True, help="destination dataset root")
    ap.add_argument("--category", default="chewinggum")
    ap.add_argument("--clip", type=float, default=2.0)
    ap.add_argument("--grid", type=int, default=8)
    args = ap.parse_args()

    src = Path(args.src) / args.category / "Data" / "Images"
    dst = Path(args.dst) / args.category / "Data" / "Images"
    if not src.exists():
        raise SystemExit(f"Not found: {src}")

    for cls in ("Normal", "Anomaly"):
        in_dir = src / cls
        out_dir = dst / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(in_dir.glob("*.png"))
        print(f"{cls:8s}: {len(files)} crops -> {out_dir}")
        for i, f in enumerate(files):
            img = Image.open(f).convert("RGB")
            clahe_img = clahe_normalize(img, args.clip, (args.grid, args.grid))
            clahe_img.save(out_dir / f.name)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(files)}")

    print(f"\nDone. Output: {args.dst}")


if __name__ == "__main__":
    main()