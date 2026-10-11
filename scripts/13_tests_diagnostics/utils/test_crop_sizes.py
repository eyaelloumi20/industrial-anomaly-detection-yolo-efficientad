"""
test_crop_sizes.py
==================
Traces every image size transformation that happens inside
build_crop_dataset.py — from raw VisA image to the final saved PNG.

Runs on the first --n_images from the split CSV and prints a detailed
per-step size log for each one.  Nothing is written to disk.

Usage:
    python test_crop_sizes.py \
        --visa_root    /home/ahmed/Downloads/VisA_pytorch \
        --yolo_weights /home/ahmed/framework/runs/detect/yolo_runs/chewinggum_finetune/weights/best.pt \
        --category     chewinggum \
        --n_images     5 \
        --device       cuda
"""

import os, sys, csv, argparse
import numpy as np
import cv2
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from utils.detection import Detector

IMAGE_SIZE = 256

W  = '─' * 68
W2 = '═' * 68


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--visa_root',    default='/home/ahmed/Downloads/VisA_pytorch')
    p.add_argument('--yolo_weights', default=None)
    p.add_argument('--category',     default='chewinggum')
    p.add_argument('--n_images',     type=int,   default=5)
    p.add_argument('--pad_ratio',    type=float, default=0.05)
    p.add_argument('--no_denoise',   action='store_true')
    p.add_argument('--conf',         type=float, default=0.25)
    p.add_argument('--iou',          type=float, default=0.45)
    p.add_argument('--device',       default='cuda')
    return p.parse_args()


# ── size-traced version of crop_and_denoise ───────────────────────────────────

def trace_crop(pil_img, raw_box, pad_ratio, denoise):
    """
    Runs exactly the same ops as build_crop_dataset.crop_and_denoise()
    but records the size at every intermediate step.
    Returns a dict of sizes + the final tensor.
    """
    img_W, img_H = pil_img.size

    # ── STEP 2: padding ───────────────────────────────────────────────────────
    x1, y1, x2, y2 = raw_box
    bw = x2 - x1
    bh = y2 - y1
    px = int(bw * pad_ratio)
    py = int(bh * pad_ratio)
    x1p = max(0,     x1 - px)
    y1p = max(0,     y1 - py)
    x2p = min(img_W, x2 + px)
    y2p = min(img_H, y2 + py)
    padded_box  = (x1p, y1p, x2p, y2p)
    padded_W    = x2p - x1p
    padded_H    = y2p - y1p
    clamped     = (x1p != x1-px) or (y1p != y1-py) or \
                  (x2p != x2+px) or (y2p != y2+py)

    # ── STEP 3: PIL crop ──────────────────────────────────────────────────────
    crop = pil_img.crop(padded_box).convert('RGB')
    size_crop = crop.size                   # (W, H) as PIL returns it

    # ── STEP 4: LANCZOS resize ────────────────────────────────────────────────
    crop = crop.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    size_resize = crop.size                 # always (256, 256)

    # ── STEP 5: Gaussian blur ─────────────────────────────────────────────────
    arr = np.array(crop, dtype=np.uint8)
    shape_pre_gauss = arr.shape             # (H, W, C) = (256, 256, 3)
    if denoise:
        arr = cv2.GaussianBlur(arr, (3, 3), sigmaX=0.8, sigmaY=0.8)
    shape_post_gauss = arr.shape

    # ── STEP 6: bilateral filter ──────────────────────────────────────────────
    if denoise:
        arr = cv2.bilateralFilter(arr, d=5, sigmaColor=15, sigmaSpace=15)
    shape_post_bilat = arr.shape

    # ── STEP 7: back to PIL Image (what build_crop_dataset saves as PNG) ────
    final_pil  = Image.fromarray(arr)
    final_size = final_pil.size          # (W, H) = (256, 256)
    final_arr  = np.array(final_pil)
    final_min  = int(final_arr.min())
    final_max  = int(final_arr.max())

    return {
        # raw box
        'raw_box':          raw_box,
        'raw_bw':           bw,
        'raw_bh':           bh,
        # padding
        'pad_px':           px,
        'pad_py':           py,
        'padded_box':       padded_box,
        'padded_W':         padded_W,
        'padded_H':         padded_H,
        'clamped':          clamped,
        # PIL crop
        'size_crop':        size_crop,      # PIL: (W, H)
        # LANCZOS
        'size_resize':      size_resize,    # PIL: (W, H)
        # numpy stages
        'shape_pre_gauss':  shape_pre_gauss,   # numpy: (H, W, C)
        'shape_post_gauss': shape_post_gauss,
        'shape_post_bilat': shape_post_bilat,
        # final PIL saved as PNG
        'final_size':       final_size,     # PIL: (W, H)
        'final_min':        final_min,      # uint8 [0, 255]
        'final_max':        final_max,
    }


def trace_mask(mask_pil, orig_size, raw_box, pad_ratio):
    """Same padding + crop + resize as build_crop_dataset.crop_mask()."""
    W_orig, H_orig = orig_size
    W_mask, H_mask = mask_pil.size
    sx = W_mask / W_orig
    sy = H_mask / H_orig

    x1, y1, x2, y2 = raw_box
    bw = x2 - x1; bh = y2 - y1
    px = int(bw * pad_ratio); py = int(bh * pad_ratio)
    x1 = max(0, x1 - px);  y1 = max(0, y1 - py)
    x2 = min(W_orig, x2 + px); y2 = min(H_orig, y2 + py)

    mx1, my1 = int(x1*sx), int(y1*sy)
    mx2, my2 = int(x2*sx), int(y2*sy)
    mask_box  = (mx1, my1, mx2, my2)

    crop = mask_pil.crop(mask_box)
    size_crop   = crop.size
    crop = crop.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
    size_resize = crop.size

    return {
        'mask_orig_size': (W_mask, H_mask),
        'scale':          (round(sx,4), round(sy,4)),
        'mask_box':       mask_box,
        'mask_crop_size': size_crop,
        'mask_final':     size_resize,
    }


# ── pretty printer ────────────────────────────────────────────────────────────

def ok(cond):
    return '✓' if cond else '✗ FAIL'


def print_trace(img_idx, row, pil_img, boxes, args, mask_pil=None):
    img_W, img_H = pil_img.size
    denoise = not args.no_denoise

    print(f'\n{W2}')
    print(f'  IMAGE {img_idx}')
    print(f'  File  : {row["image"]}')
    print(f'  Split : {row["split"]}   Label: {row["label"]}')
    print(W2)

    # step 0: raw image
    print(f'\n  STEP 0 — raw VisA image')
    print(f'           pil_img.size = {pil_img.size}   '
          f'(PIL convention: W, H)')
    print(f'           W={img_W}  H={img_H}  mode={pil_img.mode}')

    # step 1: YOLO
    print(f'\n  STEP 1 — YOLO detection')
    print(f'           numpy array fed to YOLO : ({img_H}, {img_W}, 3)  '
          f'(H, W, C)')
    print(f'           YOLO internal letterbox : 640 × 640 px  '
          f'(never stored)')
    print(f'           Output boxes coord space: original {img_W}×{img_H}')

    if not boxes:
        print(f'\n           *** NO DETECTIONS — image SKIPPED ***')
        print()
        return

    print(f'           Detections : {len(boxes)} box(es)')
    for bi, b in enumerate(boxes):
        print(f'           Box {bi} : ({b[0]}, {b[1]}, {b[2]}, {b[3]})  '
              f'size = {b[2]-b[0]} × {b[3]-b[1]} px')

    # per-box trace
    for bi, box in enumerate(boxes):
        t = trace_crop(pil_img, box, args.pad_ratio, denoise)

        print(f'\n  {W}')
        print(f'  BOX {bi}  raw=({box[0]},{box[1]},{box[2]},{box[3]})  '
              f'{t["raw_bw"]}×{t["raw_bh"]} px')
        print(f'  {W}')

        # step 2
        print(f'\n  STEP 2 — 5% context padding  (pad_ratio={args.pad_ratio})')
        print(f'           raw box size   : {t["raw_bw"]} × {t["raw_bh"]} px')
        print(f'           pad_x = int({t["raw_bw"]} × {args.pad_ratio}) '
              f'= {t["pad_px"]} px')
        print(f'           pad_y = int({t["raw_bh"]} × {args.pad_ratio}) '
              f'= {t["pad_py"]} px')
        clamp = '  ← border clamp applied' if t['clamped'] else ''
        print(f'           padded box     : {t["padded_box"]}{clamp}')
        print(f'           padded size    : {t["padded_W"]} × {t["padded_H"]} px')

        # step 3
        cW, cH = t['size_crop']
        match3 = (cW == t['padded_W'] and cH == t['padded_H'])
        print(f'\n  STEP 3 — pil_img.crop(padded_box)  →  PIL RGB')
        print(f'           crop.size = {t["size_crop"]}  '
              f'(W={cW}, H={cH})   {ok(match3)}')

        # step 4
        rW, rH = t['size_resize']
        match4 = (rW == IMAGE_SIZE and rH == IMAGE_SIZE)
        print(f'\n  STEP 4 — LANCZOS resize to {IMAGE_SIZE}×{IMAGE_SIZE}')
        print(f'           crop.size = {t["size_resize"]}  '
              f'(W={rW}, H={rH})   {ok(match4)}')

        # step 5
        h, w, c = t['shape_pre_gauss']
        print(f'\n  STEP 5 — numpy conversion + Gaussian blur (3×3, σ=0.8)')
        print(f'           np.array(crop).shape  = {t["shape_pre_gauss"]}  '
              f'(H, W, C) = ({h}, {w}, {c})')
        if denoise:
            h2, w2, c2 = t['shape_post_gauss']
            match5 = (t['shape_pre_gauss'] == t['shape_post_gauss'])
            print(f'           after GaussianBlur    = {t["shape_post_gauss"]}  '
                  f'{ok(match5)} shape unchanged')
        else:
            print(f'           (skipped — --no_denoise)')

        # step 6
        print(f'\n  STEP 6 — bilateral filter (d=5, σColor=15, σSpace=15)')
        if denoise:
            h3, w3, c3 = t['shape_post_bilat']
            match6 = (t['shape_post_gauss'] == t['shape_post_bilat'])
            print(f'           after bilateralFilter = {t["shape_post_bilat"]}  '
                  f'{ok(match6)} shape unchanged')
        else:
            print(f'           (skipped — --no_denoise)')

        # step 7
        fW, fH = t['final_size']
        match7 = (fW == IMAGE_SIZE and fH == IMAGE_SIZE)
        print(f'\n  STEP 7 — Image.fromarray(arr)  →  PIL RGB  (saved as PNG)')
        print(f'           final_pil.size = {t["final_size"]}  '
              f'(W={fW}, H={fH})   {ok(match7)}')
        print(f'           pixel values   : min={t["final_min"]}  '
              f'max={t["final_max"]}  '
              f'(expected uint8 [0, 255])   '
              f'{ok(0 <= t["final_min"] and t["final_max"] <= 255)}')

        # mask trace (anomaly images only)
        if mask_pil is not None:
            m = trace_mask(mask_pil, pil_img.size, box, args.pad_ratio)
            print(f'\n  MASK TRACE (anomaly GT)')
            print(f'           mask on disk         : {m["mask_orig_size"]}  '
                  f'(W, H)')
            print(f'           scale to image       : sx={m["scale"][0]}  '
                  f'sy={m["scale"][1]}')
            print(f'           mask box (scaled)    : {m["mask_box"]}')
            print(f'           mask.crop().size     : {m["mask_crop_size"]}')
            mW, mH = m['mask_final']
            matchM = (mW == IMAGE_SIZE and mH == IMAGE_SIZE)
            print(f'           NEAREST resize final : {m["mask_final"]}  '
                  f'{ok(matchM)}')

        print()

    # summary for this image
    print(f'  SUMMARY for image {img_idx}')
    print(f'  Raw image  : {img_W} × {img_H} px')
    print(f'  Boxes found: {len(boxes)}')
    for bi, box in enumerate(boxes):
        t = trace_crop(pil_img, box, args.pad_ratio, denoise)
        print(f'  Box {bi}  raw {t["raw_bw"]}×{t["raw_bh"]}'
              f'  → padded {t["padded_W"]}×{t["padded_H"]}'
              f'  → crop {t["size_crop"][0]}×{t["size_crop"][1]}'
              f'  → resize {t["size_resize"][0]}×{t["size_resize"][1]}'
              f'  → saved PNG {t["final_size"][0]}×{t["final_size"][1]}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args  = parse_args()

    print(f'\n{W2}')
    print(f'  test_crop_sizes.py')
    print(f'  visa_root    : {args.visa_root}')
    print(f'  category     : {args.category}')
    print(f'  yolo_weights : {args.yolo_weights}')
    print(f'  n_images     : {args.n_images}')
    print(f'  pad_ratio    : {args.pad_ratio}')
    print(f'  denoise      : {not args.no_denoise}')
    print(f'  device       : {args.device}')
    print(f'  IMAGE_SIZE   : {IMAGE_SIZE}')
    print(W2)

    # load YOLO
    print(f'\n[Test] Loading YOLO ...')
    detector = Detector(
        model_path     = args.yolo_weights,
        conf_threshold = args.conf,
        iou_threshold  = args.iou,
        device         = args.device,
    )

    # read split CSV
    split_csv = os.path.join(args.visa_root, 'split_csv', '1cls.csv')
    assert os.path.isfile(split_csv), f'CSV not found: {split_csv}'
    rows = []
    with open(split_csv) as f:
        for row in csv.DictReader(f):
            if row['object'] == args.category:
                rows.append(row)
    print(f'[Test] {len(rows)} rows for {args.category}, '
          f'testing first {args.n_images}\n')

    # counters
    total = skipped = 0

    for img_idx, row in enumerate(rows[:args.n_images]):
        img_path = os.path.join(args.visa_root, row['image'])
        if not os.path.isfile(img_path):
            print(f'[Test] WARNING: file not found: {img_path}')
            continue

        pil_img = Image.open(img_path).convert('RGB')
        boxes   = detector.detect(pil_img)

        # load mask if anomaly
        mask_pil = None
        if row['label'] == 'anomaly' and row['mask']:
            mask_path = os.path.join(args.visa_root, row['mask'])
            if os.path.isfile(mask_path):
                mask_pil = Image.open(mask_path).convert('L')

        if not boxes:
            skipped += 1
        else:
            total += 1

        print_trace(img_idx, row, pil_img, boxes, args, mask_pil)

    # final summary
    print(W2)
    print(f'  FINAL SUMMARY')
    print(f'  Images tested   : {args.n_images}')
    print(f'  Images scored   : {total}  (YOLO found boxes)')
    print(f'  Images skipped  : {skipped}  (no detection)')
    print(f'  Expected saved PNG : {IMAGE_SIZE}×{IMAGE_SIZE} RGB  uint8  [0, 255]')
    print(W2)


if __name__ == '__main__':
    main()
