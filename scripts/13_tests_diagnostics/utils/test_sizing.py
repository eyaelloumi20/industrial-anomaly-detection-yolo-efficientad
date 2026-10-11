"""
test_sizing.py
==============
Visualises exactly what build_crop_dataset.py does to each image:
  - Original image size
  - Raw YOLO box coordinates and size
  - Padded box coordinates and size
  - Final 256x256 crop

For every sampled image it saves a side-by-side PNG showing:
  Left  : original image with raw YOLO box (blue) and padded box (green)
  Right : the final 256x256 crop that EfficientAD will see

Also prints a statistics table at the end:
  - Min / max / mean box width and height
  - % of image covered by the box
  - Number of skipped images (no detection)
  - Number of multi-detection images

Usage:
    python test_sizing.py \
        --visa_root    /home/ahmed/Downloads/VisA_20220922 \
        --yolo_weights /home/ahmed/framework/runs/detect/yolo_runs/chewinggum_finetune/weights/best.pt \
        --out_dir      ./sizing_test \
        --n_samples    20 \
        --category     chewinggum \
        --device       cuda
"""

import os
import sys
import csv
import argparse
import random
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from utils.detection import Detector

IMAGE_SIZE = 256
PAD_RATIO  = 0.05


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--visa_root',    required=True)
    p.add_argument('--yolo_weights', required=True)
    p.add_argument('--out_dir',      default='./sizing_test')
    p.add_argument('--category',     default='chewinggum')
    p.add_argument('--n_samples',    type=int, default=20,
                   help='How many images to visualise (default 20)')
    p.add_argument('--conf',         type=float, default=0.25)
    p.add_argument('--iou',          type=float, default=0.45)
    p.add_argument('--target_class', type=int,   default=0)
    p.add_argument('--device',       default='cuda')
    p.add_argument('--seed',         type=int,   default=42)
    return p.parse_args()


def pad_box(box, W, H, pad_ratio=PAD_RATIO):
    x1, y1, x2, y2 = box
    bw = x2 - x1;  bh = y2 - y1
    px = int(bw * pad_ratio);  py = int(bh * pad_ratio)
    x1p = max(0, x1 - px);  y1p = max(0, y1 - py)
    x2p = min(W, x2 + px);  y2p = min(H, y2 + py)
    return x1p, y1p, x2p, y2p


def draw_box(img_np, box, color, thickness=2, label=''):
    x1, y1, x2, y2 = box
    cv2.rectangle(img_np, (x1, y1), (x2, y2), color, thickness)
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
        cv2.rectangle(img_np, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img_np, label, (x1 + 3, y1 - 3),
                    font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img_np


def make_side_by_side(pil_img, boxes, padded_boxes, crop_pil,
                      img_name, label, detection_status):
    """
    Left panel  : original image (resized to fit) with boxes drawn
    Right panel : 256x256 crop (upscaled for visibility)
    """
    PANEL_H = 400
    PANEL_W = 400
    CROP_DISPLAY = 256

    # ── left panel: original + boxes ──────────────────────────────────────────
    W, H = pil_img.size
    scale = min(PANEL_W / W, PANEL_H / H)
    disp_w = int(W * scale); disp_h = int(H * scale)

    left = np.array(pil_img.convert('RGB'), dtype=np.uint8).copy()
    left = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)

    # draw all raw boxes in blue
    for i, box in enumerate(boxes):
        draw_box(left, box, (255, 140, 0),  thickness=2,
                 label=f'raw {i}')

    # draw all padded boxes in green
    for i, pb in enumerate(padded_boxes):
        draw_box(left, pb, (0, 200, 60), thickness=2,
                 label=f'pad {i}')

    left = cv2.resize(left, (disp_w, disp_h))

    # add title bar
    bar = np.zeros((40, disp_w, 3), dtype=np.uint8)
    status_color = (0, 180, 0) if detection_status == 'detected' else (0, 0, 220)
    cv2.putText(bar, f'{img_name}  [{label}]  {detection_status}',
                (4, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                status_color, 1, cv2.LINE_AA)
    left = np.vstack([bar, left])

    # add size annotation below
    info = np.zeros((60, disp_w, 3), dtype=np.uint8)
    if boxes:
        b  = boxes[0]
        pb = padded_boxes[0]
        bw, bh = b[2]-b[0], b[3]-b[1]
        pw, ph = pb[2]-pb[0], pb[3]-pb[1]
        cv2.putText(info, f'Orig: {W}x{H}',
                    (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,180,180), 1)
        cv2.putText(info, f'Raw box: {bw}x{bh} px  ({bw/W*100:.0f}%W {bh/H*100:.0f}%H)',
                    (4, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,140,255), 1)
        cv2.putText(info, f'Pad box: {pw}x{ph} px  ({pw/W*100:.0f}%W {ph/H*100:.0f}%H)',
                    (4, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,200,60), 1)
    else:
        cv2.putText(info, 'NO DETECTION — image skipped',
                    (4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,220), 1)
    left = np.vstack([left, info])

    # ── right panel: 256x256 crop ─────────────────────────────────────────────
    if crop_pil is not None:
        crop_np = cv2.cvtColor(np.array(crop_pil), cv2.COLOR_RGB2BGR)
        crop_np = cv2.resize(crop_np, (CROP_DISPLAY, CROP_DISPLAY),
                             interpolation=cv2.INTER_NEAREST)
    else:
        crop_np = np.zeros((CROP_DISPLAY, CROP_DISPLAY, 3), dtype=np.uint8)
        cv2.putText(crop_np, 'SKIPPED', (60, 128),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 200), 2)

    # pad right panel to match left height
    left_h = left.shape[0]
    right_h = crop_np.shape[0]
    pad_top = (left_h - right_h) // 2
    pad_bot = left_h - right_h - pad_top
    right = np.vstack([
        np.zeros((pad_top, CROP_DISPLAY, 3), dtype=np.uint8),
        crop_np,
        np.zeros((pad_bot, CROP_DISPLAY, 3), dtype=np.uint8),
    ])

    # label on right panel
    rbar = np.zeros((40, CROP_DISPLAY, 3), dtype=np.uint8)
    cv2.putText(rbar, 'EfficientAD input  (256x256)',
                (4, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,180,180), 1)
    right_with_label = np.vstack([rbar, right[40:]])

    # divider
    div = np.full((left_h, 4, 3), 60, dtype=np.uint8)
    combined = np.hstack([left, div, right_with_label])
    return combined


def get_crop(pil_img, padded_box):
    """Crop + resize to 256x256 (no denoising for speed in test mode)."""
    crop = pil_img.crop(padded_box).convert('RGB')
    crop = crop.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    return crop


def main():
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # ── load YOLO ──────────────────────────────────────────────────────────────
    print(f'[SizingTest] Loading YOLO from {args.yolo_weights} ...')
    detector = Detector(
        model_path     = args.yolo_weights,
        conf_threshold = args.conf,
        iou_threshold  = args.iou,
        target_class   = args.target_class,
        device         = args.device,
    )

    # ── read CSV ───────────────────────────────────────────────────────────────
    split_csv = os.path.join(args.visa_root, 'split_csv', '1cls.csv')
    assert os.path.isfile(split_csv), f'Not found: {split_csv}'

    rows = []
    with open(split_csv) as f:
        for row in csv.DictReader(f):
            if row['object'] == args.category:
                rows.append(row)

    print(f'[SizingTest] {len(rows)} total images for {args.category}')

    # sample n_samples images — include both normal and anomaly
    normal  = [r for r in rows if r['label'] == 'normal']
    anomaly = [r for r in rows if r['label'] == 'anomaly']
    random.shuffle(normal);  random.shuffle(anomaly)

    n_each  = args.n_samples // 2
    sampled = normal[:n_each] + anomaly[:n_each]
    random.shuffle(sampled)
    print(f'[SizingTest] Sampling {len(sampled)} images '
          f'({n_each} normal + {n_each} anomaly)\n')

    # ── stats collectors ───────────────────────────────────────────────────────
    stats = {
        'orig_sizes':        [],   # (W, H)
        'raw_box_sizes':     [],   # (bw, bh)
        'pad_box_sizes':     [],   # (pw, ph)
        'coverage_pct':      [],   # padded box area / image area * 100
        'n_skipped':         0,
        'n_multi':           0,
        'n_detected':        0,
    }

    # ── process ────────────────────────────────────────────────────────────────
    for i, row in enumerate(tqdm(sampled, desc='[SizingTest]')):
        img_path = os.path.join(args.visa_root, row['image'])
        if not os.path.isfile(img_path):
            continue

        pil_img = Image.open(img_path).convert('RGB')
        W, H    = pil_img.size
        label   = row['label']
        stem    = os.path.splitext(os.path.basename(img_path))[0]

        boxes = detector.detect(pil_img)

        if not boxes:
            stats['n_skipped'] += 1
            panel = make_side_by_side(
                pil_img, [], [], None,
                stem, label, 'NO DETECTION')
            cv2.imwrite(os.path.join(args.out_dir, f'{i:03d}_{stem}_SKIP.png'),
                        panel)
            continue

        if len(boxes) > 1:
            stats['n_multi'] += 1

        stats['n_detected'] += 1
        stats['orig_sizes'].append((W, H))

        padded_boxes = [pad_box(b, W, H) for b in boxes]

        # collect stats on first box
        b  = boxes[0]
        pb = padded_boxes[0]
        bw, bh = b[2]-b[0],  b[3]-b[1]
        pw, ph = pb[2]-pb[0], pb[3]-pb[1]
        stats['raw_box_sizes'].append((bw, bh))
        stats['pad_box_sizes'].append((pw, ph))
        stats['coverage_pct'].append((pw * ph) / (W * H) * 100)

        # get the actual crop (first box)
        crop = get_crop(pil_img, padded_boxes[0])

        # save panel
        panel = make_side_by_side(
            pil_img, boxes, padded_boxes, crop,
            stem, label, f'detected {len(boxes)} box(es)')
        out_name = f'{i:03d}_{stem}_{label}.png'
        cv2.imwrite(os.path.join(args.out_dir, out_name), panel)

    # ── print summary ──────────────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'  SIZING TEST SUMMARY')
    print(f'{"="*60}')
    print(f'  Images sampled    : {len(sampled)}')
    print(f'  Detected          : {stats["n_detected"]}')
    print(f'  Skipped (no det.) : {stats["n_skipped"]}  '
          f'({stats["n_skipped"]/len(sampled)*100:.1f}%)')
    print(f'  Multi-detection   : {stats["n_multi"]}')

    if stats['orig_sizes']:
        ws = [s[0] for s in stats['orig_sizes']]
        hs = [s[1] for s in stats['orig_sizes']]
        print(f'\n  Original image sizes:')
        print(f'    W: min={min(ws)}  max={max(ws)}  mean={np.mean(ws):.0f}')
        print(f'    H: min={min(hs)}  max={max(hs)}  mean={np.mean(hs):.0f}')

    if stats['raw_box_sizes']:
        bws = [s[0] for s in stats['raw_box_sizes']]
        bhs = [s[1] for s in stats['raw_box_sizes']]
        print(f'\n  Raw YOLO box sizes (px):')
        print(f'    W: min={min(bws)}  max={max(bws)}  mean={np.mean(bws):.0f}')
        print(f'    H: min={min(bhs)}  max={max(bhs)}  mean={np.mean(bhs):.0f}')

    if stats['pad_box_sizes']:
        pws = [s[0] for s in stats['pad_box_sizes']]
        phs = [s[1] for s in stats['pad_box_sizes']]
        print(f'\n  Padded box sizes (px):')
        print(f'    W: min={min(pws)}  max={max(pws)}  mean={np.mean(pws):.0f}')
        print(f'    H: min={min(phs)}  max={max(phs)}  mean={np.mean(phs):.0f}')

    if stats['coverage_pct']:
        cov = stats['coverage_pct']
        print(f'\n  Padded box coverage (% of image area):')
        print(f'    min={min(cov):.1f}%  max={max(cov):.1f}%  mean={np.mean(cov):.1f}%')

        # warn if coverage is suspiciously high or low
        if np.mean(cov) > 80:
            print(f'\n  ⚠ WARNING: mean coverage {np.mean(cov):.1f}% is very high.')
            print(f'    YOLO may be detecting the whole image, not just the gum.')
            print(f'    Check the saved panels in: {args.out_dir}')
        elif np.mean(cov) < 10:
            print(f'\n  ⚠ WARNING: mean coverage {np.mean(cov):.1f}% is very low.')
            print(f'    YOLO boxes may be too tight (missing the full gum piece).')

    print(f'\n  Final crop size   : {IMAGE_SIZE}x{IMAGE_SIZE} px (fixed)')
    print(f'\n  Panels saved to   : {args.out_dir}/')
    print(f'{"="*60}\n')

    # save stats CSV
    csv_out = os.path.join(args.out_dir, 'sizing_stats.csv')
    with open(csv_out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['img_W', 'img_H', 'raw_bw', 'raw_bh',
                    'pad_bw', 'pad_bh', 'coverage_pct'])
        for i in range(len(stats['orig_sizes'])):
            w.writerow([
                stats['orig_sizes'][i][0],    stats['orig_sizes'][i][1],
                stats['raw_box_sizes'][i][0], stats['raw_box_sizes'][i][1],
                stats['pad_box_sizes'][i][0], stats['pad_box_sizes'][i][1],
                f'{stats["coverage_pct"][i]:.2f}',
            ])
    print(f'  Stats CSV → {csv_out}')


if __name__ == '__main__':
    main()
