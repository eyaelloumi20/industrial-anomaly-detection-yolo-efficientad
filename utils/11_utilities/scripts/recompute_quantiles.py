"""
recompute_quantiles.py
======================
Recomputes EfficientAD normalisation quantiles (mean, std, qa_st, qb_st,
qa_ae, qb_ae) using the EXACT same preprocessing chain as pipeline.py:

    full image (PIL)
        -> optional YOLO detection (or full-image box when --use_full_image)
        -> pad bbox by --pad_ratio
        -> aspect-ratio-preserving resize + black padding to 256x256  (LANCZOS)
        -> Gaussian blur (3x3, s=0.8) + Bilateral (d=5, sColor=15, sSpace=15)
        -> ToTensor()

No Otsu refinement, no portrait rotation -- those are NOT in pipeline.py and
must not be in quantile computation either.

Run this after every retrain_efficientad.py call before doing final eval.

Usage (pre-cropped dataset, no YOLO needed):
    python scripts/recompute_quantiles.py \
        --ckpt_dir       ./ckpt_cropped_real_D1 \
        --dataset_path   ./dataset1 \
        --category       chewinggum \
        --model_size     S \
        --device         cuda \
        --use_full_image

Usage (raw full-resolution dataset, YOLO crops):
    python scripts/recompute_quantiles.py \
        --ckpt_dir      ./ckpt_cropped_real_D1 \
        --dataset_path  ./dataset_raw \
        --yolo_weights  ./runs/detect/.../best.pt \
        --category      chewinggum \
        --model_size    S \
        --device        cuda \
        --pad_ratio     0.02
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import cv2

# ── path setup ────────────────────────────────────────────────────────────────
# This script lives in  <repo_root>/scripts/recompute_quantiles.py
# Modules live in       <repo_root>/utils/
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT        = os.path.dirname(_SCRIPTS_DIR)   # repo root
_UTILS       = os.path.join(_ROOT, 'utils')

for _p in (_ROOT, _UTILS, _SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models      import Teacher, Student, AutoEncoder
from data_loader import get_AD_dataset

IMAGE_SIZE   = 256
OUT_CHANNELS = 384


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Recompute EfficientAD quantiles matching pipeline.py preprocessing.')
    p.add_argument('--ckpt_dir',       required=True,
                   help='Checkpoint directory (student/AE/teacher .pth files)')
    p.add_argument('--dataset_path',   required=True,
                   help='Dataset root (same layout as used for training)')
    p.add_argument('--yolo_weights',   default=None,
                   help='YOLOv8 .pt weights. Ignored when --use_full_image is set.')
    p.add_argument('--category',       default='chewinggum')
    p.add_argument('--model_size',     default='S', choices=['S', 'M'])
    p.add_argument('--device',         default='cuda')
    p.add_argument('--pad_ratio',      type=float, default=0.02,
                   help='Context padding around YOLO box (default 0.02). '
                        'Use 0.0 when images are already cropped.')
    p.add_argument('--use_full_image', action='store_true',
                   help='Skip YOLO -- treat each image as one crop. '
                        'Use this when dataset images are already 256x256 crops.')
    p.add_argument('--no_denoise',     action='store_true',
                   help='Disable Gaussian+bilateral denoising. '
                        'Must match the flag used during training.')
    p.add_argument('--n_stats',        type=int, default=300,
                   help='Max images used for channel mean/std (Pass 1). '
                        'All normal images are used for quantile maps (Pass 2).')
    p.add_argument('--conf',           type=float, default=0.25)
    p.add_argument('--iou',            type=float, default=0.45)
    return p.parse_args()


# ── preprocessing — mirrors pipeline.py preprocess_crop() exactly ─────────────

def _resize_with_padding(pil_img, size=IMAGE_SIZE):
    """Aspect-ratio-preserving resize + black pad -- identical to pipeline.py."""
    ow, oh  = pil_img.size
    scale   = size / max(ow, oh)
    nw, nh  = int(ow * scale), int(oh * scale)
    resized = pil_img.resize((nw, nh), Image.LANCZOS)
    canvas  = Image.new('RGB', (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def preprocess_crop(pil_img, box, pad_ratio=0.02, denoise=True):
    """
    Exactly mirrors pipeline.py preprocess_crop():
      pad -> _resize_with_padding -> [Gaussian + bilateral] -> ToTensor
    Returns [3, 256, 256] float32 tensor in [0, 1].
    """
    W, H = pil_img.size
    x1, y1, x2, y2 = box
    bw = x2 - x1;  bh = y2 - y1
    px = int(bw * pad_ratio);  py = int(bh * pad_ratio)
    x1 = max(0, x1 - px);  y1 = max(0, y1 - py)
    x2 = min(W, x2 + px);  y2 = min(H, y2 + py)

    crop = pil_img.crop((x1, y1, x2, y2)).convert('RGB')
    crop = _resize_with_padding(crop, IMAGE_SIZE)

    if denoise:
        arr  = np.array(crop, dtype=np.uint8)
        arr  = cv2.GaussianBlur(arr, (3, 3), sigmaX=0.8, sigmaY=0.8)
        arr  = cv2.bilateralFilter(arr, d=5, sigmaColor=15, sigmaSpace=15)
        crop = Image.fromarray(arr)

    return transforms.ToTensor()(crop)   # [3, 256, 256], values in [0, 1]


def get_boxes(pil_img, detector, use_full_image):
    """Return list of (x1,y1,x2,y2) boxes."""
    if use_full_image or detector is None:
        w, h = pil_img.size
        return [(0, 0, w, h)]
    return detector.detect(pil_img)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    device  = args.device
    denoise = not args.no_denoise

    print(f'\n{"="*60}')
    print(f'  Recompute Quantiles')
    print(f'{"="*60}')
    print(f'  ckpt_dir       : {args.ckpt_dir}')
    print(f'  dataset_path   : {args.dataset_path}')
    print(f'  category       : {args.category}')
    print(f'  model_size     : PDN-{args.model_size}')
    print(f'  device         : {device}')
    print(f'  use_full_image : {args.use_full_image}')
    print(f'  pad_ratio      : {args.pad_ratio}')
    print(f'  denoise        : {denoise}')
    print(f'  n_stats        : {args.n_stats}')
    print(f'{"="*60}\n')

    # ── load models ───────────────────────────────────────────────────────────
    print('[Quantiles] Loading models ...')
    teacher = Teacher(args.model_size).to(device).eval()
    student = Student(args.model_size).to(device).eval()
    ae      = AutoEncoder().to(device).eval()

    teacher.load_state_dict(torch.load(
        os.path.join(args.ckpt_dir, 'best_teacher.pth'),
        map_location=device, weights_only=False))
    student.load_state_dict(torch.load(
        os.path.join(args.ckpt_dir, f'{args.category}_student.pth'),
        map_location=device, weights_only=False))
    ae.load_state_dict(torch.load(
        os.path.join(args.ckpt_dir, f'{args.category}_autoencoder.pth'),
        map_location=device, weights_only=False))
    print('[Quantiles] Models loaded.\n')

    # ── load detector (optional) ──────────────────────────────────────────────
    detector = None
    if not args.use_full_image:
        if args.yolo_weights is None:
            print('[Quantiles] WARNING: --yolo_weights not provided and '
                  '--use_full_image not set. Falling back to full-image boxes.')
        else:
            from detection import Detector
            detector = Detector(
                model_path     = args.yolo_weights,
                conf_threshold = args.conf,
                iou_threshold  = args.iou,
                device         = device,
            )

    # ── load normal training images ───────────────────────────────────────────
    tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    print('[Quantiles] Loading training split ...')
    dataset = get_AD_dataset(
        'VisA', args.dataset_path, tf, tf,
        'train', args.category, split_ratio=1.0)

    normal = [dataset[i] for i in range(len(dataset))
              if dataset[i]['label'] == 0]
    print(f'[Quantiles] Normal training images: {len(normal)}\n')

    if len(normal) == 0:
        raise RuntimeError('No normal training images found. '
                           'Check --dataset_path and --category.')

    # ── Pass 1: channel mean / std from teacher features ─────────────────────
    n_stats = min(args.n_stats, len(normal))
    print(f'[Quantiles] Pass 1 -- teacher channel stats ({n_stats} images) ...')
    all_feats = []
    for sample in tqdm(normal[:n_stats], desc='  Channel stats'):
        pil_img = Image.fromarray(sample['origin'].astype(np.uint8))
        boxes   = get_boxes(pil_img, detector, args.use_full_image)
        for box in boxes:
            crop_t = preprocess_crop(
                pil_img, box, args.pad_ratio, denoise
            ).unsqueeze(0).to(device)
            with torch.no_grad():
                all_feats.append(teacher(crop_t).cpu())

    all_t  = torch.cat(all_feats, dim=0)
    mean   = all_t.mean(dim=(0, 2, 3), keepdim=True)
    std    = all_t.std( dim=(0, 2, 3), keepdim=True)
    mean_d = mean.to(device)
    std_d  = std.to(device)

    print(f'  mean : min={mean.min():.6f}  max={mean.max():.6f}')
    print(f'  std  : min={std.min():.6f}   max={std.max():.6f}\n')

    # ── Pass 2: anomaly maps on ALL normal training images ────────────────────
    print(f'[Quantiles] Pass 2 -- anomaly maps on all {len(normal)} normal images ...')
    all_st, all_ae = [], []

    for sample in tqdm(normal, desc='  Anomaly maps'):
        pil_img = Image.fromarray(sample['origin'].astype(np.uint8))
        boxes   = get_boxes(pil_img, detector, args.use_full_image)
        for box in boxes:
            crop_t = preprocess_crop(
                pil_img, box, args.pad_ratio, denoise
            ).unsqueeze(0).to(device)

            with torch.no_grad():
                t_out = teacher(crop_t)
                s_out = student(crop_t)
                a_out = ae(crop_t)

            y_st   = s_out[:, :OUT_CHANNELS,  :, :]
            y_stae = s_out[:, -OUT_CHANNELS:, :, :]

            norm_t  = (t_out - mean_d) / (std_d + 1e-8)
            d_st    = torch.pow(norm_t - y_st,   2)
            d_stae  = torch.pow(a_out   - y_stae, 2)

            fm_st   = F.interpolate(
                torch.mean(d_st,   dim=1, keepdim=True),
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode='bilinear', align_corners=False)
            fm_stae = F.interpolate(
                torch.mean(d_stae, dim=1, keepdim=True),
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode='bilinear', align_corners=False)

            all_st.append(fm_st.cpu().numpy())
            all_ae.append(fm_stae.cpu().numpy())

    st_np = np.concatenate(all_st)
    ae_np = np.concatenate(all_ae)

    qa_st = float(np.percentile(st_np, 90))
    qb_st = float(np.percentile(st_np, 99.5))
    qa_ae = float(np.percentile(ae_np, 90))
    qb_ae = float(np.percentile(ae_np, 99.5))

    print(f'\n[Quantiles] Results:')
    print(f'  qa_st = {qa_st:.8f}   qb_st = {qb_st:.8f}  '
          f'(width = {qb_st - qa_st:.8f})')
    print(f'  qa_ae = {qa_ae:.8f}   qb_ae = {qb_ae:.8f}  '
          f'(width = {qb_ae - qa_ae:.8f})')

    # sanity check
    for name, qa, qb in [('ST', qa_st, qb_st), ('AE', qa_ae, qb_ae)]:
        width = qb - qa
        if width < 1e-6:
            print(f'  WARNING: {name} quantile range is near-zero ({width:.2e}). '
                  'Check model weights.')
        elif width < 0.001:
            print(f'  WARNING: {name} quantile range is very narrow ({width:.6f}). '
                  'Score separation may be poor.')

    # ── save ──────────────────────────────────────────────────────────────────
    quantiles = {
        'qa_st': qa_st,
        'qb_st': qb_st,
        'qa_ae': qa_ae,
        'qb_ae': qb_ae,
        'mean':  mean.numpy(),
        'std':   std.numpy(),
    }
    out_path = os.path.join(args.ckpt_dir, f'{args.category}_quantiles.npy')
    np.save(out_path, quantiles)
    print(f'\n[Quantiles] Saved -> {out_path}')
    print('[Quantiles] Done.\n')
    print('Next step -- run evaluation:')
    print(f'  python -m pipeline.evaluate_pipeline \\')
    print(f'      --dataset_path {args.dataset_path} \\')
    print(f'      --ckpt_dir     {args.ckpt_dir} \\')
    print(f'      --save_dir     ./results_recomputed \\')
    print(f'      --model_size   {args.model_size} \\')
    print(f'      --device       {device}' +
          (' \\\n      --use_full_image' if args.use_full_image else '') +
          (' \\\n      --no_denoise'     if args.no_denoise     else ''))


if __name__ == '__main__':
    main()