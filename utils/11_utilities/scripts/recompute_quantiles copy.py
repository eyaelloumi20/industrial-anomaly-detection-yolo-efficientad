"""
recompute_quantiles_fixed.py
============================
Improved version that uses EXACT same preprocessing as inference.
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

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
    
from utils.models import Teacher, Student, AutoEncoder
from utils.detection import Detector
from utils.data_loader import get_AD_dataset

IMAGE_SIZE = 256
OUT_CHANNELS = 384


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_dir', required=True)
    p.add_argument('--dataset_path', required=True)
    p.add_argument('--yolo_weights', required=True)
    p.add_argument('--category', default='chewinggum')
    p.add_argument('--model_size', default='S', choices=['S', 'M'])
    p.add_argument('--device', default='cuda')
    p.add_argument('--pad_ratio', type=float, default=0.02)   # match inference
    return p.parse_args()


# === EXACT SAME preprocessing as single_photo_inference.py ===
def get_processed_crop_pil(pil_img, box, pad_ratio=0.02, denoise=True):
    W, H = pil_img.size
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1

    px = int(bw * pad_ratio)
    py = int(bh * pad_ratio)
    x1 = max(0, x1 - px); y1 = max(0, y1 - py)
    x2 = min(W, x2 + px); y2 = min(H, y2 + py)

    crop = pil_img.crop((x1, y1, x2, y2)).convert('RGB')

    # Otsu refinement
    arr_gray = np.array(crop.convert('L'), dtype=np.uint8)
    _, mask = cv2.threshold(arr_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(mask)
    if coords is not None:
        rx, ry, rw, rh = cv2.boundingRect(coords)
        coverage = (rw * rh) / (arr_gray.shape[1] * arr_gray.shape[0])
        if coverage >= 0.30:
            margin = 2
            rx = max(0, rx - margin)
            ry = max(0, ry - margin)
            rw = min(crop.width - rx, rw + 2 * margin)
            rh = min(crop.height - ry, rh + 2 * margin)
            crop = crop.crop((rx, ry, rx + rw, ry + rh))

    # Rotate portrait to landscape
    if crop.height > crop.width:
        crop = crop.rotate(90, expand=True)

    # Resize with black padding
    ow, oh = crop.size
    scale = IMAGE_SIZE / max(ow, oh)
    nw, nh = int(ow * scale), int(oh * scale)
    resized = crop.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
    canvas.paste(resized, ((IMAGE_SIZE - nw) // 2, (IMAGE_SIZE - nh) // 2))
    crop = canvas

    if denoise:
        arr = np.array(crop, dtype=np.uint8)
        arr = cv2.GaussianBlur(arr, (3, 3), sigmaX=0.8, sigmaY=0.8)
        arr = cv2.bilateralFilter(arr, d=5, sigmaColor=15, sigmaSpace=15)
        crop = Image.fromarray(arr)

    return crop


def main():
    args = parse_args()
    device = args.device

    # Load models
    print("[Fixed Recompute] Loading models...")
    teacher = Teacher(args.model_size).to(device).eval()
    student = Student(args.model_size).to(device).eval()
    ae = AutoEncoder().to(device).eval()

    teacher.load_state_dict(torch.load(os.path.join(args.ckpt_dir, "best_teacher.pth"), map_location=device, weights_only=False))
    student.load_state_dict(torch.load(os.path.join(args.ckpt_dir, f"{args.category}_student.pth"), map_location=device, weights_only=False))
    ae.load_state_dict(torch.load(os.path.join(args.ckpt_dir, f"{args.category}_autoencoder.pth"), map_location=device, weights_only=False))

    detector = Detector(
        model_path=args.yolo_weights,
        conf_threshold=0.25,
        iou_threshold=0.45,
        device=device
    )

    # Load normal training images
    pass  # duplicate import removed
    tf = transforms.Compose([transforms.ToTensor()])
    dataset = get_AD_dataset('VisA', args.dataset_path, tf, tf, 'train', args.category, split_ratio=1.0)

    normal_samples = [dataset[i] for i in range(len(dataset)) if dataset[i]['label'] == 0]
    print(f"[Fixed Recompute] Found {len(normal_samples)} normal images")

    to_tensor = transforms.ToTensor()

    # Pass 1: channel stats
    print("Computing channel mean/std on real inference crops...")
    all_feats = []
    for sample in tqdm(normal_samples[:300]):  # limit for speed
        pil_img = Image.fromarray(sample['origin'].astype(np.uint8))
        boxes = detector.detect(pil_img)
        for box in boxes:
            crop_pil = get_processed_crop_pil(pil_img, box, pad_ratio=args.pad_ratio)
            crop_t = to_tensor(crop_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                t_out = teacher(crop_t)
            all_feats.append(t_out.cpu())

    all_t = torch.cat(all_feats, dim=0)
    mean = all_t.mean(dim=(0,2,3), keepdim=True)
    std = all_t.std(dim=(0,2,3), keepdim=True)

    # Pass 2: anomaly maps
    print("Computing anomaly map quantiles...")
    all_st, all_ae = [], []
    for sample in tqdm(normal_samples):
        pil_img = Image.fromarray(sample['origin'].astype(np.uint8))
        boxes = detector.detect(pil_img)
        for box in boxes:
            crop_pil = get_processed_crop_pil(pil_img, box, pad_ratio=args.pad_ratio)
            crop_t = to_tensor(crop_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                t_out = teacher(crop_t)
                s_out = student(crop_t)
                a_out = ae(crop_t)

            y_st = s_out[:, :OUT_CHANNELS, :, :]
            y_stae = s_out[:, -OUT_CHANNELS:, :, :]
            norm_t = (t_out - mean.to(device)) / (std.to(device) + 1e-8)

            d_st = torch.pow(norm_t - y_st, 2)
            d_stae = torch.pow(a_out - y_stae, 2)

            fm_st = F.interpolate(torch.mean(d_st, dim=1, keepdim=True), size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear')
            fm_stae = F.interpolate(torch.mean(d_stae, dim=1, keepdim=True), size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear')

            all_st.append(fm_st.cpu().numpy())
            all_ae.append(fm_stae.cpu().numpy())

    all_st_np = np.concatenate(all_st)
    all_ae_np = np.concatenate(all_ae)

    qa_st = float(np.percentile(all_st_np, 90))
    qb_st = float(np.percentile(all_st_np, 99.5))
    qa_ae = float(np.percentile(all_ae_np, 90))
    qb_ae = float(np.percentile(all_ae_np, 99.5))

    print(f"\nNew quantiles:")
    print(f"qa_st = {qa_st:.6f}   qb_st = {qb_st:.6f}")
    print(f"qa_ae = {qa_ae:.6f}   qb_ae = {qb_ae:.6f}")

    quantiles = {
        'qa_st': qa_st, 'qb_st': qb_st,
        'qa_ae': qa_ae, 'qb_ae': qb_ae,
        'mean': mean.numpy(),
        'std': std.numpy(),
    }
    np.save(os.path.join(args.ckpt_dir, f"{args.category}_quantiles.npy"), quantiles)
    print(f"Quantiles saved to {args.ckpt_dir}")


if __name__ == "__main__":
    main()
