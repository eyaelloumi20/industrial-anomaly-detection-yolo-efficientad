"""
calibrate_threshold.py
======================
Calibrate anomaly threshold from a video of known-normal pieces.
Threshold = mean score across all detections.
"""

import os
import sys
import csv
import argparse

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from utils.load_efficientad import load_efficientad_model

IMAGE_SIZE   = 192
OUT_CHANNELS = 384


def resize_with_padding(crop: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    ow, oh = crop.size
    scale  = size / max(ow, oh)
    nw, nh = int(ow * scale), int(oh * scale)
    resized = crop.resize((nw, nh), Image.LANCZOS)
    canvas  = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def get_crop(pil_img: Image.Image, box: tuple, pad_ratio: float = 0.02,
            denoise: bool = True) -> Image.Image:
    W, H = pil_img.size
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad_ratio), int(bh * pad_ratio)
    x1 = max(0, x1 - px);  y1 = max(0, y1 - py)
    x2 = min(W, x2 + px);  y2 = min(H, y2 + py)
    crop = pil_img.crop((x1, y1, x2, y2)).convert("RGB")
    crop = resize_with_padding(crop, IMAGE_SIZE)

    if denoise:
        arr = np.array(crop, dtype=np.uint8)
        arr = cv2.GaussianBlur(arr, (3, 3), sigmaX=0.8, sigmaY=0.8)
        arr = cv2.bilateralFilter(arr, d=5, sigmaColor=15, sigmaSpace=15)
        crop = Image.fromarray(arr)

    return crop


def score_crop(models: dict, crop_pil: Image.Image, device: str, ratio: float = 0.1):
    t = transforms.ToTensor()(crop_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        t_out = models['teacher'](t)
        s_out = models['student'](t)
        a_out = models['ae'](t)

    C      = OUT_CHANNELS
    y_st   = s_out[:, :C, :, :]
    y_stae = s_out[:, C:, :, :]

    norm_t = (t_out - models['mean']) / (models['std'] + 1e-8)
    d_st   = torch.pow(norm_t - y_st,   2)
    d_stae = torch.pow(a_out  - y_stae, 2)

    fm_st   = F.interpolate(torch.mean(d_st,   dim=1, keepdim=True),
                            size=(IMAGE_SIZE, IMAGE_SIZE),
                            mode='bilinear', align_corners=False)
    fm_stae = F.interpolate(torch.mean(d_stae, dim=1, keepdim=True),
                            size=(IMAGE_SIZE, IMAGE_SIZE),
                            mode='bilinear', align_corners=False)

    nm_st = (ratio * (fm_st   - models['qa_st'])) / (models['qb_st'] - models['qa_st'] + 1e-8)
    nm_ae = (ratio * (fm_stae - models['qa_ae'])) / (models['qb_ae'] - models['qa_ae'] + 1e-8)

    amap  = (0.5 * nm_st + 0.5 * nm_ae)[0, 0].cpu().numpy()
    return float(np.max(amap))


def parse_args():
    p = argparse.ArgumentParser(
        description="Calibrate anomaly threshold from a normal-only video.")
    p.add_argument("--source",       required=True)
    p.add_argument("--ckpt_dir",     required=True)
    p.add_argument("--yolo_weights", required=True)
    p.add_argument("--out_dir",      default="./calibration")
    p.add_argument("--category",     default="chewinggum")
    p.add_argument("--model_size",   default="S", choices=["S", "M"])
    p.add_argument("--image_size",   type=int, default=192,
                   help="EfficientAD crop resolution (must match the resolution the "
                        "checkpoint in --ckpt_dir was trained at, e.g. 192 or 256)")
    p.add_argument("--ratio",        type=float, default=0.1)
    p.add_argument("--conf",         type=float, default=0.8)
    p.add_argument("--iou",          type=float, default=0.45)
    p.add_argument("--pad_ratio",    type=float, default=0.02)
    p.add_argument("--no_denoise",   action="store_true",
                   help="Disable Gaussian+bilateral denoising on crops "
                        "(matches evaluate_pipeline.py / crop_dataset_video1.py)")
    p.add_argument("--device",       default="cuda")
    p.add_argument("--no_display",   action="store_true")
    return p.parse_args()


def run(args):
    global IMAGE_SIZE
    IMAGE_SIZE = args.image_size
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[Calibration] Image size : {IMAGE_SIZE}x{IMAGE_SIZE}")

    print(f"\n[Calibration] Loading EfficientAD ...")
    models = load_efficientad_model(
        ckpt_dir=args.ckpt_dir, category=args.category,
        model_size=args.model_size, device=device)

    print(f"[Calibration] Loading YOLO ...")
    from ultralytics import YOLO
    yolo = YOLO(args.yolo_weights)
    yolo.to(device)

    source = int(args.source) if args.source.isdigit() else args.source
    cap    = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {source}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    denoise = not args.no_denoise
    print(f"[Calibration] Source     : {source}  |  {src_w}x{src_h}  |  ~{total_frames} frames")
    print(f"[Calibration] Denoising  : {'ON (Gaussian + Bilateral)' if denoise else 'OFF'}")
    print(f"[Calibration] Pad ratio  : {args.pad_ratio * 100:.0f}%\n")

    log_path = os.path.join(args.out_dir, "calibration_scores.csv")
    log_file = open(log_path, "w", newline="")
    log_w    = csv.writer(log_file)
    log_w.writerow(["frame", "track_id", "raw_score", "x1", "y1", "x2", "y2"])

    all_scores = []
    frame_idx  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        annotated = frame.copy()

        results = yolo.track(
            source=frame, conf=args.conf, iou=args.iou,
            classes=[0], persist=True, tracker="bytetrack.yaml", verbose=False,
        )

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            boxes     = r.boxes.xyxy.cpu().numpy().astype(int)
            track_ids = (r.boxes.id.cpu().numpy().astype(int)
                         if r.boxes.id is not None else list(range(len(boxes))))

            for box, tid in zip(boxes, track_ids):
                x1 = max(0, box[0]);  y1 = max(0, box[1])
                x2 = min(src_w, box[2]);  y2 = min(src_h, box[3])
                if x2 <= x1 or y2 <= y1:
                    continue

                crop  = get_crop(pil_frame, (x1, y1, x2, y2), args.pad_ratio,
                                 denoise=denoise)
                score = score_crop(models, crop, device, ratio=args.ratio)
                all_scores.append(score)
                log_w.writerow([frame_idx, tid, f"{score:.6f}", x1, y1, x2, y2])

                if not args.no_display:
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (200, 150, 0), 2)
                    label = f"ID{tid} {score:.3f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    tx, ty = max(0, x1 + 3), max(th + 6, y1 + 3)
                    cv2.rectangle(annotated, (tx-2, ty-th-4), (tx+tw+4, ty+2), (200, 150, 0), -1)
                    cv2.putText(annotated, label, (tx, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        if not args.no_display:
            n   = len(all_scores)
            hud = (f"Frame {frame_idx:05d} | Detections {n} | Mean {np.mean(all_scores):.3f}"
                   if n > 0 else f"Frame {frame_idx:05d} | No detections")
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (src_w, 36), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0, annotated)
            cv2.putText(annotated, hud, (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.imshow("Calibration (normal video)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if frame_idx % 30 == 0 and all_scores:
            print(f"  [{frame_idx:05d}]  detections={len(all_scores)}  "
                  f"score_mean={np.mean(all_scores):.4f}")

    cap.release()
    log_file.close()
    if not args.no_display:
        cv2.destroyAllWindows()

    if not all_scores:
        print("[Calibration] ERROR: no detections found. Check YOLO weights and --conf.")
        return

    scores    = np.array(all_scores)
    threshold = float(np.mean(scores))

    print(f"\n=== Calibration Results ===")
    print(f"  Total detections : {len(scores)}")
    print(f"  Score mean       : {scores.mean():.4f}")
    print(f"  ─────────────────────────")
    print(f"  Threshold        : {threshold:.6f}   <- use this")

    thr_path = os.path.join(args.out_dir, "threshold.txt")
    with open(thr_path, "w") as f:
        f.write(f"{threshold:.8f}\n")

    print(f"\n  Saved to: {thr_path}  <- pass to --threshold_file")


if __name__ == "__main__":
    run(parse_args())