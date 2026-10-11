"""
video_inference_bg_removal.py
==============================
ByteTrack + EfficientAD anomaly detection with per‑crop background removal.
- Removes background from each detected object before scoring
- Optional image enhancement (sharpness, contrast, brightness, saturation)
- Clean drawing (small ID + thin border + reduced heatmap opacity)
"""

import os
import sys
import csv
import time
import argparse
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

import torch
import torch.nn.functional as F
from torchvision import transforms

# Try importing rembg (optional, but required if --remove-bg is used)
try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from utils.load_efficientad import load_efficientad_model

IMAGE_SIZE   = 256
OUT_CHANNELS = 384


# --------------------- Enhancement & Background Helpers ---------------------
def hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def enhance_image(img: Image.Image, sharpen: float = 1.5, contrast: float = 1.2,
                  brightness: float = 1.05, saturation: float = 1.1) -> Image.Image:
    """Apply sharpening, contrast, brightness and saturation to an RGB or RGBA image."""
    original_mode = img.mode
    if original_mode == "RGBA":
        rgb, alpha = img.convert("RGB"), img.split()[3]
    else:
        rgb = img.convert("RGB")
        alpha = None

    # Enhancements on RGB part
    if sharpen > 1.0:
        rgb = rgb.filter(ImageFilter.UnsharpMask(radius=1.5, percent=int((sharpen - 1) * 150), threshold=3))
    rgb = ImageEnhance.Contrast(rgb).enhance(contrast)
    rgb = ImageEnhance.Brightness(rgb).enhance(brightness)
    rgb = ImageEnhance.Color(rgb).enhance(saturation)
    rgb = ImageEnhance.Sharpness(rgb).enhance(sharpen)

    if alpha:
        rgb = rgb.convert("RGBA")
        rgb.putalpha(alpha)
    return rgb


def apply_background(img: Image.Image, bg_color: str = "#ffffff") -> Image.Image:
    """Paste an RGBA image onto a solid background and convert to RGB."""
    if img.mode != "RGBA":
        return img.convert("RGB")
    bg_rgb = hex_to_rgb(bg_color)
    background = Image.new("RGBA", img.size, bg_rgb + (255,))
    background.paste(img, mask=img.split()[3])
    return background.convert("RGB")


def resize_with_padding(crop: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    """Resize preserving aspect ratio, pad with black to target size."""
    ow, oh = crop.size
    scale = size / max(ow, oh)
    nw, nh = int(ow * scale), int(oh * scale)
    resized = crop.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB" if crop.mode == "RGB" else "RGBA", (size, size), (0, 0, 0, 0) if crop.mode == "RGBA" else (0,0,0))
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


# --------------------- Crop Preprocessing (with background removal) ---------------------
def process_crop_with_bg_removal(pil_crop: Image.Image,
                                  remove_bg: bool,
                                  enhance: bool,
                                  bg_color: str,
                                  sharpen: float,
                                  contrast: float,
                                  brightness: float,
                                  saturation: float,
                                  pad_ratio: float = 0.05,
                                  denoise: bool = True) -> Image.Image:
    """
    Full pipeline for a single crop:
        1. (optional) background removal -> RGBA
        2. (optional) image enhancement
        3. Paste onto background color -> RGB
        4. Resize with padding to 256x256
        5. (optional) denoise (Gaussian + bilateral)
    Returns a PIL Image (RGB, 256x256) ready for EfficientAD.
    """
    from io import BytesIO

    # 1. Background removal (if enabled)
    if remove_bg:
        if not REMBG_AVAILABLE:
            raise RuntimeError("rembg not installed. Install with: pip install rembg onnxruntime")
        # Encode PIL image to PNG bytes (rembg expects a valid image file)
        with BytesIO() as input_buffer:
            pil_crop.convert("RGB").save(input_buffer, format="PNG")
            input_bytes = input_buffer.getvalue()
        output_bytes = rembg_remove(input_bytes)
        pil_crop = Image.open(BytesIO(output_bytes)).convert("RGBA")

    # 2. Image enhancement (on RGBA if available, else RGB)
    if enhance:
        pil_crop = enhance_image(pil_crop, sharpen, contrast, brightness, saturation)

    # 3. Convert to solid background RGB (if we have transparency)
    if pil_crop.mode == "RGBA":
        pil_crop = apply_background(pil_crop, bg_color)
    else:
        pil_crop = pil_crop.convert("RGB")

    # 4. Resize with padding to IMAGE_SIZE (256)
    pil_crop = resize_with_padding(pil_crop, IMAGE_SIZE)

    # 5. Denoise (optional, same as original)
    if denoise:
        arr = np.array(pil_crop, dtype=np.uint8)
        arr = cv2.GaussianBlur(arr, (3, 3), sigmaX=0.8)
        arr = cv2.bilateralFilter(arr, d=5, sigmaColor=15, sigmaSpace=15)
        pil_crop = Image.fromarray(arr)

    return pil_crop

def get_processed_crop_pil_original(pil_img, box, pad_ratio=0.05, denoise=True):
    """Original version (no background removal) – kept for comparison."""
    W, H = pil_img.size
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad_ratio), int(bh * pad_ratio)
    x1 = max(0, x1 - px); y1 = max(0, y1 - py)
    x2 = min(W, x2 + px); y2 = min(H, y2 + py)

    crop = pil_img.crop((x1, y1, x2, y2)).convert("RGB")
    crop = resize_with_padding(crop, IMAGE_SIZE)

    if denoise:
        arr = np.array(crop, dtype=np.uint8)
        arr = cv2.GaussianBlur(arr, (3, 3), sigmaX=0.8)
        arr = cv2.bilateralFilter(arr, d=5, sigmaColor=15, sigmaSpace=15)
        crop = Image.fromarray(arr)
    return crop


def score_crop(models, crop_pil, device, ratio=0.1):
    """Same anomaly scoring as original."""
    t = transforms.ToTensor()(crop_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        t_out = models['teacher'](t)
        s_out = models['student'](t)
        a_out = models['ae'](t)

    y_st = s_out[:, :OUT_CHANNELS, :, :]
    y_stae = s_out[:, -OUT_CHANNELS:, :, :]

    norm_t = (t_out - models['mean']) / (models['std'] + 1e-8)
    d_st = torch.pow(norm_t - y_st, 2)
    d_stae = torch.pow(a_out - y_stae, 2)

    fm_st = F.interpolate(torch.mean(d_st, dim=1, keepdim=True), size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)
    fm_stae = F.interpolate(torch.mean(d_stae, dim=1, keepdim=True), size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)

    nm_st = (ratio * (fm_st - models['qa_st'])) / (models['qb_st'] - models['qa_st'] + 1e-8)
    nm_ae = (ratio * (fm_stae - models['qa_ae'])) / (models['qb_ae'] - models['qa_ae'] + 1e-8)

    amap = (0.5 * nm_st + 0.5 * nm_ae)[0, 0].cpu().numpy()
    return amap, float(np.max(amap))


# --------------------- Drawing Functions (unchanged from clean version) ---------------------
def draw_clean_id(frame, tid, box, is_anomaly):
    x1, y1, x2, y2 = box
    color = (0, 0, 220) if is_anomaly else (0, 200, 60)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    tag = f"ID{tid}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(tag, font, 0.55, 2)
    tx = max(0, x1 + 4)
    ty = max(th + 8, y1 + 4)
    cv2.rectangle(frame, (tx-2, ty-th-4), (tx+tw+4, ty+4), color, -1)
    cv2.putText(frame, tag, (tx, ty), font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def overlay_heatmap(frame, amap, x1, y1, x2, y2, alpha=0.38):
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    norm = ((amap - amap.min()) / (amap.max() - amap.min() + 1e-8) * 255).astype(np.uint8)
    heat = cv2.resize(cv2.applyColorMap(norm, cv2.COLORMAP_HOT), (w, h))
    roi = frame[y1:y2, x1:x2]
    if roi.shape[0] > 0 and roi.shape[1] > 0:
        frame[y1:y2, x1:x2] = cv2.addWeighted(roi, 1 - alpha, heat, alpha, 0)
    return frame


def draw_hud(frame, frame_idx, fps, worst, threshold):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 44), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    is_anom = worst >= threshold
    color = (0, 0, 200) if is_anom else (30, 140, 30)
    status = "!!! ANOMALY DETECTED !!!" if is_anom else "NORMAL"
    hud = f"Frame {frame_idx:05d} | FPS {fps:5.1f} | Score {worst:.4f} | {status}"
    cv2.putText(frame, hud, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return frame


# --------------------- Main ---------------------
def parse_args():
    p = argparse.ArgumentParser(description="Video inference with per‑crop background removal and anomaly detection")
    p.add_argument("--source", required=True, help="video path, 0 for webcam, or 'visa'")
    p.add_argument("--visa_root", default=None)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--yolo_weights", required=True)
    p.add_argument("--out_dir", default="./video_results_bg_removal")
    p.add_argument("--category", default="chewinggum")
    p.add_argument("--model_size", default="S", choices=["S", "M"])

    # Background removal & enhancement options
    p.add_argument("--remove-bg", dest="remove_bg", action="store_true", default=True,
                   help="Enable background removal (default: True)")
    p.add_argument("--no-rembg", dest="remove_bg", action="store_false",
                   help="Disable background removal")
    p.add_argument("--bg-color", default="#ffffff", help="Background color after removal (hex, e.g. #ffffff)")
    p.add_argument("--enhance", dest="enhance", action="store_true", default=True,
                   help="Enable image enhancement (default: True)")
    p.add_argument("--no-enhance", dest="enhance", action="store_false",
                   help="Disable image enhancement")
    p.add_argument("--sharpen", type=float, default=1.5, help="Sharpen level 1.0–3.0")
    p.add_argument("--contrast", type=float, default=1.2, help="Contrast level 1.0–2.0")
    p.add_argument("--brightness", type=float, default=1.05, help="Brightness 0.5–2.0")
    p.add_argument("--saturation", type=float, default=1.1, help="Saturation 0.0–2.0")

    # Detection & tracking parameters
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--ema_alpha", type=float, default=0.3)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--pad_ratio", type=float, default=0.05)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--no_display", action="store_true")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def build_visa_video(visa_root, category, out_path, fps=10.0):
    """Placeholder: reuse your original build_visa_video function if needed."""
    # ... (copy the actual implementation from your original script)
    print(f"[VideoGen] Building test video from VisA for {category}")
    # For brevity, we assume it exists. Replace with your full function.
    return out_path


def run(args):
    if args.remove_bg and not REMBG_AVAILABLE:
        print("ERROR: --remove-bg requires rembg. Install: pip install rembg onnxruntime")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"\n[Video BG Removal] Loading EfficientAD ...")
    models = load_efficientad_model(
        ckpt_dir=args.ckpt_dir, category=args.category,
        model_size=args.model_size, device=device)

    print(f"[Video BG Removal] Loading YOLOv8 + ByteTrack ...")
    from ultralytics import YOLO
    yolo = YOLO(args.yolo_weights)
    yolo.to(device)

    if args.source.lower() == "visa":
        source = build_visa_video(args.visa_root, args.category,
                                  os.path.join(args.out_dir, "visa_test_input.mp4"), args.fps)
    else:
        source = int(args.source) if args.source.isdigit() else args.source

    cap = cv2.VideoCapture(source)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = os.path.join(args.out_dir, "output_video.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (src_w, src_h))

    csv_path = os.path.join(args.out_dir, "frame_scores.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame", "track_id",
        "raw_score", "ema_score",
        "prediction",
        "x1", "y1", "x2", "y2",
        "width", "height", "area",
        "frame_ms", "frame_fps"
    ])

    track_scores = {}
    track_raw_history = {}
    track_ema_history = {}
    frame_idx = 0
    total_anomaly = 0
    fps_log = []

    print(f"[Video BG Removal] Running (background removal {'ON' if args.remove_bg else 'OFF'}, enhancement {'ON' if args.enhance else 'OFF'})")
    print("Press Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        frame_worst = 0.0
        active_ids = set()

        results = yolo.track(
            source=frame,
            conf=args.conf,
            iou=args.iou,
            classes=[0],
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        annotated = frame.copy()

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            boxes = r.boxes.xyxy.cpu().numpy().astype(int)
            track_ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else list(range(len(boxes)))

            for box, tid in zip(boxes, track_ids):
                x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(src_w, box[2]), min(src_h, box[3])
                if x2 <= x1 or y2 <= y1:
                    continue

                active_ids.add(tid)

                # Convert frame to PIL (RGB) for crop extraction
                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # Original crop with padding (same as before)
                x1_c, y1_c, x2_c, y2_c = x1, y1, x2, y2  # already padded? We add padding again inside
                bw, bh = x2_c - x1_c, y2_c - y1_c
                px, py = int(bw * args.pad_ratio), int(bh * args.pad_ratio)
                x1_pad = max(0, x1_c - px)
                y1_pad = max(0, y1_c - py)
                x2_pad = min(src_w, x2_c + px)
                y2_pad = min(src_h, y2_c + py)
                crop_raw = pil_frame.crop((x1_pad, y1_pad, x2_pad, y2_pad)).convert("RGB")

                # Apply background removal + enhancement + resize to 256x256
                try:
                    crop_processed = process_crop_with_bg_removal(
                        crop_raw,
                        remove_bg=args.remove_bg,
                        enhance=args.enhance,
                        bg_color=args.bg_color,
                        sharpen=args.sharpen,
                        contrast=args.contrast,
                        brightness=args.brightness,
                        saturation=args.saturation,
                        pad_ratio=0.0,          # we already padded, no extra padding inside
                        denoise=True
                    )
                except Exception as e:
                    print(f"  Warning: bg removal failed for ID{tid}: {e}, using original crop")
                    crop_processed = get_processed_crop_pil_original(pil_frame, (x1, y1, x2, y2), pad_ratio=args.pad_ratio)

                # Score anomaly
                amap, raw = score_crop(models, crop_processed, device)
                prev_ema = track_scores.get(tid, raw)
                ema_score = args.ema_alpha * raw + (1 - args.ema_alpha) * prev_ema
                track_scores[tid] = ema_score
                frame_worst = max(frame_worst, ema_score)

                track_raw_history.setdefault(tid, []).append(raw)
                track_ema_history.setdefault(tid, []).append(ema_score)

                is_anomaly = ema_score >= args.threshold
                prediction = "anomaly" if is_anomaly else "normal"

                box_w = x2 - x1
                box_h = y2 - y1
                box_area = box_w * box_h
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                csv_writer.writerow([
                    frame_idx, tid,
                    f"{raw:.6f}", f"{ema_score:.6f}",
                    prediction,
                    x1, y1, x2, y2,
                    box_w, box_h, box_area,
                    f"{elapsed_ms:.2f}",
                    ""
                ])

                annotated = draw_clean_id(annotated, tid, (x1, y1, x2, y2), is_anomaly)
                annotated = overlay_heatmap(annotated, amap, x1, y1, x2, y2, alpha=0.38)

        # Clean stale tracks
        for sid in list(track_scores.keys()):
            if sid not in active_ids:
                track_scores.pop(sid, None)

        if frame_worst >= args.threshold:
            total_anomaly += 1

        t1 = time.perf_counter()
        elapsed_frame_ms = (t1 - t0) * 1000.0
        fps_log.append(1000.0 / max(elapsed_frame_ms, 1e-3))
        cur_fps = float(np.mean(fps_log[-30:]))

        # Update fps in CSV rows of this frame (simple: flush and ignore backfill for brevity)
        csv_file.flush()

        annotated = draw_hud(annotated, frame_idx, cur_fps, frame_worst, args.threshold)
        writer.write(annotated)

        if not args.no_display:
            cv2.imshow("Gum Inspection - BG Removal", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if frame_idx % 20 == 0:
            print(f"  [{frame_idx:05d}]  fps={cur_fps:.1f}  score={frame_worst:.4f}")

    cap.release()
    writer.release()
    csv_file.close()

    # Per-ID summary
    summary_path = os.path.join(args.out_dir, "per_id_summary.csv")
    with open(summary_path, "w", newline="") as sf:
        sw = csv.writer(sf)
        sw.writerow([
            "track_id", "total_frames_seen", "anomaly_frames", "normal_frames",
            "anomaly_rate_%", "raw_score_min", "raw_score_max", "raw_score_mean",
            "ema_score_min", "ema_score_max", "ema_score_mean", "peak_ema_frame"
        ])
        for tid in sorted(track_raw_history.keys()):
            raws = track_raw_history[tid]
            emas = track_ema_history[tid]
            n = len(emas)
            anom_frames = sum(1 for s in emas if s >= args.threshold)
            peak_idx = int(np.argmax(emas))
            sw.writerow([
                tid, n, anom_frames, n - anom_frames,
                f"{100.0 * anom_frames / max(n, 1):.1f}",
                f"{min(raws):.6f}", f"{max(raws):.6f}", f"{np.mean(raws):.6f}",
                f"{min(emas):.6f}", f"{max(emas):.6f}", f"{np.mean(emas):.6f}",
                peak_idx,
            ])

    print(f"\n=== Finished ===\nOutput saved to: {args.out_dir}")
    print(f"   Video           → output_video.mp4")
    print(f"   Frame CSV       → frame_scores.csv  ({frame_idx} frames, {len(track_raw_history)} unique tracks)")
    print(f"   Per-ID summary  → per_id_summary.csv")


if __name__ == "__main__":
    run(parse_args())