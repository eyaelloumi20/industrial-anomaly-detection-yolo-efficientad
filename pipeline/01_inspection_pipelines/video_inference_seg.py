"""
video_inference.py
==================
Clean annotation version using YOLOv8-**seg** + ByteTrack:
- Pixel-level segmentation mask applied before EfficientAD cropping
- Small ID only (no big scores inside boxes)
- Thin coloured contour outlines (green = normal, red = anomaly)
- Semi-transparent mask fill overlay
- Reduced heatmap opacity
"""

import os
import sys
import csv
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from utils.load_efficientad import load_efficientad_model

IMAGE_SIZE   = 256
OUT_CHANNELS = 384


# ====================== PREPROCESSING ======================

def resize_with_padding(crop: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    ow, oh  = crop.size
    scale   = size / max(ow, oh)
    nw, nh  = int(ow * scale), int(oh * scale)
    resized = crop.resize((nw, nh), Image.LANCZOS)
    canvas  = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def get_processed_crop_pil(
    pil_img: Image.Image,
    box: tuple,
    mask_full: np.ndarray,      # H×W uint8 (255=foreground) at original image size
    pad_ratio: float = 0.05,
    denoise: bool = True,
) -> Image.Image:
    """
    Apply the segmentation mask (background → black) then crop, resize,
    and optionally denoise – identical pipeline to single_photo_inference.py.
    """
    W, H   = pil_img.size
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad_ratio), int(bh * pad_ratio)
    x1 = max(0, x1 - px); y1 = max(0, y1 - py)
    x2 = min(W, x2 + px); y2 = min(H, y2 + py)

    img_arr = np.array(pil_img.convert("RGB"), dtype=np.uint8)
    binary  = (mask_full > 127).astype(np.uint8)
    masked  = img_arr * binary[:, :, np.newaxis]

    crop_arr = masked[y1:y2, x1:x2]
    crop     = Image.fromarray(crop_arr)
    crop     = resize_with_padding(crop, IMAGE_SIZE)

    if denoise:
        arr  = np.array(crop, dtype=np.uint8)
        arr  = cv2.GaussianBlur(arr, (3, 3), sigmaX=0.8)
        arr  = cv2.bilateralFilter(arr, d=5, sigmaColor=15, sigmaSpace=15)
        crop = Image.fromarray(arr)
    return crop


def score_crop(models, crop_pil: Image.Image, device: str, ratio: float = 0.1):
    t = transforms.ToTensor()(crop_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        t_out = models["teacher"](t)
        s_out = models["student"](t)
        a_out = models["ae"](t)

    y_st   = s_out[:, :OUT_CHANNELS,  :, :]
    y_stae = s_out[:, -OUT_CHANNELS:, :, :]

    norm_t = (t_out - models["mean"]) / (models["std"] + 1e-8)
    d_st   = torch.pow(norm_t - y_st,   2)
    d_stae = torch.pow(a_out - y_stae, 2)

    fm_st   = F.interpolate(torch.mean(d_st,   dim=1, keepdim=True),
                            size=(IMAGE_SIZE, IMAGE_SIZE),
                            mode="bilinear", align_corners=False)
    fm_stae = F.interpolate(torch.mean(d_stae, dim=1, keepdim=True),
                            size=(IMAGE_SIZE, IMAGE_SIZE),
                            mode="bilinear", align_corners=False)

    nm_st = (ratio * (fm_st   - models["qa_st"])) / (models["qb_st"] - models["qa_st"] + 1e-8)
    nm_ae = (ratio * (fm_stae - models["qa_ae"])) / (models["qb_ae"] - models["qa_ae"] + 1e-8)

    amap  = (0.5 * nm_st + 0.5 * nm_ae)[0, 0].cpu().numpy()
    return amap, float(np.max(amap))


# ====================== DRAWING FUNCTIONS ======================

def draw_seg_annotation(
    frame: np.ndarray,
    tid: int,
    box: tuple,
    mask_full: np.ndarray,   # H×W uint8 at full frame size
    is_anomaly: bool,
) -> np.ndarray:
    """
    Draw a semi-transparent filled segmentation mask + contour outline
    and a small track-ID label.  Replaces draw_clean_id() from the
    bounding-box version.
    """
    color = (0, 0, 220) if is_anomaly else (0, 200, 60)

    # Semi-transparent fill
    fill = np.zeros_like(frame, dtype=np.uint8)
    fill[mask_full > 127] = color
    cv2.addWeighted(fill, 0.35, frame, 0.65, 0, frame)

    # Contour outline (thin)
    contours, _ = cv2.findContours(
        (mask_full > 127).astype(np.uint8),
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(frame, contours, -1, color, 2)

    # Small ID label
    x1, y1, x2, y2 = box
    tag  = f"ID{tid}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(tag, font, 0.55, 2)
    tx = max(0, x1 + 4)
    ty = max(th + 8, y1 + 4)
    cv2.rectangle(frame, (tx - 2, ty - th - 4), (tx + tw + 4, ty + 4), color, -1)
    cv2.putText(frame, tag, (tx, ty), font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def overlay_heatmap(
    frame: np.ndarray,
    amap: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    alpha: float = 0.38,
) -> np.ndarray:
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    norm = ((amap - amap.min()) / (amap.max() - amap.min() + 1e-8) * 255).astype(np.uint8)
    heat = cv2.resize(cv2.applyColorMap(norm, cv2.COLORMAP_HOT), (w, h))
    roi  = frame[y1:y2, x1:x2]
    if roi.shape[0] > 0 and roi.shape[1] > 0:
        frame[y1:y2, x1:x2] = cv2.addWeighted(roi, 1 - alpha, heat, alpha, 0)
    return frame


def draw_hud(frame, frame_idx, fps, worst, threshold):
    h, w    = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 44), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    is_anom = worst >= threshold
    color   = (0, 0, 200) if is_anom else (30, 140, 30)
    status  = "!!! ANOMALY DETECTED !!!" if is_anom else "NORMAL"
    hud     = f"Frame {frame_idx:05d} | FPS {fps:5.1f} | Score {worst:.4f} | {status}"
    cv2.putText(frame, hud, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return frame


# ====================== SEGMENTATION HELPER ======================

def extract_detections_from_results(results, src_h: int, src_w: int) -> list[dict]:
    """
    Parse ultralytics Results objects from a YOLOv8-seg model.
    Returns a list of dicts: {track_id, box, mask, conf}
      - box  : (x1, y1, x2, y2) clipped to frame
      - mask : H×W uint8 (255=foreground) at full frame resolution
    """
    detections = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue

        boxes     = r.boxes.xyxy.cpu().numpy().astype(int)
        confs     = r.boxes.conf.cpu().numpy()
        track_ids = (r.boxes.id.cpu().numpy().astype(int)
                     if r.boxes.id is not None
                     else list(range(len(boxes))))

        if r.masks is not None:
            masks_raw = (r.masks.data.cpu().numpy() * 255).astype(np.uint8)
            seg_masks = []
            for m in masks_raw:
                if m.shape != (src_h, src_w):
                    m = cv2.resize(m, (src_w, src_h), interpolation=cv2.INTER_NEAREST)
                seg_masks.append(m)
        else:
            # Fallback: rectangular mask from bounding box
            seg_masks = []
            for box in boxes:
                m = np.zeros((src_h, src_w), dtype=np.uint8)
                m[box[1]:box[3], box[0]:box[2]] = 255
                seg_masks.append(m)

        for box, mask, tid, conf in zip(boxes, seg_masks, track_ids, confs):
            x1 = max(0, box[0]); y1 = max(0, box[1])
            x2 = min(src_w, box[2]); y2 = min(src_h, box[3])
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append({
                "track_id": int(tid),
                "box":  (x1, y1, x2, y2),
                "mask": mask,
                "conf": float(conf),
            })
    return detections


# ====================== ARG PARSING ======================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source",       required=True,
                   help="Video path, 0 for webcam, or 'visa'")
    p.add_argument("--visa_root",    default=None)
    p.add_argument("--ckpt_dir",     required=True)
    p.add_argument("--yolo_weights", required=True,
                   help="Fine-tuned YOLOv8-seg .pt weights")
    p.add_argument("--out_dir",      default="./video_results_seg")
    p.add_argument("--category",     default="chewinggum")
    p.add_argument("--model_size",   default="S", choices=["S", "M"])
    p.add_argument("--threshold",    type=float, default=0.5)
    p.add_argument("--ema_alpha",    type=float, default=0.3)
    p.add_argument("--conf",         type=float, default=0.25)
    p.add_argument("--iou",          type=float, default=0.45)
    p.add_argument("--pad_ratio",    type=float, default=0.05)
    p.add_argument("--fps",          type=float, default=10.0)
    p.add_argument("--no_display",   action="store_true")
    p.add_argument("--device",       default="cuda")
    return p.parse_args()


def build_visa_video(visa_root, category, out_path, fps=10.0):
    # Preserve your original implementation here if needed.
    import glob
    print(f"[VideoGen] Building test video from VisA for {category}")
    return out_path


# ====================== MAIN ======================

def run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"\n[Video Seg] Loading EfficientAD ...")
    models = load_efficientad_model(
        ckpt_dir   = args.ckpt_dir,
        category   = args.category,
        model_size = args.model_size,
        device     = device,
    )

    print(f"[Video Seg] Loading YOLOv8-seg + ByteTrack ...")
    from ultralytics import YOLO
    yolo = YOLO(args.yolo_weights)
    yolo.to(device)

    if args.source.lower() == "visa":
        source = build_visa_video(
            args.visa_root, args.category,
            os.path.join(args.out_dir, "visa_test_input.mp4"), args.fps,
        )
    else:
        source = int(args.source) if args.source.isdigit() else args.source

    cap   = cv2.VideoCapture(source)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = os.path.join(args.out_dir, "output_video.mp4")
    writer   = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (src_w, src_h)
    )

    csv_path = os.path.join(args.out_dir, "frame_scores.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame", "track_id",
        "raw_score", "ema_score",
        "prediction",
        "x1", "y1", "x2", "y2",
        "width", "height", "area",
        "frame_ms", "frame_fps",
    ])

    track_scores      = {}
    track_raw_history = {}
    track_ema_history = {}
    frame_idx         = 0
    total_anomaly     = 0
    fps_log           = []

    print(f"[Video Seg] Running — Press Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0          = time.perf_counter()
        frame_worst = 0.0
        active_ids  = set()

        # ── Detection + Tracking ─────────────────────────────────────────────
        # Pass the raw frame to YOLO; ByteTrack is invoked via persist=True.
        results = yolo.track(
            source    = frame,
            conf      = args.conf,
            iou       = args.iou,
            classes   = [0],
            persist   = True,
            tracker   = "bytetrack.yaml",
            verbose   = False,
        )

        annotated   = frame.copy()
        pil_frame   = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detections  = extract_detections_from_results(results, src_h, src_w)

        for det in detections:
            tid  = det["track_id"]
            box  = det["box"]
            mask = det["mask"]
            x1, y1, x2, y2 = box

            active_ids.add(tid)

            # ── Crop with mask → EfficientAD ─────────────────────────────────
            crop = get_processed_crop_pil(
                pil_frame, box, mask, pad_ratio=args.pad_ratio
            )
            amap, raw = score_crop(models, crop, device)

            # EMA smoothing per track
            prev_ema  = track_scores.get(tid, raw)
            ema_score = args.ema_alpha * raw + (1 - args.ema_alpha) * prev_ema
            track_scores[tid] = ema_score
            frame_worst = max(frame_worst, ema_score)

            track_raw_history.setdefault(tid, []).append(raw)
            track_ema_history.setdefault(tid, []).append(ema_score)

            is_anomaly = ema_score >= args.threshold
            prediction = "anomaly" if is_anomaly else "normal"

            box_w    = x2 - x1
            box_h    = y2 - y1
            box_area = box_w * box_h

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            csv_writer.writerow([
                frame_idx, tid,
                f"{raw:.6f}", f"{ema_score:.6f}",
                prediction,
                x1, y1, x2, y2,
                box_w, box_h, box_area,
                f"{elapsed_ms:.2f}",
                "",   # fps back-filled after frame completes
            ])

            # ── Draw segmentation mask + heatmap ─────────────────────────────
            annotated = draw_seg_annotation(annotated, tid, box, mask, is_anomaly)
            annotated = overlay_heatmap(annotated, amap, x1, y1, x2, y2, alpha=0.38)

        # Remove stale tracks
        for s in list(track_scores.keys()):
            if s not in active_ids:
                track_scores.pop(s, None)

        if frame_worst >= args.threshold:
            total_anomaly += 1

        t1              = time.perf_counter()
        elapsed_frame   = (t1 - t0) * 1000.0
        fps_log.append(1000.0 / max(elapsed_frame, 1e-3))
        cur_fps = float(np.mean(fps_log[-30:]))
        csv_file.flush()

        annotated = draw_hud(annotated, frame_idx, cur_fps, frame_worst, args.threshold)
        writer.write(annotated)

        if not args.no_display:
            cv2.imshow("Gum Inspection – Seg", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if frame_idx % 20 == 0:
            print(f"  [{frame_idx:05d}]  fps={cur_fps:.1f}  score={frame_worst:.4f}")

    cap.release()
    writer.release()
    csv_file.close()

    # ── Per-ID summary ────────────────────────────────────────────────────────
    summary_path = os.path.join(args.out_dir, "per_id_summary.csv")
    with open(summary_path, "w", newline="") as sf:
        sw = csv.writer(sf)
        sw.writerow([
            "track_id",
            "total_frames_seen",
            "anomaly_frames", "normal_frames", "anomaly_rate_%",
            "raw_score_min",  "raw_score_max",  "raw_score_mean",
            "ema_score_min",  "ema_score_max",  "ema_score_mean",
            "peak_ema_frame",
        ])
        for tid in sorted(track_raw_history.keys()):
            raws  = track_raw_history[tid]
            emas  = track_ema_history[tid]
            n     = len(emas)
            af    = sum(1 for s in emas if s >= args.threshold)
            peak  = int(np.argmax(emas))
            sw.writerow([
                tid, n,
                af, n - af, f"{100.0 * af / max(n, 1):.1f}",
                f"{min(raws):.6f}", f"{max(raws):.6f}", f"{np.mean(raws):.6f}",
                f"{min(emas):.6f}", f"{max(emas):.6f}", f"{np.mean(emas):.6f}",
                peak,
            ])

    print(f"\n=== Finished ===\nOutput saved to: {args.out_dir}")
    print(f"   Video           → output_video.mp4")
    print(f"   Frame CSV       → frame_scores.csv  "
          f"({frame_idx} frames, {len(track_raw_history)} unique tracks)")
    print(f"   Per-ID summary  → per_id_summary.csv")


if __name__ == "__main__":
    run(parse_args())
