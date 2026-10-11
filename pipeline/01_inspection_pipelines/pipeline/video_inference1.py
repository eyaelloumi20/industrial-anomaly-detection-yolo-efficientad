"""
video_inference1_clean.py
Clean annotation version (DeepOCSORT) - Fixed imports
"""

import os
import sys
import csv
import time
import argparse
import textwrap
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F          # ← This was missing!
from torchvision import transforms

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from utils.load_efficientad import load_efficientad_model

IMAGE_SIZE   = 256
OUT_CHANNELS = 384


# ====================== PREPROCESSING ======================
def resize_with_padding(crop: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    ow, oh = crop.size
    scale = size / max(ow, oh)
    nw, nh = int(ow * scale), int(oh * scale)
    resized = crop.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def get_processed_crop_pil(pil_img: Image.Image, box: tuple, pad_ratio: float = 0.05, denoise: bool = True) -> Image.Image:
    W, H = pil_img.size
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    px = int(bw * pad_ratio)
    py = int(bh * pad_ratio)
    x1 = max(0, x1 - px); y1 = max(0, y1 - py)
    x2 = min(W, x2 + px); y2 = min(H, y2 + py)

    crop = pil_img.crop((x1, y1, x2, y2)).convert("RGB")

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
            rw = min(crop.width - rx, rw + 2*margin)
            rh = min(crop.height - ry, rh + 2*margin)
            crop = crop.crop((rx, ry, rx + rw, ry + rh))

    if crop.height > crop.width:
        crop = crop.rotate(90, expand=True)

    crop = resize_with_padding(crop, IMAGE_SIZE)

    if denoise:
        arr = np.array(crop, dtype=np.uint8)
        arr = cv2.GaussianBlur(arr, (3, 3), sigmaX=0.8)
        arr = cv2.bilateralFilter(arr, d=5, sigmaColor=15, sigmaSpace=15)
        crop = Image.fromarray(arr)

    return crop


def score_crop(models, crop_pil, device, ratio=0.1):
    t = transforms.ToTensor()(crop_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        t_out = models['teacher'](t)
        s_out = models['student'](t)
        a_out = models['ae'](t)

    y_st   = s_out[:, :OUT_CHANNELS, :, :]
    y_stae = s_out[:, -OUT_CHANNELS:, :, :]

    norm_t = (t_out - models['mean']) / (models['std'] + 1e-8)
    d_st   = torch.pow(norm_t - y_st, 2)
    d_stae = torch.pow(a_out - y_stae, 2)

    fm_st   = torch.mean(d_st,   dim=1, keepdim=True)
    fm_stae = torch.mean(d_stae, dim=1, keepdim=True)

    fm_st   = F.interpolate(fm_st,   size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)
    fm_stae = F.interpolate(fm_stae, size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)

    nm_st = (ratio * (fm_st   - models['qa_st'])) / (models['qb_st'] - models['qa_st'] + 1e-8)
    nm_ae = (ratio * (fm_stae - models['qa_ae'])) / (models['qb_ae'] - models['qa_ae'] + 1e-8)

    amap = (0.5 * nm_st + 0.5 * nm_ae)[0, 0].cpu().numpy()
    return amap, float(np.max(amap))


# ====================== CLEAN DRAWING ======================
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
    cv2.putText(frame, tag, (tx, ty), font, 0.55, (255,255,255), 2, cv2.LINE_AA)
    return frame


def overlay_heatmap(frame, amap, x1, y1, x2, y2, alpha=0.38):
    w, h = max(1, x2-x1), max(1, y2-y1)
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


# ====================== MAIN ======================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--visa_root", default=None)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--yolo_weights", required=True)
    p.add_argument("--out_dir", default="./results_video_clean")
    p.add_argument("--category", default="MyCategory")
    p.add_argument("--model_size", default="S", choices=["S", "M"])
    p.add_argument("--threshold", type=float, default=None,
                   help="Anomaly decision threshold. If omitted, --threshold_file is used.")
    p.add_argument("--threshold_file", default=None,
                   help="Path to threshold.txt written by evaluate_pipeline.py.")
    p.add_argument("--ema_alpha", type=float, default=0.3)
    p.add_argument("--conf", type=float, default=0.8)
    p.add_argument("--iou", type=float, default=0.45)
    # FIX: match evaluate_pipeline.py default pad_ratio (was 0.05)
    p.add_argument("--pad_ratio", type=float, default=0.02,
                   help="Context padding around YOLO box (default 0.02, matches evaluate_pipeline.py)")
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--no_display", action="store_true")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    # ── Resolve threshold ─────────────────────────────────────────────────────
    threshold = args.threshold
    if args.threshold_file and os.path.isfile(args.threshold_file):
        threshold = float(open(args.threshold_file).read().strip())
        print(f"[Video Clean] Threshold loaded from file: {threshold:.6f}  ({args.threshold_file})")
    elif threshold is not None:
        print(f"[Video Clean] Threshold (--threshold): {threshold:.6f}")
    else:
        threshold = 0.5
        print(f"[Video Clean] WARNING: No threshold specified. Using fallback {threshold}.")
        print(f"              Pass --threshold_file <results_dir>/threshold.txt for the calibrated value.")
    args.threshold = threshold

    print(f"\n[Video Clean] Loading EfficientAD ...")
    models = load_efficientad_model(
        ckpt_dir=args.ckpt_dir,
        category=args.category,
        model_size=args.model_size,
        device=device)

    print(f"[Video Clean] Loading YOLOv8 + DeepOCSORT ...")
    from ultralytics import YOLO
    from boxmot import DeepOcSort
    yolo = YOLO(args.yolo_weights)
    yolo.to(device)

    boxmot_device = "0" if str(device).startswith("cuda") else "cpu"
    tracker = DeepOcSort(
        reid_weights=Path("osnet_x0_25_msmt17.pt"),
        device=boxmot_device,
        half=False,
        embedding_off=True,
    )

    source = int(args.source) if str(args.source).isdigit() else args.source

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

    print(f"[Video Clean] Starting... Press Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        frame_worst = 0.0
        active_ids = set()

        # YOLO + Tracking
        results = yolo.predict(source=frame, conf=args.conf, iou=args.iou, classes=[0], verbose=False)
        dets = np.empty((0, 6), dtype=np.float32)
        if results and results[0].boxes is not None and len(results[0].boxes):
            xyxy = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy().reshape(-1, 1)
            clss = results[0].boxes.cls.cpu().numpy().reshape(-1, 1)
            dets = np.hstack([xyxy, confs, clss]).astype(np.float32)

        tracks = tracker.update(dets, frame)

        annotated = frame.copy()

        for t in tracks:
            x1 = max(0, int(t[0]))
            y1 = max(0, int(t[1]))
            x2 = min(src_w, int(t[2]))
            y2 = min(src_h, int(t[3]))
            tid = int(t[4])

            if x2 <= x1 or y2 <= y1:
                continue

            active_ids.add(tid)

            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            crop = get_processed_crop_pil(pil_frame, (x1, y1, x2, y2), pad_ratio=args.pad_ratio)

            amap, raw_score = score_crop(models, crop, device)

            # EMA
            prev_ema = track_scores.get(tid, raw_score)
            ema_score = args.ema_alpha * raw_score + (1 - args.ema_alpha) * prev_ema
            track_scores[tid] = ema_score
            frame_worst = max(frame_worst, ema_score)

            track_raw_history.setdefault(tid, []).append(raw_score)
            track_ema_history.setdefault(tid, []).append(ema_score)

            is_anomaly = ema_score >= args.threshold
            prediction = "anomaly" if is_anomaly else "normal"

            box_w   = x2 - x1
            box_h   = y2 - y1
            box_area = box_w * box_h
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            csv_writer.writerow([
                frame_idx, tid,
                f"{raw_score:.6f}", f"{ema_score:.6f}",
                prediction,
                x1, y1, x2, y2,
                box_w, box_h, box_area,
                f"{elapsed_ms:.2f}", ""
            ])

            annotated = draw_clean_id(annotated, tid, (x1, y1, x2, y2), is_anomaly)
            annotated = overlay_heatmap(annotated, amap, x1, y1, x2, y2, alpha=0.38)

        # clean stale
        for s in list(track_scores.keys()):
            if s not in active_ids:
                track_scores.pop(s, None)

        if frame_worst >= args.threshold:
            total_anomaly += 1

        t1 = time.perf_counter()
        elapsed_frame_ms = (t1 - t0) * 1000.0
        fps_log.append(1000.0 / max(elapsed_frame_ms, 1e-3))
        cur_fps = float(np.mean(fps_log[-30:]))
        csv_file.flush()

        annotated = draw_hud(annotated, frame_idx, cur_fps, frame_worst, args.threshold)
        writer.write(annotated)

        if not args.no_display:
            cv2.imshow("Gum Inspection - Clean", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if frame_idx % 20 == 0:
            print(f"  [{frame_idx:05d}]  fps={cur_fps:.1f}  score={frame_worst:.4f}")

    cap.release()
    writer.release()
    csv_file.close()

    # ── Per-ID summary ────────────────────────────────────────────────────
    summary_path = os.path.join(args.out_dir, "per_id_summary.csv")
    with open(summary_path, "w", newline="") as sf:
        sw = csv.writer(sf)
        sw.writerow([
            "track_id",
            "total_frames_seen",
            "anomaly_frames",
            "normal_frames",
            "anomaly_rate_%",
            "raw_score_min", "raw_score_max", "raw_score_mean",
            "ema_score_min", "ema_score_max", "ema_score_mean",
            "peak_ema_frame",
        ])
        for tid in sorted(track_raw_history.keys()):
            raws = track_raw_history[tid]
            emas = track_ema_history[tid]
            n = len(emas)
            anom_frames = sum(1 for s in emas if s >= args.threshold)
            peak_idx = int(np.argmax(emas))
            sw.writerow([
                tid, n,
                anom_frames, n - anom_frames,
                f"{100.0 * anom_frames / max(n, 1):.1f}",
                f"{min(raws):.6f}", f"{max(raws):.6f}", f"{np.mean(raws):.6f}",
                f"{min(emas):.6f}", f"{max(emas):.6f}", f"{np.mean(emas):.6f}",
                peak_idx,
            ])

    print(f"\nFinished! Output saved to: {args.out_dir}/output_video.mp4")
    print(f"   Frame CSV       → frame_scores.csv  ({frame_idx} frames, {len(track_raw_history)} unique tracks)")
    print(f"   Per-ID summary  → per_id_summary.csv")


if __name__ == "__main__":
    run(parse_args())