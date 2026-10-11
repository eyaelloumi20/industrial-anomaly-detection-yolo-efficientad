"""
video_inference_simple.py
=========================
Generic two-stage YOLO + EfficientAD video inference.
- No denoising
- No domain-specific assumptions
- ByteTrack multi-object tracking
- Per-track majority-vote final label
- Grace period for stale track cleanup
- Clean minimal annotation overlay

Usage:
    python video_inference_simple.py \
        --source path/to/video.mp4 \
        --ckpt_dir path/to/efficientad_ckpt \
        --yolo_weights path/to/yolo.pt \
        --threshold_file path/to/threshold.txt \
        --out_dir ./results
"""

import os
import sys
import csv
import time
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

IMAGE_SIZE   = 256
OUT_CHANNELS = 384


# ── Preprocessing ──────────────────────────────────────────────────────────────

def resize_with_padding(crop: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    """Aspect-ratio-preserving resize + black padding. Matches training pipeline."""
    ow, oh = crop.size
    scale  = size / max(ow, oh)
    nw, nh = int(ow * scale), int(oh * scale)
    resized = crop.resize((nw, nh), Image.LANCZOS)
    canvas  = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def get_crop(pil_img: Image.Image, box: tuple, pad_ratio: float = 0.02) -> Image.Image:
    """
    Crop with padding from a PIL image given an (x1,y1,x2,y2) box.
    No denoising — keeps it generic and avoids domain-specific tuning.
    """
    W, H = pil_img.size
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad_ratio), int(bh * pad_ratio)
    x1 = max(0, x1 - px);  y1 = max(0, y1 - py)
    x2 = min(W, x2 + px);  y2 = min(H, y2 + py)
    crop = pil_img.crop((x1, y1, x2, y2)).convert("RGB")
    return resize_with_padding(crop, IMAGE_SIZE)


# ── EfficientAD scoring ────────────────────────────────────────────────────────

def score_crop(models: dict, crop_pil: Image.Image, device: str, ratio: float = 0.1):
    """
    Returns (anomaly_map [H,W float32], scalar_score float).
    Student outputs 2*OUT_CHANNELS channels: first half = st, last half = stae.
    """
    t = transforms.ToTensor()(crop_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        t_out = models['teacher'](t)
        s_out = models['student'](t)
        a_out = models['ae'](t)

    C = OUT_CHANNELS
    y_st   = s_out[:, :C,  :, :]
    y_stae = s_out[:, C:,  :, :]          # last OUT_CHANNELS channels

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
    score = float(np.max(amap))
    return amap, score


# ── Drawing ────────────────────────────────────────────────────────────────────

def draw_box(frame: np.ndarray, tid: int, box: tuple, is_anomaly: bool) -> np.ndarray:
    x1, y1, x2, y2 = box
    color = (0, 0, 220) if is_anomaly else (0, 200, 60)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"ID{tid}"
    font  = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 0.50, 1)
    tx = max(0, x1 + 3)
    ty = max(th + 6, y1 + 3)
    cv2.rectangle(frame, (tx - 2, ty - th - 4), (tx + tw + 4, ty + 2), color, -1)
    cv2.putText(frame, label, (tx, ty), font, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def overlay_heatmap(frame: np.ndarray, amap: np.ndarray,
                    x1: int, y1: int, x2: int, y2: int,
                    alpha: float = 0.35) -> np.ndarray:
    w = max(1, x2 - x1);  h = max(1, y2 - y1)
    norm = ((amap - amap.min()) /
            (amap.max() - amap.min() + 1e-8) * 255).astype(np.uint8)
    heat = cv2.resize(cv2.applyColorMap(norm, cv2.COLORMAP_HOT), (w, h))
    roi  = frame[y1:y2, x1:x2]
    if roi.shape[0] > 0 and roi.shape[1] > 0:
        frame[y1:y2, x1:x2] = cv2.addWeighted(roi, 1 - alpha, heat, alpha, 0)
    return frame


def draw_hud(frame: np.ndarray, frame_idx: int, fps: float,
             worst: float, threshold: float) -> np.ndarray:
    h, w  = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    is_anom = worst >= threshold
    color   = (0, 0, 200) if is_anom else (30, 140, 30)
    status  = "ANOMALY" if is_anom else "NORMAL"
    hud     = f"Frame {frame_idx:05d} | FPS {fps:5.1f} | Score {worst:.4f} | {status}"
    cv2.putText(frame, hud, (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2, cv2.LINE_AA)
    return frame


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generic YOLO + EfficientAD video inference (no denoising)")

    p.add_argument("--source",         required=True,
                   help="Video path or 0 for webcam")
    p.add_argument("--ckpt_dir",       required=True,
                   help="EfficientAD checkpoint directory")
    p.add_argument("--yolo_weights",   required=True,
                   help="YOLO .pt weights file")
    p.add_argument("--out_dir",        default="./video_results",
                   help="Output directory (default: ./video_results)")
    p.add_argument("--category",       default="chewinggum",
                   help="EfficientAD category name (default: chewinggum)")
    p.add_argument("--model_size",     default="S", choices=["S", "M"],
                   help="EfficientAD PDN size (default: S)")

    # Threshold — file takes priority over explicit value
    p.add_argument("--threshold",      type=float, default=None,
                   help="Anomaly threshold. Ignored if --threshold_file is given.")
    p.add_argument("--threshold_file", default=None,
                   help="Path to threshold.txt (output of evaluate_pipeline.py). "
                        "Takes priority over --threshold.")

    # Inference parameters
    p.add_argument("--ratio",          type=float, default=0.1,
                   help="EfficientAD normalisation ratio (default: 0.1)")
    p.add_argument("--conf",           type=float, default=0.25,
                   help="YOLO confidence threshold (default: 0.25)")
    p.add_argument("--iou",            type=float, default=0.45,
                   help="YOLO NMS IoU threshold (default: 0.45)")
    p.add_argument("--pad_ratio",      type=float, default=0.02,
                   help="Crop padding ratio, matches evaluate_pipeline.py (default: 0.02)")

    # Tracking / smoothing
    p.add_argument("--ema_alpha",      type=float, default=0.5,
                   help="EMA weight for score smoothing — higher = more reactive "
                        "(default: 0.5)")
    p.add_argument("--vote_threshold", type=float, default=0.5,
                   help="Min anomaly frame fraction for track-level ANOMALY verdict "
                        "(default: 0.5)")
    p.add_argument("--min_frames",     type=int,   default=3,
                   help="Min frames before a track gets a final verdict (default: 3)")
    p.add_argument("--grace_period",   type=int,   default=5,
                   help="Frames to keep track EMA alive after last detection (default: 5)")

    # Visualisation
    p.add_argument("--heatmap_alpha",  type=float, default=0.35,
                   help="Heatmap overlay opacity (default: 0.35)")
    p.add_argument("--no_heatmap",     action="store_true",
                   help="Disable heatmap overlay (faster)")
    p.add_argument("--no_display",     action="store_true",
                   help="Disable live preview window")
    p.add_argument("--fps",            type=float, default=10.0,
                   help="FPS fallback when video header is missing (default: 10.0)")
    p.add_argument("--device",         default="cuda",
                   help="Compute device (default: cuda, auto-falls back to cpu)")
    return p.parse_args()


# ── Main loop ──────────────────────────────────────────────────────────────────

def run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    # Resolve threshold --------------------------------------------------------
    threshold = args.threshold
    if args.threshold_file and os.path.isfile(args.threshold_file):
        threshold = float(open(args.threshold_file).read().strip())
        print(f"[Inference] Threshold from file : {threshold:.6f}  ({args.threshold_file})")
    elif threshold is not None:
        print(f"[Inference] Threshold (--threshold) : {threshold:.6f}")
    else:
        threshold = 0.5
        print(f"[Inference] WARNING — no threshold specified, using fallback {threshold}.")
        print(f"            Run evaluate_pipeline.py first and pass --threshold_file.")
    args.threshold = threshold

    # Load models --------------------------------------------------------------
    print(f"\n[Inference] Loading EfficientAD (PDN-{args.model_size}) ...")
    models = load_efficientad_model(
        ckpt_dir   = args.ckpt_dir,
        category   = args.category,
        model_size = args.model_size,
        device     = device,
    )

    print(f"[Inference] Loading YOLO + ByteTrack ...")
    from ultralytics import YOLO
    yolo = YOLO(args.yolo_weights)
    yolo.to(device)

    # Open video ---------------------------------------------------------------
    source = int(args.source) if args.source.isdigit() else args.source
    cap    = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Inference] Source : {source}  |  {src_w}×{src_h} @ {src_fps:.1f} fps")

    # Output writers -----------------------------------------------------------
    out_video = os.path.join(args.out_dir, "output_video.mp4")
    writer    = cv2.VideoWriter(
        out_video, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (src_w, src_h))

    csv_path = os.path.join(args.out_dir, "frame_scores.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_w    = csv.writer(csv_file)
    csv_w.writerow([
        "frame", "track_id",
        "raw_score", "ema_score", "frame_prediction",
        "x1", "y1", "x2", "y2",
        "frame_ms",
    ])

    # Per-track state ----------------------------------------------------------
    # ema_state  : tid -> current EMA score
    # last_seen  : tid -> last frame the track was active (for grace period)
    # raw_hist   : tid -> [raw scores]
    # ema_hist   : tid -> [ema scores]
    ema_state  = {}
    last_seen  = {}
    raw_hist   = {}
    ema_hist   = {}

    frame_idx      = 0
    fps_log        = []

    print(f"\n[Inference] Running — press Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0           = time.perf_counter()
        frame_worst  = 0.0
        active_ids   = set()

        # Convert once per frame (not once per detection)
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        annotated = frame.copy()

        # YOLO + ByteTrack detection -------------------------------------------
        results = yolo.track(
            source      = frame,
            conf        = args.conf,
            iou         = args.iou,
            classes     = [0],
            persist     = True,
            tracker     = "bytetrack.yaml",
            verbose     = False,
        )

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue

            boxes     = r.boxes.xyxy.cpu().numpy().astype(int)
            track_ids = (r.boxes.id.cpu().numpy().astype(int)
                         if r.boxes.id is not None
                         else list(range(len(boxes))))

            for box, tid in zip(boxes, track_ids):
                x1 = max(0,     box[0]);  y1 = max(0,     box[1])
                x2 = min(src_w, box[2]);  y2 = min(src_h, box[3])
                if x2 <= x1 or y2 <= y1:
                    continue

                active_ids.add(tid)
                last_seen[tid] = frame_idx

                # Crop + score -------------------------------------------------
                crop       = get_crop(pil_frame, (x1, y1, x2, y2), args.pad_ratio)
                amap, raw  = score_crop(models, crop, device, ratio=args.ratio)

                # EMA smoothing (initialise with raw on first appearance) ------
                prev_ema  = ema_state.get(tid, raw)
                ema_score = args.ema_alpha * raw + (1 - args.ema_alpha) * prev_ema
                ema_state[tid] = ema_score
                frame_worst    = max(frame_worst, ema_score)

                raw_hist.setdefault(tid, []).append(raw)
                ema_hist.setdefault(tid, []).append(ema_score)

                is_anomaly  = ema_score >= args.threshold
                prediction  = "anomaly" if is_anomaly else "normal"

                t_box = time.perf_counter()
                csv_w.writerow([
                    frame_idx, tid,
                    f"{raw:.6f}", f"{ema_score:.6f}", prediction,
                    x1, y1, x2, y2,
                    f"{(t_box - t0) * 1000:.1f}",
                ])

                annotated = draw_box(annotated, tid, (x1, y1, x2, y2), is_anomaly)
                if not args.no_heatmap:
                    annotated = overlay_heatmap(annotated, amap, x1, y1, x2, y2,
                                                alpha=args.heatmap_alpha)

        # Grace-period stale track cleanup ------------------------------------
        # Keep EMA alive for `grace_period` frames after last detection;
        # this avoids losing a track's smoothed history on a single missed frame.
        for tid in list(ema_state.keys()):
            if tid not in active_ids:
                if frame_idx - last_seen.get(tid, 0) > args.grace_period:
                    ema_state.pop(tid, None)

        # HUD + write frame ----------------------------------------------------
        t1              = time.perf_counter()
        frame_ms        = (t1 - t0) * 1000.0
        fps_log.append(1000.0 / max(frame_ms, 1e-3))
        cur_fps         = float(np.mean(fps_log[-30:]))

        annotated = draw_hud(annotated, frame_idx, cur_fps, frame_worst, args.threshold)
        writer.write(annotated)

        if not args.no_display:
            cv2.imshow("Inspection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  [{frame_idx:05d}]  fps={cur_fps:.1f}  score={frame_worst:.4f}  "
                  f"active_tracks={len(active_ids)}")

    cap.release()
    writer.release()
    csv_file.close()
    if not args.no_display:
        cv2.destroyAllWindows()

    # ── Per-track summary with majority-vote final label ──────────────────────
    summary_path = os.path.join(args.out_dir, "per_track_summary.csv")
    with open(summary_path, "w", newline="") as sf:
        sw = csv.writer(sf)
        sw.writerow([
            "track_id",
            "total_frames",
            "anomaly_frames", "normal_frames",
            "anomaly_rate_%",
            "final_label",          # majority-vote verdict
            "verdict_confidence",   # how far from 50% the vote was
            "raw_min", "raw_max", "raw_mean",
            "ema_min", "ema_max", "ema_mean",
            "peak_ema_score", "peak_ema_frame",
        ])
        for tid in sorted(raw_hist.keys()):
            raws  = raw_hist[tid]
            emas  = ema_hist[tid]
            n     = len(emas)

            anom_frames  = sum(1 for s in emas if s >= args.threshold)
            anom_rate    = anom_frames / max(n, 1)

            # Majority vote — requires min_frames for a real verdict
            if n < args.min_frames:
                final_label = "uncertain"
                confidence  = 0.0
            else:
                final_label = "anomaly" if anom_rate >= args.vote_threshold else "normal"
                confidence  = abs(anom_rate - 0.5) * 2   # 0=borderline, 1=unanimous

            peak_frame = int(np.argmax(emas))
            sw.writerow([
                tid, n,
                anom_frames, n - anom_frames,
                f"{anom_rate * 100:.1f}",
                final_label,
                f"{confidence:.3f}",
                f"{min(raws):.6f}", f"{max(raws):.6f}", f"{np.mean(raws):.6f}",
                f"{min(emas):.6f}", f"{max(emas):.6f}", f"{np.mean(emas):.6f}",
                f"{max(emas):.6f}", peak_frame,
            ])

    n_anomaly_tracks = sum(
        1 for tid in raw_hist
        if len(ema_hist[tid]) >= args.min_frames
        and (sum(1 for s in ema_hist[tid] if s >= args.threshold)
             / len(ema_hist[tid])) >= args.vote_threshold
    )

    print(f"\n=== Done ===")
    print(f"  Frames processed : {frame_idx}")
    print(f"  Unique tracks    : {len(raw_hist)}")
    print(f"  Anomaly tracks   : {n_anomaly_tracks}  "
          f"(vote_threshold={args.vote_threshold}, min_frames={args.min_frames})")
    print(f"  Video            → {out_video}")
    print(f"  Frame CSV        → {csv_path}")
    print(f"  Track summary    → {summary_path}")


if __name__ == "__main__":
    run(parse_args())
