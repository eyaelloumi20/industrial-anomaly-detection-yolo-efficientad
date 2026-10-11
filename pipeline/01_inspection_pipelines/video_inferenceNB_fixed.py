"""
video_inferenceNB_fixed.py  (ByteTrack + frame-level rembg)
============================================================

KEY FIX — Training/Inference Preprocessing Mismatch
----------------------------------------------------
During training (batch_image_processor.py), rembg was applied to FULL
RESOLUTION photos BEFORE any cropping. The EfficientAD model therefore
learned to score background-free 256×256 crops.

The original video_inferenceNB.py applied rembg AFTER cropping to a small
YOLO bounding box. This caused three stacked errors:

  1. ASYNC LAG: new track IDs are always scored on a raw (bg-present) crop
     for their first frame, because rembg_mgr.get(tid) returns None until
     the background thread finishes.

  2. SMALL CROP DEGRADATION: rembg segmentation quality drops significantly
     on tiny bbox regions (e.g. 60×90 px). During training, rembg ran on
     full ~1080p frames and produced clean masks.

  3. OTSU CONFLICT: post_process_crop() runs Otsu on the already-bg-removed
     crop. The black background can trick Otsu into aggressively
     over-cropping the subject.

FIX STRATEGY — match training exactly:
  - Run rembg once per frame on the full-resolution frame.
  - Use this background-free frame for ALL crop extractions.
  - Remove Otsu from the scoring path (it was designed for raw frames, not
    already-cleaned ones; keep it only as a fallback when rembg is off).
  - Background thread now processes the FULL frame, not individual crops.
    One rembg call per frame (instead of one per track) is also faster.

Usage (same CLI as before):
    python video_inferenceNB_fixed.py \
        --source video.mp4 \
        --ckpt_dir ./checkpoints \
        --yolo_weights best.pt \
        --out_dir ./video_results_fixed \
        --category chewinggum \
        --threshold 1.971

    # Disable background removal (if model was trained on raw crops):
    python video_inferenceNB_fixed.py ... --no_rembg
"""

import os
import sys
import csv
import time
import argparse
import concurrent.futures
from io import BytesIO
from collections import deque

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


# ====================== BACKGROUND REMOVAL ======================

def _hex_to_rgba(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (r, g, b, 255)


def init_rembg_session(model_name: str = "u2netp"):
    from rembg import new_session
    print(f"[rembg] Loading {model_name} model ...")
    session = new_session(model_name)
    print("[rembg] Model ready.")
    return session


def remove_bg_full_frame(
    pil_frame: Image.Image,
    rembg_session,
    bg_color_hex: str = "#000000",
) -> Image.Image:
    """
    Remove background from a FULL FRAME and composite onto bg_color.
    This matches batch_image_processor.py which processed full photos.
    Called only from the background thread.
    """
    from rembg import remove as rembg_remove
    buf = BytesIO()
    pil_frame.save(buf, format="PNG")
    result_bytes = rembg_remove(buf.getvalue(), session=rembg_session)
    rgba = Image.open(BytesIO(result_bytes)).convert("RGBA")
    bg_rgba = _hex_to_rgba(bg_color_hex)
    background = Image.new("RGBA", rgba.size, bg_rgba)
    background.paste(rgba, mask=rgba.split()[3])
    return background.convert("RGB")


# ====================== ASYNC FRAME-LEVEL REMBG MANAGER ======================

class AsyncFrameRembg:
    """
    Processes FULL FRAMES in a background thread.
    The main loop uses the bg-removed frame from the PREVIOUS frame index
    and submits the current frame immediately after.

    This is fundamentally different from the original per-crop approach:
      - One rembg call per frame (not per track) → less total work
      - Crops are extracted from the cleaned full frame → matches training
      - No per-track async state needed
    """

    def __init__(self, rembg_session, bg_color_hex: str, max_workers: int = 1):
        self.session      = rembg_session
        self.bg_color_hex = bg_color_hex
        self._executor    = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._future      = None        # Future for the current in-flight job
        self._cached_frame: Image.Image | None = None  # last completed result
        self._timings     = deque(maxlen=60)

    def submit(self, pil_frame: Image.Image):
        """Queue rembg for a full frame. Non-blocking. Call once per frame."""
        if self._future is not None and not self._future.done():
            self._future.cancel()

        frame_copy = pil_frame.copy()
        session    = self.session
        bg         = self.bg_color_hex

        def _run():
            t0  = time.perf_counter()
            out = remove_bg_full_frame(frame_copy, session, bg)
            self._timings.append((time.perf_counter() - t0) * 1000.0)
            return out

        self._future = self._executor.submit(_run)

    def get(self) -> Image.Image | None:
        """
        Return the last completed bg-removed frame, or None if not ready.
        Drains the finished future into the cache.
        """
        if self._future is not None and self._future.done():
            try:
                self._cached_frame = self._future.result()
            except Exception as e:
                print(f"[rembg] Frame processing error: {e}")
            self._future = None
        return self._cached_frame

    def avg_rembg_ms(self) -> float:
        return float(np.mean(self._timings)) if self._timings else 0.0

    def shutdown(self):
        self._executor.shutdown(wait=False)


# ====================== PREPROCESSING ======================

def resize_with_padding(crop: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    ow, oh = crop.size
    scale = size / max(ow, oh)
    nw, nh = int(ow * scale), int(oh * scale)
    resized = crop.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def extract_and_preprocess_crop(
    pil_frame: Image.Image,
    box: tuple,
    pad_ratio: float = 0.05,
    denoise: bool = True,
    use_otsu: bool = False,
) -> Image.Image:
    """
    Extract a crop from pil_frame (which may already be bg-removed)
    and apply resize + optional denoise.

    use_otsu: disabled by default when rembg is active (the black
    background that Otsu was designed to remove is already gone, and
    re-running it can aggressively over-crop the subject).
    Enable it only when rembg is off and the frame still has background.
    """
    W, H = pil_frame.size
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    px = int(bw * pad_ratio)
    py = int(bh * pad_ratio)
    x1 = max(0, x1 - px);  y1 = max(0, y1 - py)
    x2 = min(W, x2 + px);  y2 = min(H, y2 + py)

    crop = pil_frame.crop((x1, y1, x2, y2)).convert("RGB")

    # Otsu: only useful when the crop still contains background pixels.
    # Skip when rembg has already cleaned the frame.
    if use_otsu:
        arr_gray = np.array(crop.convert("L"), dtype=np.uint8)
        _, mask  = cv2.threshold(arr_gray, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        coords = cv2.findNonZero(mask)
        if coords is not None:
            rx, ry, rw, rh = cv2.boundingRect(coords)
            coverage = (rw * rh) / (arr_gray.shape[1] * arr_gray.shape[0])
            if coverage >= 0.30:
                margin = 2
                rx = max(0, rx - margin)
                ry = max(0, ry - margin)
                rw = min(crop.width  - rx, rw + 2 * margin)
                rh = min(crop.height - ry, rh + 2 * margin)
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


# ====================== SCORING ======================

def score_crop(models, crop_pil, device, ratio=0.1):
    t = transforms.ToTensor()(crop_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        t_out = models["teacher"](t)
        s_out = models["student"](t)
        a_out = models["ae"](t)

    y_st   = s_out[:, :OUT_CHANNELS, :, :]
    y_stae = s_out[:, -OUT_CHANNELS:, :, :]

    norm_t = (t_out - models["mean"]) / (models["std"] + 1e-8)
    d_st   = torch.pow(norm_t - y_st,   2)
    d_stae = torch.pow(a_out  - y_stae, 2)

    fm_st   = F.interpolate(torch.mean(d_st,   dim=1, keepdim=True),
                            size=(IMAGE_SIZE, IMAGE_SIZE),
                            mode="bilinear", align_corners=False)
    fm_stae = F.interpolate(torch.mean(d_stae, dim=1, keepdim=True),
                            size=(IMAGE_SIZE, IMAGE_SIZE),
                            mode="bilinear", align_corners=False)

    nm_st = (ratio * (fm_st   - models["qa_st"])) / (models["qb_st"] - models["qa_st"] + 1e-8)
    nm_ae = (ratio * (fm_stae - models["qa_ae"])) / (models["qb_ae"] - models["qa_ae"] + 1e-8)

    amap = (0.5 * nm_st + 0.5 * nm_ae)[0, 0].cpu().numpy()
    return amap, float(np.max(amap))


# ====================== DRAWING ======================

def draw_clean_id(frame, tid, box, is_anomaly):
    x1, y1, x2, y2 = box
    color = (0, 0, 220) if is_anomaly else (0, 200, 60)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    tag = f"ID{tid}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(tag, font, 0.55, 2)
    tx = max(0, x1 + 4)
    ty = max(th + 8, y1 + 4)
    cv2.rectangle(frame, (tx - 2, ty - th - 4), (tx + tw + 4, ty + 4), color, -1)
    cv2.putText(frame, tag, (tx, ty), font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def overlay_heatmap(frame, amap, x1, y1, x2, y2, alpha=0.38):
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    norm = ((amap - amap.min()) / (amap.max() - amap.min() + 1e-8) * 255).astype(np.uint8)
    heat = cv2.resize(cv2.applyColorMap(norm, cv2.COLORMAP_HOT), (w, h))
    roi  = frame[y1:y2, x1:x2]
    if roi.shape[0] > 0 and roi.shape[1] > 0:
        frame[y1:y2, x1:x2] = cv2.addWeighted(roi, 1 - alpha, heat, alpha, 0)
    return frame


def draw_hud(frame, frame_idx, fps, worst, threshold, rembg_on, rembg_ms=0.0,
             using_cached=False):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 72), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    is_anom = worst >= threshold
    color   = (0, 0, 200) if is_anom else (30, 140, 30)
    status  = "!!! ANOMALY DETECTED !!!" if is_anom else "NORMAL"
    if rembg_on:
        cache_tag = "[cached ✓]" if using_cached else "[raw — rembg pending]"
        bg_tag = f"[rembg ON | {rembg_ms:.0f}ms/frame avg | {cache_tag}]"
    else:
        bg_tag = "[rembg OFF — raw crops]"
    line1 = f"Frame {frame_idx:05d} | FPS {fps:5.1f} | Score {worst:.4f} | {status}"
    line2 = bg_tag
    cv2.putText(frame, line1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, color, 2, cv2.LINE_AA)
    cv2.putText(frame, line2, (10, 54), cv2.FONT_HERSHEY_SIMPLEX,
                0.44, (180, 180, 180), 1, cv2.LINE_AA)
    return frame


# ====================== CLI ======================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source",         required=True)
    p.add_argument("--ckpt_dir",       required=True)
    p.add_argument("--yolo_weights",   required=True)
    p.add_argument("--out_dir",        default="./video_results_fixed")
    p.add_argument("--category",       default="chewinggum")
    p.add_argument("--model_size",     default="S", choices=["S", "M"])
    p.add_argument("--threshold",      type=float, default=1.971)
    p.add_argument("--ema_alpha",      type=float, default=0.3)
    p.add_argument("--conf",           type=float, default=0.25)
    p.add_argument("--iou",            type=float, default=0.45)
    p.add_argument("--pad_ratio",      type=float, default=0.05)
    p.add_argument("--fps",            type=float, default=10.0)
    p.add_argument("--no_display",     action="store_true")
    p.add_argument("--device",         default="cuda")
    p.add_argument("--no_rembg",       action="store_true",
                   help="Disable background removal. Use this only if the model "
                        "was trained on raw (non-bg-removed) crops.")
    p.add_argument("--rembg_bg_color", default="#000000",
                   help="Background fill colour after removal (default: black)")
    p.add_argument("--rembg_model",    default="u2netp",
                   help="rembg model name: u2netp (fast), u2net, isnet-general-use, "
                        "birefnet-general (default: u2netp)")
    return p.parse_args()


# ====================== RUN ======================

def run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"\n[Video Fixed] Loading EfficientAD ...")
    models = load_efficientad_model(
        ckpt_dir=args.ckpt_dir, category=args.category,
        model_size=args.model_size, device=device)

    print(f"[Video Fixed] Loading YOLOv8 + ByteTrack ...")
    from ultralytics import YOLO
    yolo = YOLO(args.yolo_weights)
    yolo.to(device)

    # ── Frame-level rembg manager ────────────────────────────────────────
    rembg_mgr = None
    if not args.no_rembg:
        rembg_session = init_rembg_session(args.rembg_model)
        rembg_mgr = AsyncFrameRembg(
            rembg_session=rembg_session,
            bg_color_hex=args.rembg_bg_color,
            max_workers=1,   # one worker is enough; one frame at a time
        )
        print(f"[rembg] Async FRAME-LEVEL rembg ready — model={args.rembg_model}")
        print(f"  Fix: rembg now processes full frames (matches training).")
    else:
        print("[rembg] Disabled — scoring raw crops.")

    source  = int(args.source) if args.source.isdigit() else args.source
    cap     = cv2.VideoCapture(source)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = os.path.join(args.out_dir, "output_video.mp4")
    writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                               src_fps, (src_w, src_h))

    csv_path   = os.path.join(args.out_dir, "frame_scores.csv")
    csv_file   = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame", "track_id",
        "raw_score", "ema_score", "prediction",
        "x1", "y1", "x2", "y2",
        "width", "height", "area",
        "frame_ms", "frame_fps",
        "rembg_frame_cached",   # 1 = bg-removed frame was available, 0 = raw fallback
    ])

    track_scores      = {}
    track_raw_history = {}
    track_ema_history = {}
    frame_idx = 0
    fps_log   = []
    yolo_times  = deque(maxlen=30)
    score_times = deque(maxlen=30)

    print(f"\n[Video Fixed] Running — Press Q to quit.")
    print(f"  threshold : {args.threshold}")
    print(f"  ema_alpha : {args.ema_alpha}")
    print(f"  rembg     : {'FRAME-LEVEL ' + args.rembg_model if rembg_mgr else 'OFF'}\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_frame_start = time.perf_counter()
        frame_worst   = 0.0
        active_ids    = set()

        # ── Convert raw frame to PIL (full resolution) ────────────────────
        pil_frame_raw = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # ── Get bg-removed frame (or fall back to raw) ────────────────────
        # Try to get the result of the PREVIOUS frame's rembg job.
        if rembg_mgr is not None:
            pil_frame_clean = rembg_mgr.get()       # non-blocking
            using_cached    = pil_frame_clean is not None
            if not using_cached:
                # No result yet (first frame). Fall back to raw for this frame.
                pil_frame_clean = pil_frame_raw
            # Submit current frame for the NEXT iteration.
            rembg_mgr.submit(pil_frame_raw)
        else:
            pil_frame_clean = pil_frame_raw
            using_cached    = False

        # ── YOLO detection always runs on the original frame ──────────────
        # (Detectors generalise better on the original image;
        #  bg removal is only needed for the anomaly scorer.)
        t_yolo0 = time.perf_counter()
        results = yolo.track(
            source=frame,
            conf=args.conf,
            iou=args.iou,
            classes=[0],
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        yolo_times.append((time.perf_counter() - t_yolo0) * 1000.0)

        annotated = frame.copy()

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            boxes     = r.boxes.xyxy.cpu().numpy().astype(int)
            track_ids = (r.boxes.id.cpu().numpy().astype(int)
                         if r.boxes.id is not None else list(range(len(boxes))))

            for box, tid in zip(boxes, track_ids):
                x1 = max(0, box[0]);    y1 = max(0, box[1])
                x2 = min(src_w, box[2]); y2 = min(src_h, box[3])
                if x2 <= x1 or y2 <= y1:
                    continue

                active_ids.add(tid)

                # ── Extract crop from the CLEANED frame ──────────────────
                # use_otsu=False because the black background Otsu was
                # designed to remove is already gone after rembg.
                # use_otsu=True only when rembg is disabled.
                crop = extract_and_preprocess_crop(
                    pil_frame=pil_frame_clean,
                    box=(x1, y1, x2, y2),
                    pad_ratio=args.pad_ratio,
                    denoise=True,
                    use_otsu=(rembg_mgr is None),
                )

                # ── Score ──────────────────────────────────────────────────
                t_score0 = time.perf_counter()
                amap, raw = score_crop(models, crop, device)
                score_times.append((time.perf_counter() - t_score0) * 1000.0)

                prev_ema  = track_scores.get(tid, raw)
                ema_score = args.ema_alpha * raw + (1 - args.ema_alpha) * prev_ema
                track_scores[tid] = ema_score
                frame_worst = max(frame_worst, ema_score)

                track_raw_history.setdefault(tid, []).append(raw)
                track_ema_history.setdefault(tid, []).append(ema_score)

                is_anomaly = ema_score >= args.threshold
                prediction = "anomaly" if is_anomaly else "normal"
                box_w, box_h = x2 - x1, y2 - y1
                elapsed_ms   = (time.perf_counter() - t_frame_start) * 1000.0

                csv_writer.writerow([
                    frame_idx, tid,
                    f"{raw:.6f}", f"{ema_score:.6f}", prediction,
                    x1, y1, x2, y2,
                    box_w, box_h, box_w * box_h,
                    f"{elapsed_ms:.2f}", "",
                    int(using_cached),
                ])

                annotated = draw_clean_id(annotated, tid, (x1, y1, x2, y2), is_anomaly)
                annotated = overlay_heatmap(annotated, amap, x1, y1, x2, y2, alpha=0.38)

        # ── Evict stale tracks ────────────────────────────────────────────
        for s in list(track_scores.keys()):
            if s not in active_ids:
                track_scores.pop(s, None)

        t1 = time.perf_counter()
        fps_log.append(1000.0 / max((t1 - t_frame_start) * 1000.0, 1e-3))
        cur_fps = float(np.mean(fps_log[-30:]))
        csv_file.flush()

        rembg_ms = rembg_mgr.avg_rembg_ms() if rembg_mgr else 0.0
        annotated = draw_hud(annotated, frame_idx, cur_fps, frame_worst,
                             args.threshold, rembg_mgr is not None,
                             rembg_ms, using_cached)
        writer.write(annotated)

        if not args.no_display:
            cv2.imshow("Gum Inspection [Fixed Preprocessing]", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if frame_idx % 20 == 0:
            avg_yolo  = float(np.mean(yolo_times))  if yolo_times  else 0.0
            avg_score = float(np.mean(score_times)) if score_times else 0.0
            cached_pct = int(using_cached) * 100
            print(
                f"  [{frame_idx:05d}]  fps={cur_fps:.1f}  score={frame_worst:.4f}"
                f"  | yolo={avg_yolo:.1f}ms  rembg≈{rembg_ms:.1f}ms/frame"
                f"  score={avg_score:.1f}ms  cached={cached_pct}%"
            )

    cap.release()
    writer.release()
    csv_file.close()
    if rembg_mgr:
        rembg_mgr.shutdown()

    # ── Per-ID summary ────────────────────────────────────────────────────
    summary_path = os.path.join(args.out_dir, "per_id_summary.csv")
    with open(summary_path, "w", newline="") as sf:
        sw = csv.writer(sf)
        sw.writerow([
            "track_id", "total_frames_seen",
            "anomaly_frames", "normal_frames", "anomaly_rate_%",
            "raw_score_min", "raw_score_max", "raw_score_mean",
            "ema_score_min", "ema_score_max", "ema_score_mean",
            "peak_ema_frame",
        ])
        for tid in sorted(track_raw_history.keys()):
            raws = track_raw_history[tid]
            emas = track_ema_history[tid]
            n    = len(emas)
            anom = sum(1 for s in emas if s >= args.threshold)
            sw.writerow([
                tid, n, anom, n - anom,
                f"{100.0 * anom / max(n, 1):.1f}",
                f"{min(raws):.6f}", f"{max(raws):.6f}", f"{np.mean(raws):.6f}",
                f"{min(emas):.6f}", f"{max(emas):.6f}", f"{np.mean(emas):.6f}",
                int(np.argmax(emas)),
            ])

    print(f"\n=== Finished ===")
    print(f"  Video          → {out_path}")
    print(f"  Frame CSV      → frame_scores.csv  "
          f"({frame_idx} frames, {len(track_raw_history)} unique tracks)")
    print(f"  Per-ID summary → per_id_summary.csv")


if __name__ == "__main__":
    run(parse_args())
