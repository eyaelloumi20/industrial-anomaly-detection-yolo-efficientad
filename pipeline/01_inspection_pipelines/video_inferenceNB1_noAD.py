"""
video_inferenceNB1.py  (DeepOCSORT + async rembg preprocessing)
===============================================================
Speed improvements vs original:
  - rembg runs in a background ThreadPoolExecutor (non-blocking).
    The main loop never waits for BG removal — it uses the result
    from the PREVIOUS frame for each track ID and submits a new job
    in parallel. Display stays real-time.
  - Model switched to "u2netp" (lite, ~6x faster than u2net) with
    minimal quality loss on small crops.
  - PIL frame conversion is done once per frame (not per track).
  - Timing breakdown printed every 20 frames: yolo_ms / rembg_ms / score_ms.

CLI flags (unchanged):
    --no_rembg         skip background removal entirely
    --rembg_bg_color   hex colour to fill behind subject (default: #000000)
    --rembg_workers    thread-pool size for async rembg (default: 2)

Usage:
    python video_inferenceNB1_noAD.py \
        --source video.mp4 \
        --yolo_weights best.pt \
        --out_dir ./video_results
"""

import os
import sys
import csv
import time
import argparse
import concurrent.futures
from io import BytesIO
from pathlib import Path
from collections import deque

import cv2
import numpy as np
from PIL import Image

IMAGE_SIZE = 256


# ====================== BACKGROUND REMOVAL HELPERS ======================

def _hex_to_rgba(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (r, g, b, 255)


def init_rembg_session():
    """Load the u2netp (lite) model once at startup — ~6x faster than u2net."""
    from rembg import new_session
    print("[rembg] Loading u2netp model (lite, ~6x faster than u2net) ...")
    session = new_session("u2netp")
    print("[rembg] Model ready.")
    return session


def remove_bg_black(pil_img: Image.Image, rembg_session, bg_color_hex: str = "#000000") -> Image.Image:
    """
    Remove background via rembg and composite onto a solid colour.
    Called only from background threads — never blocks the main loop.
    """
    from rembg import remove as rembg_remove
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    result_bytes = rembg_remove(buf.getvalue(), session=rembg_session)
    rgba = Image.open(BytesIO(result_bytes)).convert("RGBA")
    bg_rgba = _hex_to_rgba(bg_color_hex)
    background = Image.new("RGBA", rgba.size, bg_rgba)
    background.paste(rgba, mask=rgba.split()[3])
    return background.convert("RGB")


# ====================== ASYNC REMBG MANAGER ======================

class AsyncRembgManager:
    """
    Manages a pool of background threads that run rembg.
    The main loop calls `get(tid)` to retrieve the latest finished result
    and `submit(tid, crop)` to queue new work — both are non-blocking.
    """

    def __init__(self, rembg_session, bg_color_hex: str, max_workers: int = 2):
        self.session      = rembg_session
        self.bg_color_hex = bg_color_hex
        self._executor    = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._futures     = {}   # tid -> Future
        self._cache       = {}   # tid -> latest completed PIL crop
        self._timings     = deque(maxlen=60)

    def submit(self, tid: int, crop: Image.Image):
        """Queue a new rembg job for track `tid`. Non-blocking."""
        old = self._futures.get(tid)
        if old is not None and not old.done():
            old.cancel()

        crop_copy = crop.copy()
        session   = self.session
        bg        = self.bg_color_hex

        def _run():
            t0  = time.perf_counter()
            out = remove_bg_black(crop_copy, session, bg)
            self._timings.append((time.perf_counter() - t0) * 1000.0)
            return out

        self._futures[tid] = self._executor.submit(_run)

    def get(self, tid: int) -> Image.Image | None:
        """
        Return the latest completed rembg result for `tid`, or None if
        no result is ready yet. Drains finished futures into the cache.
        """
        fut = self._futures.get(tid)
        if fut is not None and fut.done():
            try:
                self._cache[tid] = fut.result()
            except Exception:
                pass
            del self._futures[tid]
        return self._cache.get(tid)

    def evict(self, tid: int):
        """Remove stale track data to prevent memory growth."""
        self._futures.pop(tid, None)
        self._cache.pop(tid, None)

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


def get_raw_crop(pil_img: Image.Image, box: tuple, pad_ratio: float = 0.05) -> Image.Image:
    """Extract and pad a crop — no BG removal. Used to feed AsyncRembgManager."""
    W, H = pil_img.size
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    px = int(bw * pad_ratio)
    py = int(bh * pad_ratio)
    x1 = max(0, x1 - px); y1 = max(0, y1 - py)
    x2 = min(W, x2 + px); y2 = min(H, y2 + py)
    return pil_img.crop((x1, y1, x2, y2)).convert("RGB")


def post_process_crop(crop: Image.Image, denoise: bool = True) -> Image.Image:
    """
    Otsu refinement + aspect-ratio fix + resize + optional denoise.
    Applied AFTER background removal (or on the raw crop when rembg is off).
    """
    arr_gray = np.array(crop.convert("L"), dtype=np.uint8)
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
    """Anomaly detection disabled — returns a zero score and blank heatmap."""
    amap = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    return amap, 0.0


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


def draw_hud(frame, frame_idx, fps, worst, threshold, rembg_on, rembg_ms=0.0):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 62), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    is_anom = worst >= threshold
    color  = (0, 0, 200) if is_anom else (30, 140, 30)
    status = "!!! ANOMALY DETECTED !!!" if is_anom else "NORMAL"
    bg_tag = f"[rembg ON | async ~{rembg_ms:.0f}ms/crop]" if rembg_on else "[rembg OFF]"
    line1 = f"Frame {frame_idx:05d} | FPS {fps:5.1f} | Score {worst:.4f} | {status}"
    line2 = bg_tag
    cv2.putText(frame, line1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)
    cv2.putText(frame, line2, (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)
    return frame


# ====================== CLI ======================

def parse_args():
    p = argparse.ArgumentParser(description="DeepOCSORT inference — anomaly detection disabled")
    p.add_argument("--source",          required=True)
    p.add_argument("--yolo_weights",    required=True)
    p.add_argument("--out_dir",         default="./results_video_clean_1")
    p.add_argument("--threshold",       type=float, default=0.5)
    p.add_argument("--ema_alpha",       type=float, default=0.3)
    p.add_argument("--conf",            type=float, default=0.25)
    p.add_argument("--iou",             type=float, default=0.45)
    p.add_argument("--pad_ratio",       type=float, default=0.05)
    p.add_argument("--fps",             type=float, default=10.0)
    p.add_argument("--no_display",      action="store_true")
    p.add_argument("--device",          default="cuda")
    p.add_argument("--no_rembg",        action="store_true",
                   help="Disable background removal on crops")
    p.add_argument("--rembg_bg_color",  default="#000000",
                   help="Hex colour to fill behind subject after BG removal (default: black)")
    p.add_argument("--rembg_workers",   type=int, default=2,
                   help="Thread-pool size for async rembg (default: 2)")
    return p.parse_args()


# ====================== RUN ======================

def run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device if __import__("torch").cuda.is_available() else "cpu"
    models = None  # anomaly detection disabled

    print(f"[Video] Loading YOLOv8 + DeepOCSORT ...")
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

    # ── Async rembg manager ───────────────────────────────────────────────
    rembg_mgr = None
    if not args.no_rembg:
        rembg_session = init_rembg_session()
        rembg_mgr = AsyncRembgManager(
            rembg_session=rembg_session,
            bg_color_hex=args.rembg_bg_color,
            max_workers=args.rembg_workers,
        )
        print(f"[rembg] Async pool started — {args.rembg_workers} worker(s), model=u2netp.")
    else:
        print("[rembg] Background removal DISABLED.")

    source  = int(args.source) if str(args.source).isdigit() else args.source
    cap     = cv2.VideoCapture(source)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = os.path.join(args.out_dir, "output_video.mp4")
    writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (src_w, src_h))

    csv_path   = os.path.join(args.out_dir, "frame_scores.csv")
    csv_file   = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame", "track_id",
        "raw_score", "ema_score", "prediction",
        "x1", "y1", "x2", "y2",
        "width", "height", "area",
        "frame_ms", "frame_fps",
        "rembg_cached",
    ])

    track_scores      = {}
    track_raw_history = {}
    track_ema_history = {}
    frame_idx = 0
    fps_log   = []

    yolo_times  = deque(maxlen=30)
    score_times = deque(maxlen=30)

    print(f"\n[Video] Starting ... Press Q to quit.")
    print(f"  rembg : {'ASYNC u2netp  bg=' + args.rembg_bg_color + '  workers=' + str(args.rembg_workers) if rembg_mgr else 'OFF'}")
    print(f"  threshold: {args.threshold}   ema_alpha: {args.ema_alpha}\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_frame_start = time.perf_counter()
        frame_worst = 0.0
        active_ids  = set()

        # ── Convert once per frame ────────────────────────────────────────
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # ── YOLO detect → DeepOCSORT track ────────────────────────────────
        t_yolo0 = time.perf_counter()
        results = yolo.predict(source=frame, conf=args.conf, iou=args.iou, classes=[0], verbose=False)
        dets = np.empty((0, 6), dtype=np.float32)
        if results and results[0].boxes is not None and len(results[0].boxes):
            xyxy  = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy().reshape(-1, 1)
            clss  = results[0].boxes.cls.cpu().numpy().reshape(-1, 1)
            dets  = np.hstack([xyxy, confs, clss]).astype(np.float32)
        tracks = tracker.update(dets, frame)
        yolo_times.append((time.perf_counter() - t_yolo0) * 1000.0)

        annotated = frame.copy()

        for t in tracks:
            x1  = max(0, int(t[0]));     y1  = max(0, int(t[1]))
            x2  = min(src_w, int(t[2])); y2  = min(src_h, int(t[3]))
            tid = int(t[4])

            if x2 <= x1 or y2 <= y1:
                continue

            active_ids.add(tid)

            # ── Extract raw crop ──────────────────────────────────────────
            raw_crop = get_raw_crop(pil_frame, (x1, y1, x2, y2), args.pad_ratio)

            # ── Async rembg: get last frame's result, submit new job ──────
            rembg_cached = False
            if rembg_mgr is not None:
                cached = rembg_mgr.get(tid)          # non-blocking
                if cached is not None:
                    crop_for_scoring = post_process_crop(cached)
                    rembg_cached = True
                else:
                    crop_for_scoring = post_process_crop(raw_crop)
                rembg_mgr.submit(tid, raw_crop)
            else:
                crop_for_scoring = post_process_crop(raw_crop)

            # ── Score ─────────────────────────────────────────────────────
            t_score0 = time.perf_counter()
            amap, raw_score = score_crop(models, crop_for_scoring, device)
            score_times.append((time.perf_counter() - t_score0) * 1000.0)

            prev_ema  = track_scores.get(tid, raw_score)
            ema_score = args.ema_alpha * raw_score + (1 - args.ema_alpha) * prev_ema
            track_scores[tid] = ema_score
            frame_worst = max(frame_worst, ema_score)

            track_raw_history.setdefault(tid, []).append(raw_score)
            track_ema_history.setdefault(tid, []).append(ema_score)

            is_anomaly = ema_score >= args.threshold
            prediction = "anomaly" if is_anomaly else "normal"
            box_w, box_h = x2 - x1, y2 - y1
            elapsed_ms = (time.perf_counter() - t_frame_start) * 1000.0

            csv_writer.writerow([
                frame_idx, tid,
                f"{raw_score:.6f}", f"{ema_score:.6f}", prediction,
                x1, y1, x2, y2,
                box_w, box_h, box_w * box_h,
                f"{elapsed_ms:.2f}", "",
                int(rembg_cached),
            ])

            annotated = draw_clean_id(annotated, tid, (x1, y1, x2, y2), is_anomaly)
            annotated = overlay_heatmap(annotated, amap, x1, y1, x2, y2, alpha=0.38)

        # ── Evict stale track IDs ─────────────────────────────────────────
        for s in list(track_scores.keys()):
            if s not in active_ids:
                track_scores.pop(s, None)
                if rembg_mgr:
                    rembg_mgr.evict(s)

        t1 = time.perf_counter()
        fps_log.append(1000.0 / max((t1 - t_frame_start) * 1000.0, 1e-3))
        cur_fps = float(np.mean(fps_log[-30:]))
        csv_file.flush()

        rembg_ms = rembg_mgr.avg_rembg_ms() if rembg_mgr else 0.0
        annotated = draw_hud(annotated, frame_idx, cur_fps, frame_worst,
                             args.threshold, rembg_mgr is not None, rembg_ms)
        writer.write(annotated)

        if not args.no_display:
            cv2.imshow("Gum Inspection [DeepOCSORT + async rembg]", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if frame_idx % 20 == 0:
            avg_yolo  = float(np.mean(yolo_times))  if yolo_times  else 0.0
            avg_score = float(np.mean(score_times)) if score_times else 0.0
            print(
                f"  [{frame_idx:05d}]  fps={cur_fps:.1f}  score={frame_worst:.4f}"
                f"  | yolo={avg_yolo:.1f}ms  rembg≈{rembg_ms:.1f}ms(async)  score={avg_score:.1f}ms"
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
                f"{100.0 * anom / max(n,1):.1f}",
                f"{min(raws):.6f}", f"{max(raws):.6f}", f"{np.mean(raws):.6f}",
                f"{min(emas):.6f}", f"{max(emas):.6f}", f"{np.mean(emas):.6f}",
                int(np.argmax(emas)),
            ])

    print(f"\n=== Finished ===")
    print(f"  Video          → {out_path}")
    print(f"  Frame CSV      → frame_scores.csv  ({frame_idx} frames, {len(track_raw_history)} unique tracks)")
    print(f"  Per-ID summary → per_id_summary.csv")


if __name__ == "__main__":
    run(parse_args())