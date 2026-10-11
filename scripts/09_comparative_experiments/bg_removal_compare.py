"""
bg_removal_compare.py  —  Background Removal Method Comparison
===============================================================
Samples N frames from a video (or uses a single image), runs each
detected/installed BG-removal method on every crop found by YOLO,
then saves a side-by-side grid image per frame to the output folder.

Methods compared (whichever are installed):
  1. none          — raw crop, no removal (baseline)
  2. rembg/u2netp  — U²-Net lite  (fast, ~6x faster than u2net)
  3. rembg/u2net   — U²-Net full  (better quality, slower)
  4. rembg/isnet-general-use — IS-Net (strong general segmenter)
  5. rembg/birefnet-general  — BiRefNet (state-of-the-art 2024)
  6. cv2_grabcut  — classical GrabCut (no DL, very fast, weakest)
  7. transparent_bg — InSPyReNet (ACCV 2022, salient object detection)

Install as needed:
    pip install rembg              # methods 2-5
    pip install transparent-background  # method 7
    (method 6 needs only opencv-python)

Usage:
    # Compare on a video (samples --n_frames evenly spaced frames):
    python bg_removal_compare.py \
        --source video_ch4.mp4 \
        --yolo_weights best.pt \
        --n_frames 5 \
        --out_dir ./bg_compare_results

    # Compare on a single image:
    python bg_removal_compare.py \
        --source frame.jpg \
        --yolo_weights best.pt \
        --out_dir ./bg_compare_results

    # Skip heavy models, only run fast ones:
    python bg_removal_compare.py \
        --source video_ch4.mp4 \
        --yolo_weights best.pt \
        --methods none,u2netp,grabcut
"""

import os
import sys
import cv2
import math
import time
import argparse
import textwrap
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ─────────────────────── optional imports (graceful) ────────────────────────

def _try_import_rembg():
    try:
        from rembg import remove as rembg_remove, new_session
        return rembg_remove, new_session
    except ImportError:
        return None, None

def _try_import_transparent_bg():
    try:
        from transparent_background import Remover
        return Remover
    except ImportError:
        return None


# ─────────────────────── BG removal implementations ─────────────────────────

def apply_none(crop: Image.Image, _state) -> Image.Image:
    """Baseline — return crop unchanged."""
    return crop.convert("RGB")


def apply_rembg(crop: Image.Image, state: dict) -> Image.Image:
    """rembg wrapper — works for any rembg model stored in state['session']."""
    from io import BytesIO
    rembg_remove = state["remove_fn"]
    session      = state["session"]
    buf = BytesIO()
    crop.save(buf, format="PNG")
    result = rembg_remove(buf.getvalue(), session=session)
    rgba = Image.open(BytesIO(result)).convert("RGBA")
    bg   = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
    bg.paste(rgba, mask=rgba.split()[3])
    return bg.convert("RGB")


def apply_grabcut(crop: Image.Image, _state) -> Image.Image:
    """
    Classical GrabCut — assumes the subject is roughly centred.
    Pure OpenCV, no neural network.
    """
    arr = np.array(crop.convert("RGB"))
    mask   = np.zeros(arr.shape[:2], np.uint8)
    h, w   = arr.shape[:2]
    margin = max(4, min(w, h) // 8)
    rect   = (margin, margin, w - 2 * margin, h - 2 * margin)
    bgd    = np.zeros((1, 65), np.float64)
    fgd    = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(arr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return crop.convert("RGB")
    mask2  = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    result = arr * mask2[:, :, np.newaxis]
    return Image.fromarray(result.astype(np.uint8))


def apply_transparent_bg(crop: Image.Image, state: dict) -> Image.Image:
    """transparent-background (InSPyReNet) wrapper."""
    remover = state["remover"]
    out = remover.process(crop.convert("RGB"), type="rgba")
    rgba = out.convert("RGBA")
    bg   = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
    bg.paste(rgba, mask=rgba.split()[3])
    return bg.convert("RGB")


# ─────────────────────── method registry ────────────────────────────────────

def build_methods(requested: list[str]) -> list[dict]:
    """
    Returns a list of method dicts:
        { name, label, fn, state }
    Only methods that are both requested AND whose dependencies are installed
    are included.
    """
    rembg_remove, new_session = _try_import_rembg()
    TransparentRemover         = _try_import_transparent_bg()

    registry = []

    def _add(key, label, fn, state, available, install_hint):
        if key not in requested:
            return
        if not available:
            print(f"  [skip] {label} — not installed. ({install_hint})")
            return
        registry.append({"name": key, "label": label, "fn": fn, "state": state})
        print(f"  [ok]   {label}")

    print("\n[Methods] Loading ...")

    _add("none", "Raw (no removal)", apply_none, {},
         available=True, install_hint="always available")

    if "u2netp" in requested and rembg_remove:
        print("  [load] rembg/u2netp ...")
        s = new_session("u2netp")
        _add("u2netp", "rembg / U²-Net lite", apply_rembg,
             {"remove_fn": rembg_remove, "session": s},
             available=True, install_hint="pip install rembg")
    elif "u2netp" in requested:
        print("  [skip] rembg/u2netp — not installed. (pip install rembg)")

    if "u2net" in requested and rembg_remove:
        print("  [load] rembg/u2net ...")
        s = new_session("u2net")
        _add("u2net", "rembg / U²-Net", apply_rembg,
             {"remove_fn": rembg_remove, "session": s},
             available=True, install_hint="pip install rembg")
    elif "u2net" in requested:
        print("  [skip] rembg/u2net — not installed. (pip install rembg)")

    if "isnet" in requested and rembg_remove:
        print("  [load] rembg/isnet-general-use ...")
        s = new_session("isnet-general-use")
        _add("isnet", "rembg / IS-Net", apply_rembg,
             {"remove_fn": rembg_remove, "session": s},
             available=True, install_hint="pip install rembg")
    elif "isnet" in requested:
        print("  [skip] rembg/isnet — not installed. (pip install rembg)")

    if "birefnet" in requested and rembg_remove:
        print("  [load] rembg/birefnet-general ...")
        s = new_session("birefnet-general")
        _add("birefnet", "rembg / BiRefNet", apply_rembg,
             {"remove_fn": rembg_remove, "session": s},
             available=True, install_hint="pip install rembg")
    elif "birefnet" in requested:
        print("  [skip] rembg/birefnet — not installed. (pip install rembg)")

    _add("grabcut", "GrabCut (classical)", apply_grabcut, {},
         available=True, install_hint="needs only opencv")

    if "transparent_bg" in requested and TransparentRemover:
        print("  [load] transparent-background / InSPyReNet ...")
        remover = TransparentRemover(fast=True)
        _add("transparent_bg", "InSPyReNet (transparent-bg)", apply_transparent_bg,
             {"remover": remover},
             available=True, install_hint="pip install transparent-background")
    elif "transparent_bg" in requested:
        print("  [skip] transparent-background — not installed. (pip install transparent-background)")

    print(f"  → {len(registry)} method(s) active.\n")
    return registry


# ─────────────────────── crop helpers ───────────────────────────────────────

def get_crop(frame_rgb: np.ndarray, box, pad: float = 0.05) -> Image.Image:
    h, w = frame_rgb.shape[:2]
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad), int(bh * pad)
    x1 = max(0, x1 - px);  y1 = max(0, y1 - py)
    x2 = min(w, x2 + px);  y2 = min(h, y2 + py)
    return Image.fromarray(frame_rgb[y1:y2, x1:x2])


# ─────────────────────── grid rendering ─────────────────────────────────────

CELL_SIZE   = 224   # each crop thumbnail
LABEL_H     = 28    # height of label bar under each cell
TITLE_H     = 36    # height of top title bar
PADDING     = 6     # gap between cells
BG_COLOR    = (30, 30, 30)
LABEL_COLOR = (20, 20, 20)
TEXT_COLOR  = (240, 240, 240)
GRID_COLS   = 4     # max columns before wrapping


def _label_font(size=13):
    # Try to load a decent font, fall back to default
    for name in ["DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Arial.ttf"]:
        for base in ["/usr/share/fonts", "/usr/local/share/fonts",
                     "/Library/Fonts", "C:/Windows/Fonts"]:
            p = Path(base) / name
            if p.exists():
                try:
                    return ImageFont.truetype(str(p), size)
                except Exception:
                    pass
    return ImageFont.load_default()


def make_grid(results: list[tuple[str, Image.Image]], title: str,
              timing: dict[str, float]) -> Image.Image:
    """
    results : [(label, rgb_image), ...]
    Returns a single PIL Image with all crops in a grid.
    """
    n      = len(results)
    cols   = min(n, GRID_COLS)
    rows   = math.ceil(n / cols)
    cell_w = CELL_SIZE
    cell_h = CELL_SIZE + LABEL_H
    total_w = cols * cell_w + (cols + 1) * PADDING
    total_h = TITLE_H + rows * cell_h + (rows + 1) * PADDING

    canvas = Image.new("RGB", (total_w, total_h), BG_COLOR)
    draw   = ImageDraw.Draw(canvas)
    font   = _label_font(13)
    tfont  = _label_font(15)

    # Title bar
    draw.rectangle([(0, 0), (total_w, TITLE_H)], fill=(50, 50, 80))
    draw.text((PADDING, 8), title, font=tfont, fill=TEXT_COLOR)

    for i, (label, img) in enumerate(results):
        col = i % cols
        row = i // cols
        cx  = PADDING + col * (cell_w + PADDING)
        cy  = TITLE_H + PADDING + row * (cell_h + PADDING)

        # Thumbnail
        thumb = img.resize((cell_w, CELL_SIZE), Image.LANCZOS)
        canvas.paste(thumb, (cx, cy))

        # Label bar
        t_ms  = timing.get(label, 0.0)
        ltext = f"{label}  ({t_ms:.0f} ms)"
        ly    = cy + CELL_SIZE
        draw.rectangle([(cx, ly), (cx + cell_w, ly + LABEL_H)], fill=LABEL_COLOR)
        draw.text((cx + 4, ly + 6), ltext, font=font, fill=TEXT_COLOR)

    return canvas


# ─────────────────────── main loop ──────────────────────────────────────────

def run(args):
    os.makedirs(args.out_dir, exist_ok=True)

    requested = [m.strip() for m in args.methods.split(",")]
    methods   = build_methods(requested)

    if not methods:
        print("[Error] No methods available. Check installations.")
        sys.exit(1)

    print(f"[YOLO] Loading {args.yolo_weights} ...")
    from ultralytics import YOLO
    yolo = YOLO(args.yolo_weights)

    source = args.source
    is_image = source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))

    if is_image:
        frames = [cv2.imread(source)]
        frame_indices = [0]
    else:
        cap = cv2.VideoCapture(source)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        step  = max(1, total // args.n_frames)
        frame_indices = list(range(0, total, step))[:args.n_frames]
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, fr = cap.read()
            if ret:
                frames.append(fr)
        cap.release()

    print(f"[Frames] Processing {len(frames)} frame(s) ...\n")

    saved = []
    for fi, (frame_bgr, fidx) in enumerate(zip(frames, frame_indices)):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        results_yolo = yolo.predict(
            source=frame_bgr, conf=args.conf, iou=args.iou,
            classes=[0], verbose=False)

        boxes = []
        if results_yolo and results_yolo[0].boxes is not None:
            boxes = results_yolo[0].boxes.xyxy.cpu().numpy()

        if len(boxes) == 0:
            print(f"  Frame {fidx:05d}: no detections, skipping.")
            continue

        # Use only the first --max_crops crops per frame
        boxes = boxes[:args.max_crops]
        print(f"  Frame {fidx:05d}: {len(boxes)} crop(s)")

        for ci, box in enumerate(boxes):
            crop = get_crop(frame_rgb, box, args.pad_ratio)
            grid_results = []
            timing       = {}

            for m in methods:
                t0 = time.perf_counter()
                try:
                    out = m["fn"](crop.copy(), m["state"])
                except Exception as e:
                    print(f"    [{m['name']}] error: {e}")
                    out = Image.new("RGB", (CELL_SIZE, CELL_SIZE), (80, 0, 0))
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                timing[m["label"]] = elapsed_ms
                grid_results.append((m["label"], out))
                print(f"    {m['label']:35s} {elapsed_ms:6.1f} ms")

            title = f"Frame {fidx:05d}  |  Crop {ci}  |  box=[{int(box[0])},{int(box[1])},{int(box[2])},{int(box[3])}]"
            grid  = make_grid(grid_results, title, timing)

            out_name = f"frame{fidx:05d}_crop{ci}_compare.jpg"
            out_path = os.path.join(args.out_dir, out_name)
            grid.save(out_path, quality=92)
            saved.append(out_path)
            print(f"    → saved {out_name}\n")

    print(f"=== Done — {len(saved)} grid image(s) saved to '{args.out_dir}' ===")
    for p in saved:
        print(f"  {p}")


# ─────────────────────── CLI ─────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
        Compare background-removal methods on YOLO-detected crops.

        Available method keys (comma-separated for --methods):
          none          raw crop (baseline)
          u2netp        rembg U²-Net lite  (fast)
          u2net         rembg U²-Net full
          isnet         rembg IS-Net general
          birefnet      rembg BiRefNet (best quality, 2024)
          grabcut       classical OpenCV GrabCut
          transparent_bg InSPyReNet (transparent-background package)
        """))
    p.add_argument("--source",       required=True,
                   help="Video file or image path")
    p.add_argument("--yolo_weights", required=True,
                   help="Path to YOLO .pt weights")
    p.add_argument("--out_dir",      default="./bg_compare_results")
    p.add_argument("--n_frames",     type=int, default=5,
                   help="Number of frames to sample from a video (default: 5)")
    p.add_argument("--max_crops",    type=int, default=3,
                   help="Max crops per frame to compare (default: 3)")
    p.add_argument("--conf",         type=float, default=0.25)
    p.add_argument("--iou",          type=float, default=0.45)
    p.add_argument("--pad_ratio",    type=float, default=0.05)
    p.add_argument("--methods",
                   default="none,u2netp,u2net,isnet,birefnet,grabcut,transparent_bg",
                   help="Comma-separated list of methods to run (default: all)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
