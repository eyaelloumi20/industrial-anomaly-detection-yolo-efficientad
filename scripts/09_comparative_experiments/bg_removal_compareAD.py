"""
bg_removal_compare.py  —  Background Removal Method Comparison + Anomaly Detection
====================================================================================
Samples N frames from a video (or uses a single image), runs each
detected/installed BG-removal method on every crop found by YOLO,
scores each crop with EfficientAD for anomaly detection,
then saves a side-by-side grid image per frame to the output folder.

Methods compared (whichever are installed):
  1. none          — raw crop, no removal (baseline)
  2. rembg/u2netp  — U²-Net lite  (fast, ~6x faster than u2net)
  3. rembg/u2net   — U²-Net full  (better quality, slower)
  4. rembg/isnet-general-use — IS-Net (strong general segmenter)
  5. rembg/birefnet-general  — BiRefNet (state-of-the-art 2024)
  6. cv2_grabcut  — classical GrabCut (no DL, very fast, weakest)

Anomaly detection (EfficientAD):
  - Each crop is scored by a teacher/student/AE model.
  - The grid title shows: anomaly score + NORMAL / ANOMALY label.
  - A heatmap column is added to the grid for each crop.

Install as needed:
    pip install rembg              # methods 2-5

Usage:
    # Compare on a video (samples --n_frames evenly spaced frames):
    python bg_removal_compare.py \
        --source video_ch4.mp4 \
        --yolo_weights best.pt \
        --ckpt_dir ./checkpoints \
        --category chewinggum \
        --n_frames 5 \
        --out_dir ./bg_compare_results

    # Compare on a single image:
    python bg_removal_compare.py \
        --source frame.jpg \
        --yolo_weights best.pt \
        --ckpt_dir ./checkpoints \
        --category chewinggum \
        --out_dir ./bg_compare_results

    # Skip heavy models, only run fast ones:
    python bg_removal_compare.py \
        --source video_ch4.mp4 \
        --yolo_weights best.pt \
        --ckpt_dir ./checkpoints \
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

import torch
import torch.nn.functional as F
from torchvision import transforms

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from utils.load_efficientad import load_efficientad_model

# ─────────────────────── EfficientAD constants ───────────────────────────────

IMAGE_SIZE   = 256
OUT_CHANNELS = 384


# ─────────────────────── anomaly detection helpers ───────────────────────────

def resize_with_padding(crop: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    ow, oh = crop.size
    scale  = size / max(ow, oh)
    nw, nh = int(ow * scale), int(oh * scale)
    resized = crop.resize((nw, nh), Image.LANCZOS)
    canvas  = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def get_processed_crop_pil(
    pil_img: Image.Image,
    box: tuple,
    pad_ratio: float = 0.05,
    denoise: bool = True,
) -> Image.Image:
    """Mirror of video_inference.py — same preprocessing pipeline."""
    W, H       = pil_img.size
    x1, y1, x2, y2 = box
    bw, bh     = x2 - x1, y2 - y1
    px         = int(bw * pad_ratio)
    py         = int(bh * pad_ratio)
    x1 = max(0, x1 - px);  y1 = max(0, y1 - py)
    x2 = min(W, x2 + px);  y2 = min(H, y2 + py)

    crop = pil_img.crop((x1, y1, x2, y2)).convert("RGB")

    # Otsu refinement
    arr_gray = np.array(crop.convert("L"), dtype=np.uint8)
    _, mask   = cv2.threshold(arr_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords    = cv2.findNonZero(mask)
    if coords is not None:
        rx, ry, rw, rh = cv2.boundingRect(coords)
        coverage       = (rw * rh) / (arr_gray.shape[1] * arr_gray.shape[0])
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


def score_crop(models, crop_pil: Image.Image, device: str, ratio: float = 0.1):
    """
    Run the EfficientAD teacher/student/AE pipeline on a single crop.
    Returns (amap, score) where amap is a numpy array and score is a float.
    """
    t = transforms.ToTensor()(crop_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        t_out = models["teacher"](t)
        s_out = models["student"](t)
        a_out = models["ae"](t)

    y_st   = s_out[:, :OUT_CHANNELS, :, :]
    y_stae = s_out[:, -OUT_CHANNELS:, :, :]

    norm_t = (t_out - models["mean"]) / (models["std"] + 1e-8)
    d_st   = torch.pow(norm_t - y_st,   2)
    d_stae = torch.pow(a_out - y_stae,  2)

    fm_st   = F.interpolate(torch.mean(d_st,   dim=1, keepdim=True),
                            size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    fm_stae = F.interpolate(torch.mean(d_stae, dim=1, keepdim=True),
                            size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)

    nm_st = (ratio * (fm_st   - models["qa_st"])) / (models["qb_st"] - models["qa_st"] + 1e-8)
    nm_ae = (ratio * (fm_stae - models["qa_ae"])) / (models["qb_ae"] - models["qa_ae"] + 1e-8)

    amap  = (0.5 * nm_st + 0.5 * nm_ae)[0, 0].cpu().numpy()
    return amap, float(np.max(amap))


def amap_to_pil(amap: np.ndarray, size: int = IMAGE_SIZE) -> Image.Image:
    """Convert a float anomaly map to a colour heatmap PIL image."""
    norm = ((amap - amap.min()) / (amap.max() - amap.min() + 1e-8) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(norm, cv2.COLORMAP_HOT)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return Image.fromarray(heat).resize((size, size), Image.LANCZOS)


# ─────────────────────── optional imports (graceful) ─────────────────────────

def _try_import_rembg():
    try:
        from rembg import remove as rembg_remove, new_session
        return rembg_remove, new_session
    except ImportError:
        return None, None


# ─────────────────────── BG removal implementations ──────────────────────────

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
    rgba   = Image.open(BytesIO(result)).convert("RGBA")
    bg     = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
    bg.paste(rgba, mask=rgba.split()[3])
    return bg.convert("RGB")


def apply_grabcut(crop: Image.Image, _state) -> Image.Image:
    """
    Classical GrabCut — assumes the subject is roughly centred.
    Pure OpenCV, no neural network.
    """
    arr    = np.array(crop.convert("RGB"))
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


# ─────────────────────── method registry ─────────────────────────────────────

def build_methods(requested: list[str]) -> list[dict]:
    """
    Returns a list of method dicts:
        { name, label, fn, state }
    Only methods that are both requested AND whose dependencies are installed
    are included.
    """
    rembg_remove, new_session = _try_import_rembg()

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

    print(f"  → {len(registry)} method(s) active.\n")
    return registry


# ─────────────────────── crop helpers ────────────────────────────────────────

def get_crop(frame_rgb: np.ndarray, box, pad: float = 0.05) -> Image.Image:
    h, w = frame_rgb.shape[:2]
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad), int(bh * pad)
    x1 = max(0, x1 - px);  y1 = max(0, y1 - py)
    x2 = min(w, x2 + px);  y2 = min(h, y2 + py)
    return Image.fromarray(frame_rgb[y1:y2, x1:x2])


# ─────────────────────── grid rendering ──────────────────────────────────────

CELL_SIZE    = 224   # each crop thumbnail
LABEL_H      = 28    # height of label bar under each cell
TITLE_H      = 48    # height of top title bar (taller to fit anomaly info)
PADDING      = 6     # gap between cells
BG_COLOR     = (30, 30, 30)
LABEL_COLOR  = (20, 20, 20)
TEXT_COLOR   = (240, 240, 240)
ANOM_COLOR   = (220, 60, 60)    # red tint for anomaly title bar
NORMAL_COLOR = (40, 120, 40)    # green tint for normal title bar
GRID_COLS    = 4                # max columns before wrapping


def _label_font(size=13):
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


def make_grid(
    results: list[tuple[str, Image.Image]],
    title: str,
    timing: dict[str, float],
    anomaly_score: float | None = None,
    threshold: float = 0.5,
) -> Image.Image:
    """
    results        : [(label, rgb_image), ...]
    timing         : {label: ms} — for BG-removal labels use bg_ms,
                     for heatmap labels use ad_ms
    anomaly_score  : float score from EfficientAD, or None if models not loaded
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
    tfont  = _label_font(14)

    # ── Title bar (coloured by anomaly status) ────────────────────────────
    if anomaly_score is not None:
        is_anom   = anomaly_score >= threshold
        bar_color = ANOM_COLOR if is_anom else NORMAL_COLOR
        status    = f"  ⚠ ANOMALY  score={anomaly_score:.4f}" if is_anom \
                    else f"  ✓ NORMAL   score={anomaly_score:.4f}"
    else:
        bar_color = (50, 50, 80)
        status    = ""

    draw.rectangle([(0, 0), (total_w, TITLE_H)], fill=bar_color)
    draw.text((PADDING, 6),  title,  font=tfont, fill=TEXT_COLOR)
    draw.text((PADDING, 26), status, font=tfont, fill=TEXT_COLOR)

    # ── Crop cells ────────────────────────────────────────────────────────
    for i, (label, img) in enumerate(results):
        col = i % cols
        row = i // cols
        cx  = PADDING + col * (cell_w + PADDING)
        cy  = TITLE_H + PADDING + row * (cell_h + PADDING)

        thumb = img.resize((cell_w, CELL_SIZE), Image.LANCZOS)
        canvas.paste(thumb, (cx, cy))

        t_ms  = timing.get(label, 0.0)
        # Show timing; labels ending in "[BG removal]" or "[Heatmap]" already
        # carry context, so just append the ms value.
        ltext = f"{label}  ({t_ms:.0f} ms)"
        ly    = cy + CELL_SIZE
        draw.rectangle([(cx, ly), (cx + cell_w, ly + LABEL_H)], fill=LABEL_COLOR)
        draw.text((cx + 4, ly + 6), ltext, font=font, fill=TEXT_COLOR)

    return canvas


# ─────────────────────── main loop ───────────────────────────────────────────

def run(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # ── BG removal methods ────────────────────────────────────────────────
    requested = [m.strip() for m in args.methods.split(",")]
    methods   = build_methods(requested)

    if not methods:
        print("[Error] No BG-removal methods available. Check installations.")
        sys.exit(1)

    # ── EfficientAD models (optional) ─────────────────────────────────────
    eff_models = None
    device     = "cpu"
    if args.ckpt_dir:
        try:
            import torch
            device = args.device if torch.cuda.is_available() else "cpu"
            print(f"[EfficientAD] Loading from {args.ckpt_dir} (device={device}) ...")
            eff_models = load_efficientad_model(
                ckpt_dir=args.ckpt_dir,
                category=args.category,
                model_size=args.model_size,
                device=device,
            )
            print("[EfficientAD] Models loaded — anomaly scoring enabled.\n")
        except Exception as exc:
            print(f"[EfficientAD] WARNING: could not load models — {exc}")
            print("              Anomaly scoring will be skipped.\n")
    else:
        print("[EfficientAD] --ckpt_dir not provided. Anomaly scoring disabled.\n")

    # ── YOLO ──────────────────────────────────────────────────────────────
    print(f"[YOLO] Loading {args.yolo_weights} ...")
    from ultralytics import YOLO
    yolo = YOLO(args.yolo_weights)

    # ── Collect frames ────────────────────────────────────────────────────
    source   = args.source
    is_image = source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))

    if is_image:
        frames        = [cv2.imread(source)]
        frame_indices = [0]
    else:
        cap    = cv2.VideoCapture(source)
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        step   = max(1, total // args.n_frames)
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
        pil_frame = Image.fromarray(frame_rgb)

        results_yolo = yolo.predict(
            source=frame_bgr, conf=args.conf, iou=args.iou,
            classes=[0], verbose=False)

        boxes = []
        if results_yolo and results_yolo[0].boxes is not None:
            boxes = results_yolo[0].boxes.xyxy.cpu().numpy()

        if len(boxes) == 0:
            print(f"  Frame {fidx:05d}: no detections, skipping.")
            continue

        boxes = boxes[:args.max_crops]
        print(f"  Frame {fidx:05d}: {len(boxes)} crop(s)")

        for ci, box in enumerate(boxes):
            raw_crop = get_crop(frame_rgb, box, args.pad_ratio)

            # ── Per-method: BG removal → preprocessing → AD scoring → heatmap ──
            # Each method gets its own AD score, heatmap, and timing breakdown.
            #
            # Timing columns per method:
            #   bg_ms   = time spent on background removal only
            #   ad_ms   = time spent on EfficientAD scoring only
            #   total_ms = bg_ms + ad_ms
            #
            # The "best" (lowest) anomaly score is used for the grid title bar.

            method_results = []   # list of dicts, one per method

            for m in methods:
                entry = {
                    "name":          m["name"],
                    "label":         m["label"],
                    "bg_ms":         0.0,
                    "ad_ms":         0.0,
                    "anomaly_score": None,
                    "is_anomaly":    None,
                    "crop_pil":      None,
                    "heatmap_pil":   None,
                }

                # ── 1. Background removal ─────────────────────────────────
                t0 = time.perf_counter()
                try:
                    bg_crop = m["fn"](raw_crop.copy(), m["state"])
                except Exception as e:
                    print(f"    [{m['name']}] BG removal error: {e}")
                    bg_crop = Image.new("RGB", (CELL_SIZE, CELL_SIZE), (80, 0, 0))
                entry["bg_ms"]   = (time.perf_counter() - t0) * 1000.0
                entry["crop_pil"] = bg_crop

                # ── 2. Anomaly detection on the BG-removed crop ───────────
                if eff_models is not None:
                    try:
                        # Preprocess the BG-removed PIL image the same way
                        # as the main pipeline (resize + denoise).
                        proc = resize_with_padding(bg_crop.convert("RGB"), IMAGE_SIZE)
                        arr  = np.array(proc, dtype=np.uint8)
                        arr  = cv2.GaussianBlur(arr, (3, 3), sigmaX=0.8)
                        arr  = cv2.bilateralFilter(arr, d=5, sigmaColor=15, sigmaSpace=15)
                        proc_pil = Image.fromarray(arr)

                        t0 = time.perf_counter()
                        amap, score = score_crop(eff_models, proc_pil, device)
                        entry["ad_ms"]         = (time.perf_counter() - t0) * 1000.0
                        entry["anomaly_score"] = score
                        entry["is_anomaly"]    = score >= args.threshold
                        entry["heatmap_pil"]   = amap_to_pil(amap)

                        status = "ANOMALY" if entry["is_anomaly"] else "NORMAL"
                        print(
                            f"    {m['label']:35s} "
                            f"bg={entry['bg_ms']:6.1f} ms  "
                            f"ad={entry['ad_ms']:6.1f} ms  "
                            f"score={score:.4f}  → {status}"
                        )
                    except Exception as exc:
                        print(f"    [{m['name']}] AD error: {exc}")
                        print(
                            f"    {m['label']:35s} "
                            f"bg={entry['bg_ms']:6.1f} ms  ad=ERROR"
                        )
                else:
                    print(
                        f"    {m['label']:35s} "
                        f"bg={entry['bg_ms']:6.1f} ms  (no AD model)"
                    )

                method_results.append(entry)

            # ── Build per-method output grids and save ────────────────────
            # Each method gets its own saved image:
            #   frame<N>_crop<C>_<method>.jpg   — BG-removed crop + heatmap side-by-side
            # Plus a combined comparison grid:
            #   frame<N>_crop<C>_compare.jpg

            # --- individual per-method image (crop | heatmap) ----------------
            for entry in method_results:
                cells   = []
                timing  = {}
                total_ms = entry["bg_ms"] + entry["ad_ms"]

                # BG-removed crop cell
                label_crop = f"{entry['label']}  [BG removal]"
                cells.append((label_crop, entry["crop_pil"]))
                timing[label_crop] = entry["bg_ms"]

                # Heatmap cell (if available)
                if entry["heatmap_pil"] is not None:
                    label_heat = f"{entry['label']}  [Heatmap]"
                    cells.append((label_heat, entry["heatmap_pil"]))
                    timing[label_heat] = entry["ad_ms"]

                per_title = (
                    f"Frame {fidx:05d}  |  Crop {ci}  |  Method: {entry['label']}  "
                    f"|  total={total_ms:.0f} ms"
                )
                per_grid = make_grid(
                    cells, per_title, timing,
                    anomaly_score=entry["anomaly_score"],
                    threshold=args.threshold,
                )
                per_name = (
                    f"frame{fidx:05d}_crop{ci}_{entry['name']}.jpg"
                )
                per_path = os.path.join(args.out_dir, per_name)
                per_grid.save(per_path, quality=92)
                saved.append(per_path)
                print(f"      → saved {per_name}")

            # --- combined comparison grid (all methods) ----------------------
            # Grid layout: for each method one column = [crop, heatmap]
            # We flatten: crop_m1, heat_m1, crop_m2, heat_m2, ...
            grid_results = []
            timing_combined = {}
            # Use the highest score across methods for the combined title bar
            all_scores = [e["anomaly_score"] for e in method_results
                          if e["anomaly_score"] is not None]
            combined_score = max(all_scores) if all_scores else None

            for entry in method_results:
                total_ms = entry["bg_ms"] + entry["ad_ms"]
                lbl = entry["label"]
                grid_results.append((lbl, entry["crop_pil"]))
                timing_combined[lbl] = total_ms          # shown as bg+ad total

                if entry["heatmap_pil"] is not None:
                    hlbl = f"{lbl} ▸ heatmap"
                    grid_results.append((hlbl, entry["heatmap_pil"]))
                    timing_combined[hlbl] = entry["ad_ms"]

            cmp_title = (
                f"Frame {fidx:05d}  |  Crop {ci}  |  "
                f"box=[{int(box[0])},{int(box[1])},{int(box[2])},{int(box[3])}]"
            )
            cmp_grid = make_grid(
                grid_results, cmp_title, timing_combined,
                anomaly_score=combined_score,
                threshold=args.threshold,
            )
            cmp_name = f"frame{fidx:05d}_crop{ci}_compare.jpg"
            cmp_path = os.path.join(args.out_dir, cmp_name)
            cmp_grid.save(cmp_path, quality=92)
            saved.append(cmp_path)
            print(f"    → saved comparison grid: {cmp_name}\n")

    print(f"=== Done — {len(saved)} grid image(s) saved to '{args.out_dir}' ===")
    for p in saved:
        print(f"  {p}")


# ─────────────────────── CLI ─────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
        Compare background-removal methods on YOLO-detected crops,
        with optional EfficientAD anomaly detection scoring.

        Available BG method keys (comma-separated for --methods):
          none          raw crop (baseline)
          u2netp        rembg U²-Net lite  (fast)
          u2net         rembg U²-Net full
          isnet         rembg IS-Net general
          birefnet      rembg BiRefNet (best quality, 2024)
          grabcut       classical OpenCV GrabCut
        """))
    p.add_argument("--source",       required=True,
                   help="Video file or image path")
    p.add_argument("--yolo_weights", required=True,
                   help="Path to YOLO .pt weights")
    p.add_argument("--out_dir",      default="./bg_compare_resultsAD")
    p.add_argument("--n_frames",     type=int,   default=5,
                   help="Number of frames to sample from a video (default: 5)")
    p.add_argument("--max_crops",    type=int,   default=3,
                   help="Max crops per frame to compare (default: 3)")
    p.add_argument("--conf",         type=float, default=0.25)
    p.add_argument("--iou",          type=float, default=0.45)
    p.add_argument("--pad_ratio",    type=float, default=0.05)
    p.add_argument("--methods",
                   default="none,u2netp,u2net,isnet,birefnet,grabcut",
                   help="Comma-separated list of BG-removal methods to run")
    # ── EfficientAD options ──────────────────────────────────────────────
    p.add_argument("--ckpt_dir",     default=None,
                   help="EfficientAD checkpoint directory (enables anomaly scoring)")
    p.add_argument("--category",     default="chewinggum",
                   help="VisA / EfficientAD category name")
    p.add_argument("--model_size",   default="S", choices=["S", "M"])
    p.add_argument("--threshold",    type=float, default=0.5,
                   help="Anomaly score threshold (default: 0.5)")
    p.add_argument("--device",       default="cuda",
                   help="Torch device for EfficientAD (default: cuda, falls back to cpu)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())