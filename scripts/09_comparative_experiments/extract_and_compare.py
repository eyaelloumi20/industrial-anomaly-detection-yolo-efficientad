"""
extract_and_compare.py
----------------------
Extracts individual chewing-gum pieces from a **video** using three methods
and saves each piece into its own folder:

    output/
    ├── A_rembg/       ← YOLO-det box crop → rembg AI matting
    ├── B_yolo_seg/    ← YOLO-seg polygon mask (your current approach)
    ├── C_sam2/        ← YOLO-det box crop → SAM2 mask
    └── strips/        ← side-by-side comparison strips (optional)

Usage (minimum — only YOLO-seg, skips A & C automatically):
    python extract_and_compare.py \\
        --video   path/to/video.mp4 \\
        --yolo_seg path/to/seg_best.pt

With detection model + rembg:
    python extract_and_compare.py \\
        --video    path/to/video.mp4 \\
        --yolo_seg path/to/seg_best.pt \\
        --yolo_det path/to/det_best.pt

Full (all three methods):
    python extract_and_compare.py \\
        --video     path/to/video.mp4 \\
        --yolo_seg  path/to/seg_best.pt \\
        --yolo_det  path/to/det_best.pt \\
        --sam2_ckpt ./sam2.1_hiera_small.pt \\
        --output    ./gum_results

Requirements:
    pip install ultralytics opencv-python-headless numpy Pillow tqdm
    pip install rembg onnxruntime          # for Method A
    pip install sam2                       # for Method C
    # SAM2 checkpoint (Method C):
    # wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
"""

import argparse
import sys
import time
from collections import deque
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ── Optional heavy deps ───────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    print("[ERROR] ultralytics not installed.  pip install ultralytics")
    sys.exit(1)

try:
    from rembg import remove as rembg_remove, new_session
    HAS_REMBG = True
except ImportError:
    HAS_REMBG = False
    print("[WARN] rembg not found — Method A (rembg) will be skipped.  "
          "pip install rembg onnxruntime")

try:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    HAS_SAM2 = True
except ImportError:
    HAS_SAM2 = False
    print("[WARN] SAM2 not found — Method C (SAM2) will be skipped.  "
          "pip install sam2")

# ── Visual constants ──────────────────────────────────────────────────────────
OUTPUT_SIZE = 256       # every saved crop is 256×256
CONTRAST    = 1.8
SHARPNESS   = 2.0
LABEL_H     = 44        # label bar height on comparison strips
GAP         = 6         # gap between tiles in a strip
FONT_PATH   = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# ── Dedup / detection constants ───────────────────────────────────────────────
PADDING      = 10
MIN_AREA     = 500
DEDUP_IOU    = 0.85
DEDUP_WINDOW = 10


# ═══════════════════════════════════════════════════════════════════════════════
#  Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0


def enhance(img: Image.Image) -> Image.Image:
    """Boost contrast + sharpness on an RGB or RGBA image."""
    if img.mode == "RGBA":
        r, g, b, a = img.split()
        rgb = Image.merge("RGB", (r, g, b))
        rgb = ImageEnhance.Contrast(rgb).enhance(CONTRAST)
        rgb = ImageEnhance.Sharpness(rgb).enhance(SHARPNESS)
        r2, g2, b2 = rgb.split()
        return Image.merge("RGBA", (r2, g2, b2, a))
    img = ImageEnhance.Contrast(img).enhance(CONTRAST)
    img = ImageEnhance.Sharpness(img).enhance(SHARPNESS)
    return img


def rgba_on_white(rgba: Image.Image) -> Image.Image:
    bg = Image.new("RGB", rgba.size, (255, 255, 255))
    bg.paste(rgba, mask=rgba.split()[3])
    return bg


def mask_to_rgba(frame_rgb: np.ndarray, binary_mask: np.ndarray) -> Image.Image:
    alpha = (binary_mask * 255).astype(np.uint8)
    return Image.fromarray(np.dstack([frame_rgb, alpha]), "RGBA")


def crop_to_mask(rgba: Image.Image) -> Image.Image:
    alpha = np.array(rgba.split()[3])
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0:
        return rgba
    x1 = max(xs.min() - PADDING, 0)
    y1 = max(ys.min() - PADDING, 0)
    x2 = min(xs.max() + PADDING, rgba.width  - 1)
    y2 = min(ys.max() + PADDING, rgba.height - 1)
    return rgba.crop((x1, y1, x2 + 1, y2 + 1))


def to_256(img: Image.Image) -> Image.Image:
    return img.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.LANCZOS)


def save_256(img: Image.Image, folder: Path, name: str):
    to_256(img).save(folder / name, format="PNG")


# ── Strip / label helpers ─────────────────────────────────────────────────────

def get_font(size):
    try:    return ImageFont.truetype(FONT_PATH, size)
    except: return ImageFont.load_default()


def labeled_tile(img: Image.Image, line1: str, line2: str, bar_color=(30, 30, 30)):
    tile   = to_256(img)
    canvas = Image.new("RGB", (OUTPUT_SIZE, OUTPUT_SIZE + LABEL_H), bar_color)
    draw   = ImageDraw.Draw(canvas)
    f1, f2 = get_font(13), get_font(11)
    for text, font, y in [(line1, f1, 4), (line2, f2, 22)]:
        try:    tw = draw.textlength(text, font=font)
        except: tw = len(text) * 7
        draw.text(((OUTPUT_SIZE - tw) / 2, y), text, fill=(255, 255, 255), font=font)
    canvas.paste(tile, (0, LABEL_H))
    return canvas


def build_strip(tiles):
    h = tiles[0].height
    w = sum(t.width for t in tiles) + GAP * (len(tiles) - 1)
    strip = Image.new("RGB", (w, h), (50, 50, 50))
    x = 0
    for t in tiles:
        strip.paste(t, (x, 0)); x += t.width + GAP
    return strip


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-method extractors  (return list of RGB PIL crops + list of timings)
# ═══════════════════════════════════════════════════════════════════════════════

def method_a_rembg(frame_bgr, boxes_xyxy, session):
    """Method A: tight box crop from YOLO-det → rembg AI matting."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    crops, timings = [], []
    for box in boxes_xyxy:
        x1, y1, x2, y2 = map(int, box)
        crop_pil = Image.fromarray(frame_rgb[y1:y2, x1:x2])
        buf = BytesIO(); crop_pil.save(buf, format="PNG")
        t0      = time.perf_counter()
        result  = rembg_remove(buf.getvalue(), session=session)
        timing  = time.perf_counter() - t0
        rgba    = Image.open(BytesIO(result)).convert("RGBA")
        crops.append(enhance(rgba_on_white(rgba)))
        timings.append(timing)
    return crops, timings


def method_b_yolo_seg(frame_bgr, seg_result, t_yolo_seg):
    """Method B: YOLO-seg polygon mask (your existing approach)."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_bgr.shape[:2]
    crops, timings = [], []
    if seg_result.masks is None:
        return [], []
    for poly in seg_result.masks.xy:
        if len(poly) < 3:
            continue
        t0   = time.perf_counter()
        pts  = poly.astype(np.int32).reshape((-1, 1, 2))
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        if mask.sum() < MIN_AREA:
            continue
        rgba    = mask_to_rgba(frame_rgb, mask.astype(bool))
        cropped = crop_to_mask(rgba)
        timings.append(t_yolo_seg + (time.perf_counter() - t0))
        crops.append(enhance(rgba_on_white(cropped)))
    return crops, timings


def method_c_sam2(frame_bgr, boxes_xyxy, predictor, t_yolo_det):
    """Method C: YOLO-det box prompts → SAM2 instance mask."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    crops, timings = [], []
    t0_set = time.perf_counter()
    predictor.set_image(frame_rgb)
    t_set = time.perf_counter() - t0_set
    for box in boxes_xyxy:
        t0 = time.perf_counter()
        masks, scores, _ = predictor.predict(
            box=np.array(box)[None], multimask_output=False
        )
        best    = masks[np.argmax(scores)]
        rgba    = mask_to_rgba(frame_rgb, best)
        cropped = crop_to_mask(rgba)
        timings.append(t_yolo_det + t_set + (time.perf_counter() - t0))
        crops.append(enhance(rgba_on_white(cropped)))
    return crops, timings


# ═══════════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run(
    video_path:    str,
    yolo_seg_path: str,
    yolo_det_path: str  = None,
    output_dir:    str  = "gum_results",
    sam2_cfg:      str  = "configs/sam2.1/sam2.1_hiera_s.yaml",
    sam2_ckpt:     str  = None,
    conf:          float = 0.25,
    iou:           float = 0.45,
    frame_skip:    int   = 1,
    save_strips:   bool  = True,
):
    out = Path(output_dir)

    # ── Create the 3 method folders (+ strips) ────────────────────────────────
    dir_a      = out / "A_rembg"
    dir_b      = out / "B_yolo_seg"
    dir_c      = out / "C_sam2"
    dir_strips = out / "strips"
    for d in [dir_a, dir_b, dir_c, dir_strips]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Output layout:")
    print(f"    {dir_a}       ← YOLO-det + rembg")
    print(f"    {dir_b}  ← YOLO-seg polygon mask")
    print(f"    {dir_c}        ← YOLO-det + SAM2")
    print(f"    {dir_strips}       ← comparison strips")
    print("=" * 60)

    # ── Load YOLO models ──────────────────────────────────────────────────────
    print(f"\nLoading YOLO-seg: {yolo_seg_path}")
    yolo_seg = YOLO(yolo_seg_path)

    if yolo_det_path:
        print(f"Loading YOLO-det: {yolo_det_path}")
        yolo_det = YOLO(yolo_det_path)
    else:
        print("No --yolo_det supplied — using seg model for box crops too.")
        yolo_det = yolo_seg

    # ── Load rembg session ────────────────────────────────────────────────────
    rembg_sess = None
    if HAS_REMBG:
        print("Loading rembg u2net session…")
        rembg_sess = new_session("u2net")
    else:
        print("Skipping Method A (rembg not installed).")

    # ── Load SAM2 ─────────────────────────────────────────────────────────────
    sam2_predictor = None
    if HAS_SAM2 and sam2_ckpt and Path(sam2_ckpt).exists():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SAM2 on {device}…")
        sam2_predictor = SAM2ImagePredictor(
            build_sam2(sam2_cfg, sam2_ckpt, device=device)
        )
    elif sam2_ckpt:
        print(f"[WARN] SAM2 checkpoint not found at '{sam2_ckpt}' — Method C skipped.")
    else:
        print("Skipping Method C (no --sam2_ckpt supplied).")

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video    = cap.get(cv2.CAP_PROP_FPS)
    print(f"\nVideo: {total_frames} frames @ {fps_video:.1f} fps")
    print(f"Processing every {frame_skip} frame(s).\n")

    # ── Dedup state (per class, per method) ───────────────────────────────────
    recent: dict = {m: {} for m in ("A", "B", "C")}

    # ── Counters ──────────────────────────────────────────────────────────────
    saved   = {"A": 0, "B": 0, "C": 0}
    skipped = {"A": 0, "B": 0, "C": 0}
    all_times = {"A": [], "B": [], "C": []}

    frames_to_process = (total_frames + frame_skip - 1) // frame_skip
    pbar = tqdm(total=frames_to_process, unit="frame") if HAS_TQDM else None

    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        # ── YOLO-seg inference ────────────────────────────────────────────────
        t0 = time.perf_counter()
        seg_result = yolo_seg(frame_bgr, conf=conf, iou=iou, verbose=False)[0]
        t_seg = time.perf_counter() - t0

        # ── YOLO-det inference (for A + C box prompts) ────────────────────────
        t0 = time.perf_counter()
        det_result = yolo_det(frame_bgr, conf=conf, iou=iou, verbose=False)[0]
        t_det = time.perf_counter() - t0

        has_det = det_result.boxes is not None and len(det_result.boxes) > 0
        boxes_xyxy = det_result.boxes.xyxy.cpu().numpy() if has_det else np.empty((0, 4))

        # ── Run the three methods ─────────────────────────────────────────────
        crops_a, ta = (
            method_a_rembg(frame_bgr, boxes_xyxy, rembg_sess)
            if (HAS_REMBG and rembg_sess is not None and len(boxes_xyxy) > 0)
            else ([], [])
        )
        crops_b, tb = method_b_yolo_seg(frame_bgr, seg_result, t_seg)
        crops_c, tc = (
            method_c_sam2(frame_bgr, boxes_xyxy, sam2_predictor, t_det)
            if (sam2_predictor is not None and len(boxes_xyxy) > 0)
            else ([], [])
        )

        all_times["A"].extend(ta)
        all_times["B"].extend(tb)
        all_times["C"].extend(tc)

        # ── Dedup + save helper ───────────────────────────────────────────────
        def dedup_and_save(crops, timings, method_key, dir_out, boxes=None):
            """
            Save each crop if it isn't a duplicate of a recent box.
            For method B (seg) boxes come from seg_result; for A/C from det boxes.
            """
            recent_m = recent[method_key]
            for i, crop in enumerate(crops):
                # Derive a box for dedup
                if boxes is not None and i < len(boxes):
                    x1, y1, x2, y2 = map(int, boxes[i])
                    box = (x1, y1, x2, y2)
                    cls_id = 0          # det model may not expose class here; use 0
                else:
                    # Method B: derive box from the crop content (approximate)
                    cls_id = 0
                    box = (0, 0, crop.width, crop.height)  # always unique → no dedup

                if cls_id not in recent_m:
                    recent_m[cls_id] = deque(maxlen=DEDUP_WINDOW)

                is_dup = any(_box_iou(box, prev) >= DEDUP_IOU
                             for prev in recent_m[cls_id])
                recent_m[cls_id].append(box)

                if is_dup:
                    skipped[method_key] += 1
                    continue

                tag = f"f{frame_idx:06d}_d{i:02d}"
                if i < len(timings):
                    ms = timings[i] * 1000
                    tag += f"_t{ms:.0f}ms"
                save_256(crop, dir_out, f"{tag}.png")
                saved[method_key] += 1

        dedup_and_save(crops_a, ta, "A", dir_a, boxes=boxes_xyxy)
        dedup_and_save(crops_b, tb, "B", dir_b, boxes=None)   # B: seg, no box dedup
        dedup_and_save(crops_c, tc, "C", dir_c, boxes=boxes_xyxy)

        # ── Build comparison strip ────────────────────────────────────────────
        if save_strips:
            n_det = len(boxes_xyxy)
            frame_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

            for di in range(max(n_det, len(crops_b))):
                tiles = []

                # Original raw box crop (or full frame thumbnail if no box)
                if di < n_det:
                    x1, y1, x2, y2 = map(int, boxes_xyxy[di])
                    orig = frame_pil.crop((x1, y1, x2, y2))
                else:
                    orig = frame_pil
                tiles.append(labeled_tile(orig, "Original", "", bar_color=(20, 20, 80)))

                def ms_label(tlist, idx):
                    if idx < len(tlist):
                        t = tlist[idx]
                        return f"{t*1000:.0f}ms  {1/t:.1f}fps"
                    return "skipped"

                if di < len(crops_a):
                    tiles.append(labeled_tile(
                        crops_a[di], "A: YOLO-det + rembg",
                        ms_label(ta, di), bar_color=(20, 80, 20)
                    ))
                if di < len(crops_b):
                    tiles.append(labeled_tile(
                        crops_b[di], "B: YOLO-seg",
                        ms_label(tb, di), bar_color=(80, 40, 20)
                    ))
                if di < len(crops_c):
                    tiles.append(labeled_tile(
                        crops_c[di], "C: YOLO-det + SAM2",
                        ms_label(tc, di), bar_color=(80, 20, 60)
                    ))

                if len(tiles) > 1:
                    strip_name = f"f{frame_idx:06d}_d{di:02d}.png"
                    build_strip(tiles).save(dir_strips / strip_name)

        if pbar:
            pbar.set_postfix(A=saved["A"], B=saved["B"], C=saved["C"])
            pbar.update(1)
        elif frame_idx % 50 == 0:
            print(f"  Frame {frame_idx}/{total_frames}  "
                  f"saved A:{saved['A']} B:{saved['B']} C:{saved['C']}")

        frame_idx += 1

    cap.release()
    if pbar:
        pbar.close()

    # ── Final summary ─────────────────────────────────────────────────────────
    def avg_ms(tlist):
        return (sum(tlist) / len(tlist) * 1000) if tlist else 0.0
    def avg_fps(tlist):
        return (1.0 / (sum(tlist) / len(tlist))) if tlist else 0.0
    def rt_label(f):
        if   f >= 10: return "✅ real-time"
        elif f >=  4: return "⚠️  borderline"
        else:         return "❌ too slow"

    print(f"\n{'═'*62}")
    print("  EXTRACTION COMPLETE")
    print(f"{'─'*62}")
    rows = [
        ("A_rembg/   (YOLO-det + rembg)", "A"),
        ("B_yolo_seg/ (YOLO-seg polygon)", "B"),
        ("C_sam2/    (YOLO-det + SAM2)",   "C"),
    ]
    for label, key in rows:
        tl = all_times[key]
        if tl:
            f = avg_fps(tl)
            print(f"  {label}")
            print(f"    saved={saved[key]:4d}  dedup-skip={skipped[key]:4d}  "
                  f"avg={avg_ms(tl):.1f}ms/crop  ~{f:.1f}fps  {rt_label(f)}")
        else:
            print(f"  {label}")
            print(f"    saved={saved[key]:4d}  [not run]")
    print(f"{'═'*62}")
    print(f"  Output: {out.resolve()}/")
    print(f"    ├── A_rembg/       ({saved['A']} crops)")
    print(f"    ├── B_yolo_seg/    ({saved['B']} crops)")
    print(f"    ├── C_sam2/        ({saved['C']} crops)")
    print(f"    └── strips/        (comparison images)")
    print(f"{'═'*62}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract gum pieces from video into 3 method folders",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Required
    p.add_argument("--video",      required=True,
                   help="Input video file (.mp4, .avi, …)")
    p.add_argument("--yolo_seg",   required=True,
                   help="YOLO segmentation model (best.pt)")
    # Optional models
    p.add_argument("--yolo_det",   default="",
                   help="YOLO detection model for box prompts (falls back to seg model)")
    p.add_argument("--sam2_ckpt",  default="",
                   help="SAM2 checkpoint .pt (enables Method C)")
    p.add_argument("--sam2_cfg",   default="configs/sam2.1/sam2.1_hiera_s.yaml",
                   help="SAM2 config yaml")
    # Output
    p.add_argument("--output",     default="gum_results",
                   help="Root output folder")
    p.add_argument("--no_strips",  action="store_true",
                   help="Skip saving comparison strips (faster)")
    # Inference
    p.add_argument("--conf",       type=float, default=0.25)
    p.add_argument("--iou",        type=float, default=0.45)
    p.add_argument("--frame_skip", type=int,   default=1,
                   help="Process every Nth frame (1 = all frames)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        video_path    = args.video,
        yolo_seg_path = args.yolo_seg,
        yolo_det_path = args.yolo_det  or None,
        output_dir    = args.output,
        sam2_cfg      = args.sam2_cfg,
        sam2_ckpt     = args.sam2_ckpt or None,
        conf          = args.conf,
        iou           = args.iou,
        frame_skip    = args.frame_skip,
        save_strips   = not args.no_strips,
    )
