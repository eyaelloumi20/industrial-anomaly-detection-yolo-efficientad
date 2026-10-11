"""
extract_yolo_grabcut.py
-----------------------
Extracts chewing-gum pieces from a video using three methods:

    output/
    ├── B_yolo_seg/        ← YOLO-seg polygon mask (raw)
    ├── B2_yolo_seg_morph/ ← YOLO-seg + morphological cleanup
    ├── C_grabcut/         ← YOLO-det box → GrabCut refinement
    └── strips/            ← side-by-side comparison strips

Usage:
    python extract_yolo_grabcut.py \
        --video     video_ch4.mp4 \
        --yolo_seg  path/to/seg_best.pt \
        --yolo_det  path/to/det_best.pt \
        --frame_skip 3

Requirements:
    pip install ultralytics opencv-python numpy Pillow tqdm
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] pip install ultralytics")
    sys.exit(1)

# ── Constants ─────────────────────────────────────────────────────────────────
OUTPUT_SIZE  = 256
CONTRAST     = 1.8
SHARPNESS    = 2.0
LABEL_H      = 44
GAP          = 6
FONT_PATH    = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
PADDING      = 10
MIN_AREA     = 500
DEDUP_IOU    = 0.60
DEDUP_WINDOW = 15

# Morphological cleanup kernel sizes
MORPH_CLOSE_K = 7   # closes small holes inside the mask
MORPH_OPEN_K  = 3   # removes small speckles outside the mask

# GrabCut iterations (more = slower but better)
GRABCUT_ITER  = 5


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
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
    img = ImageEnhance.Contrast(img).enhance(CONTRAST)
    return ImageEnhance.Sharpness(img).enhance(SHARPNESS)


def rgba_on_white(rgba: Image.Image) -> Image.Image:
    bg = Image.new("RGB", rgba.size, (255, 255, 255))
    bg.paste(rgba, mask=rgba.split()[3])
    return bg


def mask_to_rgba(frame_rgb: np.ndarray, mask_bool: np.ndarray) -> Image.Image:
    alpha = (mask_bool.astype(np.uint8) * 255)
    return Image.fromarray(np.dstack([frame_rgb, alpha]), "RGBA")


def tight_crop(rgba: Image.Image) -> Image.Image:
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


def get_font(size):
    try:    return ImageFont.truetype(FONT_PATH, size)
    except: return ImageFont.load_default()


def labeled_tile(img, line1, line2, bar_color=(30, 30, 30)):
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
#  Deduplicator
# ═══════════════════════════════════════════════════════════════════════════════

class BoxDeduplicator:
    def __init__(self, iou_thresh=DEDUP_IOU, window=DEDUP_WINDOW):
        self.thresh  = iou_thresh
        self.history = deque(maxlen=window)

    def check_and_register(self, box) -> bool:
        dup = any(_box_iou(box, prev) >= self.thresh for prev in self.history)
        self.history.append(box)
        return dup


# ═══════════════════════════════════════════════════════════════════════════════
#  Method B — YOLO-seg raw polygon
# ═══════════════════════════════════════════════════════════════════════════════

def method_b_yolo_seg(frame_bgr, seg_result, t_yolo_seg):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_bgr.shape[:2]
    crops, timings, boxes = [], [], []

    if seg_result.masks is None:
        return [], [], []

    for poly in seg_result.masks.xy:
        if len(poly) < 3:
            continue
        t0  = time.perf_counter()
        pts = poly.astype(np.int32).reshape((-1, 1, 2))
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        if mask.sum() < MIN_AREA:
            continue

        ys, xs = np.where(mask)
        x1 = max(int(xs.min()) - PADDING, 0)
        y1 = max(int(ys.min()) - PADDING, 0)
        x2 = min(int(xs.max()) + PADDING, w - 1)
        y2 = min(int(ys.max()) + PADDING, h - 1)

        rgba = mask_to_rgba(frame_rgb, mask.astype(bool))
        crops.append(enhance(rgba_on_white(tight_crop(rgba))))
        timings.append(t_yolo_seg + (time.perf_counter() - t0))
        boxes.append((x1, y1, x2, y2))

    return crops, timings, boxes


# ═══════════════════════════════════════════════════════════════════════════════
#  Method B2 — YOLO-seg + morphological cleanup
#  Close holes → remove speckles → largest connected component only
# ═══════════════════════════════════════════════════════════════════════════════

def morph_clean(mask: np.ndarray) -> np.ndarray:
    """
    1. Close small holes inside the gum piece.
    2. Remove small external speckles.
    3. Keep only the largest connected component.
    """
    k_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_CLOSE_K, MORPH_CLOSE_K))
    k_open  = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_OPEN_K,  MORPH_OPEN_K))

    m = mask.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)  # fill holes
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k_open)   # remove speckles

    # Keep largest connected component
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n_labels <= 1:
        return m
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8)


def method_b2_yolo_seg_morph(frame_bgr, seg_result, t_yolo_seg):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_bgr.shape[:2]
    crops, timings, boxes = [], [], []

    if seg_result.masks is None:
        return [], [], []

    for poly in seg_result.masks.xy:
        if len(poly) < 3:
            continue
        t0  = time.perf_counter()
        pts = poly.astype(np.int32).reshape((-1, 1, 2))
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        if mask.sum() < MIN_AREA:
            continue

        mask = morph_clean(mask)   # ← the only difference from B
        if mask.sum() < MIN_AREA:
            continue

        ys, xs = np.where(mask)
        x1 = max(int(xs.min()) - PADDING, 0)
        y1 = max(int(ys.min()) - PADDING, 0)
        x2 = min(int(xs.max()) + PADDING, w - 1)
        y2 = min(int(ys.max()) + PADDING, h - 1)

        rgba = mask_to_rgba(frame_rgb, mask.astype(bool))
        crops.append(enhance(rgba_on_white(tight_crop(rgba))))
        timings.append(t_yolo_seg + (time.perf_counter() - t0))
        boxes.append((x1, y1, x2, y2))

    return crops, timings, boxes


# ═══════════════════════════════════════════════════════════════════════════════
#  Method C — YOLO-det + GrabCut
#  Uses the detection box as ROI, YOLO-seg mask (if available) as hint,
#  otherwise falls back to pure box-based GrabCut.
# ═══════════════════════════════════════════════════════════════════════════════

def grabcut_with_mask_hint(frame_bgr, box, seg_mask_full=None):
    """
    Run GrabCut on the region defined by `box`.
    If seg_mask_full is provided (same size as frame), use it to seed
    probable foreground — gives much better results than box alone.
    """
    x1, y1, x2, y2 = box
    # GrabCut needs at least a 3px border inside the rect
    x1c = max(x1 + 2, 0)
    y1c = max(y1 + 2, 0)
    x2c = min(x2 - 2, frame_bgr.shape[1] - 1)
    y2c = min(y2 - 2, frame_bgr.shape[0] - 1)
    if x2c <= x1c or y2c <= y1c:
        return None

    bgd = np.zeros((1, 65), dtype=np.float64)
    fgd = np.zeros((1, 65), dtype=np.float64)

    if seg_mask_full is not None:
        # Seed GrabCut with YOLO-seg polygon as probable foreground
        gc_mask = np.full(frame_bgr.shape[:2], cv2.GC_BGD, dtype=np.uint8)
        # Border of box = definite background
        gc_mask[y1:y2, x1:x2] = cv2.GC_PR_BGD
        # Seg mask region = probable foreground
        gc_mask[seg_mask_full == 1] = cv2.GC_PR_FGD
        # Eroded core of seg mask = definite foreground
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        core = cv2.erode(seg_mask_full, kern, iterations=2)
        gc_mask[core == 1] = cv2.GC_FGD

        cv2.grabCut(frame_bgr, gc_mask, None, bgd, fgd,
                    GRABCUT_ITER, cv2.GC_INIT_WITH_MASK)
    else:
        # Pure box-based GrabCut
        rect = (x1c, y1c, x2c - x1c, y2c - y1c)
        gc_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
        cv2.grabCut(frame_bgr, gc_mask, rect, bgd, fgd,
                    GRABCUT_ITER, cv2.GC_INIT_WITH_RECT)

    fg_mask = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0
    ).astype(np.uint8)

    # Morphological cleanup on GrabCut result
    fg_mask = morph_clean(fg_mask)
    return fg_mask if fg_mask.sum() >= MIN_AREA else None


def method_c_grabcut(frame_bgr, frame_rgb, boxes_xyxy, seg_result, t_yolo_det):
    h, w = frame_bgr.shape[:2]
    crops, timings, boxes_out = [], [], []

    # Build a quick lookup: for each det box, find the best-overlapping seg mask
    seg_masks = []
    if seg_result is not None and seg_result.masks is not None:
        for poly in seg_result.masks.xy:
            if len(poly) < 3:
                continue
            pts  = poly.astype(np.int32).reshape((-1, 1, 2))
            m    = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(m, [pts], 1)
            seg_masks.append(m)

    for box in boxes_xyxy:
        t0 = time.perf_counter()
        x1, y1, x2, y2 = map(int, box)
        box_t = (x1, y1, x2, y2)

        # Find best-matching seg mask for this det box (by IoU of bounding boxes)
        hint_mask = None
        best_iou  = 0.0
        for sm in seg_masks:
            ys, xs = np.where(sm)
            if len(xs) == 0:
                continue
            sm_box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            iou    = _box_iou(box_t, sm_box)
            if iou > best_iou:
                best_iou  = iou
                hint_mask = sm

        fg = grabcut_with_mask_hint(frame_bgr, box_t,
                                     hint_mask if best_iou > 0.3 else None)
        if fg is None:
            continue

        rgba = mask_to_rgba(frame_rgb, fg.astype(bool))
        crops.append(enhance(rgba_on_white(tight_crop(rgba))))
        timings.append(t_yolo_det + (time.perf_counter() - t0))
        boxes_out.append(box_t)

    return crops, timings, boxes_out


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def run(
    video_path:    str,
    yolo_seg_path: str,
    yolo_det_path: str,
    output_dir:    str  = "gum_results",
    conf:          float = 0.25,
    iou_nms:       float = 0.45,
    frame_skip:    int   = 1,
    save_strips:   bool  = True,
):
    out     = Path(output_dir)
    dir_b   = out / "B_yolo_seg"
    dir_b2  = out / "B2_yolo_seg_morph"
    dir_c   = out / "C_grabcut"
    dir_st  = out / "strips"
    for d in [dir_b, dir_b2, dir_c, dir_st]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Loading YOLO-seg : {yolo_seg_path}")
    yolo_seg = YOLO(yolo_seg_path)
    print(f"Loading YOLO-det : {yolo_det_path}")
    yolo_det = YOLO(yolo_det_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fpsv  = cap.get(cv2.CAP_PROP_FPS)
    print(f"\nVideo : {total} frames @ {fpsv:.1f} fps  |  skip={frame_skip}")
    print(f"Output: {out.resolve()}/")
    print(f"  ├── B_yolo_seg/        raw YOLO-seg polygon")
    print(f"  ├── B2_yolo_seg_morph/ YOLO-seg + morph cleanup")
    print(f"  ├── C_grabcut/         YOLO-det + GrabCut")
    print(f"  └── strips/\n")

    dedup = {k: BoxDeduplicator() for k in ("B", "B2", "C")}
    saved   = {k: 0 for k in ("B", "B2", "C")}
    skipped = {k: 0 for k in ("B", "B2", "C")}
    times   = {k: [] for k in ("B", "B2", "C")}

    n_proc = (total + frame_skip - 1) // frame_skip
    pbar   = tqdm(total=n_proc, unit="frame") if HAS_TQDM else None
    fidx   = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if fidx % frame_skip != 0:
            fidx += 1
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # ── Inference ─────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        seg_res = yolo_seg(frame_bgr, conf=conf, iou=iou_nms, verbose=False)[0]
        t_seg   = time.perf_counter() - t0

        t0 = time.perf_counter()
        det_res = yolo_det(frame_bgr, conf=conf, iou=iou_nms, verbose=False)[0]
        t_det   = time.perf_counter() - t0

        has_det    = det_res.boxes is not None and len(det_res.boxes) > 0
        boxes_xyxy = det_res.boxes.xyxy.cpu().numpy() if has_det else np.empty((0, 4))

        # ── Extract ───────────────────────────────────────────────────────────
        crops_b,  tb,  boxes_b  = method_b_yolo_seg(frame_bgr, seg_res, t_seg)
        crops_b2, tb2, boxes_b2 = method_b2_yolo_seg_morph(frame_bgr, seg_res, t_seg)
        crops_c,  tc,  boxes_c  = (
            method_c_grabcut(frame_bgr, frame_rgb, boxes_xyxy, seg_res, t_det)
            if len(boxes_xyxy) > 0 else ([], [], [])
        )

        times["B"].extend(tb);   times["B2"].extend(tb2);  times["C"].extend(tc)

        # ── Dedup + save ──────────────────────────────────────────────────────
        def save_method(key, crops, timings, boxes, folder):
            for i, (crop, box) in enumerate(zip(crops, boxes)):
                if dedup[key].check_and_register(box):
                    skipped[key] += 1
                    continue
                ms   = timings[i] * 1000 if i < len(timings) else 0
                name = f"f{fidx:06d}_d{i:02d}_{ms:.0f}ms.png"
                save_256(crop, folder, name)
                saved[key] += 1

        save_method("B",  crops_b,  tb,  boxes_b,  dir_b)
        save_method("B2", crops_b2, tb2, boxes_b2, dir_b2)
        save_method("C",  crops_c,  tc,  boxes_c,  dir_c)

        # ── Strips ────────────────────────────────────────────────────────────
        if save_strips:
            frame_pil = Image.fromarray(frame_rgb)

            def ms_lbl(tl, idx):
                if idx < len(tl):
                    t = tl[idx]; return f"{t*1000:.0f}ms  {1/t:.1f}fps"
                return "—"

            n = max(len(crops_b), len(crops_c))
            for di in range(n):
                tiles = []
                if di < len(boxes_xyxy):
                    x1, y1, x2, y2 = map(int, boxes_xyxy[di])
                    tiles.append(labeled_tile(
                        frame_pil.crop((x1, y1, x2, y2)),
                        "Original", "", (20, 20, 80)))
                if di < len(crops_b):
                    tiles.append(labeled_tile(
                        crops_b[di], "B: YOLO-seg raw",
                        ms_lbl(tb, di), (60, 30, 10)))
                if di < len(crops_b2):
                    tiles.append(labeled_tile(
                        crops_b2[di], "B2: YOLO-seg+morph",
                        ms_lbl(tb2, di), (80, 50, 10)))
                if di < len(crops_c):
                    tiles.append(labeled_tile(
                        crops_c[di], "C: GrabCut",
                        ms_lbl(tc, di), (20, 60, 80)))
                if len(tiles) > 1:
                    build_strip(tiles).save(dir_st / f"f{fidx:06d}_d{di:02d}.png")

        if pbar:
            pbar.set_postfix(B=saved["B"], B2=saved["B2"], C=saved["C"])
            pbar.update(1)
        elif fidx % 50 == 0:
            print(f"  Frame {fidx}/{total}  "
                  f"B:{saved['B']}  B2:{saved['B2']}  C:{saved['C']}")
        fidx += 1

    cap.release()
    if pbar: pbar.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    def avg_ms(t):  return (sum(t)/len(t)*1000) if t else 0.0
    def avg_fps(t): return (1/(sum(t)/len(t)))  if t else 0.0
    def rt(f): return "✅ real-time" if f>=10 else ("⚠️  borderline" if f>=4 else "❌ too slow")

    print(f"\n{'═'*65}")
    print("  EXTRACTION COMPLETE")
    print(f"{'─'*65}")
    for label, key in [
        ("B   YOLO-seg raw        ", "B"),
        ("B2  YOLO-seg + morph    ", "B2"),
        ("C   YOLO-det + GrabCut  ", "C"),
    ]:
        tl = times[key]
        if tl:
            f = avg_fps(tl)
            print(f"  {label}  saved={saved[key]:4d}  skip={skipped[key]:4d}  "
                  f"{avg_ms(tl):.1f}ms  ~{f:.1f}fps  {rt(f)}")
        else:
            print(f"  {label}  saved={saved[key]:4d}  [not run]")
    print(f"{'─'*65}")
    print(f"  {out.resolve()}/")
    print(f"    ├── B_yolo_seg/        ({saved['B']} crops)")
    print(f"    ├── B2_yolo_seg_morph/ ({saved['B2']} crops)")
    print(f"    ├── C_grabcut/         ({saved['C']} crops)")
    print(f"    └── strips/")
    print(f"{'═'*65}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract gum: YOLO-seg raw vs morph vs GrabCut",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video",      required=True)
    p.add_argument("--yolo_seg",   required=True,  help="YOLO segmentation model")
    p.add_argument("--yolo_det",   required=True,  help="YOLO detection model")
    p.add_argument("--output",     default="gum_results")
    p.add_argument("--conf",       type=float, default=0.25)
    p.add_argument("--iou",        type=float, default=0.45)
    p.add_argument("--frame_skip", type=int,   default=1)
    p.add_argument("--no_strips",  action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        video_path    = args.video,
        yolo_seg_path = args.yolo_seg,
        yolo_det_path = args.yolo_det,
        output_dir    = args.output,
        conf          = args.conf,
        iou_nms       = args.iou,
        frame_skip    = args.frame_skip,
        save_strips   = not args.no_strips,
    )
