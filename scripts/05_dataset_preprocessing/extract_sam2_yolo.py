"""
extract_sam2_yolo.py
--------------------
Extracts individual chewing-gum pieces from a video using two methods:

    output/
    ├── B_yolo_seg/    ← YOLO-seg polygon mask
    ├── C_sam2/        ← YOLO-det box crop → SAM2 mask
    └── strips/        ← side-by-side comparison strips

Usage:
    python extract_sam2_yolo.py \
        --video     video_ch4.mp4 \
        --yolo_seg  path/to/seg_best.pt \
        --yolo_det  path/to/det_best.pt \
        --sam2_ckpt sam2.1_hiera_base_plus.pt \
        --sam2_cfg  configs/sam2.1/sam2.1_hiera_b+.yaml \
        --frame_skip 3

Requirements:
    pip install ultralytics opencv-python-headless numpy Pillow tqdm hydra-core
    # SAM2 source already copied to site-packages
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
    print("[ERROR] ultralytics not installed.  pip install ultralytics")
    sys.exit(1)

try:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    HAS_SAM2 = True
except ImportError as e:
    HAS_SAM2 = False
    print(f"[ERROR] SAM2 not available: {e}")
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


def mask_to_rgba(frame_rgb: np.ndarray, binary_mask: np.ndarray) -> Image.Image:
    alpha = (binary_mask * 255).astype(np.uint8)
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
        """Returns True if duplicate (skip). Always registers the box."""
        dup = any(_box_iou(box, prev) >= self.thresh for prev in self.history)
        self.history.append(box)
        return dup


# ═══════════════════════════════════════════════════════════════════════════════
#  Method B — YOLO-seg
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

        rgba    = mask_to_rgba(frame_rgb, mask.astype(bool))
        cropped = tight_crop(rgba)
        crops.append(enhance(rgba_on_white(cropped)))
        timings.append(t_yolo_seg + (time.perf_counter() - t0))
        boxes.append((x1, y1, x2, y2))

    return crops, timings, boxes


# ═══════════════════════════════════════════════════════════════════════════════
#  Method C — YOLO-det + SAM2
# ═══════════════════════════════════════════════════════════════════════════════

def method_c_sam2(frame_bgr, boxes_xyxy, predictor, t_yolo_det):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    crops, timings, boxes_out = [], [], []

    t0 = time.perf_counter()
    predictor.set_image(frame_rgb)
    t_set = time.perf_counter() - t0

    for box in boxes_xyxy:
        t0 = time.perf_counter()
        masks, scores, _ = predictor.predict(
            box=np.array(box)[None], multimask_output=False
        )
        best    = masks[np.argmax(scores)]
        rgba    = mask_to_rgba(frame_rgb, best)
        cropped = tight_crop(rgba)
        crops.append(enhance(rgba_on_white(cropped)))
        timings.append(t_yolo_det + t_set + (time.perf_counter() - t0))
        boxes_out.append(tuple(map(int, box)))

    return crops, timings, boxes_out


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def run(
    video_path:    str,
    yolo_seg_path: str,
    yolo_det_path: str,
    sam2_ckpt:     str,
    sam2_cfg:      str  = "configs/sam2.1/sam2.1_hiera_b+.yaml",
    output_dir:    str  = "gum_results",
    conf:          float = 0.25,
    iou:           float = 0.45,
    frame_skip:    int   = 1,
    save_strips:   bool  = True,
):
    out    = Path(output_dir)
    dir_b  = out / "B_yolo_seg"
    dir_c  = out / "C_sam2"
    dir_st = out / "strips"
    for d in [dir_b, dir_c, dir_st]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Load models ───────────────────────────────────────────────────────────
    print(f"Loading YOLO-seg : {yolo_seg_path}")
    yolo_seg = YOLO(yolo_seg_path)

    print(f"Loading YOLO-det : {yolo_det_path}")
    yolo_det = YOLO(yolo_det_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM2     : {sam2_ckpt}  [{device}]")
    sam2_predictor = SAM2ImagePredictor(
        build_sam2(sam2_cfg, sam2_ckpt, device=device)
    )

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_v  = cap.get(cv2.CAP_PROP_FPS)
    print(f"\nVideo : {total} frames @ {fps_v:.1f} fps")
    print(f"Output: {out.resolve()}/")
    print(f"  ├── B_yolo_seg/")
    print(f"  ├── C_sam2/")
    print(f"  └── strips/\n")

    dedup_b = BoxDeduplicator()
    dedup_c = BoxDeduplicator()

    saved    = {"B": 0, "C": 0}
    skipped  = {"B": 0, "C": 0}
    times    = {"B": [], "C": []}

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

        # ── Inference ─────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        seg_res = yolo_seg(frame_bgr, conf=conf, iou=iou, verbose=False)[0]
        t_seg   = time.perf_counter() - t0

        t0 = time.perf_counter()
        det_res = yolo_det(frame_bgr, conf=conf, iou=iou, verbose=False)[0]
        t_det   = time.perf_counter() - t0

        has_det    = det_res.boxes is not None and len(det_res.boxes) > 0
        boxes_xyxy = det_res.boxes.xyxy.cpu().numpy() if has_det else np.empty((0, 4))

        # ── Extract ───────────────────────────────────────────────────────────
        crops_b, tb, boxes_b = method_b_yolo_seg(frame_bgr, seg_res, t_seg)
        crops_c, tc, boxes_c = (
            method_c_sam2(frame_bgr, boxes_xyxy, sam2_predictor, t_det)
            if len(boxes_xyxy) > 0 else ([], [], [])
        )

        times["B"].extend(tb)
        times["C"].extend(tc)

        # ── Dedup + save ──────────────────────────────────────────────────────
        def save_method(key, crops, timings, boxes, folder, dedup):
            for i, (crop, box) in enumerate(zip(crops, boxes)):
                if dedup.check_and_register(box):
                    skipped[key] += 1
                    continue
                ms  = timings[i] * 1000 if i < len(timings) else 0
                name = f"f{fidx:06d}_d{i:02d}_{ms:.0f}ms.png"
                save_256(crop, folder, name)
                saved[key] += 1

        save_method("B", crops_b, tb, boxes_b, dir_b, dedup_b)
        save_method("C", crops_c, tc, boxes_c, dir_c, dedup_c)

        # ── Comparison strips ─────────────────────────────────────────────────
        if save_strips:
            frame_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

            def ms_lbl(tlist, idx):
                if idx < len(tlist):
                    t = tlist[idx]
                    return f"{t*1000:.0f}ms  {1/t:.1f}fps"
                return "—"

            n = max(len(crops_b), len(crops_c))
            for di in range(n):
                tiles = []
                # Original box crop as reference
                if di < len(boxes_xyxy):
                    x1, y1, x2, y2 = map(int, boxes_xyxy[di])
                    orig = frame_pil.crop((x1, y1, x2, y2))
                    tiles.append(labeled_tile(orig, "Original", "", (20, 20, 80)))

                if di < len(crops_b):
                    tiles.append(labeled_tile(
                        crops_b[di], "B: YOLO-seg",
                        ms_lbl(tb, di), (80, 40, 20)
                    ))
                if di < len(crops_c):
                    tiles.append(labeled_tile(
                        crops_c[di], "C: YOLO-det + SAM2",
                        ms_lbl(tc, di), (80, 20, 60)
                    ))
                if len(tiles) > 1:
                    build_strip(tiles).save(dir_st / f"f{fidx:06d}_d{di:02d}.png")

        if pbar:
            pbar.set_postfix(B=saved["B"], C=saved["C"])
            pbar.update(1)
        elif fidx % 50 == 0:
            print(f"  Frame {fidx}/{total}  B:{saved['B']}  C:{saved['C']}")

        fidx += 1

    cap.release()
    if pbar: pbar.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    def avg_ms(t): return (sum(t)/len(t)*1000) if t else 0.0
    def avg_fps(t): return (1/(sum(t)/len(t))) if t else 0.0
    def rt(f): return "✅ real-time" if f>=10 else ("⚠️  borderline" if f>=4 else "❌ too slow")

    print(f"\n{'═'*60}")
    print("  EXTRACTION COMPLETE")
    print(f"{'─'*60}")
    for label, key in [("B  YOLO-seg          ", "B"),
                        ("C  YOLO-det + SAM2   ", "C")]:
        tl = times[key]
        if tl:
            f = avg_fps(tl)
            print(f"  {label}  saved={saved[key]:4d}  dup-skip={skipped[key]:4d}  "
                  f"{avg_ms(tl):.1f}ms/crop  ~{f:.1f}fps  {rt(f)}")
        else:
            print(f"  {label}  saved={saved[key]:4d}  [not run]")
    print(f"{'─'*60}")
    print(f"  {out.resolve()}/")
    print(f"    ├── B_yolo_seg/   ({saved['B']} crops)")
    print(f"    ├── C_sam2/       ({saved['C']} crops)")
    print(f"    └── strips/       (comparison images)")
    print(f"{'═'*60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract gum pieces: YOLO-seg vs YOLO-det+SAM2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video",      required=True)
    p.add_argument("--yolo_seg",   required=True,  help="YOLO segmentation model")
    p.add_argument("--yolo_det",   required=True,  help="YOLO detection model")
    p.add_argument("--sam2_ckpt",  required=True,  help="SAM2 checkpoint .pt")
    p.add_argument("--sam2_cfg",   default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    p.add_argument("--output",     default="gum_results")
    p.add_argument("--conf",       type=float, default=0.25)
    p.add_argument("--iou",        type=float, default=0.45)
    p.add_argument("--frame_skip", type=int,   default=1)
    p.add_argument("--no_strips",  action="store_true", help="Skip comparison strips")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        video_path    = args.video,
        yolo_seg_path = args.yolo_seg,
        yolo_det_path = args.yolo_det,
        sam2_ckpt     = args.sam2_ckpt,
        sam2_cfg      = args.sam2_cfg,
        output_dir    = args.output,
        conf          = args.conf,
        iou           = args.iou,
        frame_skip    = args.frame_skip,
        save_strips   = not args.no_strips,
    )
