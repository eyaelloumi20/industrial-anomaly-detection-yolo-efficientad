"""
crop_dataset.py
===============
Run YOLOv8 on a folder of raw images, crop every detected object,
apply EfficientAD-compatible preprocessing, and save all crops
into a single output folder.

Preprocessing (mirrors build_crop_dataset.py):
    1. Pad bbox by pad_ratio (default 5%)
    2. Otsu refinement: tighten crop to foreground (skipped if coverage < 30%)
    3. Resize preserving aspect ratio + black padding to 256×256 (LANCZOS)
    4. GaussianBlur (3×3, sigma=0.8)  ← skipped with --no-denoise
    5. BilateralFilter (d=5, sigmaColor=15, sigmaSpace=15)  ← skipped with --no-denoise

Output structure:
    out_dir/
    └── crops/
        ├── img001_crop000.png
        ├── img001_crop001.png
        └── ...


"""

import argparse
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from ultralytics import YOLO


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
IMAGE_SIZE      = 256


def get_images(folder):
    return sorted(
        p for p in Path(folder).iterdir()
        if p.suffix.lower() in IMG_EXTENSIONS
    )


def resize_with_padding(crop: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    """
    Resize keeping aspect ratio, then pad with black to fill size×size.
    Ensures the full object is always visible without stretching.
    """
    original_w, original_h = crop.size
    scale  = size / max(original_w, original_h)
    new_w  = int(original_w * scale)
    new_h  = int(original_h * scale)

    resized = crop.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    pad_x  = (size - new_w) // 2
    pad_y  = (size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas


def process_crop(pil_img: Image.Image, box: tuple,
                 pad_ratio: float = 0.05, denoise: bool = True) -> Image.Image:
    """
    Crop tightly on the detected ROI, refine with Otsu to remove background,
    resize preserving aspect ratio with black padding to 256×256,
    optionally apply Gaussian + bilateral denoising.
    Returns a PIL Image (RGB).
    """
    W, H = pil_img.size
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1

    px = int(bw * pad_ratio)
    py = int(bh * pad_ratio)
    x1 = max(0, x1 - px);  y1 = max(0, y1 - py)
    x2 = min(W, x2 + px);  y2 = min(H, y2 + py)

    crop = pil_img.crop((x1, y1, x2, y2)).convert("RGB")

    # ── Otsu refinement: remove dark background inside the YOLO box ──────────
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

    # ── Resize keeping aspect ratio + black padding to 256×256 ───────────────
    crop = resize_with_padding(crop, IMAGE_SIZE)

    if denoise:
        arr = np.array(crop, dtype=np.uint8)
        arr = cv2.GaussianBlur(arr, (3, 3), sigmaX=0.8, sigmaY=0.8)
        arr = cv2.bilateralFilter(arr, d=5, sigmaColor=15, sigmaSpace=15)
        crop = Image.fromarray(arr)

    return crop


def main(args):
    out_dir = Path(args.out_dir) / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)

    denoise = not args.no_denoise

    print(f"\nLoading YOLO model: {args.model}")
    model = YOLO(args.model)

    images = get_images(args.input_dir)
    print(f"Found {len(images)} image(s) in {args.input_dir}\n")

    total_crops  = 0
    total_missed = 0

    for img_path in images:
        pil_img = Image.open(img_path).convert("RGB")
        results  = model(np.array(pil_img), conf=args.conf, verbose=False)
        saved    = 0

        for result in results:
            for i, box in enumerate(result.boxes):
                if int(box.cls[0]) != args.class_id:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop_pil = process_crop(pil_img, (x1, y1, x2, y2),
                                        pad_ratio=args.pad_ratio,
                                        denoise=denoise)
                fname = f"{img_path.stem}_crop{i:03d}.png"
                crop_pil.save(str(out_dir / fname))
                saved += 1

        if saved == 0:
            print(f"  [WARN] No detection : {img_path.name}")
            total_missed += 1
        else:
            print(f"  {img_path.name:45s} -> {saved} crop(s)")
            total_crops += saved

    print(f"\n{'='*50}")
    print(f"  Input images   : {len(images)}")
    print(f"  No detection   : {total_missed} image(s)")
    print(f"  Total crops    : {total_crops}")
    print(f"  Saved to       : {out_dir.resolve()}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",   required=True,            help="Folder containing all raw images")
    parser.add_argument("--out-dir",     required=True,            help="Output root directory")
    parser.add_argument("--model",       default="best.pt",        help="YOLO .pt weights")
    parser.add_argument("--class-id",    type=int,   default=0,    help="YOLO class index")
    parser.add_argument("--conf",        type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--pad-ratio",   type=float, default=0.05, help="Bbox padding ratio (default 5%%)")
    parser.add_argument("--no-denoise",  action="store_true",      help="Disable Gaussian + bilateral denoising")
    args = parser.parse_args()
    main(args)