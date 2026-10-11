"""
Batch Image Processor — YOLOv8 Segmentation Edition
=====================================================
Uses a YOLOv8-seg model to remove background via predicted masks,
then optionally enhances clarity. Drops-in as a replacement for rembg.

Requirements:
    pip install ultralytics Pillow numpy opencv-python

Usage:
    # Background removal via YOLO + enhancement (default)
    python batch_image_processor_yolo.py \
        --model /home/ahmed/bbox2seg/runs/segment/train/weights/best.pt \
        --input ./photos \
        --output ./output

    # Remove background only (no enhancement)
    python batch_image_processor_yolo.py \
        --model best.pt --input ./photos --output ./output --no-enhance

    # Enhancement only (skip segmentation)
    python batch_image_processor_yolo.py \
        --model best.pt --input ./photos --output ./output --no-seg

    # White background instead of transparency
    python batch_image_processor_yolo.py \
        --model best.pt --input ./photos --output ./output --bg-color "#ffffff"

    # Run on GPU 0, save as JPEG
    python batch_image_processor_yolo.py \
        --model best.pt --input ./photos --output ./output \
        --device 0 --format JPEG
"""

import argparse
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from ultralytics import YOLO

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


# ── Helpers ────────────────────────────────────────────────────────────────

def hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def enhance_image(img: Image.Image, sharpen=1.5, contrast=1.2,
                  brightness=1.05, saturation=1.1) -> Image.Image:
    """Unsharp-mask + contrast/brightness/saturation/sharpness boosts."""
    original_mode = img.mode

    if sharpen > 1.0:
        img = img.filter(
            ImageFilter.UnsharpMask(radius=1.5,
                                    percent=int((sharpen - 1) * 150),
                                    threshold=3)
        )

    if original_mode == "RGBA":
        rgb, alpha = img.convert("RGB"), img.split()[3]
    else:
        rgb  = img.convert("RGB")
        alpha = None

    rgb = ImageEnhance.Contrast(rgb).enhance(contrast)
    rgb = ImageEnhance.Brightness(rgb).enhance(brightness)
    rgb = ImageEnhance.Color(rgb).enhance(saturation)
    rgb = ImageEnhance.Sharpness(rgb).enhance(sharpen)

    if alpha:
        rgb = rgb.convert("RGBA")
        rgb.putalpha(alpha)
    return rgb


def apply_background(img: Image.Image, bg_color: str) -> Image.Image:
    """Flatten RGBA onto a solid color background."""
    if img.mode != "RGBA":
        return img
    bg_rgb = hex_to_rgb(bg_color)
    background = Image.new("RGBA", img.size, bg_rgb + (255,))
    background.paste(img, mask=img.split()[3])
    return background.convert("RGB")


# ── Core segmentation step ────────────────────────────────────────────────

def remove_background_yolo(model: YOLO, image_path: Path,
                            conf: float, device: str,
                            imgsz: int) -> Image.Image:
    """
    Run YOLOv8-seg inference on one image.
    Merges all predicted instance masks into a single alpha channel,
    returning an RGBA PIL image with the background set to transparent.
    """
    img_pil = Image.open(image_path).convert("RGB")
    w, h    = img_pil.size

    results = model.predict(
        str(image_path),
        conf=conf,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )

    result = results[0]

    # No detections → return original image unchanged (fully opaque)
    if result.masks is None or len(result.masks) == 0:
        print(f"    [warn] No detections in {image_path.name} — returning original.")
        rgba = img_pil.convert("RGBA")
        return rgba

    # Merge all instance masks with OR (union of all foreground objects)
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    for mask_tensor in result.masks.data:
        # masks.data shape: (N, mask_h, mask_w) — values in [0,1]
        mask_np = mask_tensor.cpu().numpy()
        # Resize to original image resolution
        mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_LINEAR)
        binary = (mask_resized > 0.5).astype(np.uint8) * 255
        combined_mask = cv2.bitwise_or(combined_mask, binary)

    # Optional: slight Gaussian blur on mask edges for a cleaner cut
    combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)
    _, combined_mask = cv2.threshold(combined_mask, 127, 255, cv2.THRESH_BINARY)

    # Build RGBA image: paste mask as alpha channel
    img_np   = np.array(img_pil)
    rgba_np  = np.dstack([img_np, combined_mask])
    return Image.fromarray(rgba_np, "RGBA")


# ── Per-image worker ──────────────────────────────────────────────────────

def process_image(model: YOLO,
                  input_path: Path, output_path: Path,
                  use_seg: bool, enhance: bool,
                  bg_color: str, out_format: str,
                  conf: float, device: str, imgsz: int,
                  sharpen: float, contrast: float,
                  brightness: float, saturation: float) -> str:
    try:
        if use_seg:
            img = remove_background_yolo(model, input_path, conf, device, imgsz)
        else:
            img = Image.open(input_path).convert("RGBA")

        if enhance:
            img = enhance_image(img, sharpen=sharpen, contrast=contrast,
                                brightness=brightness, saturation=saturation)

        if bg_color:
            img = apply_background(img, bg_color)

        # Format & save
        fmt = out_format.upper()
        if fmt == "JPEG" and img.mode == "RGBA":
            img = img.convert("RGB")
        ext = ".jpg" if fmt == "JPEG" else f".{fmt.lower()}"
        final_path = output_path.with_suffix(ext)

        img.save(final_path, format=fmt, quality=95)
        return f"✓  {input_path.name}  →  {final_path.name}"

    except Exception as e:
        return f"✗  {input_path.name}: {e}"


# ── Main ──────────────────────────────────────────────────────────────────

def collect_images(input_dir: Path):
    return sorted([p for p in input_dir.rglob("*") if p.suffix.lower() in SUPPORTED])


def main():
    parser = argparse.ArgumentParser(
        description="Batch background removal via YOLOv8-seg + optional clarity enhancement"
    )
    parser.add_argument("--model",      required=True, help="Path to YOLOv8-seg best.pt")
    parser.add_argument("--input",  "-i", required=True, help="Input folder")
    parser.add_argument("--output", "-o", required=True, help="Output folder")
    parser.add_argument("--no-seg",       action="store_true", help="Skip segmentation (enhance only)")
    parser.add_argument("--no-enhance",   action="store_true", help="Skip enhancement")
    parser.add_argument("--bg-color",     default=None,
                        help="Solid background hex color e.g. #ffffff (default: transparent)")
    parser.add_argument("--format",       default="PNG", choices=["PNG","JPEG","WEBP"],
                        help="Output format (default: PNG)")
    parser.add_argument("--conf",         type=float, default=0.5,  help="Detection confidence threshold")
    parser.add_argument("--imgsz",        type=int,   default=640,  help="Inference image size")
    parser.add_argument("--device",       default="0", help="cuda device index or 'cpu'")
    parser.add_argument("--workers", "-w",type=int,   default=4,    help="Parallel workers (default: 4)")
    parser.add_argument("--sharpen",      type=float, default=1.5)
    parser.add_argument("--contrast",     type=float, default=1.2)
    parser.add_argument("--brightness",   type=float, default=1.05)
    parser.add_argument("--saturation",   type=float, default=1.1)
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Error: input folder '{input_dir}' does not exist.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(input_dir)
    if not images:
        print(f"No supported images found in '{input_dir}'.")
        sys.exit(0)

    use_seg = not args.no_seg
    enhance = not args.no_enhance

    print(f"\nBatch Image Processor — YOLOv8-seg")
    print(f"{'─'*50}")
    print(f"  Model      : {args.model}")
    print(f"  Input      : {input_dir.resolve()}")
    print(f"  Output     : {output_dir.resolve()}")
    print(f"  Images     : {len(images)} found")
    print(f"  Segmentation: {'yes' if use_seg else 'no'}")
    print(f"  Enhance    : {'yes' if enhance else 'no'}")
    print(f"  BG color   : {args.bg_color or 'transparent'}")
    print(f"  Format     : {args.format}")
    print(f"  Conf       : {args.conf}   imgsz: {args.imgsz}   device: {args.device}")
    print(f"  Workers    : {args.workers}")
    print(f"{'─'*50}\n")

    # Load model once — shared across threads (YOLO predict is thread-safe per call)
    print("Loading model...")
    model = YOLO(args.model)
    print("Model loaded.\n")

    jobs = []
    for img_path in images:
        rel      = img_path.relative_to(input_dir)
        out_path = output_dir / rel.with_suffix(f".{args.format.lower()}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        jobs.append((img_path, out_path))

    results_log = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_image,
                model, inp, out,
                use_seg, enhance,
                args.bg_color, args.format,
                args.conf, args.device, args.imgsz,
                args.sharpen, args.contrast,
                args.brightness, args.saturation
            ): inp for inp, out in jobs
        }
        done = 0
        for future in as_completed(futures):
            done += 1
            msg = future.result()
            results_log.append(msg)
            print(f"  [{done:>4}/{len(images)}] {msg}")

    ok  = sum(1 for r in results_log if r.startswith("✓"))
    err = sum(1 for r in results_log if r.startswith("✗"))
    print(f"\n{'─'*50}")
    print(f"  Done : {ok} succeeded, {err} failed")
    print(f"  Saved: {output_dir.resolve()}")
    print(f"{'─'*50}\n")


if __name__ == "__main__":
    main()
