"""
Batch Image Processor
=====================
Removes background and enhances clarity for an entire folder of photos.

Requirements:
    pip install rembg onnxruntime Pillow numpy

Usage:
    python batch_image_processor.py --input ./photos --output ./output

    # Remove background only
    python batch_image_processor.py --input /home/ahmed/two-stage-efficientad-inspection/dataset/chewinggum/Data/Images/Anomaly --output ./output_pictures --no-enhance

    # Enhance only (no background removal)
    python batch_image_processor.py --input ./photos --output ./output --no-rembg

    # Custom background color (hex)
    python batch_image_processor.py --input ./photos --output ./output --bg-color "#ffffff"

    # Keep original format (default saves as PNG to preserve transparency)
    python batch_image_processor.py --input ./photos --output ./output --format JPEG
"""

import argparse
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def enhance_image(img: Image.Image, sharpen: float = 1.5, contrast: float = 1.2,
                  brightness: float = 1.05, saturation: float = 1.1) -> Image.Image:
    """Apply sharpening, contrast, brightness and saturation boosts."""
    # Work on RGB(A) copy
    original_mode = img.mode

    # Sharpen using unsharp mask for edge clarity
    if sharpen > 1.0:
        img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=int((sharpen - 1) * 150), threshold=3))

    # Apply enhancements (skip alpha channel)
    if original_mode == "RGBA":
        rgb, alpha = img.convert("RGB"), img.split()[3]
    else:
        rgb = img.convert("RGB")
        alpha = None

    rgb = ImageEnhance.Contrast(rgb).enhance(contrast)
    rgb = ImageEnhance.Brightness(rgb).enhance(brightness)
    rgb = ImageEnhance.Color(rgb).enhance(saturation)
    rgb = ImageEnhance.Sharpness(rgb).enhance(sharpen)

    if alpha:
        rgb = rgb.convert("RGBA")
        rgb.putalpha(alpha)

    return rgb


def apply_background(img: Image.Image, bg_color: str = None) -> Image.Image:
    """Paste the image (with transparency) onto a solid background color."""
    if bg_color is None or img.mode != "RGBA":
        return img
    bg_rgb = hex_to_rgb(bg_color)
    background = Image.new("RGBA", img.size, bg_rgb + (255,))
    background.paste(img, mask=img.split()[3])
    return background.convert("RGB")


def process_image(input_path: Path, output_path: Path,
                  remove_bg: bool, enhance: bool,
                  bg_color: str, out_format: str,
                  sharpen: float, contrast: float,
                  brightness: float, saturation: float) -> str:
    try:
        img = Image.open(input_path).convert("RGBA")

        if remove_bg:
            from rembg import remove as rembg_remove
            img_bytes = open(input_path, "rb").read()
            from io import BytesIO
            result_bytes = rembg_remove(img_bytes)
            img = Image.open(BytesIO(result_bytes)).convert("RGBA")

        if enhance:
            img = enhance_image(img, sharpen=sharpen, contrast=contrast,
                                brightness=brightness, saturation=saturation)

        if bg_color:
            img = apply_background(img, bg_color)

        # Decide save format and extension
        fmt = out_format.upper()
        if fmt == "JPEG" and img.mode == "RGBA":
            img = img.convert("RGB")  # JPEG doesn't support alpha
        ext = ".jpg" if fmt == "JPEG" else f".{fmt.lower()}"
        final_path = output_path.with_suffix(ext)

        img.save(final_path, format=fmt, quality=95)
        return f"  ✓  {input_path.name}  →  {final_path.name}"

    except Exception as e:
        return f"  ✗  {input_path.name}: {e}"


def collect_images(input_dir: Path):
    return sorted([p for p in input_dir.rglob("*") if p.suffix.lower() in SUPPORTED])


def main():
    parser = argparse.ArgumentParser(description="Batch image background remover & clarity enhancer")
    parser.add_argument("--input",     "-i", required=True,  help="Input folder containing images")
    parser.add_argument("--output",    "-o", required=True,  help="Output folder for processed images")
    parser.add_argument("--no-rembg",        action="store_true", help="Skip background removal")
    parser.add_argument("--no-enhance",      action="store_true", help="Skip clarity enhancement")
    parser.add_argument("--bg-color",        default=None,    help="Solid background color after removal (e.g. #ffffff)")
    parser.add_argument("--format",          default="PNG",   choices=["PNG","JPEG","WEBP"], help="Output format (default: PNG)")
    parser.add_argument("--workers",  "-w",  type=int, default=4, help="Parallel workers (default: 4)")
    parser.add_argument("--sharpen",         type=float, default=1.5, help="Sharpen level 1.0–3.0 (default: 1.5)")
    parser.add_argument("--contrast",        type=float, default=1.2, help="Contrast level 1.0–2.0 (default: 1.2)")
    parser.add_argument("--brightness",      type=float, default=1.05,help="Brightness 0.5–2.0 (default: 1.05)")
    parser.add_argument("--saturation",      type=float, default=1.1, help="Saturation 0.0–2.0 (default: 1.1)")
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
        print(f"Supported formats: {', '.join(SUPPORTED)}")
        sys.exit(0)

    remove_bg = not args.no_rembg
    enhance   = not args.no_enhance

    print(f"\nBatch Image Processor")
    print(f"{'─'*45}")
    print(f"  Input   : {input_dir.resolve()}")
    print(f"  Output  : {output_dir.resolve()}")
    print(f"  Images  : {len(images)} found")
    print(f"  Remove BG: {'yes' if remove_bg else 'no'}")
    print(f"  Enhance  : {'yes' if enhance else 'no'}")
    print(f"  BG color : {args.bg_color or 'transparent'}")
    print(f"  Format   : {args.format}")
    print(f"  Workers  : {args.workers}")
    print(f"{'─'*45}\n")

    jobs = []
    for img_path in images:
        rel = img_path.relative_to(input_dir)
        out_path = output_dir / rel.with_suffix(f".{args.format.lower()}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        jobs.append((img_path, out_path))

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_image,
                inp, out,
                remove_bg, enhance,
                args.bg_color, args.format,
                args.sharpen, args.contrast,
                args.brightness, args.saturation
            ): inp for inp, out in jobs
        }
        done = 0
        for future in as_completed(futures):
            done += 1
            msg = future.result()
            results.append(msg)
            print(f"[{done}/{len(images)}] {msg}")

    ok  = sum(1 for r in results if "✓" in r)
    err = sum(1 for r in results if "✗" in r)
    print(f"\n{'─'*45}")
    print(f"  Done: {ok} succeeded, {err} failed")
    print(f"  Saved to: {output_dir.resolve()}")
    print(f"{'─'*45}\n")


if __name__ == "__main__":
    main()
