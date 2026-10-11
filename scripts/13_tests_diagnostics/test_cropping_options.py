"""
test_cropping_options.py
========================
Test different cropping options on a single frame from a video.
Generates an HTML report comparing the effects of:
  - pad_ratio (0.0, 0.02, 0.05, 0.1)
  - Otsu refinement (ON / OFF)
  - Denoising (ON / OFF)

Usage:
    python test_cropping_options.py \
        --video "/path/to/video.mp4" \
        --frame 100 \
        --model /path/to/yolo.pt \
        --conf 0.25 \
        --out-dir ./cropping_test
"""

import argparse
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from ultralytics import YOLO

IMAGE_SIZE = 256

def resize_with_padding(crop: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    original_w, original_h = crop.size
    scale = size / max(original_w, original_h)
    new_w = int(original_w * scale)
    new_h = int(original_h * scale)
    resized = crop.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas

def process_crop(pil_img: Image.Image, box: tuple,
                 pad_ratio: float = 0.05,
                 use_otsu: bool = True,
                 denoise: bool = True,
                 return_intermediates: bool = False):
    """
    Process a single crop with the given parameters.
    box: (x1, y1, x2, y2) in pixel coordinates.
    """
    W, H = pil_img.size
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1

    px = int(bw * pad_ratio)
    py = int(bh * pad_ratio)
    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(W, x2 + px)
    y2 = min(H, y2 + py)

    # ── Step 1: padded crop ──────────────────────────────────────────────
    padded = pil_img.crop((x1, y1, x2, y2)).convert("RGB")

    # ── Step 2: Otsu refinement (optional) ──────────────────────────────
    otsu_crop = padded
    if use_otsu:
        arr_gray = np.array(padded.convert("L"), dtype=np.uint8)
        _, mask = cv2.threshold(arr_gray, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        coords = cv2.findNonZero(mask)
        if coords is not None:
            rx, ry, rw, rh = cv2.boundingRect(coords)
            coverage = (rw * rh) / (arr_gray.shape[1] * arr_gray.shape[0])
            if coverage >= 0.30:
                margin = 2
                rx = max(0, rx - margin)
                ry = max(0, ry - margin)
                rw = min(padded.width - rx, rw + 2 * margin)
                rh = min(padded.height - ry, rh + 2 * margin)
                otsu_crop = padded.crop((rx, ry, rx + rw, ry + rh))

    # ── Step 3: Resize + pad to 256×256 ─────────────────────────────────
    final = resize_with_padding(otsu_crop, IMAGE_SIZE)

    if denoise:
        arr = np.array(final, dtype=np.uint8)
        arr = cv2.GaussianBlur(arr, (3, 3), sigmaX=0.8, sigmaY=0.8)
        arr = cv2.bilateralFilter(arr, d=5, sigmaColor=15, sigmaSpace=15)
        final = Image.fromarray(arr)

    if return_intermediates:
        return {
            'padded': padded,
            'otsu_refined': otsu_crop if use_otsu else padded,
            'final': final,
        }
    return final

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--frame", type=int, default=None, help="Frame index (0-based). If not provided, uses first frame with detections.")
    parser.add_argument("--model", default="best.pt", help="YOLO weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence")
    parser.add_argument("--class-id", type=int, default=0, help="YOLO class ID to keep")
    parser.add_argument("--out-dir", default="./cropping_test", help="Output directory for images and HTML")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # ── Load YOLO ────────────────────────────────────────────────────────
    print(f"Loading YOLO: {args.model}")
    model = YOLO(args.model)

    # ── Open video ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {args.video}  |  {total_frames} frames")

    # ── Find frame with detections ──────────────────────────────────────
    if args.frame is None:
        for f_idx in range(total_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            results = model(frame_rgb, conf=args.conf, verbose=False)
            dets = []
            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) == args.class_id:
                        dets.append(box)
            if dets:
                frame_idx = f_idx
                break
        else:
            raise RuntimeError("No frame with detections found.")
    else:
        frame_idx = args.frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = cap.read()
        if not ret:
            raise RuntimeError(f"Could not read frame {frame_idx}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        results = model(frame_rgb, conf=args.conf, verbose=False)
        dets = []
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == args.class_id:
                    dets.append(box)
        if not dets:
            raise RuntimeError(f"No detections in frame {frame_idx}")

    cap.release()

    # ── Use the first detection ──────────────────────────────────────────
    box_obj = dets[0]
    x1, y1, x2, y2 = map(int, box_obj.xyxy[0])   # <--- FIX: extract coordinates
    print(f"Frame {frame_idx}: using detection at ({x1},{y1})-({x2},{y2})")

    # ── Parameter combinations ────────────────────────────────────────────
    pad_ratios = [0.0, 0.02, 0.05, 0.1]
    otsu_options = [True, False]
    denoise_options = [True, False]

    # ── Generate all crops ──────────────────────────────────────────────
    all_images = []
    for pad in pad_ratios:
        for use_otsu in otsu_options:
            for denoise in denoise_options:
                label = f"pad={pad:.2f} otsu={'ON' if use_otsu else 'OFF'} denoise={'ON' if denoise else 'OFF'}"
                result = process_crop(pil_img, (x1, y1, x2, y2),
                                      pad_ratio=pad,
                                      use_otsu=use_otsu,
                                      denoise=denoise,
                                      return_intermediates=True)
                # Save intermediate images
                for stage, img in [('padded', result['padded']),
                                   ('otsu_refined', result['otsu_refined']),
                                   ('final', result['final'])]:
                    fname = f"frame{frame_idx:07d}_pad{pad:.2f}_otsu{int(use_otsu)}_denoise{int(denoise)}_{stage}.png"
                    img.save(img_dir / fname)
                all_images.append({
                    'pad': pad,
                    'otsu': use_otsu,
                    'denoise': denoise,
                    'label': label,
                    'padded': result['padded'],
                    'otsu_refined': result['otsu_refined'],
                    'final': result['final'],
                    'filenames': {
                        'padded': f"frame{frame_idx:07d}_pad{pad:.2f}_otsu{int(use_otsu)}_denoise{int(denoise)}_padded.png",
                        'otsu_refined': f"frame{frame_idx:07d}_pad{pad:.2f}_otsu{int(use_otsu)}_denoise{int(denoise)}_otsu_refined.png",
                        'final': f"frame{frame_idx:07d}_pad{pad:.2f}_otsu{int(use_otsu)}_denoise{int(denoise)}_final.png",
                    }
                })

    # ── Generate HTML report ──────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Cropping Options Comparison</title>
    <style>
        body {{ font-family: -apple-system, sans-serif; background: #1a1a2e; color: #eaeaea; padding: 20px; }}
        h1, h2 {{ color: #eaeaea; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #444; padding: 12px; text-align: center; vertical-align: middle; }}
        th {{ background: #2a2a4e; color: #9ca3af; font-weight: 600; }}
        td {{ background: #1e1e3a; }}
        img {{ max-width: 200px; height: auto; border-radius: 4px; }}
        .label {{ font-size: 0.8em; color: #9ca3af; margin-top: 4px; }}
        .stage-label {{ font-weight: bold; color: #60a5fa; }}
    </style>
</head>
<body>
    <h1>Cropping Options Comparison</h1>
    <p>Video: <b>{args.video}</b> &nbsp;|&nbsp; Frame: <b>{frame_idx}</b> &nbsp;|&nbsp; Detection: <b>({x1},{y1})-({x2},{y2})</b></p>
    <p>Image size: <b>{IMAGE_SIZE}×{IMAGE_SIZE}</b> &nbsp;|&nbsp; Otsu coverage threshold: <b>30%</b> &nbsp;|&nbsp; Denoise: GaussianBlur(3×3) + BilateralFilter(d=5)</p>

    <h2>Final Crops (256×256)</h2>
    <table>
        <tr>
            <th>pad_ratio</th>
            <th>Otsu</th>
            <th>Denoise</th>
            <th>Final crop</th>
        </tr>
"""
    for img_info in all_images:
        final_path = img_info['filenames']['final']
        html += f"""
        <tr>
            <td>{img_info['pad']:.2f}</td>
            <td>{'ON' if img_info['otsu'] else 'OFF'}</td>
            <td>{'ON' if img_info['denoise'] else 'OFF'}</td>
            <td><img src="images/{final_path}" alt="{img_info['label']}"></td>
        </tr>
"""
    html += """
    </table>

    <h2>Intermediate Stages (for selected combinations)</h2>
    <table>
        <tr>
            <th>pad_ratio</th>
            <th>Otsu</th>
            <th>Denoise</th>
            <th>Padded (before Otsu)</th>
            <th>Otsu-refined</th>
            <th>Final (256×256)</th>
        </tr>
"""
    # Show only a subset: all pad_ratios with Otsu ON and Denoise OFF
    selected = [img for img in all_images if img['otsu'] and not img['denoise']]
    for img_info in selected:
        padded_path = img_info['filenames']['padded']
        otsu_path = img_info['filenames']['otsu_refined']
        final_path = img_info['filenames']['final']
        html += f"""
        <tr>
            <td>{img_info['pad']:.2f}</td>
            <td>ON</td>
            <td>OFF</td>
            <td><img src="images/{padded_path}" alt="padded"></td>
            <td><img src="images/{otsu_path}" alt="otsu"></td>
            <td><img src="images/{final_path}" alt="final"></td>
        </tr>
"""
    # Also show with Denoise ON for pad_ratio=0.05
    html += f"""
        <tr><td colspan="6" style="background:#2a2a4e; text-align:center;"><b>With Denoising ON (pad_ratio=0.05, Otsu=ON)</b></td></tr>
"""
    for img_info in all_images:
        if img_info['pad'] == 0.05 and img_info['otsu'] and img_info['denoise']:
            padded_path = img_info['filenames']['padded']
            otsu_path = img_info['filenames']['otsu_refined']
            final_path = img_info['filenames']['final']
            html += f"""
        <tr>
            <td>{img_info['pad']:.2f}</td>
            <td>ON</td>
            <td>ON</td>
            <td><img src="images/{padded_path}" alt="padded"></td>
            <td><img src="images/{otsu_path}" alt="otsu"></td>
            <td><img src="images/{final_path}" alt="final"></td>
        </tr>
"""
    html += """
    </table>

    <h2>All Combinations (Final Crops only)</h2>
    <table>
        <tr>
            <th>pad_ratio</th>
            <th>Otsu</th>
            <th>Denoise</th>
            <th>Final crop</th>
        </tr>
"""
    for img_info in all_images:
        final_path = img_info['filenames']['final']
        html += f"""
        <tr>
            <td>{img_info['pad']:.2f}</td>
            <td>{'ON' if img_info['otsu'] else 'OFF'}</td>
            <td>{'ON' if img_info['denoise'] else 'OFF'}</td>
            <td><img src="images/{final_path}" alt="{img_info['label']}"></td>
        </tr>
"""
    html += """
    </table>
</body>
</html>
"""

    html_path = out_dir / "cropping_comparison.html"
    with open(html_path, 'w') as f:
        f.write(html)

    print(f"HTML report saved to: {html_path.resolve()}")
    print("Open it in a browser to view the comparison.")

if __name__ == "__main__":
    main()