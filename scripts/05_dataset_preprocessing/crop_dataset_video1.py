"""
crop_dataset_video.py
=====================
Run YOLOv8 on a video file (or every video file in a folder), crop every
detected object per frame, apply EfficientAD-compatible preprocessing,
and save all crops into an output folder.

Preprocessing (mirrors build_crop_dataset.py):
    1. Pad bbox by pad_ratio (default 5%)
    2. Otsu refinement: tighten crop to foreground (skipped if coverage < 30%)
    3. Resize preserving aspect ratio + black padding to 256×256 (LANCZOS)
    4. GaussianBlur (3×3, sigma=0.8)  ← skipped with --no-denoise
    5. BilateralFilter (d=5, sigmaColor=15, sigmaSpace=15)  ← skipped with --no-denoise

Input (choose one):
    --video PATH        Single video file
    --video-dir PATH     Folder of videos — every file matching --video-ext
                         is processed in turn, using the SAME loaded model.

Options:
    --video-ext EXT [EXT ...]  Extensions to match in --video-dir
                                (default: .mp4 .avi .mov .mkv)
    --frame-step N       Only process every N-th frame (default: 1 = all frames)
    --max-frames N       Stop after N frames PER VIDEO (default: unlimited)
    --dedup-iou FLOAT    Skip a crop if IoU with last saved crop box > threshold
                         (default: 0.85). Set to 0.0 to disable deduplication.
                         Dedup state resets at the start of each video.

Output structure (crops are prefixed with the source video's stem so files
from different videos never collide):
    out_dir/
    └── crops/
        ├── videoA_frame0000001_crop000.png
        ├── videoA_frame0000001_crop001.png
        ├── videoB_frame0000001_crop000.png
        └── ...
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
    W, H = pil_img.size
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1

    px = int(bw * pad_ratio)
    py = int(bh * pad_ratio)
    x1 = max(0, x1 - px);  y1 = max(0, y1 - py)
    x2 = min(W, x2 + px);  y2 = min(H, y2 + py)

    crop = pil_img.crop((x1, y1, x2, y2)).convert("RGB")

    # ── Otsu refinement ───────────────────────────────────────────────────────
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

    # ── Resize + pad ──────────────────────────────────────────────────────────
    crop = resize_with_padding(crop, IMAGE_SIZE)

    if denoise:
        arr = np.array(crop, dtype=np.uint8)
        arr = cv2.GaussianBlur(arr, (3, 3), sigmaX=0.8, sigmaY=0.8)
        arr = cv2.bilateralFilter(arr, d=5, sigmaColor=15, sigmaSpace=15)
        crop = Image.fromarray(arr)

    return crop


def box_iou(a: tuple, b: tuple) -> float:
    """Compute IoU between two (x1,y1,x2,y2) boxes."""
    ix1 = max(a[0], b[0]);  iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]);  iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter)


def process_video(video_path: Path, out_dir: Path, model: YOLO, args) -> dict:
    """Run detection+crop extraction on a single video. Returns a stats dict."""
    denoise = not args.no_denoise
    prefix  = video_path.stem  # used to namespace crop filenames across videos

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ⚠ Cannot open video, skipping: {video_path}")
        return {"video": str(video_path), "opened": False}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    print(f"\nVideo: {video_path.name}  |  {total_frames} frames @ {fps:.1f} fps")

    total_crops       = 0
    processed_frames  = 0
    skipped_dedup     = 0

    # Per-detection-slot: last saved box for cross-frame dedup.
    # Reset per video — slots aren't meaningful across different videos.
    last_boxes: dict[int, tuple] = {}

    frame_idx = -1
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_idx += 1

        # ── frame-step filter ─────────────────────────────────────────────────
        if frame_idx % args.frame_step != 0:
            continue

        # ── max-frames limit (per video) ────────────────────────────────────
        if args.max_frames and processed_frames >= args.max_frames:
            break

        processed_frames += 1

        # Convert BGR→RGB for YOLO + PIL
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img   = Image.fromarray(frame_rgb)

        results = model(frame_rgb, conf=args.conf, verbose=False)

        crop_idx = 0
        for result in results:
            for i, box in enumerate(result.boxes):
                if int(box.cls[0]) != args.class_id:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                current_box = (x1, y1, x2, y2)

                # ── cross-frame deduplication ─────────────────────────────────
                if args.dedup_iou > 0.0:
                    prev = last_boxes.get(i)
                    if prev is not None and box_iou(prev, current_box) >= args.dedup_iou:
                        skipped_dedup += 1
                        continue
                last_boxes[i] = current_box

                crop_pil = process_crop(pil_img, current_box,
                                        pad_ratio=args.pad_ratio,
                                        denoise=denoise)

                fname = f"{prefix}_frame{frame_idx:07d}_crop{crop_idx:03d}.png"
                crop_pil.save(str(out_dir / fname))
                crop_idx  += 1
                total_crops += 1

        if processed_frames % 50 == 0:
            pct = frame_idx / max(total_frames, 1) * 100
            print(f"  Frame {frame_idx:6d}/{total_frames}  ({pct:5.1f}%)  "
                  f"crops so far: {total_crops}")

    cap.release()

    print(f"  Processed      : {processed_frames} (step={args.frame_step})")
    print(f"  Dedup skipped  : {skipped_dedup}")
    print(f"  Crops saved    : {total_crops}")

    return {
        "video": str(video_path),
        "opened": True,
        "total_frames": total_frames,
        "processed_frames": processed_frames,
        "skipped_dedup": skipped_dedup,
        "total_crops": total_crops,
    }


def main(args):
    out_dir = Path(args.out_dir) / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve list of videos to process ───────────────────────────────────
    if args.video_dir:
        exts = {e.lower() if e.startswith(".") else f".{e.lower()}"
                for e in args.video_ext}
        video_paths = sorted(
            p for p in Path(args.video_dir).iterdir()
            if p.is_file() and p.suffix.lower() in exts
        )
        if not video_paths:
            raise RuntimeError(
                f"No videos matching {sorted(exts)} found in {args.video_dir}"
            )
    else:
        video_paths = [Path(args.video)]

    print(f"\nLoading YOLO model: {args.model}")
    model = YOLO(args.model)  # loaded once, reused across all videos

    print(f"Found {len(video_paths)} video(s) to process\n")

    all_stats = [process_video(vp, out_dir, model, args) for vp in video_paths]

    # ── Aggregate summary ────────────────────────────────────────────────────
    opened      = [s for s in all_stats if s["opened"]]
    failed      = [s for s in all_stats if not s["opened"]]
    total_crops = sum(s["total_crops"] for s in opened)
    total_proc_frames = sum(s["processed_frames"] for s in opened)
    total_dedup = sum(s["skipped_dedup"] for s in opened)

    print(f"\n{'='*55}")
    print(f"  Videos processed : {len(opened)}/{len(video_paths)}"
          f"{f'  ({len(failed)} failed to open)' if failed else ''}")
    print(f"  Frames processed : {total_proc_frames} (step={args.frame_step})")
    print(f"  Dedup skipped    : {total_dedup}")
    print(f"  Total crops      : {total_crops}")
    print(f"  Saved to         : {out_dir.resolve()}")
    print(f"{'='*55}\n")

    if failed:
        print("Videos that failed to open:")
        for s in failed:
            print(f"  - {s['video']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crop YOLO detections from a video into EfficientAD-ready 256×256 images."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video",       help="Path to a single input video file")
    src.add_argument("--video-dir",   help="Path to a folder of video files to process")
    parser.add_argument("--video-ext", nargs="+", default=[".mp4", ".avi", ".mov", ".mkv"],
                        help="Extensions to match when using --video-dir "
                             "(default: .mp4 .avi .mov .mkv)")
    parser.add_argument("--out-dir",     required=True,              help="Output root directory")
    parser.add_argument("--model",       default="best.pt",          help="YOLO .pt weights")
    parser.add_argument("--class-id",    type=int,   default=0,      help="YOLO class index to keep")
    parser.add_argument("--conf",        type=float, default=0.25,   help="Detection confidence threshold")
    parser.add_argument("--pad-ratio",   type=float, default=0.05,   help="Bbox padding ratio (default 5%%)")
    parser.add_argument("--frame-step",  type=int,   default=1,      help="Process every N-th frame (default: 1)")
    parser.add_argument("--max-frames",  type=int,   default=0,      help="Stop after N processed frames (0=unlimited)")
    parser.add_argument("--dedup-iou",   type=float, default=0.85,   help="Skip crop if IoU with last saved box >= threshold (0=off)")
    parser.add_argument("--no-denoise",  action="store_true",        help="Disable Gaussian + bilateral denoising")
    args = parser.parse_args()
    main(args)
