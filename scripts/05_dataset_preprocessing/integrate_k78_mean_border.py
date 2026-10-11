#!/usr/bin/env python3
from pathlib import Path

ROOT = Path("/home/ahmed/two-stage-efficientad-inspection")
SRC = ROOT / "video_streaming_detector_simple_optimized_v2.py"
DST = ROOT / "video_streaming_detector_simple_optimized_v3.py"

if not SRC.exists():
    raise SystemExit(f"ERROR: missing source file: {SRC}")

text = SRC.read_text(encoding="utf-8")

anchor = "_NORM_CACHE = {}\n"
if anchor not in text:
    raise SystemExit("ERROR: could not find _NORM_CACHE anchor.")

helper = r"""
_MEAN_BORDER_MASK_CACHE = {}


def _mean_border_mask(height, width, keep_ratio=0.78, feather_ratio=0.08):
    key = (
        int(height),
        int(width),
        round(float(keep_ratio), 6),
        round(float(feather_ratio), 6),
    )
    if key in _MEAN_BORDER_MASK_CACHE:
        return _MEAN_BORDER_MASK_CACHE[key]

    keep_ratio = float(np.clip(keep_ratio, 0.40, 0.98))
    feather_ratio = float(np.clip(feather_ratio, 0.0, 0.25))

    mask = np.zeros((int(height), int(width)), dtype=np.float32)

    xpad = int(round((1.0 - keep_ratio) * 0.5 * width))
    ypad = int(round((1.0 - keep_ratio) * 0.5 * height))

    x1 = max(0, xpad)
    y1 = max(0, ypad)
    x2 = min(width, max(x1 + 1, width - xpad))
    y2 = min(height, max(y1 + 1, height - ypad))

    mask[y1:y2, x1:x2] = 1.0

    feather = max(1, int(round(feather_ratio * min(height, width))))
    kernel = 2 * feather + 1
    mask = cv2.GaussianBlur(mask, (kernel, kernel), 0)
    mask = np.clip(mask, 0.0, 1.0)[..., None]

    _MEAN_BORDER_MASK_CACHE[key] = mask
    return mask


def apply_mean_border_suppression(
    crop_pil,
    keep_ratio=0.78,
    feather_ratio=0.08,
    strength=1.0,
):
    if strength <= 0.0:
        return crop_pil

    arr = np.asarray(crop_pil.convert("RGB"), dtype=np.uint8)
    height, width = arr.shape[:2]

    mask = _mean_border_mask(
        height,
        width,
        keep_ratio=keep_ratio,
        feather_ratio=feather_ratio,
    )

    center_ratio = 0.50
    xpad = int(round((1.0 - center_ratio) * 0.5 * width))
    ypad = int(round((1.0 - center_ratio) * 0.5 * height))

    center = arr[
        ypad:max(ypad + 1, height - ypad),
        xpad:max(xpad + 1, width - xpad),
    ]
    if center.size == 0:
        center = arr

    fill_rgb = center.reshape(-1, 3).mean(axis=0).astype(np.float32)

    original = arr.astype(np.float32)
    fill = np.empty_like(original, dtype=np.float32)
    fill[...] = fill_rgb

    strength = float(np.clip(strength, 0.0, 1.0))
    outer = (1.0 - strength) * original + strength * fill
    output = original * mask + outer * (1.0 - mask)

    return Image.fromarray(
        np.clip(output, 0, 255).astype(np.uint8),
        mode="RGB",
    )


def apply_crop_preprocess(crop_pil, args):
    if args.crop_preprocess == "current":
        return crop_pil

    if args.crop_preprocess == "mean_border":
        return apply_mean_border_suppression(
            crop_pil,
            keep_ratio=args.mean_border_keep_ratio,
            feather_ratio=args.mean_border_feather_ratio,
            strength=args.mean_border_strength,
        )

    raise ValueError(f"Unknown crop preprocessing: {args.crop_preprocess}")

"""

text = text.replace(anchor, anchor + helper, 1)

old_build = """            def build_crop(task):
                pil, box = task
                return get_crop(
                    pil,
                    box,
                    scorer.image_size,
                    args.pad_ratio,
                    args.denoise,
                )
"""

new_build = """            def build_crop(task):
                pil, box = task

                crop = get_crop(
                    pil,
                    box,
                    scorer.image_size,
                    args.pad_ratio,
                    args.denoise,
                )

                crop = apply_crop_preprocess(
                    crop,
                    args,
                )

                return crop
"""

if old_build not in text:
    raise SystemExit("ERROR: could not find build_crop block.")
text = text.replace(old_build, new_build, 1)

old_cli = """    p.add_argument("--pad_ratio", type=float, default=0.02)
    p.add_argument("--denoise", action="store_true")
"""

new_cli = """    p.add_argument("--pad_ratio", type=float, default=0.02)
    p.add_argument("--denoise", action="store_true")

    p.add_argument(
        "--crop_preprocess",
        choices=["current", "mean_border"],
        default="mean_border",
        help=(
            "Crop preprocessing after the original get_crop(). "
            "mean_border is the selected generic preprocessing; "
            "current reproduces v2 exactly."
        ),
    )

    p.add_argument(
        "--mean_border_keep_ratio",
        type=float,
        default=0.78,
        help="Central crop fraction preserved.",
    )

    p.add_argument(
        "--mean_border_feather_ratio",
        type=float,
        default=0.08,
        help="Soft transition width.",
    )

    p.add_argument(
        "--mean_border_strength",
        type=float,
        default=1.0,
        help="Outer-context mean replacement strength.",
    )
"""

if old_cli not in text:
    raise SystemExit("ERROR: could not find CLI insertion anchor.")
text = text.replace(old_cli, new_cli, 1)

old_validation = """    if args.batch_frames < 1 or args.batch_crops < 1:
        raise ValueError("batch sizes must be >= 1")

    os.makedirs(args.save_dir, exist_ok=True)
"""

new_validation = """    if args.batch_frames < 1 or args.batch_crops < 1:
        raise ValueError("batch sizes must be >= 1")

    if not (0.40 <= args.mean_border_keep_ratio <= 0.98):
        raise ValueError("--mean_border_keep_ratio must be between 0.40 and 0.98")

    if not (0.0 <= args.mean_border_feather_ratio <= 0.25):
        raise ValueError("--mean_border_feather_ratio must be between 0 and 0.25")

    if not (0.0 <= args.mean_border_strength <= 1.0):
        raise ValueError("--mean_border_strength must be between 0 and 1")

    os.makedirs(args.save_dir, exist_ok=True)
"""

if old_validation not in text:
    raise SystemExit("ERROR: could not find validation anchor.")
text = text.replace(old_validation, new_validation, 1)

old_print = """    print(
        f"batch          : frames={args.batch_frames}, "
        f"chunk_crops={args.batch_crops}, "
        f"gpu_crops={args.gpu_batch_crops}"
    )
"""

new_print = """    print(
        f"batch          : frames={args.batch_frames}, "
        f"chunk_crops={args.batch_crops}, "
        f"gpu_crops={args.gpu_batch_crops}"
    )
    print(f"crop preprocess: {args.crop_preprocess}")
    if args.crop_preprocess == "mean_border":
        print(
            f"mean border    : keep={args.mean_border_keep_ratio:.2f}, "
            f"feather={args.mean_border_feather_ratio:.2f}, "
            f"strength={args.mean_border_strength:.2f}"
        )
"""

if old_print not in text:
    raise SystemExit("ERROR: could not find startup print anchor.")
text = text.replace(old_print, new_print, 1)

text = text.replace(
    "video_streaming_detector_simple_optimized.py",
    "video_streaming_detector_simple_optimized_v3.py",
    1,
)

DST.write_text(text, encoding="utf-8")

print("=" * 72)
print("INTEGRATION COMPLETE")
print("=" * 72)
print(f"Original kept : {SRC}")
print(f"New detector  : {DST}")
print("Defaults: mean_border, keep=0.78, feather=0.08, strength=1.0")
print("Use --crop_preprocess current to reproduce v2 behavior.")
