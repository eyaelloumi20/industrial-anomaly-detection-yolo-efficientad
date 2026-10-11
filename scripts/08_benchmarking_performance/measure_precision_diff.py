"""
measure_precision_diff.py
==========================
STEP 3 DIAGNOSTIC INSTRUMENTATION — purely additive, read-only.

Measures how much the EfficientAD anomaly score for the SAME crop differs
between:
  - FP32 forward pass  (the precision evaluate_pipeline.py / pipeline.py /
    efficientad_inference.py currently use)
  - FP16 forward pass  (the precision calibrate_threshold_all_fp16.py
    ALWAYS uses under CUDA autocast)

This script does NOT modify, re-implement, or re-derive the EfficientAD
scoring math, YOLO detection, or ByteTrack tracking. It imports and calls
the actual functions already living in your project:

    load_efficientad.py            -> load_efficientad_model()   (model loading)
    efficientad_inference.py       -> run_inference()             (FP32 scoring path)
    calibrate_threshold_all_fp16.py-> get_crop(), score_crop_batch() (crop + FP16 scoring path)
    detection.py                   -> Detector, full_image_box()  (YOLO detection)

It does NOT touch pipeline.py, the EfficientAD architecture, or the
model-compression variants (lowres/thin/qat/pruned). It only reads your
checkpoint plus your images/videos, and writes new files into --out_dir.

Because the four source files above may live in different locations in
your repo, each is located by an explicit file path you pass on the
command line rather than assumed via package imports.

TWO INPUT MODES (mutually exclusive)
-------------------------------------
1. --input_dir  : a flat folder of already-extracted images (pre-cropped
   objects, or full frames if --yolo_weights is also given). Unchanged
   from the previous version of this script.

2. --video_dir  : a folder of CONDITION SUBFOLDERS, each containing video
   files, e.g.:
       precision_test/
       ├── original/
       │   ├── video_01.mp4
       │   └── video_02.mp4
       └── video/
           ├── video_01.mp4
           └── video_02.mp4
   Every video is decoded frame-by-frame (subsampled by --frame_stride),
   each sampled frame is run through the existing YOLO Detector, and every
   detected box is cropped via get_crop() -- IDENTICAL to how
   calibrate_threshold_all_fp16.py itself turns a video into scored crops.
   --yolo_weights is REQUIRED in this mode (frames must be detected to
   obtain object crops).

WHERE PREPROCESSING HAPPENS (both modes)
------------------------------------------
For every box, get_crop() (imported unmodified from
calibrate_threshold_all_fp16.py) runs EXACTLY ONCE, producing one PIL
crop. That same crop object is then passed to BOTH the FP32 branch
(run_inference) and the FP16 branch (score_crop_batch) -- so detection and
cropping variability can never be mistaken for FP32-vs-FP16 error; only
forward-pass precision differs between the two scores recorded for that
crop.

Outputs, all written under --out_dir:
    precision_diff_per_crop.csv   -- one row per crop: condition, video,
                                      frame_idx, box, fp32, fp16, abs/rel diff
    precision_diff_summary.json   -- aggregate stats: overall, per-condition,
                                      per-video
    precision_diff_hist_abs_overall.png / precision_diff_hist_rel_overall.png
    precision_diff_hist_abs_<condition>.png / _rel_<condition>.png
                                   -- one pair per condition (video mode only)

IMPORTANT CAVEAT
-----------------
score_crop_batch() only runs its forward pass under real fp16 autocast when
device.startswith("cuda"). On CPU it silently falls back to fp32 -- so on
CPU this script will report near-zero differences that are NOT
representative of your real GPU deployment. Run this on your CUDA machine.

Usage — video mode (recommended for this experiment)
-------------------------------------------------------
    python measure_precision_diff.py \\
        --ckpt_dir /path/to/baseline_ckpt_dir \\
        --category chewinggum --model_size S --device cuda \\
        --video_dir ./precision_test \\
        --yolo_weights /path/to/best.pt \\
        --frame_stride 5 \\
        --load_efficientad_path       /path/to/load_efficientad.py \\
        --efficientad_inference_path  /path/to/efficientad_inference.py \\
        --detection_path              /path/to/detection.py \\
        --calibrate_script_path       /path/to/calibrate_threshold_all_fp16.py \\
        --out_dir ./results_precision_diff

Usage — image mode (unchanged from before)
----------------------------------------------
    python measure_precision_diff.py \\
        --ckpt_dir /path/to/baseline_ckpt_dir \\
        --category chewinggum --model_size S --device cuda \\
        --input_dir /path/to/reference_crops_or_frames \\
        --load_efficientad_path       /path/to/load_efficientad.py \\
        --efficientad_inference_path  /path/to/efficientad_inference.py \\
        --detection_path              /path/to/detection.py \\
        --calibrate_script_path       /path/to/calibrate_threshold_all_fp16.py \\
        --out_dir ./results_precision_diff
    # add --yolo_weights if --input_dir holds full frames rather than pre-cropped tiles
"""

import os
import sys
import glob
import json
import csv
import argparse
import importlib.util

import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}


# ── dynamic, path-based imports (no assumption about package layout) ─────────
#
# The files we reuse have their OWN internal imports that assume a certain
# repo layout, e.g.:
#   utils/load_efficientad.py   does  "from models import Teacher, ..."
#                                (a flat sibling import -- needs utils/ on sys.path)
#   calibrate_threshold_all_fp16.py (repo root)  does
#                                "from utils.load_efficientad_adaptive import ..."
#                                (a package-qualified import -- needs repo ROOT on sys.path)
# Rather than guess which pattern a given file uses, we add BOTH the file's
# own directory and its parent directory to sys.path before importing it.
# This is additive only (sys.path.insert), never removes or reorders
# anything already on sys.path, and touches no existing project file.

def _add_to_syspath(directory: str):
    if directory and directory not in sys.path:
        sys.path.insert(0, directory)


def _load_module(name: str, path: str):
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            f"Could not find '{name}' source file at: {path!r}\n"
            f"Pass the correct location with the corresponding --*_path argument."
        )
    own_dir = os.path.dirname(os.path.abspath(path))
    parent_dir = os.path.dirname(own_dir)
    _add_to_syspath(own_dir)
    _add_to_syspath(parent_dir)

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except ImportError as e:
        raise ImportError(
            f"Failed to import '{name}' from {path}: {e}\n"
            f"This usually means one of its OWN internal imports needs a directory "
            f"that isn't on sys.path yet (e.g. a sibling module, or a package one "
            f"level up). Added to sys.path for this file: {own_dir!r} and {parent_dir!r}. "
            f"If the real dependency lives elsewhere, that's a repo-layout detail this "
            f"script can't guess -- check the traceback above for the missing module name."
        ) from e
    return module


# ── args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Measure FP32 vs FP16 EfficientAD score difference on identical crops. "
                    "Purely additive / read-only diagnostic -- does not modify existing code.")

    p.add_argument("--ckpt_dir",   required=True, help="Baseline EfficientAD checkpoint directory")
    p.add_argument("--category",   default="chewinggum")
    p.add_argument("--model_size", default="S", choices=["S", "M"])
    p.add_argument("--device",     default="cuda",
                   help="cuda strongly recommended -- see CPU caveat in the module docstring")
    p.add_argument("--ratio",      type=float, default=0.1,
                   help="EfficientAD normalisation ratio (default 0.1, matches production)")

    # ── input mode 1: flat folder of images ───────────────────────────────────
    p.add_argument("--input_dir",  default=None,
                   help="Directory of images: either pre-cropped tiles, or full frames "
                        "(if --yolo_weights is also given). Mutually exclusive with --video_dir.")
    p.add_argument("--max_images", type=int, default=0, help="0 = no cap (--input_dir mode)")

    # ── input mode 2: condition subfolders of videos ──────────────────────────
    p.add_argument("--video_dir",  default=None,
                   help="Folder of condition subfolders, each containing video files, e.g. "
                        "precision_test/original/*.mp4, precision_test/video/*.mp4. "
                        "Mutually exclusive with --input_dir. Requires --yolo_weights.")
    p.add_argument("--frame_stride", type=int, default=1,
                   help="Sample every Nth frame from each video (default 1 = every frame).")
    p.add_argument("--max_frames_per_video", type=int, default=0,
                   help="Cap sampled frames PER VIDEO after striding (0 = no cap).")

    p.add_argument("--yolo_weights", default=None,
                   help="Path to YOLO .pt weights. REQUIRED for --video_dir mode. Optional for "
                        "--input_dir mode (omit it there to treat each image as one full-image box).")
    p.add_argument("--conf",       type=float, default=0.25)
    p.add_argument("--iou",        type=float, default=0.45)
    p.add_argument("--pad_ratio",  type=float, default=0.02,
                   help="Context padding around each box (default 0.02, matches "
                        "evaluate_pipeline.py / calibrate_threshold_all_fp16.py current defaults)")
    p.add_argument("--denoise",    action="store_true", default=True,
                   help="Apply the existing Gaussian+bilateral denoise in get_crop (default on, "
                        "matches production default)")
    p.add_argument("--no_denoise", dest="denoise", action="store_false")
    p.add_argument("--target_size", type=int, default=256)

    p.add_argument("--load_efficientad_path",       required=True,
                   help="Path to your load_efficientad.py")
    p.add_argument("--efficientad_inference_path",  required=True,
                   help="Path to your efficientad_inference.py")
    p.add_argument("--detection_path",               required=True,
                   help="Path to your detection.py")
    p.add_argument("--calibrate_script_path",        required=True,
                   help="Path to your calibrate_threshold_all_fp16.py "
                        "(get_crop() and score_crop_batch() are reused from here)")

    p.add_argument("--out_dir", default="./results_precision_diff")
    return p.parse_args()


# ── input gathering ────────────────────────────────────────────────────────────

def gather_images(input_dir, max_images):
    paths = sorted(
        p for p in glob.glob(os.path.join(input_dir, "**", "*"), recursive=True)
        if os.path.splitext(p)[1].lower() in IMAGE_EXTS
    )
    if not paths:
        raise SystemExit(f"No images found under --input_dir={input_dir} "
                         f"(looked for extensions: {sorted(IMAGE_EXTS)})")
    if max_images:
        paths = paths[:max_images]
    return paths


def gather_videos(video_dir):
    """Expects video_dir/<condition>/<video files>. Returns [(condition, video_path), ...]."""
    if not os.path.isdir(video_dir):
        raise SystemExit(f"--video_dir not found or not a directory: {video_dir}")
    conditions = sorted(
        d for d in os.listdir(video_dir)
        if os.path.isdir(os.path.join(video_dir, d))
    )
    if not conditions:
        raise SystemExit(
            f"No condition subfolders found under --video_dir={video_dir}. "
            f"Expected layout: {video_dir}/<condition_name>/<video files>"
        )
    items = []
    for cond in conditions:
        cond_dir = os.path.join(video_dir, cond)
        vids = sorted(
            p for p in glob.glob(os.path.join(cond_dir, "**", "*"), recursive=True)
            if os.path.splitext(p)[1].lower() in VIDEO_EXTS
        )
        for v in vids:
            items.append((cond, v))
    if not items:
        raise SystemExit(f"No video files found under --video_dir={video_dir} "
                         f"(looked for extensions: {sorted(VIDEO_EXTS)})")
    return items


def iter_video_frames(video_path, frame_stride, max_frames_per_video):
    """Decode video_path frame by frame, yielding (frame_idx, pil_frame) every
    frame_stride-th frame. frame_idx is the TRUE index in the source video
    (not the sampled index), so it stays traceable back to the original file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frame_idx = 0
    yielded = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_stride == 0:
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            yield frame_idx, pil_frame
            yielded += 1
            if max_frames_per_video and yielded >= max_frames_per_video:
                break
        frame_idx += 1
    cap.release()


# ── core scoring: detect once, crop once, score BOTH precisions on that crop ──

def score_image_boxes(pil_img, detector, detection_mod, calibrate_mod,
                      efficientad_inference_mod, models, device, args, to_tensor):
    """Detect (or full-image-box), crop ONCE per box, then score that SAME
    crop under FP32 and FP16. Returns a list of dicts (no image/video/
    condition identifying fields -- callers add those)."""
    if detector is not None:
        boxes = detector.detect(pil_img)
    else:
        boxes = detection_mod.full_image_box(pil_img)

    out = []
    for box_idx, box in enumerate(boxes):
        # Preprocessing happens EXACTLY ONCE here. Both branches below receive
        # this identical crop_pil object -- detection/crop variability cannot
        # leak into the FP32-vs-FP16 comparison.
        crop_pil = calibrate_mod.get_crop(
            pil_img, box, args.target_size, args.pad_ratio, args.denoise)

        # FP32 branch: reuse efficientad_inference.run_inference() unmodified
        crop_t = to_tensor(crop_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            _, fp32_scores = efficientad_inference_mod.run_inference(
                crop_t, models, ratio=args.ratio)
        fp32_score = float(fp32_scores[0].cpu())

        # FP16 branch: reuse calibrate_threshold_all_fp16.score_crop_batch() unmodified
        fp16_scores = calibrate_mod.score_crop_batch(
            models, [crop_pil], device, args.target_size, ratio=args.ratio)
        fp16_score = float(fp16_scores[0])

        abs_diff = abs(fp32_score - fp16_score)
        rel_diff = abs_diff / (abs(fp32_score) + 1e-8)

        out.append({
            "box_idx": box_idx, "box": box,
            "fp32_score": fp32_score, "fp16_score": fp16_score,
            "abs_diff": abs_diff, "rel_diff": rel_diff,
        })
    return out


def percentile(arr, q):
    return float(np.percentile(arr, q)) if len(arr) else float("nan")


def compute_stats(rows):
    abs_arr = np.array([r["abs_diff"] for r in rows])
    rel_arr = np.array([r["rel_diff"] for r in rows])
    return {
        "n_crops": len(rows),
        "abs_diff": {
            "mean": float(abs_arr.mean()), "median": float(np.median(abs_arr)),
            "std": float(abs_arr.std()), "max": float(abs_arr.max()),
            "p90": percentile(abs_arr, 90), "p95": percentile(abs_arr, 95),
            "p99": percentile(abs_arr, 99),
        },
        "rel_diff": {
            "mean": float(rel_arr.mean()), "median": float(np.median(rel_arr)),
            "std": float(rel_arr.std()), "max": float(rel_arr.max()),
            "p90": percentile(rel_arr, 90), "p95": percentile(rel_arr, 95),
            "p99": percentile(rel_arr, 99),
        },
    }


def save_histograms(rows, out_dir, label_suffix, device):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    abs_arr = np.array([r["abs_diff"] for r in rows])
    rel_arr = np.array([r["rel_diff"] for r in rows])
    for arr, axis_label, kind in [
        (abs_arr, "Absolute difference (FP32 - FP16 score)", "abs"),
        (rel_arr, "Relative difference (|FP32-FP16| / |FP32|)", "rel"),
    ]:
        plt.figure(figsize=(7, 4))
        plt.hist(arr, bins=50)
        plt.xlabel(axis_label)
        plt.ylabel("Count (crops)")
        plt.title(f"{label_suffix}  |  n={len(arr)} crops  |  device={device}")
        plt.tight_layout()
        out_png = os.path.join(out_dir, f"precision_diff_hist_{kind}_{label_suffix}.png")
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"[Precision Diff] Histogram -> {out_png}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if bool(args.input_dir) == bool(args.video_dir):
        raise SystemExit("Pass exactly one of --input_dir or --video_dir (not both, not neither).")
    if args.video_dir and not args.yolo_weights:
        raise SystemExit("--video_dir requires --yolo_weights (frames must be detected to obtain crops).")

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARNING] --device=cuda requested but CUDA is not available here. "
              "Falling back to cpu -- FP16 autocast will be a no-op, so the measured "
              "differences will NOT be representative of your GPU deployment.")
        device = "cpu"
    elif device == "cpu":
        print("[WARNING] Running on CPU: score_crop_batch()'s fp16 autocast branch only "
              "activates for device.startswith('cuda'). On CPU it silently runs at fp32, "
              "so this run will measure ~0 difference by construction -- NOT a real "
              "precision comparison. Re-run on your CUDA machine for a meaningful result.")

    print(f"\n{'='*70}\n  FP32 vs FP16 EfficientAD score comparison  (Step 3 diagnostic)\n{'='*70}")
    print(f"  ckpt_dir   : {args.ckpt_dir}")
    print(f"  category   : {args.category}   model_size: {args.model_size}")
    print(f"  device     : {device}")
    if args.video_dir:
        print(f"  mode       : VIDEO  (--video_dir={args.video_dir}, frame_stride={args.frame_stride})")
    else:
        print(f"  mode       : IMAGE  (--input_dir={args.input_dir})")
    print(f"  yolo_weights: {args.yolo_weights or '(none -- full-image box per image)'}")
    print(f"  pad_ratio  : {args.pad_ratio}   denoise: {args.denoise}")
    print(f"{'='*70}\n")

    # ── locate and import the real, existing project code ────────────────────
    load_efficientad_mod = _load_module("load_efficientad_reused", args.load_efficientad_path)
    efficientad_inference_mod = _load_module("efficientad_inference_reused", args.efficientad_inference_path)
    detection_mod = _load_module("detection_reused", args.detection_path)
    calibrate_mod = _load_module("calibrate_threshold_all_fp16_reused", args.calibrate_script_path)

    # ── load the model bundle ONCE, exactly as evaluate_pipeline.py does ──────
    models = load_efficientad_mod.load_efficientad_model(
        ckpt_dir=args.ckpt_dir, category=args.category,
        model_size=args.model_size, device=device,
    )

    # ── YOLO detector, exactly as detection.py already implements ────────────
    detector = None
    if args.yolo_weights:
        detector = detection_mod.Detector(
            model_path=args.yolo_weights, conf_threshold=args.conf,
            iou_threshold=args.iou, device=device,
        )

    to_tensor = transforms.ToTensor()
    rows = []

    if args.video_dir:
        video_items = gather_videos(args.video_dir)
        conditions = sorted(set(c for c, _ in video_items))
        print(f"[Precision Diff] {len(video_items)} video(s) across "
              f"{len(conditions)} condition folder(s): {conditions}\n")

        for cond, video_path in video_items:
            video_name = os.path.basename(video_path)
            print(f"  [{cond}] {video_name} -- decoding (stride={args.frame_stride}) ...")
            n_frames_used = 0
            n_crops_this_video = 0
            for frame_idx, pil_frame in iter_video_frames(
                    video_path, args.frame_stride, args.max_frames_per_video):
                box_results = score_image_boxes(
                    pil_frame, detector, detection_mod, calibrate_mod,
                    efficientad_inference_mod, models, device, args, to_tensor)
                for r in box_results:
                    r["condition"] = cond
                    r["video"] = video_name
                    r["frame_idx"] = frame_idx
                    r["image"] = f"{video_name}#frame{frame_idx}"
                    rows.append(r)
                n_frames_used += 1
                n_crops_this_video += len(box_results)
            print(f"  [{cond}] {video_name} -- {n_frames_used} frame(s) sampled, "
                  f"{n_crops_this_video} crop(s) scored")
    else:
        image_paths = gather_images(args.input_dir, args.max_images)
        print(f"[Precision Diff] {len(image_paths)} source image(s) to process.\n")
        for idx, img_path in enumerate(image_paths, 1):
            pil_img = Image.open(img_path).convert("RGB")
            box_results = score_image_boxes(
                pil_img, detector, detection_mod, calibrate_mod,
                efficientad_inference_mod, models, device, args, to_tensor)
            for r in box_results:
                r["condition"] = ""
                r["video"] = ""
                r["frame_idx"] = ""
                r["image"] = os.path.basename(img_path)
                rows.append(r)
            if idx % 20 == 0 or idx == len(image_paths):
                print(f"  processed {idx}/{len(image_paths)} images "
                      f"({len(rows)} crops scored so far)")

    if not rows:
        raise SystemExit("No crops were scored -- check --input_dir/--video_dir and --yolo_weights.")

    # ── write per-crop CSV (every row traceable to condition/video/frame) ─────
    csv_path = os.path.join(args.out_dir, "precision_diff_per_crop.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "condition", "video", "frame_idx", "image", "box_idx", "box",
            "fp32_score", "fp16_score", "abs_diff", "rel_diff"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\n[Precision Diff] Per-crop results -> {csv_path}")

    # ── aggregate statistics: overall + per-condition + per-video ─────────────
    summary = {"overall": compute_stats(rows)}

    conditions = sorted(set(r["condition"] for r in rows if r["condition"]))
    if conditions:
        summary["by_condition"] = {
            cond: compute_stats([r for r in rows if r["condition"] == cond])
            for cond in conditions
        }
        videos = sorted(set((r["condition"], r["video"]) for r in rows if r["video"]))
        summary["by_video"] = {
            f"{cond}/{vid}": compute_stats(
                [r for r in rows if r["condition"] == cond and r["video"] == vid])
            for cond, vid in videos
        }

    json_path = os.path.join(args.out_dir, "precision_diff_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Precision Diff] Summary statistics  -> {json_path}")

    # ── histograms: overall, plus one pair per condition if in video mode ─────
    save_histograms(rows, args.out_dir, "overall", device)
    for cond in conditions:
        save_histograms([r for r in rows if r["condition"] == cond], args.out_dir, cond, device)

    # ── print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}\n  Summary  ({len(rows)} crops total, device={device})\n{'='*70}")
    ov = summary["overall"]
    print(f"  OVERALL       abs: mean={ov['abs_diff']['mean']:.6f} median={ov['abs_diff']['median']:.6f} "
          f"std={ov['abs_diff']['std']:.6f} max={ov['abs_diff']['max']:.6f}")
    print(f"                rel: mean={ov['rel_diff']['mean']:.6%} median={ov['rel_diff']['median']:.6%} "
          f"std={ov['rel_diff']['std']:.6%} max={ov['rel_diff']['max']:.6%}")
    for cond in conditions:
        s = summary["by_condition"][cond]
        print(f"  [{cond:<12s}] n={s['n_crops']:<6d} "
              f"abs mean={s['abs_diff']['mean']:.6f}  rel mean={s['rel_diff']['mean']:.6%}")
    if device == "cpu" or not torch.cuda.is_available():
        print(f"\n  NOTE: this run did not exercise real FP16 autocast (see CPU caveat "
              f"above). Re-run on CUDA before drawing any conclusion from these numbers.")
    print(f"{'='*70}\n")
    print("This script only reports measured differences -- interpretation "
          "(whether this magnitude is negligible or significant enough to standardize "
          "precision before later steps, and whether it grows under domain shift) is "
          "left for you to decide against these numbers.")


if __name__ == "__main__":
    main()