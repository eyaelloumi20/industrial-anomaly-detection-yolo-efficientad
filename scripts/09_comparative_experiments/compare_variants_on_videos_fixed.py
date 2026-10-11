"""
compare_variants_on_videos_fixed.py
===================================
Evaluate every trained EfficientAD variant on the held-out videos using the
optimized video detector.

Improvements over the previous runner:
- Uses video_streaming_detector_optimized1.py when present.
- Keeps FP16 enabled; the detector now owns dtype-safe scoring.
- Supports resume: completed run summaries can be reused instead of rerunning
  expensive YOLO + EfficientAD inference.
- More useful subprocess error output (stdout + stderr tail).
- Saves elapsed wall-clock time as well as detector processing FPS.
- Keeps runs sequential on one GPU to avoid VRAM contention and invalid timing.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/ahmed/two-stage-efficientad-inspection")

# Prefer the script containing the dtype-safe optimized scorer.
_detector_candidates = [
    ROOT / "video_streaming_detector_optimized1.py",
    ROOT / "video_streaming_detector_optimized.py",
]
DETECTOR = next((p for p in _detector_candidates if p.is_file()), _detector_candidates[0])

YOLO = ROOT / "runs/detect/yolo_runs/chewinggum_yolov8n_v2/weights/best.pt"
THRESHOLDS = ROOT / "video_calibration_new/video_thresholds_baseline.json"
VIDEO_DIR = ROOT / "video_test"
OUT_ROOT = ROOT / "variant_runs"

VARIANTS = {
    "baseline":   "ckpt_cropped_multi",
    "lowres_192": "ckpt_lowres_192",
    "lowres_224": "ckpt_lowres_224",
    "thin_075":   "ckpt_thin_075",
    "thin_050":   "ckpt_thin_050",
    "pruned_030": "ckpt_pruned_030",
}

GROUND_TRUTH = {
    "video_A":  {1, 4, 12, 18, 34, 36},
    "video_A3": {6, 8, 18, 27, 34, 50},
    "video_A4": {2, 15, 18, 30, 49, 55},
}

TAU_TRACK = 0.45

# Performance-oriented detector settings.
DEVICE = "cuda:0"
BATCH_FRAMES = 16
BATCH_CROPS = 96
CROP_WORKERS = 12
USE_FP16 = True

# If True, reuse the latest valid summary.json for a pair that has already run.
RESUME = True


def latest_summary(save_dir: Path):
    """Return (summary_path, summary_dict) for the latest complete run, else None."""
    run_dirs = sorted(
        save_dir.glob("run_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for run_dir in run_dirs:
        summary_path = run_dir / "summary.json"
        if not summary_path.is_file():
            continue
        try:
            data = json.loads(summary_path.read_text())
        except Exception:
            continue
        # Only reuse finalized runs.
        if data.get("reason") in ("end_of_video", "max_frames") and "fired_track_ids" in data:
            return summary_path, data
    return None


def run_one(variant, ckpt_rel, video_name):
    ckpt_abs = ROOT / ckpt_rel
    if not ckpt_abs.is_dir():
        print(f"    [skip] {variant}: {ckpt_abs} missing")
        return None

    video_path = VIDEO_DIR / f"{video_name}.mp4"
    if not video_path.is_file():
        print(f"    [skip] {variant}/{video_name}: {video_path} missing")
        return None

    if not DETECTOR.is_file():
        raise FileNotFoundError(
            "No optimized detector found. Expected one of:\n  "
            + "\n  ".join(str(p) for p in _detector_candidates)
        )

    save_dir = OUT_ROOT / variant / video_name
    save_dir.mkdir(parents=True, exist_ok=True)

    if RESUME:
        previous = latest_summary(save_dir)
        if previous is not None:
            summary_path, summary = previous
            print(f" reused {summary_path.parent.name}", end="")
            summary["_reused"] = True
            summary["_elapsed_s"] = 0.0
            return summary

    cmd = [
        sys.executable,
        "-u",
        str(DETECTOR),
        "--video", str(video_path),
        "--yolo_det_weights", str(YOLO),
        "--ckpt_dir", str(ckpt_abs),
        "--thresholds_json", str(THRESHOLDS),
        "--category", "chewinggum",
        "--model_size", "S",
        "--tau_track", str(TAU_TRACK),
        "--inactive_frames", "30",
        "--min_samples", "10",
        "--clip_margin", "5",
        "--min_box_frac", "0.5",
        "--batch_frames", str(BATCH_FRAMES),
        "--batch_crops", str(BATCH_CROPS),
        "--crop_workers", str(CROP_WORKERS),
        "--device", DEVICE,
        "--no_video",
        "--log_every", "0",
        "--save_dir", str(save_dir),
    ]
    if USE_FP16:
        cmd.append("--fp16")

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(ROOT),
    )
    dt = time.perf_counter() - t0

    if proc.returncode != 0:
        print(f" FAILED (exit {proc.returncode}, {dt:.1f}s)")
        combined = []
        if proc.stdout.strip():
            combined.extend(proc.stdout.strip().splitlines())
        if proc.stderr.strip():
            combined.extend(proc.stderr.strip().splitlines())
        for line in combined[-12:]:
            print(f"      | {line}")
        return None

    completed = latest_summary(save_dir)
    if completed is None:
        print(" FAILED (detector exited 0 but no complete summary.json)")
        return None

    summary_path, summary = completed
    summary["_elapsed_s"] = dt
    summary["_reused"] = False
    return summary


def evaluate(fired_ids, truth):
    fired = set(fired_ids)
    tp = len(fired & truth)
    fp = len(fired - truth)
    fn = len(truth - fired)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fired_ids": sorted(fired),
        "false_positives": sorted(fired - truth),
        "false_negatives": sorted(truth - fired),
    }


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Detector: {DETECTOR}")
    print(
        f"Runtime : device={DEVICE}, fp16={USE_FP16}, "
        f"batch_frames={BATCH_FRAMES}, batch_crops={BATCH_CROPS}, "
        f"crop_workers={CROP_WORKERS}, resume={RESUME}"
    )

    results = {}

    # Sequential execution is deliberate: multiple detector processes on the
    # same GPU would contend for VRAM/compute and make FPS comparisons unfair.
    for variant, ckpt_rel in VARIANTS.items():
        print(f"\n=== {variant} ===")
        per_video = {}

        for video_name, truth in GROUND_TRUTH.items():
            print(f"  {variant} / {video_name} ...", end="", flush=True)
            summary = run_one(variant, ckpt_rel, video_name)

            if summary is None:
                print(" SKIP")
                per_video[video_name] = None
                continue

            metrics = evaluate(summary["fired_track_ids"], truth)
            metrics["processing_fps"] = float(summary.get("processing_fps", 0.0))
            metrics["n_frames"] = int(summary.get("n_frames", 0))
            metrics["elapsed_s"] = float(summary.get("_elapsed_s", 0.0))
            metrics["reused"] = bool(summary.get("_reused", False))
            per_video[video_name] = metrics

            reuse_tag = " [cached]" if metrics["reused"] else ""
            print(
                f" fired={metrics['fired_ids']} "
                f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} "
                f"({metrics['processing_fps']:.1f} fps){reuse_tag}"
            )

        agg = {"tp": 0, "fp": 0, "fn": 0}
        fps_list = []

        for data in per_video.values():
            if data is None:
                continue
            agg["tp"] += data["tp"]
            agg["fp"] += data["fp"]
            agg["fn"] += data["fn"]
            if data["processing_fps"] > 0:
                fps_list.append(data["processing_fps"])

        precision = agg["tp"] / max(agg["tp"] + agg["fp"], 1)
        recall = agg["tp"] / max(agg["tp"] + agg["fn"], 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        mean_fps = sum(fps_list) / len(fps_list) if fps_list else 0.0

        results[variant] = {
            "per_video": per_video,
            "agg": agg,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_fps": mean_fps,
        }

    print("\n" + "=" * 118)
    print(
        f"{'variant':<14} {'TP':>4} {'FP':>4} {'FN':>4} "
        f"{'Prec':>7} {'Recall':>7} {'F1':>7} {'fps':>7}   per-video fired IDs"
    )
    print("=" * 118)

    for variant, result in results.items():
        per_v = "  ".join(
            (
                f"{video_name[6:]}:{result['per_video'][video_name]['fired_ids']}"
                if result["per_video"][video_name] is not None
                else f"{video_name[6:]}:NA"
            )
            for video_name in GROUND_TRUTH
        )

        print(
            f"{variant:<14} {result['agg']['tp']:>4} {result['agg']['fp']:>4} "
            f"{result['agg']['fn']:>4} {result['precision']:>7.3f} "
            f"{result['recall']:>7.3f} {result['f1']:>7.3f} "
            f"{result['mean_fps']:>7.1f}   {per_v}"
        )

    print("=" * 118)

    output = OUT_ROOT / "comparison.json"
    output.write_text(json.dumps(results, indent=2))
    print(f"\nFull results saved to {output}")

    # Keep the existing project criterion so results remain comparable.
    candidates = [
        (name, result)
        for name, result in results.items()
        if result["f1"] >= 0.72 and name != "baseline"
    ]
    if candidates:
        best = max(candidates, key=lambda item: item[1]["mean_fps"])
        print(
            f"\n>>> FASTEST VIABLE VARIANT: {best[0]} "
            f"(F1={best[1]['f1']:.3f}, {best[1]['mean_fps']:.1f} fps)"
        )
    else:
        print("\n>>> No non-baseline variant reached the F1=0.72 criterion.")


if __name__ == "__main__":
    main()
