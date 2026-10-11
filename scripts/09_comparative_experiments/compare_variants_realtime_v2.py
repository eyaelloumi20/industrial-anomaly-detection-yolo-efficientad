"""
compare_variants_realtime.py
============================
Variant comparison runner for the optimized real-time detector.

Important:
1. Speed profile uses YOLO FP16 and the optimized EfficientAD scorer.
2. By default YOLO stays at 640 to preserve tracking comparability.
3. Set YOLO_IMGSZ=512 only after revalidating track IDs / detection quality.
4. Each EfficientAD variant SHOULD have its own calibrated thresholds JSON.
   Reusing baseline thresholds for lowres/thin/pruned gives unfair F1 results.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/ahmed/two-stage-efficientad-inspection")
DETECTOR = ROOT / "video_streaming_detector_optimized3.py"
YOLO = ROOT / "runs/detect/yolo_runs/chewinggum_yolov8n_v2/weights/best.pt"
VIDEO_DIR = ROOT / "video_test"
OUT_ROOT = ROOT / "variant_runs_realtime"

VARIANTS = {
    "baseline":   "ckpt_cropped_multi",
    "lowres_192": "ckpt_lowres_192",
    "lowres_224": "ckpt_lowres_224",
    "thin_075":   "ckpt_thin_075",
    "thin_050":   "ckpt_thin_050",
    "pruned_030": "ckpt_pruned_030",
}

# Prefer variant-specific calibration. Files that do not yet exist will fall
# back to baseline ONLY for a temporary speed run, with a loud warning.
THRESHOLDS = {
    "baseline":   ROOT / "video_calibration_new/video_thresholds_baseline.json",
    "lowres_192": ROOT / "video_calibration_new/video_thresholds_lowres_192.json",
    "lowres_224": ROOT / "video_calibration_new/video_thresholds_lowres_224.json",
    "thin_075":   ROOT / "video_calibration_new/video_thresholds_thin_075.json",
    "thin_050":   ROOT / "video_calibration_new/video_thresholds_thin_050.json",
    "pruned_030": ROOT / "video_calibration_new/video_thresholds_pruned_030.json",
}
BASELINE_THRESHOLDS = ROOT / "video_calibration_new/video_thresholds_baseline.json"

# Per-variant track thresholds should also be calibrated. Until you have them,
# these defaults retain your existing 0.45 behavior.
TAU_TRACK = {
    "baseline": 0.45,
    "lowres_192": 0.45,
    "lowres_224": 0.45,
    "thin_075": 0.45,
    "thin_050": 0.45,
    "pruned_030": 0.45,
}

GROUND_TRUTH = {
    "video_A":  {1, 4, 12, 18, 34, 36},
    "video_A3": {6, 8, 18, 27, 34, 50},
    "video_A4": {2, 15, 18, 30, 49, 55},
}

# -------- real-time speed profile ------------------------------------------
DEVICE = "cuda:0"
YOLO_FP16 = True
YOLO_IMGSZ = 640       # Try 512 later; must revalidate tracking/IDs.
YOLO_MAX_DET = 32      # Gum line should not need huge max_det.
COMPILE_EFFICIENTAD = False
NATIVE_SCORE = False   # True is faster but requires FULL threshold recalibration.

# Throughput/latency balance. 16/96 maximizes throughput but increases delay.
# 8/48 is a better starting point for an interactive real-time stream.
BATCH_FRAMES = 8
BATCH_CROPS = 48
CROP_WORKERS = 8

RESUME = False  # False while comparing new speed settings.


def evaluate(fired_ids, truth):
    fired = set(fired_ids)
    tp = len(fired & truth)
    fp = len(fired - truth)
    fn = len(truth - fired)
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": p, "recall": r, "f1": f1,
        "fired_ids": sorted(fired),
        "false_positives": sorted(fired - truth),
        "false_negatives": sorted(truth - fired),
    }


def latest_summary(save_dir):
    runs = sorted(save_dir.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for d in runs:
        f = d / "summary.json"
        if f.is_file():
            try:
                data = json.loads(f.read_text())
            except Exception:
                continue
            if data.get("reason") in ("end_of_video", "max_frames"):
                return f, data
    return None


def threshold_for(variant):
    specific = THRESHOLDS[variant]
    if specific.is_file():
        return specific, True
    print(
        f"\n    [CALIBRATION WARNING] {specific.name} is missing. "
        f"Using BASELINE thresholds temporarily. FPS is usable, "
        f"but F1 for {variant} is NOT a fair calibrated result."
    )
    return BASELINE_THRESHOLDS, variant == "baseline"


def run_one(variant, ckpt_rel, video_name):
    ckpt = ROOT / ckpt_rel
    video = VIDEO_DIR / f"{video_name}.mp4"

    if not ckpt.is_dir() or not video.is_file():
        return None

    save_dir = OUT_ROOT / variant / video_name
    save_dir.mkdir(parents=True, exist_ok=True)

    if RESUME:
        prev = latest_summary(save_dir)
        if prev:
            _, data = prev
            data["_reused"] = True
            return data

    thresholds_json, calibrated = threshold_for(variant)

    cmd = [
        sys.executable, "-u", str(DETECTOR),
        "--video", str(video),
        "--yolo_det_weights", str(YOLO),
        "--ckpt_dir", str(ckpt),
        "--thresholds_json", str(thresholds_json),
        "--category", "chewinggum",
        "--model_size", "S",
        "--device", DEVICE,
        "--tau_track", str(TAU_TRACK[variant]),
        "--inactive_frames", "30",
        "--min_samples", "10",
        "--clip_margin", "5",
        "--min_box_frac", "0.5",
        "--batch_frames", str(BATCH_FRAMES),
        "--batch_crops", str(BATCH_CROPS),
        "--crop_workers", str(CROP_WORKERS),
        "--yolo_imgsz", str(YOLO_IMGSZ),
        "--yolo_max_det", str(YOLO_MAX_DET),
        "--fp16",
        "--yolo_fp16",
        "--no_video",
        "--log_every", "0",
        "--save_dir", str(save_dir),
    ]

    if COMPILE_EFFICIENTAD:
        cmd.append("--compile_efficientad")
    if NATIVE_SCORE:
        cmd.append("--native_score")

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(ROOT),
    )
    elapsed = time.perf_counter() - t0

    if proc.returncode != 0:
        print(f" FAILED ({elapsed:.1f}s)")
        lines = proc.stdout.splitlines() + proc.stderr.splitlines()
        for line in lines[-18:]:
            print(f"      | {line}")
        return None

    found = latest_summary(save_dir)
    if not found:
        print(" FAILED: no summary")
        return None

    _, summary = found
    summary["_elapsed_s"] = elapsed
    summary["_calibrated_variant_thresholds"] = calibrated
    return summary


def main():
    if not DETECTOR.is_file():
        raise FileNotFoundError(f"Copy optimized detector to {DETECTOR}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(
        f"Detector={DETECTOR.name} | YOLO fp16={YOLO_FP16} imgsz={YOLO_IMGSZ} "
        f"| EA compile={COMPILE_EFFICIENTAD} native={NATIVE_SCORE} "
        f"| chunk={BATCH_FRAMES}f/{BATCH_CROPS}c"
    )

    results = {}

    for variant, ckpt_rel in VARIANTS.items():
        print(f"\n=== {variant} ===")
        per_video = {}

        for video_name, truth in GROUND_TRUTH.items():
            print(f"  {video_name} ...", end="", flush=True)
            summary = run_one(variant, ckpt_rel, video_name)
            if summary is None:
                print(" SKIP")
                per_video[video_name] = None
                continue

            m = evaluate(summary["fired_track_ids"], truth)
            m["processing_fps"] = float(summary.get("processing_fps", 0))
            m["calibrated"] = bool(summary.get("_calibrated_variant_thresholds", False))
            per_video[video_name] = m

            cal = "" if m["calibrated"] else " [UNCALIBRATED]"
            print(
                f" {m['processing_fps']:.1f} fps | fired={m['fired_ids']} "
                f"| F1={m['f1']:.3f}{cal}"
            )

        valid = [v for v in per_video.values() if v is not None]
        tp = sum(v["tp"] for v in valid)
        fp = sum(v["fp"] for v in valid)
        fn = sum(v["fn"] for v in valid)
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        fps = sum(v["processing_fps"] for v in valid) / max(len(valid), 1)

        results[variant] = {
            "per_video": per_video,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": p, "recall": r, "f1": f1,
            "mean_fps": fps,
            "fully_calibrated": all(v["calibrated"] for v in valid) if valid else False,
        }

    print("\n" + "=" * 92)
    print(f"{'variant':<14} {'FPS':>8} {'P':>7} {'R':>7} {'F1':>7} {'calibrated':>12}")
    print("=" * 92)
    for name, r in results.items():
        print(
            f"{name:<14} {r['mean_fps']:>8.1f} {r['precision']:>7.3f} "
            f"{r['recall']:>7.3f} {r['f1']:>7.3f} "
            f"{str(r['fully_calibrated']):>12}"
        )
    print("=" * 92)

    out = OUT_ROOT / "comparison_realtime.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")
    print("\nFor a production decision, do not rank uncalibrated variants by F1.")


if __name__ == "__main__":
    main()
