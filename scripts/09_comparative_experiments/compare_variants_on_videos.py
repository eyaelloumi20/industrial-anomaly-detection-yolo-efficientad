"""
compare_variants_on_videos.py
=============================
Run every trained EfficientAD variant through the full video pipeline on the
held-out videos, then compute precision / recall / F1 against ground truth.

Calls video_streaming_detector_optimized.py once per (variant, video) pair,
then reads each run's summary.json to get the final fired track IDs.

Held-out videos (never in the training split of dataset_multi):
    video_A   gt = {1, 4, 12, 18, 34, 36}
    video_A3  gt = {6, 8, 18, 27, 34, 50}
    video_A4  gt = {2, 15, 18, 30, 49, 55}

Output:
    variant_runs/<variant>/<video>/run_*/summary.json  per-run details
    variant_runs/comparison.json                       aggregated table

Usage:
    cd /home/ahmed/two-stage-efficientad-inspection
    python compare_variants_on_videos.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

# ---- configuration ---------------------------------------------------------
ROOT        = Path("/home/ahmed/two-stage-efficientad-inspection")
DETECTOR    = ROOT / "video_streaming_detector_optimized.py"
YOLO        = ROOT / "runs/detect/yolo_runs/chewinggum_yolov8n_v2/weights/best.pt"
THRESHOLDS  = ROOT / "video_calibration_new/video_thresholds_baseline.json"
VIDEO_DIR   = ROOT / "video_test"
OUT_ROOT    = ROOT / "variant_runs"

# variant_name -> ckpt_dir relative to ROOT
VARIANTS = {
    "baseline":   "ckpt_cropped_multi",
    "lowres_192": "ckpt_lowres_192",
    "lowres_224": "ckpt_lowres_224",
    "thin_075":   "ckpt_thin_075",
    "thin_050":   "ckpt_thin_050",
    "pruned_030": "ckpt_pruned_030",
}

# Ground-truth defect track IDs for each held-out video
GROUND_TRUTH = {
    "video_A":  {1, 4, 12, 18, 34, 36},
    "video_A3": {6, 8, 18, 27, 34, 50},
    "video_A4": {2, 15, 18, 30, 49, 55},
}

TAU_TRACK = 0.45


# ---- runner ----------------------------------------------------------------
def run_one(variant, ckpt_rel, video_name):
    ckpt_abs = ROOT / ckpt_rel
    if not ckpt_abs.is_dir():
        print(f"    [skip] {variant}: {ckpt_abs} missing")
        return None

    video_path = VIDEO_DIR / f"{video_name}.mp4"
    if not video_path.is_file():
        print(f"    [skip] {variant}/{video_name}: {video_path} missing")
        return None

    save_dir = OUT_ROOT / variant / video_name
    save_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(DETECTOR),
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
        "--batch_frames", "16",
        "--batch_crops", "96",
        "--crop_workers", "12",
        "--fp16",
        "--device", "cuda:0",
        "--no_video",
        "--log_every", "0",
        "--save_dir", str(save_dir),
    ]

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    dt = time.perf_counter() - t0

    if proc.returncode != 0:
        print(f"    FAILED (exit {proc.returncode}, {dt:.1f}s)")
        tail = proc.stderr.strip().splitlines()[-5:]
        for line in tail:
            print(f"      | {line}")
        return None

    # The detector creates a unique run_<tmp>/ subdir. Find the latest.
    run_dirs = sorted(save_dir.glob("run_*"), key=lambda p: p.stat().st_mtime)
    if not run_dirs:
        print(f"    no run_* dir produced")
        return None

    summary_file = run_dirs[-1] / "summary.json"
    if not summary_file.is_file():
        print(f"    no summary.json in {run_dirs[-1]}")
        return None

    summary = json.loads(summary_file.read_text())
    summary["_elapsed_s"] = dt
    return summary


# ---- metrics ---------------------------------------------------------------
def evaluate(fired_ids, truth):
    fired = set(fired_ids)
    tp = len(fired & truth)
    fp = len(fired - truth)
    fn = len(truth - fired)
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "fired_ids": sorted(fired),
        "false_positives": sorted(fired - truth),
        "false_negatives": sorted(truth - fired),
    }


# ---- main ------------------------------------------------------------------
def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = {}

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
            m = evaluate(summary["fired_track_ids"], truth)
            m["processing_fps"] = summary.get("processing_fps", 0.0)
            m["n_frames"] = summary.get("n_frames", 0)
            per_video[video_name] = m
            print(f" fired={m['fired_ids']}  "
                  f"TP={m['tp']} FP={m['fp']} FN={m['fn']}  "
                  f"({summary.get('processing_fps', 0):.1f} fps)")

        # Aggregate across all three held-out videos
        agg = {"tp": 0, "fp": 0, "fn": 0}
        fps_list = []
        for v in per_video.values():
            if v is not None:
                agg["tp"] += v["tp"]
                agg["fp"] += v["fp"]
                agg["fn"] += v["fn"]
                if v["processing_fps"] > 0:
                    fps_list.append(v["processing_fps"])

        p = agg["tp"] / max(agg["tp"] + agg["fp"], 1)
        r = agg["tp"] / max(agg["tp"] + agg["fn"], 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        mean_fps = sum(fps_list) / len(fps_list) if fps_list else 0.0

        results[variant] = {
            "per_video": per_video,
            "agg": agg,
            "precision": p,
            "recall": r,
            "f1": f1,
            "mean_fps": mean_fps,
        }

    # ---- print comparison table --------------------------------------------
    print("\n" + "=" * 118)
    print(f"{'variant':<14} {'TP':>4} {'FP':>4} {'FN':>4} "
          f"{'Prec':>7} {'Recall':>7} {'F1':>7} {'fps':>7}   per-video fired IDs")
    print("=" * 118)
    for variant, r in results.items():
        per_v = "  ".join(
            f"{v[6:]}:{r['per_video'][v]['fired_ids']}"
            if r['per_video'][v] is not None else f"{v[6:]}:NA"
            for v in GROUND_TRUTH
        )
        print(f"{variant:<14} {r['agg']['tp']:>4} {r['agg']['fp']:>4} "
              f"{r['agg']['fn']:>4} {r['precision']:>7.3f} {r['recall']:>7.3f} "
              f"{r['f1']:>7.3f} {r['mean_fps']:>7.1f}   {per_v}")
    print("=" * 118)

    # Save
    (OUT_ROOT / "comparison.json").write_text(json.dumps(results, indent=2))
    print(f"\nFull results saved to {OUT_ROOT / 'comparison.json'}")

    # Highlight winner
    candidates = [(n, r) for n, r in results.items()
                  if r["f1"] >= 0.72 and n != "baseline"]
    if candidates:
        best = max(candidates, key=lambda x: x[1]["mean_fps"])
        print(f"\n>>> FASTEST VIABLE VARIANT: {best[0]}  "
              f"(F1={best[1]['f1']:.3f}, {best[1]['mean_fps']:.1f} fps)")
    else:
        print("\n>>> No variant beat the F1=0.72 threshold. "
              "Keep the baseline.")


if __name__ == "__main__":
    main()