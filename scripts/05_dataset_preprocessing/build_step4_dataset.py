"""
build_step4_dataset.py
=======================
Step 4A (diagnosis phase, read-only).

Joins the per-detection score CSVs already produced by
compare_pipeline_methods_batched.py (frame, track_id, raw_score, ema_score,
verdict, x1, y1, x2, y2, score_ms) with:
  - video_dims.csv        (from get_video_dims.py)
  - manifest.csv           (split / background_id / lighting_id / session)

background_id and lighting_id are tracked as TWO SEPARATE factors (a video
can vary one, the other, both, or neither vs. the reference) -- not one
combined "condition" label. This matters when real footage varies both at
once in different combinations (e.g. 4 backgrounds mostly under one
lighting, plus 1 lighting variant on only 1 background).

Excluding known-defective tracks
---------------------------------
Controlled-validation videos should show normal product only, but real
footage sometimes has 1-2 genuinely defective items mixed in. Pass
--exclude_tracks_csv (video_filename, track_id) to drop specific tracks
from BOTH output files before they contaminate the "normal score"
statistics. Identify them by inspecting step4_track_aggregated.csv for
outlier scores within a video (a real defect should score far above its
neighbours if the model works), cross-checked visually.

and derives the normalized position/scale features Step 4 needs, then
aggregates to the TRACK level (the primary statistical unit per the Step 4
statistical rule -- frames within a track are correlated, tracks are not).

This script is READ-ONLY with respect to the existing pipeline: it does not
call YOLO, ByteTrack, or EfficientAD, and does not import or modify
retrain_efficientad.py, pipeline.py, evaluate_pipeline.py,
calibrate_threshold_all_fp16.py, or compare_pipeline_methods_batched.py. It
only reads CSVs those scripts already wrote to disk.

Linking detections to a video
------------------------------
scores_<method>.csv (compare_pipeline_methods_batched.py's output) has no
video filename column -- it's one file per video per method. To join
unambiguously across many videos, pass a small "runs" CSV that maps each
video to its scores file:

    runs.csv columns: video_filename, scores_csv, method (method optional,
    defaults to 'baseline')

Example runs.csv:
    video_filename,scores_csv,method
    videoN1.mp4,./results_compare_all/videoN1/baseline/scores_baseline.csv,baseline
    video_N3.mp4,./results_compare_all/video_N3/baseline/scores_baseline.csv,baseline
    bg1_plain.mp4,./results_compare_all/bg1_plain/baseline/scores_baseline.csv,baseline

Usage
-----
    python build_step4_dataset.py \
        --runs_csv    ./runs.csv \
        --video_dims  ./dataset_step4/video_dims.csv \
        --manifest    ./dataset_step4/manifest.csv \
        --out_dir     ./dataset_step4 \
        --frozen_threshold 5.266783

Outputs (written to --out_dir)
-------------------------------
step4_master_metadata.csv   one row per detection (frame x track x video)
step4_track_aggregated.csv  one row per (video_filename, track_id) --
                             median raw score per track, per-track
                             position/scale bins, condition labels
"""

import os
import csv
import argparse
import statistics
from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np


REQUIRED_SCORE_COLS = ["frame", "track_id", "raw_score", "ema_score",
                        "verdict", "x1", "y1", "x2", "y2", "score_ms"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Step 4A: join detection scores with video dims + "
                    "manifest, derive position/scale features, aggregate "
                    "to track level.")
    p.add_argument("--runs_csv", required=True,
                   help="CSV mapping video_filename -> scores_<method>.csv "
                        "path (+ optional method column). See module "
                        "docstring for format.")
    p.add_argument("--video_dims", required=True,
                   help="video_dims.csv from get_video_dims.py")
    p.add_argument("--manifest", required=True,
                   help="manifest.csv (video_filename, split, background_id, "
                        "lighting_id, session, ...)")
    p.add_argument("--exclude_tracks_csv", default=None,
                   help="Optional CSV with columns (video_filename, "
                        "track_id) -- tracks to drop before writing outputs "
                        "(e.g. genuinely defective items found in otherwise "
                        "known-good footage). See module docstring.")
    p.add_argument("--out_dir", required=True,
                   help="Directory to write step4_master_metadata.csv and "
                        "step4_track_aggregated.csv into.")
    p.add_argument("--frozen_threshold", type=float, default=5.266783,
                   help="Existing pooled p99.5 EMA threshold from "
                        "calibrate_threshold_all_fp16.py (default "
                        "5.266783). Used ONLY to compute an exceedance "
                        "flag for diagnosis -- never recomputed here.")
    p.add_argument("--n_position_bins", type=int, default=3, choices=[3],
                   help="Vertical/horizontal position bins (fixed at 3: "
                        "top/middle/bottom, left/center/right, by equal "
                        "normalized-coordinate thirds).")
    p.add_argument("--scale_quantiles", type=float, nargs=2,
                   default=[1.0 / 3.0, 2.0 / 3.0],
                   help="Quantiles (low, high) of relative_box_area used "
                        "to cut small/medium/large, computed data-driven "
                        "over ALL detections pooled (default tercile "
                        "0.333/0.667). Pass e.g. 0.25 0.75 for quartile "
                        "cuts instead.")
    return p.parse_args()


# ── loaders ────────────────────────────────────────────────────────────────

def load_runs(runs_csv: str) -> List[dict]:
    with open(runs_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            if not r.get("video_filename", "").strip():
                continue
            r["method"] = (r.get("method") or "baseline").strip()
            rows.append(r)
    if not rows:
        raise ValueError(f"--runs_csv '{runs_csv}' has no usable rows.")
    return rows


def load_video_dims(video_dims_csv: str) -> Dict[str, dict]:
    out = {}
    with open(video_dims_csv, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            fn = r["video_filename"].strip()
            if str(r.get("found", "")).strip().lower() not in ("true", "1"):
                out[fn] = None
                continue
            out[fn] = dict(
                width=int(r["width"]), height=int(r["height"]),
                fps=float(r["fps"]) if r["fps"] else 0.0,
                frame_count=int(r["frame_count"]) if r["frame_count"] else 0,
            )
    return out


def load_manifest(manifest_csv: str) -> Dict[str, dict]:
    out = {}
    with open(manifest_csv, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            fn = r["video_filename"].strip()
            out[fn] = {k: (v.strip() if v else "") for k, v in r.items()}
    return out


def load_exclude_tracks(path: Optional[str]) -> set:
    if not path:
        return set()
    excl = set()
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            excl.add((r["video_filename"].strip(), int(r["track_id"])))
    print(f"[exclude_tracks] {len(excl)} track(s) will be dropped as "
          f"known-defective: {sorted(excl)}")
    return excl


def load_scores(scores_csv: str) -> List[dict]:
    with open(scores_csv, newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_SCORE_COLS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                f"'{scores_csv}' is missing expected column(s) {missing}. "
                f"Found columns: {reader.fieldnames}. Was this really "
                f"written by compare_pipeline_methods_batched.py?")
        return list(reader)


# ── feature derivation ───────────────────────────────────────────────────────

def position_bin_3(norm_coord: float) -> str:
    """Equal-thirds bin on a [0,1] normalized coordinate."""
    if norm_coord < 1.0 / 3.0:
        return "low"     # top (for y) / left (for x)
    elif norm_coord < 2.0 / 3.0:
        return "mid"     # middle / center
    else:
        return "high"    # bottom / right


def build_master_rows(runs: List[dict], dims_map: Dict[str, dict],
                      manifest_map: Dict[str, dict],
                      frozen_threshold: float,
                      exclude_tracks: set) -> List[dict]:
    master_rows = []
    n_excluded = 0
    for run in runs:
        video_filename = run["video_filename"].strip()
        method = run["method"]
        scores_csv = run["scores_csv"].strip()

        dims = dims_map.get(video_filename)
        if dims is None:
            print(f"  [SKIP] '{video_filename}': no valid entry in "
                  f"video_dims.csv (missing or unreadable video) -- run "
                  f"get_video_dims.py first.")
            continue
        cond = manifest_map.get(video_filename)
        if cond is None:
            print(f"  [SKIP] '{video_filename}': not found in manifest.csv "
                  f"-- add a row for it before joining.")
            continue
        if not os.path.isfile(scores_csv):
            print(f"  [SKIP] '{video_filename}': --scores_csv path "
                  f"'{scores_csv}' does not exist.")
            continue

        W, H = dims["width"], dims["height"]
        if W <= 0 or H <= 0:
            print(f"  [SKIP] '{video_filename}': invalid dims {W}x{H}.")
            continue

        detections = load_scores(scores_csv)
        print(f"  [JOIN] '{video_filename}'  method={method}  "
              f"{len(detections)} detections  dims={W}x{H}")

        for d in detections:
            if (video_filename, int(d["track_id"])) in exclude_tracks:
                n_excluded += 1
                continue
            x1, y1, x2, y2 = float(d["x1"]), float(d["y1"]), float(d["x2"]), float(d["y2"])
            bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
            cx, cy = x1 + bw / 2.0, y1 + bh / 2.0

            cx_norm = cx / W
            cy_norm = cy / H
            bw_norm = bw / W
            bh_norm = bh / H
            rel_area = (bw * bh) / (W * H)

            raw_score = float(d["raw_score"])
            ema_score = float(d["ema_score"])

            # ── clipping geometry facts only -- NOT a policy/eligibility
            # decision. Mirrors the EXACT clamp already applied at detection
            # time in compare_pipeline_methods_batched.py (x1=max(0,box[0]),
            # x2=min(src_w,box[2]), same for y) -- no invented tolerance.
            touches_top    = (y1 <= 0)
            touches_bottom = (y2 >= H)
            touches_left   = (x1 <= 0)
            touches_right  = (x2 >= W)
            vertical_clipped   = touches_top or touches_bottom
            horizontal_clipped = touches_left or touches_right
            if vertical_clipped and horizontal_clipped:
                bbox_clipping_status = "both"
            elif vertical_clipped:
                bbox_clipping_status = "vertical"
            elif horizontal_clipped:
                bbox_clipping_status = "horizontal"
            else:
                bbox_clipping_status = "none"

            master_rows.append(dict(
                video_filename=video_filename,
                method=method,
                split=cond.get("split", ""),
                background_id=cond.get("background_id", ""),
                lighting_id=cond.get("lighting_id", ""),
                session=cond.get("session", ""),
                frame=int(d["frame"]),
                track_id=int(d["track_id"]),
                raw_score=raw_score,
                ema_score=ema_score,
                verdict=d.get("verdict", ""),
                x1=x1, y1=y1, x2=x2, y2=y2,
                score_ms=float(d["score_ms"]) if d.get("score_ms") else "",
                cx_norm=cx_norm,
                cy_norm=cy_norm,
                bw_norm=bw_norm,
                bh_norm=bh_norm,
                relative_box_area=rel_area,
                vertical_bin_raw=position_bin_3(cy_norm),   # relabeled below
                horizontal_bin_raw=position_bin_3(cx_norm),
                raw_exceeds_threshold=int(raw_score >= frozen_threshold),
                ema_exceeds_threshold=int(ema_score >= frozen_threshold),
                touches_top=int(touches_top),
                touches_bottom=int(touches_bottom),
                touches_left=int(touches_left),
                touches_right=int(touches_right),
                vertical_clipped=int(vertical_clipped),
                horizontal_clipped=int(horizontal_clipped),
                bbox_clipping_status=bbox_clipping_status,
            ))

    # relabel generic low/mid/high -> top/middle/bottom and left/center/right
    vlabel = {"low": "top", "mid": "middle", "high": "bottom"}
    hlabel = {"low": "left", "mid": "center", "high": "right"}
    for r in master_rows:
        r["vertical_position"] = vlabel[r.pop("vertical_bin_raw")]
        r["horizontal_position"] = hlabel[r.pop("horizontal_bin_raw")]

    if n_excluded:
        print(f"  [exclude_tracks] dropped {n_excluded} detection row(s) "
              f"belonging to excluded tracks.")

    return master_rows


def assign_scale_bins(master_rows: List[dict], q_low: float, q_high: float) -> None:
    """Data-driven small/medium/large cut on relative_box_area, pooled over
    ALL detections in this build (in place)."""
    areas = np.array([r["relative_box_area"] for r in master_rows], dtype=float)
    if len(areas) == 0:
        return
    lo_cut = float(np.quantile(areas, q_low))
    hi_cut = float(np.quantile(areas, q_high))
    print(f"[scale bins] relative_box_area cuts: small < {lo_cut:.5f} <= "
          f"medium < {hi_cut:.5f} <= large  (quantiles {q_low:.3f}/{q_high:.3f}, "
          f"n={len(areas)})")
    for r in master_rows:
        a = r["relative_box_area"]
        r["scale_bin"] = "small" if a < lo_cut else ("large" if a >= hi_cut else "medium")


# ── track-level aggregation ─────────────────────────────────────────────────

def percentile(vals: List[float], p: float) -> float:
    return float(np.percentile(np.asarray(vals, dtype=float), p)) if vals else float("nan")


def aggregate_tracks(master_rows: List[dict], frozen_threshold: float) -> List[dict]:
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for r in master_rows:
        groups[(r["video_filename"], r["track_id"])].append(r)

    track_rows = []
    for (video_filename, track_id), rows in groups.items():
        raw_scores = [r["raw_score"] for r in rows]
        ema_scores = [r["ema_score"] for r in rows]

        # Track-level position/scale: median of the continuous normalized
        # coordinates across the track's frames, THEN bin once -- more
        # stable than taking the mode of per-frame bins for a moving object.
        cy_med = statistics.median(r["cy_norm"] for r in rows)
        cx_med = statistics.median(r["cx_norm"] for r in rows)
        area_med = statistics.median(r["relative_box_area"] for r in rows)

        vlabel = {"low": "top", "mid": "middle", "high": "bottom"}
        hlabel = {"low": "left", "mid": "center", "high": "right"}
        vertical_position = vlabel[position_bin_3(cy_med)]
        horizontal_position = hlabel[position_bin_3(cx_med)]

        # reuse whichever scale cuts were used at detection level: recompute
        # membership vs. the same pooled thresholds is not directly available
        # here, so take the majority scale_bin among the track's own frames
        # (already consistent with the single global cut applied upstream).
        scale_bin = statistics.mode([r["scale_bin"] for r in rows])

        n = len(rows)
        n_raw_exceed = sum(r["raw_exceeds_threshold"] for r in rows)
        n_ema_exceed = sum(r["ema_exceeds_threshold"] for r in rows)

        first = rows[0]
        track_rows.append(dict(
            video_filename=video_filename,
            track_id=track_id,
            split=first["split"],
            background_id=first["background_id"],
            lighting_id=first["lighting_id"],
            session=first["session"],
            n_frames=n,
            median_raw_score=statistics.median(raw_scores),
            mean_raw_score=float(np.mean(raw_scores)),
            std_raw_score=float(np.std(raw_scores, ddof=1)) if n > 1 else 0.0,
            p90_raw_score=percentile(raw_scores, 90),
            p95_raw_score=percentile(raw_scores, 95),
            p99_raw_score=percentile(raw_scores, 99),
            p99_5_raw_score=percentile(raw_scores, 99.5),
            max_raw_score=max(raw_scores),
            median_ema_score=statistics.median(ema_scores),
            mean_ema_score=float(np.mean(ema_scores)),
            pct_frames_raw_exceed_threshold=100.0 * n_raw_exceed / n,
            pct_frames_ema_exceed_threshold=100.0 * n_ema_exceed / n,
            track_median_exceeds_threshold=int(
                statistics.median(raw_scores) >= frozen_threshold),
            vertical_position=vertical_position,
            horizontal_position=horizontal_position,
            scale_bin=scale_bin,
            median_cx_norm=cx_med,
            median_cy_norm=cy_med,
            median_relative_box_area=area_med,
        ))

    return track_rows


# ── main ──────────────────────────────────────────────────────────────────────

def write_csv(rows: List[dict], path: str) -> None:
    if not rows:
        print(f"  [WARN] no rows to write for '{path}' -- skipping.")
        return
    fieldnames = list(rows[0].keys())
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {len(rows)} row(s) -> {path}")


def main():
    args = parse_args()

    print(f"\n{'='*70}\n  Step 4A -- build_step4_dataset.py\n{'='*70}")
    print(f"  runs_csv          : {args.runs_csv}")
    print(f"  video_dims        : {args.video_dims}")
    print(f"  manifest          : {args.manifest}")
    print(f"  frozen_threshold  : {args.frozen_threshold}  (diagnosis-only, "
          f"NOT recalculated)")
    print(f"{'='*70}\n")

    runs = load_runs(args.runs_csv)
    dims_map = load_video_dims(args.video_dims)
    manifest_map = load_manifest(args.manifest)
    exclude_tracks = load_exclude_tracks(args.exclude_tracks_csv)

    print(f"[build_step4_dataset] Joining {len(runs)} run(s) ...")
    master_rows = build_master_rows(runs, dims_map, manifest_map,
                                    args.frozen_threshold, exclude_tracks)
    if not master_rows:
        raise SystemExit("No detections joined successfully -- check the "
                         "[SKIP] messages above.")

    assign_scale_bins(master_rows, args.scale_quantiles[0], args.scale_quantiles[1])

    track_rows = aggregate_tracks(master_rows, args.frozen_threshold)

    print(f"\n[build_step4_dataset] {len(master_rows)} detection rows, "
          f"{len(track_rows)} track rows across "
          f"{len({r['video_filename'] for r in master_rows})} video(s).")

    write_csv(master_rows, os.path.join(args.out_dir, "step4_master_metadata.csv"))
    write_csv(track_rows, os.path.join(args.out_dir, "step4_track_aggregated.csv"))

    print(f"\n[build_step4_dataset] Done. Next: "
          f"python analyze_step4_factors.py --tracks_csv "
          f"{os.path.join(args.out_dir, 'step4_track_aggregated.csv')}")


if __name__ == "__main__":
    main()