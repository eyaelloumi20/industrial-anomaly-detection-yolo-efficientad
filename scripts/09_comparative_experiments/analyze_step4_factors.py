"""
analyze_step4_factors.py
=========================
Step 4 statistical diagnosis (read-only).

Determines whether each environmental/acquisition factor -- background,
lighting, vertical position, horizontal position, scale, session -- shifts
the track-level EfficientAD raw-score distribution, and how much.

Primary statistical unit: TRACK. Per the Step 4 statistical rule, frames
within a track are highly correlated (same object, consecutive frames), so
analyzing at frame level would pseudo-replicate and inflate significance.
Each track contributes exactly one value: its median raw score across all
frames in that track (already computed by build_step4_dataset.py as
`median_raw_score` in step4_track_aggregated.csv).

For each factor, this script:
  1. Prints descriptive statistics per level: n tracks, mean, median, std,
     P90, P95, P99, P99.5, max, and % of tracks whose median score exceeds
     the frozen threshold.
  2. Runs a Kruskal-Wallis H-test across the factor's levels (non-parametric
     one-way ANOVA -- appropriate given the right-skewed anomaly-score
     distributions already seen in Step 3, and avoids assuming normality
     or equal variance per level).
  3. Reports epsilon-squared effect size: eps^2 = H / (N - 1)
     (Tomczak & Tomczak, 2014) -- 0 = no effect, ~1 = maximal.
  4. If Kruskal-Wallis is significant (p < --alpha) AND the factor has more
     than 2 levels, runs Dunn's pairwise post-hoc test (rank-based, with
     the standard tie correction) across every pair of levels, with
     Holm-Bonferroni correction applied across all pairs for that factor.
     (With exactly 2 levels, the KW test IS the pairwise test -- no
     post-hoc needed.)

This script does NOT propose or compute a new thresholding method, and does
NOT touch scoring/YOLO/ByteTrack/calibration/model weights -- it only reads
step4_track_aggregated.csv and writes results tables. Per the project's
Step 4 rule, interactions / mixed models are NOT auto-run; if two or more
factors each show a significant main effect, this script prints a note
suggesting a manual follow-up rather than computing one.

Usage
-----
    python analyze_step4_factors.py \
        --tracks_csv ./dataset_step4/step4_track_aggregated.csv \
        --out_dir    ./dataset_step4/step4_analysis \
        --alpha      0.05
"""

import os
import csv
import argparse
import itertools
from typing import Dict, List

import numpy as np
from scipy import stats

FACTOR_COLUMNS = {
    "background_id": "background_id",
    "lighting_id": "lighting_id",
    "split (sanity check, not a real factor)": "split",
    "vertical_position": "vertical_position",
    "horizontal_position": "horizontal_position",
    "scale_bin": "scale_bin",
    "session": "session",
}

DESCRIPTIVE_METRIC = "median_raw_score"
EXCEED_COLUMN = "track_median_exceeds_threshold"


def parse_args():
    p = argparse.ArgumentParser(
        description="Step 4 factor diagnosis: descriptive stats + "
                    "Kruskal-Wallis + epsilon-squared + Dunn post-hoc on "
                    "track-level median raw scores.")
    p.add_argument("--tracks_csv", required=True,
                   help="step4_track_aggregated.csv from build_step4_dataset.py")
    p.add_argument("--out_dir", required=True,
                   help="Directory to write step4_factor_results.csv and "
                        "step4_posthoc_results.csv into.")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Significance threshold for Kruskal-Wallis and, "
                        "after Holm correction, for Dunn's post-hoc "
                        "(default 0.05).")
    p.add_argument("--min_n_per_level", type=int, default=5,
                   help="Skip a factor level with fewer than this many "
                        "tracks (too few for a meaningful comparison; "
                        "default 5). The level is reported but excluded "
                        "from the KW test.")
    return p.parse_args()


def load_tracks(path: str) -> List[dict]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"'{path}' has no rows.")
    for r in rows:
        r[DESCRIPTIVE_METRIC] = float(r[DESCRIPTIVE_METRIC])
        r[EXCEED_COLUMN] = int(r[EXCEED_COLUMN])
    return rows


def percentile(vals: List[float], p: float) -> float:
    return float(np.percentile(np.asarray(vals, dtype=float), p)) if vals else float("nan")


def describe_level(vals: List[float], exceed_flags: List[int]) -> dict:
    arr = np.asarray(vals, dtype=float)
    return dict(
        n=len(arr),
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        std=float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        p90=percentile(vals, 90),
        p95=percentile(vals, 95),
        p99=percentile(vals, 99),
        p99_5=percentile(vals, 99.5),
        max=float(np.max(arr)),
        pct_exceed_threshold=100.0 * sum(exceed_flags) / len(exceed_flags),
    )


# ── epsilon-squared effect size for Kruskal-Wallis ────────────────────────────

def epsilon_squared(H: float, N: int) -> float:
    """eps^2 = H / (N - 1)  (Tomczak & Tomczak, 2014)."""
    if N <= 1:
        return float("nan")
    return H / (N - 1)


# ── Dunn's post-hoc test with tie correction + Holm-Bonferroni ────────────────

def dunn_posthoc(groups: Dict[str, List[float]], alpha: float) -> List[dict]:
    """Pairwise Dunn's test across every pair of levels in `groups`.
    Ranks are computed ONCE over the pooled data (standard Dunn's test),
    with the usual tie correction in the standard-error term. Returns one
    row per pair with z, raw p (two-sided), Holm-adjusted p, and
    significance at `alpha` after correction."""
    level_names = list(groups.keys())
    all_vals = np.concatenate([np.asarray(groups[k], dtype=float) for k in level_names])
    N = len(all_vals)
    ranks = stats.rankdata(all_vals)

    # split ranks back out per level, and mean rank per level
    idx = 0
    mean_rank: Dict[str, float] = {}
    n_per_level: Dict[str, int] = {}
    for k in level_names:
        n_k = len(groups[k])
        mean_rank[k] = float(np.mean(ranks[idx:idx + n_k]))
        n_per_level[k] = n_k
        idx += n_k

    # tie correction term: sum(t^3 - t) over all tied groups in the pooled ranks
    _, counts = np.unique(all_vals, return_counts=True)
    tie_term = float(np.sum(counts ** 3 - counts))

    pairs = list(itertools.combinations(level_names, 2))
    raw_results = []
    for a, b in pairs:
        n_a, n_b = n_per_level[a], n_per_level[b]
        se = np.sqrt(
            ((N * (N + 1) / 12.0) - (tie_term / (12.0 * (N - 1))))
            * (1.0 / n_a + 1.0 / n_b)
        )
        if se == 0 or np.isnan(se):
            z, p = 0.0, 1.0
        else:
            z = (mean_rank[a] - mean_rank[b]) / se
            p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
        raw_results.append(dict(level_a=a, level_b=b, z=z, p_raw=p))

    # Holm-Bonferroni correction across all pairs for this factor
    order = sorted(range(len(raw_results)), key=lambda i: raw_results[i]["p_raw"])
    m = len(raw_results)
    holm_adj = [None] * m
    running_max = 0.0
    for rank_i, idx_i in enumerate(order):
        p_raw = raw_results[idx_i]["p_raw"]
        adj = min(1.0, p_raw * (m - rank_i))
        running_max = max(running_max, adj)
        holm_adj[idx_i] = running_max

    for i, r in enumerate(raw_results):
        r["p_holm"] = holm_adj[i]
        r["significant"] = holm_adj[i] < alpha

    return raw_results


# ── main analysis loop ─────────────────────────────────────────────────────

def analyze_factor(factor_label: str, column: str, tracks: List[dict],
                   alpha: float, min_n_per_level: int) -> tuple:
    print(f"\n{'-'*70}")
    print(f"  Factor: {factor_label}   (column: {column})")
    print(f"{'-'*70}")

    by_level: Dict[str, List[dict]] = {}
    for r in tracks:
        level = r.get(column, "").strip()
        if not level:
            continue
        by_level.setdefault(level, []).append(r)

    if len(by_level) < 2:
        print(f"  Only {len(by_level)} level(s) present in this dataset -- "
              f"skipping (need >= 2 to compare).")
        return [], []

    kept_levels = {}
    for level, rows in sorted(by_level.items()):
        vals = [r[DESCRIPTIVE_METRIC] for r in rows]
        flags = [r[EXCEED_COLUMN] for r in rows]
        d = describe_level(vals, flags)
        print(f"  [{level:<12s}] n={d['n']:<4d} mean={d['mean']:.4f}  "
              f"median={d['median']:.4f}  std={d['std']:.4f}  "
              f"P90={d['p90']:.4f}  P95={d['p95']:.4f}  P99={d['p99']:.4f}  "
              f"P99.5={d['p99_5']:.4f}  max={d['max']:.4f}  "
              f"%exceed={d['pct_exceed_threshold']:.1f}%")
        if d["n"] < min_n_per_level:
            print(f"    -> excluded from Kruskal-Wallis: n={d['n']} < "
                  f"--min_n_per_level={min_n_per_level}")
            continue
        kept_levels[level] = vals

    factor_rows = []
    posthoc_rows = []

    if len(kept_levels) < 2:
        print(f"  Fewer than 2 levels have enough tracks -- skipping "
              f"Kruskal-Wallis for this factor.")
        return factor_rows, posthoc_rows

    H, p = stats.kruskal(*kept_levels.values())
    N = sum(len(v) for v in kept_levels.values())
    eps2 = epsilon_squared(H, N)
    significant = p < alpha

    print(f"\n  Kruskal-Wallis: H={H:.4f}  df={len(kept_levels)-1}  p={p:.6g}  "
          f"epsilon^2={eps2:.4f}  N={N}  "
          f"-> {'SIGNIFICANT' if significant else 'not significant'} "
          f"(alpha={alpha})")

    factor_rows.append(dict(
        factor=factor_label, column=column, n_levels=len(kept_levels),
        levels=";".join(sorted(kept_levels.keys())), N=N,
        H=H, df=len(kept_levels) - 1, p_value=p,
        epsilon_squared=eps2, significant=significant, alpha=alpha,
    ))

    if significant and len(kept_levels) > 2:
        print(f"\n  Dunn's post-hoc (Holm-corrected, alpha={alpha}):")
        results = dunn_posthoc(kept_levels, alpha)
        for r in sorted(results, key=lambda x: x["p_holm"]):
            flag = "*** SIG ***" if r["significant"] else ""
            print(f"    {r['level_a']:<12s} vs {r['level_b']:<12s}   "
                  f"z={r['z']:+.3f}  p_raw={r['p_raw']:.4g}  "
                  f"p_holm={r['p_holm']:.4g}  {flag}")
            posthoc_rows.append(dict(
                factor=factor_label, column=column,
                level_a=r["level_a"], level_b=r["level_b"],
                z=r["z"], p_raw=r["p_raw"], p_holm=r["p_holm"],
                significant=r["significant"],
            ))
    elif significant and len(kept_levels) == 2:
        print(f"  Only 2 levels -- the Kruskal-Wallis test above IS the "
              f"pairwise comparison; no separate post-hoc needed.")

    return factor_rows, posthoc_rows


def write_csv(rows: List[dict], path: str) -> None:
    if not rows:
        print(f"  [WARN] no rows to write for '{path}' -- skipping.")
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {len(rows)} row(s) -> {path}")


def main():
    args = parse_args()
    tracks = load_tracks(args.tracks_csv)

    print(f"\n{'='*70}")
    print(f"  Step 4 factor diagnosis")
    print(f"  tracks_csv : {args.tracks_csv}")
    print(f"  n tracks   : {len(tracks)}")
    print(f"  metric     : {DESCRIPTIVE_METRIC}  (median raw score per track)")
    print(f"  alpha      : {args.alpha}")
    print(f"{'='*70}")

    all_factor_rows = []
    all_posthoc_rows = []
    sig_factors = []

    for label, column in FACTOR_COLUMNS.items():
        f_rows, ph_rows = analyze_factor(label, column, tracks, args.alpha,
                                         args.min_n_per_level)
        all_factor_rows.extend(f_rows)
        all_posthoc_rows.extend(ph_rows)
        sig_factors.extend(r["factor"] for r in f_rows if r["significant"])

    print(f"\n{'='*70}")
    print(f"  Summary")
    print(f"{'='*70}")
    if all_factor_rows:
        for r in all_factor_rows:
            flag = "SIGNIFICANT" if r["significant"] else "n.s."
            print(f"  {r['factor']:<40s} H={r['H']:.3f}  p={r['p_value']:.4g}  "
                  f"eps^2={r['epsilon_squared']:.4f}  [{flag}]")
    else:
        print("  No factor had >= 2 levels with enough tracks to test -- "
              "collect more controlled_validation videos before drawing "
              "conclusions.")

    if len(sig_factors) >= 2:
        print(f"\n  NOTE: {len(sig_factors)} factors show a significant main "
              f"effect ({', '.join(sig_factors)}). Per the Step 4 protocol, "
              f"interaction / mixed-effects modelling is NOT run "
              f"automatically -- consider it as a manual follow-up once "
              f"single-factor diagnosis is reviewed, not before.")

    write_csv(all_factor_rows, os.path.join(args.out_dir, "step4_factor_results.csv"))
    write_csv(all_posthoc_rows, os.path.join(args.out_dir, "step4_posthoc_results.csv"))

    print(f"\n[analyze_step4_factors] Done. This is diagnosis only -- do "
          f"NOT change scoring/thresholding based on this until the full "
          f"Step 4 dataset (multiple sessions, all planned background/"
          f"lighting conditions) has been collected and reviewed.")


if __name__ == "__main__":
    main()
