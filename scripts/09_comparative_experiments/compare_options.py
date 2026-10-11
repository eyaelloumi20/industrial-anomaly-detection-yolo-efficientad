"""
compare_options.py
==================
Aggregates the 4 base runs from ./options/ and computes the ensembles.
Prints a single comparison table.
"""
import json
from pathlib import Path

# Ground truth (from defected pieces.txt)
TRUTH = {
    "video_A":  {1, 4, 12, 18, 34, 36},
    "video_A1": {8, 11, 12, 15, 23, 30},
    "video_A2": {3, 5, 10, 18, 29, 33},
    "video_A3": {6, 8, 18, 27, 34, 50},
    "video_A4": {2, 15, 18, 30, 49, 55},
}
VIDEOS = list(TRUTH.keys())


def fired_set(config, video):
    """Read records_*.jsonl for one config+video and return fired track IDs."""
    base = Path(f"./options/{config}/{video}")
    if not base.exists():
        return None
    # Filename depends on --method: records_baseline.jsonl or records_lowres.jsonl
    for fname in ("records_baseline.jsonl", "records_lowres.jsonl"):
        p = base / fname
        if p.exists():
            s = set()
            for line in p.open():
                rec = json.loads(line)
                for f in rec.get("fired_tracks", []):
                    s.add(f["track_id"])
            return s
    return None


def metrics(fired_by_video, label):
    tp = fp = fn = 0
    for v in VIDEOS:
        fired = fired_by_video.get(v)
        if fired is None:
            print(f"  [MISSING] {label} {v}")
            continue
        truth = TRUTH[v]
        tp += len(truth & fired)
        fp += len(fired - truth)
        fn += len(truth - fired)
    total_pos = sum(len(TRUTH[v]) for v in VIDEOS)
    total_fired = tp + fp
    tpr = tp / max(total_pos, 1)
    fpr = fp / max(total_fired, 1)
    prec = tp / max(total_fired, 1)
    j = tpr - fpr
    return {
        "label": label, "TP": tp, "FP": fp, "FN": fn,
        "TPR": tpr, "FPR": fpr, "Prec": prec, "J": j,
    }


def main():
    # Load the 4 base configs
    base_configs = {
        "A_new_median":    {},
        "B_new_mean":      {},
        "C_old_median":    {},
        "D_old_framerel":  {},
    }
    for cfg in base_configs:
        for v in VIDEOS:
            base_configs[cfg][v] = fired_set(cfg, v)

    # Compute ensembles
    ensembles = {
        "E_intersect":  {},  # A ∩ D
        "F_union":      {},  # A ∪ D
        "G_A_and_C":    {},  # A ∩ C
    }
    for v in VIDEOS:
        A = base_configs["A_new_median"][v]
        C = base_configs["C_old_median"][v]
        D = base_configs["D_old_framerel"][v]
        if A is None or D is None:
            ensembles["E_intersect"][v] = None
            ensembles["F_union"][v]     = None
        else:
            ensembles["E_intersect"][v] = A & D
            ensembles["F_union"][v]     = A | D
        if A is not None and C is not None:
            ensembles["G_A_and_C"][v] = A & C
        else:
            ensembles["G_A_and_C"][v] = None

    # Print the comparison table
    all_results = []
    for name, data in base_configs.items():
        all_results.append(metrics(data, name))
    for name, data in ensembles.items():
        all_results.append(metrics(data, name))

    print()
    print("=" * 90)
    print(f"{'config':<20} {'TP':>4} {'FP':>5} {'FN':>4} "
          f"{'TPR':>7} {'FPR':>7} {'Prec':>7} {'J':>7}")
    print("-" * 90)
    for r in all_results:
        print(f"{r['label']:<20} {r['TP']:>4} {r['FP']:>5} {r['FN']:>4} "
              f"{r['TPR']:>7.3f} {r['FPR']:>7.3f} "
              f"{r['Prec']:>7.3f} {r['J']:>7.3f}")
    print("=" * 90)


if __name__ == "__main__":
    main()