"""
generate_csv_video_split.py
===========================
Build split_csv/1cls.csv for EfficientAD training, splitting by SOURCE VIDEO
rather than by crop.

Why this matters
----------------
The old generate_csv.py shuffled crops. The same physical gum appears in many
crops across adjacent frames. Shuffling crops puts some of gum X's crops in
train and others in validation → the student memorizes gum X → inflated AUROC.
This script groups crops by their source video, then splits videos.

Label format
------------
The EfficientAD VisA loader (utils.data_loader.get_AD_dataset) expects the
label column to contain the STRINGS 'normal' and 'anomaly', not integers.
Writing 0/1 causes every row to be read as anomaly, which makes AUROC stuck
at 0.0000.

Split convention (EfficientAD paper, 3 splits)
----------------------------------------------
    train      normal only            student/AE training
    test       normal only (held out) quantile calibration
    validation normal + anomaly       AUROC monitoring

Usage
-----
    python scripts/generate_csv_video_split.py

Edit the constants at the top if your dataset path or ratios differ.
"""

import csv
import random
from collections import defaultdict
from pathlib import Path

# ── config ─────────────────────────────────────────────────────────────────
DATASET_ROOT = Path('/home/ahmed/two-stage-efficientad-inspection/dataset_multi')
CATEGORY     = 'chewinggum'
TRAIN_RATIO  = 0.70
TEST_RATIO   = 0.15
SEED         = 42

# Guarantee at least this many anomaly-bearing videos land in validation.
MIN_SHARED_IN_VAL = 4

# Fail hard if fewer than this many anomaly crops end up in validation.
MIN_VAL_ANOMALY   = 100

# Force specific videos into validation by name (stem only, no extension).
# These must be shared videos (contribute both normal and anomaly crops).
FORCE_VAL_VIDEOS = ['video_A', 'video_A3', 'video_A4', 'video1']


def video_stem(path: Path):
    """
    Return the source video stem, or None if it can't be determined.

    Names like 'video1_frame0000100_crop000.png'      → 'video1'
    Names like 'frame2_0000000_crop002.png'          → None (unknown source)
    Names like 'frame0000000_crop000.png'            → None (unknown source)

    Crops with unknown source are DROPPED to prevent crop-level leakage:
    two frames of the same physical gum would otherwise end up in different
    splits.
    """
    name = path.stem
    if '_frame' in name:
        return name.split('_frame')[0]
    return None


def main():
    rng = random.Random(SEED)

    img_root = DATASET_ROOT / CATEGORY / 'Data' / 'Images'

    def load(root):
        out = []
        dropped = 0
        for p in sorted(root.glob('*.png')):
            stem = video_stem(p)
            if stem is None:
                dropped += 1
                continue
            out.append((stem, p))
        return out, dropped

    normals,   n_dropped = load(img_root / 'Normal')
    anomalies, a_dropped = load(img_root / 'Anomaly')

    print(f'Dataset root    : {DATASET_ROOT}')
    print(f'Category        : {CATEGORY}')
    print(f'Normals loaded  : {len(normals):5d}   (dropped {n_dropped} with unknown source)')
    print(f'Anomalies loaded: {len(anomalies):5d}   (dropped {a_dropped} with unknown source)')
    print()

    normals_by_video   = defaultdict(list)
    anomalies_by_video = defaultdict(list)
    for stem, p in normals:
        normals_by_video[stem].append(p)
    for stem, p in anomalies:
        anomalies_by_video[stem].append(p)

    shared_videos = sorted(set(normals_by_video) & set(anomalies_by_video))
    normal_only   = sorted(set(normals_by_video) - set(anomalies_by_video))

    print(f'Shared videos (N+A): {len(shared_videos)}')
    for v in shared_videos:
        print(f'  {v:50s}  N={len(normals_by_video[v]):5d}  A={len(anomalies_by_video[v]):4d}')
    print()

    # ── split shared videos ────────────────────────────────────────────────
    forced_val = [v for v in FORCE_VAL_VIDEOS if v in shared_videos]
    remaining  = [v for v in shared_videos if v not in forced_val]
    rng.shuffle(remaining)

    n_remaining = len(remaining)
    if n_remaining < 3:
        raise SystemExit(
            f'Only {n_remaining} shared videos left after forcing '
            f'{len(forced_val)} to val — need at least 3. Reduce FORCE_VAL_VIDEOS.')

    n_train_s = max(1, int(n_remaining * TRAIN_RATIO))
    n_test_s  = max(1, int(n_remaining * TEST_RATIO))
    if n_train_s + n_test_s >= n_remaining:
        n_test_s = max(1, n_remaining - n_train_s - 1)

    train_shared = remaining[:n_train_s]
    test_shared  = remaining[n_train_s:n_train_s + n_test_s]
    val_shared   = remaining[n_train_s + n_test_s:] + forced_val

    # ── split normal-only videos ───────────────────────────────────────────
    rng.shuffle(normal_only)
    n_no = len(normal_only)
    n_train_n = max(1, int(n_no * TRAIN_RATIO)) if n_no else 0
    n_test_n  = max(1, int(n_no * TEST_RATIO))  if n_no else 0
    train_norm_only = normal_only[:n_train_n]
    test_norm_only  = normal_only[n_train_n:n_train_n + n_test_n]
    val_norm_only   = normal_only[n_train_n + n_test_n:]

    # ── build file lists ───────────────────────────────────────────────────
    def collect(d, vids):
        return [p for v in vids for p in d.get(v, [])]

    train_normal = collect(normals_by_video,   train_shared + train_norm_only)
    test_normal  = collect(normals_by_video,   test_shared  + test_norm_only)
    val_normal   = collect(normals_by_video,   val_shared   + val_norm_only)
    val_anomaly  = collect(anomalies_by_video, val_shared)

    # ── video-level leak check ─────────────────────────────────────────────
    def vids(files):
        return {video_stem(f) for f in files}

    tv, ev, vv, av = vids(train_normal), vids(test_normal), vids(val_normal), vids(val_anomaly)

    print('Video assignment:')
    print(f'  train      : {sorted(tv)[:8]}'
          + (f'  ... ({len(tv)} total)' if len(tv) > 8 else ''))
    print(f'  test       : {sorted(ev)}')
    print(f'  val normal : {sorted(vv)[:8]}'
          + (f'  ... ({len(vv)} total)' if len(vv) > 8 else ''))
    print(f'  val anomaly: {sorted(av)}')
    print()
    print('Video-level overlap check:')
    print(f'  train ∩ test        : {sorted(tv & ev)}')
    print(f'  train ∩ val_normals : {sorted(tv & vv)}')
    print(f'  test  ∩ val_normals : {sorted(ev & vv)}')
    print(f'  train ∩ val_anom    : {sorted(tv & av)}')

    leaks = (tv & ev) | (tv & vv) | (ev & vv) | (tv & av)
    if leaks:
        print(f'\n⚠ VIDEO-LEVEL LEAKAGE DETECTED: {sorted(leaks)}')
        print('  Re-run with a different SEED or adjust FORCE_VAL_VIDEOS.')
        return
    print('  ✓ no overlap')
    print()

    print('Split sizes:')
    print(f'  train normal   : {len(train_normal):5d} crops ({len(tv)} videos)')
    print(f'  test  normal   : {len(test_normal):5d} crops ({len(ev)} videos)')
    print(f'  val   normal   : {len(val_normal):5d} crops ({len(vv)} videos)')
    print(f'  val   anomaly  : {len(val_anomaly):5d} crops ({len(av)} videos)')
    print()

    if len(val_anomaly) < MIN_VAL_ANOMALY:
        raise SystemExit(
            f'Only {len(val_anomaly)} val anomalies — need at least {MIN_VAL_ANOMALY}. '
            f'Add more videos to FORCE_VAL_VIDEOS or increase the anomaly count.')

    # ── write CSV ──────────────────────────────────────────────────────────
    def rel(p):
        try:
            return str(p.relative_to(DATASET_ROOT))
        except ValueError:
            return str(p)

    rows = []
    for f in train_normal:
        rows.append((CATEGORY, 'train',      'normal',  rel(f), ''))
    for f in test_normal:
        rows.append((CATEGORY, 'test',       'normal',  rel(f), ''))
    for f in val_normal:
        rows.append((CATEGORY, 'validation', 'normal',  rel(f), ''))
    for f in val_anomaly:
        rows.append((CATEGORY, 'validation', 'anomaly', rel(f), ''))

    out_csv = DATASET_ROOT / 'split_csv' / '1cls.csv'
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['object', 'split', 'label', 'image', 'mask'])
        w.writerows(rows)

    print(f'CSV written → {out_csv}  ({len(rows)} rows)')
    print(f'  train      : {len(train_normal)} normals')
    print(f'  test       : {len(test_normal)} normals')
    print(f'  validation : {len(val_normal)} normals + {len(val_anomaly)} anomalies')
    print()
    print('Label column uses STRINGS ("normal" / "anomaly") — required by the')
    print('EfficientAD VisA loader. Integer labels cause AUROC = 0.0000.')


if __name__ == '__main__':
    main()