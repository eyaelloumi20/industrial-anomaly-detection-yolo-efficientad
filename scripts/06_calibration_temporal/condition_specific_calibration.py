"""
condition_specific_calibration.py

Computes condition-specific thresholds from the Step 4 track-level data,
using the 99.5th percentile of per-track median scores (non-clipped only).
Optionally supports different percentiles and margins per condition.

Usage:
    python condition_specific_calibration.py [--score_type raw|ema] [--percentile 99.5] [--margin 1.0]
"""

import pandas as pd
import numpy as np
import json
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--score_type', choices=['raw', 'ema'], default='ema',
                        help='Which track-level median score to use (raw or ema)')
    parser.add_argument('--percentile', type=float, default=99.5,
                        help='Percentile to use for calibration (default 99.5)')
    parser.add_argument('--margin', type=float, default=1.0,
                        help='Margin multiplier for thresholds (default 1.0)')
    parser.add_argument('--conditions', nargs='*', default=None,
                        help='Specific conditions to calibrate (bg|light). If not specified, all conditions are calibrated.')
    args = parser.parse_args()

    # Paths
    MASTER_CSV = './dataset_step4/step4_master_metadata.csv'
    TRACK_CSV = './dataset_step4/step4_track_aggregated.csv'
    OUT_DIR = './dataset_step4/condition_thresholds_v2'
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load data
    master = pd.read_csv(MASTER_CSV)
    tracks = pd.read_csv(TRACK_CSV)

    # Ensure key columns are consistent
    for df in [master, tracks]:
        df['video_filename'] = df['video_filename'].astype(str)
        df['track_id'] = df['track_id'].astype(int)

    # 1. Compute per-track median EMA score from master
    ema_medians = master.groupby(['video_filename', 'track_id'], as_index=False)['ema_score'].median()
    ema_medians.rename(columns={'ema_score': 'median_ema_score'}, inplace=True)

    # Merge into tracks
    tracks = tracks.merge(ema_medians, on=['video_filename', 'track_id'], how='left')
    if 'median_ema_score' not in tracks.columns:
        ema_dict = ema_medians.set_index(['video_filename', 'track_id'])['median_ema_score'].to_dict()
        tracks['median_ema_score'] = tracks.apply(
            lambda row: ema_dict.get((row['video_filename'], row['track_id']), np.nan), axis=1
        )

    # 2. Filter out vertically-clipped tracks
    clipped_tracks = master[master['vertical_clipped'] == 1][['video_filename', 'track_id']].drop_duplicates()
    clipped_tracks['clipped'] = True
    tracks = tracks.merge(clipped_tracks, on=['video_filename', 'track_id'], how='left')
    tracks['clipped'] = tracks['clipped'].fillna(False)
    tracks = tracks.infer_objects(copy=False)

    # 3. Calibration set: both calibration and controlled_validation splits
    calib_df = tracks[tracks['split'].isin(['calibration', 'controlled_validation'])].copy()

    # 4. Choose score column
    if args.score_type == 'raw':
        score_col = 'median_raw_score'
    else:
        score_col = 'median_ema_score'

    print(f"Using score: {score_col}")
    print(f"Percentile: {args.percentile}")
    print(f"Margin: {args.margin}x")
    print("="*70)

    # 5. For each condition in calibration, compute threshold
    thresholds = {}
    condition_details = {}
    all_conditions = calib_df.groupby(['background_id', 'lighting_id']).size().reset_index()
    all_conditions.columns = ['background_id', 'lighting_id', 'n_tracks']

    # Parse specific conditions if provided
    specific_conditions = []
    if args.conditions:
        for cond in args.conditions:
            parts = cond.split('|')
            if len(parts) == 2:
                specific_conditions.append((parts[0], parts[1]))

    for (bg, light), group in calib_df.groupby(['background_id', 'lighting_id']):
        # Skip if specific conditions are provided and this isn't in the list
        if specific_conditions and (bg, light) not in specific_conditions:
            continue

        clean = group[~group['clipped']].copy()
        if len(clean) < 30:
            print(f"Skipping condition {bg}-{light}: only {len(clean)} tracks")
            continue

        # Check if this is the problem condition (BG2_dark_textured)
        # Use a higher percentile for BG2_dark_textured
        if bg == 'BG2_dark_textured' and light == 'L0_normal':
            percentile = 99.9
            print(f"  {bg} / {light}: USING 99.9th PERCENTILE (problem condition)")
        else:
            percentile = args.percentile

        raw_thr = np.percentile(clean[score_col], percentile)
        # Apply margin
        thr = raw_thr * args.margin

        thresholds[(bg, light)] = thr
        condition_details[(bg, light)] = {
            'percentile': percentile,
            'margin': args.margin,
            'n_tracks': len(clean),
            'raw_percentile': raw_thr,
            'threshold': thr
        }
        print(f"{bg} / {light}: threshold = {thr:.4f} (p{percentile}={raw_thr:.4f} x {args.margin}) n={len(clean)}")

    # 6. Save thresholds to JSON
    with open(os.path.join(OUT_DIR, 'condition_thresholds.json'), 'w') as f:
        json.dump({f"{bg}|{light}": thr for (bg, light), thr in thresholds.items()}, f, indent=2)

    # 7. Save detailed summary
    summary = []
    for (bg, light), details in condition_details.items():
        summary.append({
            'background_id': bg,
            'lighting_id': light,
            'n_tracks': details['n_tracks'],
            'percentile': details['percentile'],
            'margin': details['margin'],
            'raw_percentile': details['raw_percentile'],
            'threshold': details['threshold']
        })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(OUT_DIR, 'calibration_summary.csv'), index=False)

    print("\n" + "="*70)
    print("Calibration Summary")
    print("="*70)
    print(summary_df.to_string(index=False))
    print(f"\nSaved thresholds to {os.path.join(OUT_DIR, 'condition_thresholds.json')}")
    print(f"Saved summary to {os.path.join(OUT_DIR, 'calibration_summary.csv')}")

if __name__ == '__main__':
    main()