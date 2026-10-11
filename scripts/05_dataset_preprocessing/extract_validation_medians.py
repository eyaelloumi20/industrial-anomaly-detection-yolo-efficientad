# extract_validation_medians.py
# Extracts per-track median scores from the validation run's scores_baseline.csv files.

import pandas as pd
import os
import glob

results_dir = './results_validation'
pattern = os.path.join(results_dir, '*', 'baseline', 'scores_baseline.csv')
csv_files = glob.glob(pattern)

if not csv_files:
    print("No scores_baseline.csv files found. Did you run the inference with --save_dir ./results_validation?")
    exit()

all_tracks = []
for csv_path in csv_files:
    # Skip if file is empty
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        print(f"Skipping empty/missing: {csv_path}")
        continue
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        print(f"Skipping empty CSV: {csv_path}")
        continue
    # Extract video name from folder structure: ./results_validation/video_name/baseline/scores_baseline.csv
    parts = csv_path.split(os.sep)
    video_name = parts[-3]  # assuming the structure above
    print(f"Processing: {video_name} ({len(df)} detections)")
    medians = df.groupby('track_id').agg(
        median_raw_score=('raw_score', 'median'),
        median_ema_score=('ema_score', 'median'),
        n_frames=('frame', 'count')
    ).reset_index()
    medians['video_filename'] = video_name
    all_tracks.append(medians)

if all_tracks:
    all_tracks_df = pd.concat(all_tracks, ignore_index=True)
    output_path = './results_validation/validation_track_medians.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    all_tracks_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(all_tracks_df)} tracks to {output_path}")
    print("\nFirst few rows:")
    print(all_tracks_df.head())
else:
    print("No track medians extracted. Check that scores_baseline.csv files exist and contain data.")