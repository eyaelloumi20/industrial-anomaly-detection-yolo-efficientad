# calibrate_bg2_only.py
import pandas as pd
import numpy as np
import json
import os

# Load the existing thresholds
with open('./dataset_step4/condition_thresholds_v2/condition_thresholds.json', 'r') as f:
    thresholds = json.load(f)

# Load data
MASTER_CSV = './dataset_step4/step4_master_metadata.csv'
TRACK_CSV = './dataset_step4/step4_track_aggregated.csv'

master = pd.read_csv(MASTER_CSV)
tracks = pd.read_csv(TRACK_CSV)

# Ensure consistent types
for df in [master, tracks]:
    df['video_filename'] = df['video_filename'].astype(str)
    df['track_id'] = df['track_id'].astype(int)

# Compute per-track median EMA score directly from master
ema_medians = master.groupby(['video_filename', 'track_id'], as_index=False)['ema_score'].median()
ema_medians.rename(columns={'ema_score': 'median_ema_score'}, inplace=True)

# Merge into tracks
tracks = tracks.merge(ema_medians, on=['video_filename', 'track_id'], how='left')

# If merge failed, compute it manually
if 'median_ema_score' not in tracks.columns:
    print("Merge failed, computing manually...")
    ema_dict = ema_medians.set_index(['video_filename', 'track_id'])['median_ema_score'].to_dict()
    tracks['median_ema_score'] = tracks.apply(
        lambda row: ema_dict.get((row['video_filename'], row['track_id']), np.nan), axis=1
    )

# Filter clipped tracks
clipped_tracks = master[master['vertical_clipped'] == 1][['video_filename', 'track_id']].drop_duplicates()
clipped_tracks['clipped'] = True
tracks = tracks.merge(clipped_tracks, on=['video_filename', 'track_id'], how='left')
tracks['clipped'] = tracks['clipped'].fillna(False)

# Calibration set
calib_df = tracks[tracks['split'].isin(['calibration', 'controlled_validation'])].copy()

# Get BG2_dark_textured tracks
bg2 = calib_df[(calib_df['background_id'] == 'BG2_dark_textured') & 
               (calib_df['lighting_id'] == 'L0_normal')]
bg2_clean = bg2[~bg2['clipped']].copy()

print(f"BG2_dark_textured tracks: {len(bg2_clean)}")

if len(bg2_clean) == 0:
    print("No BG2_dark_textured tracks found. Checking available conditions...")
    print(calib_df[['background_id', 'lighting_id']].drop_duplicates().to_string(index=False))
    exit()

# Try different percentiles
print("\nPercentile options for BG2_dark_textured:")
for p in [99.9, 99.95, 99.99, 99.995, 99.999]:
    raw_thr = np.percentile(bg2_clean['median_ema_score'], p)
    thr = raw_thr * 1.05
    print(f"  p{p:.3f}: raw={raw_thr:.4f}, with margin=1.05: {thr:.4f}")

# Use 99.99th percentile with 1.05 margin
selected_p = 99.99
raw_thr = np.percentile(bg2_clean['median_ema_score'], selected_p)
new_thr = raw_thr * 1.05

print(f"\nUpdating BG2_dark_textured|L0_normal to {new_thr:.4f}")
print(f"  (was {thresholds.get('BG2_dark_textured|L0_normal', 'not set')})")

thresholds['BG2_dark_textured|L0_normal'] = new_thr

# Save updated thresholds
output_path = './dataset_step4/condition_thresholds_v2/condition_thresholds_bg2_fixed.json'
with open(output_path, 'w') as f:
    json.dump(thresholds, f, indent=2)

print(f"\nSaved to {output_path}")

# Show all thresholds for comparison
print("\nAll thresholds:")
for key, val in thresholds.items():
    print(f"  {key}: {val:.4f}")