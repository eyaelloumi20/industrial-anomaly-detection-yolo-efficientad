# evaluate_validation_with_manifest.py
import pandas as pd
import json
import os
import glob

# 1. Load track medians
tracks_path = './results_validation/validation_track_medians.csv'
if not os.path.exists(tracks_path):
    print(f"Track medians file not found: {tracks_path}")
    print("Run extract_validation_medians.py first.")
    exit()

tracks = pd.read_csv(tracks_path)
print(f"Loaded {len(tracks)} tracks from medians file.")

# 2. Load manifest
manifest_path = 'manifest_33_only_bg2_split.csv'
if not os.path.exists(manifest_path):
    print(f"Manifest not found: {manifest_path}")
    exit()

manifest = pd.read_csv(manifest_path)
print(f"Loaded manifest with {len(manifest)} videos.")

# 3. Normalize filenames for matching
def normalize_for_match(name):
    name = name.replace('.mp4', '')
    name = name.replace(' ', '_')
    name = name.replace('(', '_').replace(')', '_')
    while '__' in name:
        name = name.replace('__', '_')
    name = name.rstrip('_')
    return name

manifest['folder_name'] = manifest['video_filename'].apply(normalize_for_match)
tracks['folder_name'] = tracks['video_filename'].apply(normalize_for_match)

# Merge on folder_name
tracks = tracks.merge(manifest, on='folder_name', how='inner')
print(f"Merged to {len(tracks)} tracks with condition information.")

if len(tracks) == 0:
    print("No tracks matched.")
    exit()

# Clean up column names after merge
if 'video_filename_x' in tracks.columns and 'video_filename_y' in tracks.columns:
    tracks = tracks.drop(columns=['video_filename_x'])
    tracks = tracks.rename(columns={'video_filename_y': 'video_filename'})

# 4. Exclude defective video
exclude_video = "WhatsApp Video 2026-09-08 at 21.42.07 (4).mp4"
tracks = tracks[tracks['video_filename'] != exclude_video]
print(f"After excluding defective video: {len(tracks)} tracks")

# 5. Load condition thresholds - USE THE NEW THRESHOLDS
thr_path = './dataset_step4/condition_thresholds_v2/condition_thresholds.json'
if not os.path.exists(thr_path):
    print(f"Threshold file not found: {thr_path}")
    print("Looking for new thresholds from condition_specific_calibration.py")
    # Fallback to old thresholds
    thr_path = './dataset_step4/condition_thresholds/condition_thresholds.json'
    if not os.path.exists(thr_path):
        print("No threshold file found.")
        exit()

with open(thr_path, 'r') as f:
    thresholds = json.load(f)

# Add threshold column
tracks['condition_key'] = tracks['background_id'] + '|' + tracks['lighting_id']
tracks['threshold_cond'] = tracks['condition_key'].map(thresholds)

# 6. Compute condition-specific FPR for known conditions
known = tracks[tracks['threshold_cond'].notna()].copy()
known['anomaly_cond'] = known['median_ema_score'] >= known['threshold_cond']

if len(known) > 0:
    fpr_cond = known.groupby(['background_id', 'lighting_id']).agg(
        total=('anomaly_cond', 'count'),
        anomalous=('anomaly_cond', 'sum')
    ).reset_index()
    fpr_cond['fpr'] = fpr_cond['anomalous'] / fpr_cond['total']
    print("\n" + "="*70)
    print("Condition-specific FPR (known conditions) - WITH NEW THRESHOLDS")
    print("="*70)
    print(fpr_cond.to_string(index=False))
else:
    print("\nNo tracks with known condition thresholds.")

# 7. Global baseline threshold (5.2668) for comparison
global_threshold = 5.2668
tracks['anomaly_global'] = tracks['median_ema_score'] >= global_threshold

fpr_global = tracks.groupby(['background_id', 'lighting_id']).agg(
    total=('anomaly_global', 'count'),
    anomalous=('anomaly_global', 'sum')
).reset_index()
fpr_global['fpr_global'] = fpr_global['anomalous'] / fpr_global['total']
print("\n" + "="*70)
print("Global threshold FPR (5.2668) per condition (including new)")
print("="*70)
print(fpr_global.to_string(index=False))

# 8. Overall FPR
if len(known) > 0:
    overall_cond = known['anomaly_cond'].sum() / len(known)
    print(f"\nOverall FPR (condition-specific): {overall_cond:.2%}  (on {len(known)} tracks)")
else:
    overall_cond = None

overall_global = tracks['anomaly_global'].sum() / len(tracks)
print(f"Overall FPR (global threshold): {overall_global:.2%}  (on {len(tracks)} tracks)")

# 9. Per-video FPR for known conditions
if len(known) > 0:
    video_fpr = known.groupby(['video_filename', 'background_id', 'lighting_id']).agg(
        total=('anomaly_cond', 'count'),
        anomalous=('anomaly_cond', 'sum'),
        threshold=('threshold_cond', 'first')
    ).reset_index()
    video_fpr['fpr'] = video_fpr['anomalous'] / video_fpr['total']
    print("\n" + "="*70)
    print("Per-video FPR (condition-specific) - WITH NEW THRESHOLDS")
    print("="*70)
    print(video_fpr.to_string(index=False))

# 10. Per-video FPR with global threshold
video_global = tracks.groupby(['video_filename', 'background_id', 'lighting_id']).agg(
    total=('anomaly_global', 'count'),
    anomalous=('anomaly_global', 'sum')
).reset_index()
video_global['fpr_global'] = video_global['anomalous'] / video_global['total']
print("\n" + "="*70)
print("Per-video FPR (global threshold)")
print("="*70)
print(video_global.to_string(index=False))