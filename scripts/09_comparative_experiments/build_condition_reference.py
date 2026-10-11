# build_condition_reference.py
import pandas as pd
import numpy as np
import cv2
import os
import json
import glob
from sklearn.preprocessing import StandardScaler

# Paths
VIDEO_DIR = '/home/ahmed/two-stage-efficientad-inspection/validation_videos'
MANIFEST_PATH = 'manifest_33_only.csv'
OUTPUT_REFERENCE = './condition_reference.json'

# 1. Load manifest
manifest = pd.read_csv(MANIFEST_PATH)
print(f"Loaded manifest with {len(manifest)} videos")

# 2. Function to extract features from a video
def extract_video_features(video_path, n_frames=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return None
    
    # Sample frames evenly
    frame_indices = np.linspace(0, total_frames-1, min(n_frames, total_frames), dtype=int)
    
    features = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Features (same as before)
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = np.var(laplacian)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges > 0)
        contrast = np.max(gray) - np.min(gray)
        
        features.append([mean_brightness, std_brightness, texture_variance, edge_density, contrast])
    
    cap.release()
    
    if len(features) == 0:
        return None
    
    avg_features = np.mean(features, axis=0)
    return {
        'mean_brightness': avg_features[0],
        'std_brightness': avg_features[1],
        'texture_variance': avg_features[2],
        'edge_density': avg_features[3],
        'contrast': avg_features[4]
    }

# 3. Function to normalize filename for matching
def normalize_name(name):
    name = name.replace('.mp4', '')
    name = name.replace(' ', '_')
    name = name.replace('(', '_').replace(')', '_')
    while '__' in name:
        name = name.replace('__', '_')
    return name.rstrip('_')

# 4. Get all video files in the directory
actual_files = {}
for f in glob.glob(os.path.join(VIDEO_DIR, '*.mp4')):
    basename = os.path.basename(f)
    actual_files[basename] = f

# 5. Build feature database per condition
condition_features = {}
condition_videos = {}

for idx, row in manifest.iterrows():
    video_name = row['video_filename']
    norm = normalize_name(video_name)
    
    if norm not in actual_files:
        # Try to match with the normalized name
        # The actual file might have a slightly different format
        for actual_name, actual_path in actual_files.items():
            if normalize_name(actual_name) == norm:
                video_path = actual_path
                break
        else:
            print(f"Could not find video: {video_name}")
            continue
    else:
        video_path = actual_files[norm]
    
    # Get condition
    bg = row['background_id']
    light = row['lighting_id']
    condition = f"{bg}|{light}"
    
    print(f"Processing: {video_name} -> {condition}")
    
    features = extract_video_features(video_path)
    if features is None:
        continue
    
    if condition not in condition_features:
        condition_features[condition] = []
        condition_videos[condition] = []
    
    condition_features[condition].append([features[f] for f in ['mean_brightness', 'std_brightness', 'texture_variance', 'edge_density', 'contrast']])
    condition_videos[condition].append(video_name)

# 6. Compute mean features per condition
feature_names = ['mean_brightness', 'std_brightness', 'texture_variance', 'edge_density', 'contrast']
reference = {}

print("\n" + "="*70)
print("Condition Reference Features")
print("="*70)

for condition, features_list in condition_features.items():
    if len(features_list) < 2:
        print(f"Skipping {condition}: only {len(features_list)} videos")
        continue
    
    mean_features = np.mean(features_list, axis=0)
    std_features = np.std(features_list, axis=0)
    
    reference[condition] = {
        'mean': mean_features.tolist(),
        'std': std_features.tolist(),
        'n_videos': len(features_list),
        'videos': condition_videos[condition]
    }
    
    print(f"\n{condition} (n={len(features_list)}):")
    for name, val in zip(feature_names, mean_features):
        print(f"  {name}: {val:.2f}")

# 7. Save reference
with open(OUTPUT_REFERENCE, 'w') as f:
    json.dump(reference, f, indent=2)

print(f"\nSaved reference to {OUTPUT_REFERENCE}")
print(f"Conditions available: {list(reference.keys())}")