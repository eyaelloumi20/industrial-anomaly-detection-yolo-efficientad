# split_bg2_conditions.py
import pandas as pd
import numpy as np
import cv2
import os
import glob
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import re

# Paths
VIDEO_DIR = '/home/ahmed/two-stage-efficientad-inspection/validation_videos'
MANIFEST_PATH = 'manifest_33_only.csv'
OUTPUT_MANIFEST = 'manifest_33_only_bg2_split.csv'

# 1. Load manifest
manifest = pd.read_csv(MANIFEST_PATH)
print(f"Loaded manifest with {len(manifest)} videos")

# 2. Get all actual video files in the directory
actual_files = {}
for f in glob.glob(os.path.join(VIDEO_DIR, '*.mp4')):
    basename = os.path.basename(f)
    actual_files[basename] = f

print(f"Found {len(actual_files)} videos in {VIDEO_DIR}")

# 3. Function to normalize filename for matching
def normalize_name(name):
    # Remove .mp4 extension
    name = name.replace('.mp4', '')
    # Replace spaces with underscores
    name = name.replace(' ', '_')
    # Replace parentheses with underscores
    name = name.replace('(', '_').replace(')', '_')
    # Remove duplicate underscores
    while '__' in name:
        name = name.replace('__', '_')
    name = name.rstrip('_')
    return name

# 4. Get BG2_dark_textured videos and match them to actual files
bg2_videos = manifest[manifest['background_id'] == 'BG2_dark_textured']
print(f"\nFound {len(bg2_videos)} BG2_dark_textured videos in manifest")

# Build mapping from normalized name to actual file
normalized_to_actual = {}
for actual_name, actual_path in actual_files.items():
    norm = normalize_name(actual_name)
    normalized_to_actual[norm] = actual_path

# Match manifest entries to actual files
matched_videos = []
for idx, row in bg2_videos.iterrows():
    video_name = row['video_filename']
    norm = normalize_name(video_name)
    
    if norm in normalized_to_actual:
        matched_videos.append({
            'video_filename': video_name,
            'actual_path': normalized_to_actual[norm],
            'actual_name': os.path.basename(normalized_to_actual[norm]),
            'background_id': row['background_id'],
            'lighting_id': row['lighting_id']
        })
        print(f"  Matched: {video_name} -> {os.path.basename(normalized_to_actual[norm])}")
    else:
        print(f"  No match found for: {video_name}")

print(f"\nMatched {len(matched_videos)} out of {len(bg2_videos)} BG2 videos")

if len(matched_videos) < 5:
    print("\nNot enough BG2 videos matched. Showing actual files in directory:")
    for f in sorted(actual_files.keys())[:20]:
        print(f"  {f}")
    exit()

# 5. Extract features from each matched video
video_features = {}
for video_info in matched_videos:
    video_path = video_info['actual_path']
    video_name = video_info['video_filename']
    
    print(f"Processing: {video_name}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Could not open {video_path}")
        continue
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        continue
    
    # Sample frames evenly (max 10)
    n_frames = min(10, total_frames)
    frame_indices = np.linspace(0, total_frames-1, n_frames, dtype=int)
    
    features = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Features
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
        continue
    
    avg_features = np.mean(features, axis=0)
    video_features[video_name] = {
        'mean_brightness': avg_features[0],
        'std_brightness': avg_features[1],
        'texture_variance': avg_features[2],
        'edge_density': avg_features[3],
        'contrast': avg_features[4]
    }

print(f"\nExtracted features for {len(video_features)} videos")

if len(video_features) < 3:
    print("Not enough videos with features extracted. Exiting.")
    exit()

# 6. Create feature matrix for clustering
feature_names = ['mean_brightness', 'std_brightness', 'texture_variance', 'edge_density', 'contrast']
feature_matrix = []
video_names = []

for video_name, features in video_features.items():
    feature_matrix.append([features[f] for f in feature_names])
    video_names.append(video_name)

feature_matrix = np.array(feature_matrix)
scaler = StandardScaler()
feature_matrix_scaled = scaler.fit_transform(feature_matrix)

# 7. Cluster into 2 groups
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(feature_matrix_scaled)

# 8. Display clusters
print(f"\nClustering into {n_clusters} groups:")
cluster_assignments = {}
for video_name, label in zip(video_names, labels):
    cluster_assignments[video_name] = label
    print(f"  {video_name} -> Cluster {label}")

# 9. Show feature differences
cluster_features = {0: [], 1: []}
for video_name, label in cluster_assignments.items():
    features = video_features[video_name]
    cluster_features[label].append([features[f] for f in feature_names])

print("\nFeature means per cluster:")
for label in [0, 1]:
    if cluster_features[label]:
        mean_feat = np.mean(cluster_features[label], axis=0)
        print(f"\nCluster {label} ({len(cluster_features[label])} videos):")
        for name, val in zip(feature_names, mean_feat):
            print(f"  {name}: {val:.2f}")

# 10. Assign sub-conditions
# Cluster with higher texture_variance is "A" (problematic), lower is "B" (good)
cluster_a = 0
cluster_b = 1
if np.mean([video_features[v]['texture_variance'] for v, l in cluster_assignments.items() if l == 1]) > \
   np.mean([video_features[v]['texture_variance'] for v, l in cluster_assignments.items() if l == 0]):
    cluster_a = 1
    cluster_b = 0

print(f"\nAssigning Cluster {cluster_a} -> BG2_dark_textured_A (problematic, higher texture)")
print(f"Assigning Cluster {cluster_b} -> BG2_dark_textured_B (good, lower texture)")

# 11. Update manifest
manifest_updated = manifest.copy()
for video_name, label in cluster_assignments.items():
    if label == cluster_a:
        new_bg = 'BG2_dark_textured_A'
    else:
        new_bg = 'BG2_dark_textured_B'
    manifest_updated.loc[manifest_updated['video_filename'] == video_name, 'background_id'] = new_bg
    print(f"  {video_name} -> {new_bg}")

# 12. Save updated manifest
manifest_updated.to_csv(OUTPUT_MANIFEST, index=False)
print(f"\nSaved updated manifest to {OUTPUT_MANIFEST}")

# 13. Print summary
print("\n" + "="*70)
print("Summary of BG2 split:")
print("="*70)
bg2_updated = manifest_updated[manifest_updated['background_id'].str.startswith('BG2_dark_textured')]
print(bg2_updated[['video_filename', 'background_id']].to_string(index=False))

# 14. Show which videos need manual review (unmatched)
unmatched = [v for v in bg2_videos['video_filename'].tolist() if v not in cluster_assignments]
if unmatched:
    print("\nUnmatched videos (need manual review):")
    for v in unmatched:
        print(f"  {v}")