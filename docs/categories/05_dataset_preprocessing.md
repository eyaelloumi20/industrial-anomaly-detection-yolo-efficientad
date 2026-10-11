# 05 Dataset Preprocessing

Crop extraction, segmentation, data preparation

| Script | Original path | Description | Classification reason |
|---|---|---|---|
| [`batch_image_processor.py`](../../scripts/05_dataset_preprocessing/batch_image_processor.py) | `batch_image_processor.py` | Batch Image Processor | Preprocessing helper |
| [`build_gum_dataset.py`](../../scripts/05_dataset_preprocessing/build_gum_dataset.py) | `build_gum_dataset.py` | build_gum_dataset.py | Dataset / crop / segmentation preprocessing |
| [`build_multibg_fast_dataset_approved.py`](../../scripts/05_dataset_preprocessing/build_multibg_fast_dataset_approved.py) | `build_multibg_fast_dataset_approved.py` | Fast multi-background dataset builder for LowRes192. | Dataset / crop / segmentation preprocessing |
| [`build_step4_dataset.py`](../../scripts/05_dataset_preprocessing/build_step4_dataset.py) | `build_step4_dataset.py` | build_step4_dataset.py | Dataset / crop / segmentation preprocessing |
| [`clahe_normalize_crops.py`](../../scripts/05_dataset_preprocessing/clahe_normalize_crops.py) | `clahe_normalize_crops.py` | clahe_normalize_crops.py | Dataset / crop / segmentation preprocessing |
| [`crop_dataset_video.py`](../../scripts/05_dataset_preprocessing/crop_dataset_video.py) | `crop_dataset_video.py` | crop_dataset_video.py | Dataset / crop / segmentation preprocessing |
| [`crop_dataset_video1.py`](../../scripts/05_dataset_preprocessing/crop_dataset_video1.py) | `crop_dataset_video1.py` | crop_dataset_video.py | Dataset / crop / segmentation preprocessing |
| [`crop_strategy_development.py`](../../scripts/05_dataset_preprocessing/crop_strategy_development.py) | `crop_strategy_development.py` | crop_strategy_development.py | Dataset / crop / segmentation preprocessing |
| [`dump_all_frames.py`](../../scripts/05_dataset_preprocessing/dump_all_frames.py) | `dump_all_frames.py` | dump_all_frames.py | Dataset / crop / segmentation preprocessing |
| [`extract_gum_pieces.py`](../../scripts/05_dataset_preprocessing/extract_gum_pieces.py) | `extract_gum_pieces.py` | extract_gum_pieces.py | Dataset / crop / segmentation preprocessing |
| [`extract_raw_256_batch.py`](../../scripts/05_dataset_preprocessing/extract_raw_256_batch.py) | `extract_raw_256_batch.py` | extract_raw_256_batch.py | Dataset / crop / segmentation preprocessing |
| [`extract_sam2_yolo.py`](../../scripts/05_dataset_preprocessing/extract_sam2_yolo.py) | `extract_sam2_yolo.py` | extract_sam2_yolo.py | Dataset / crop / segmentation preprocessing |
| [`extract_segments.py`](../../scripts/05_dataset_preprocessing/extract_segments.py) | `extract_segments.py` | extract_segments.py | Dataset / crop / segmentation preprocessing |
| [`extract_segments_images.py`](../../scripts/05_dataset_preprocessing/extract_segments_images.py) | `extract_segments_images.py` | extract_segments_images.py | Dataset / crop / segmentation preprocessing |
| [`extract_used_video_frames.py`](../../scripts/05_dataset_preprocessing/extract_used_video_frames.py) | `extract_used_video_frames.py` | from pathlib import Path | Dataset / crop / segmentation preprocessing |
| [`extract_validation_medians.py`](../../scripts/05_dataset_preprocessing/extract_validation_medians.py) | `extract_validation_medians.py` | extract_validation_medians.py | Dataset / crop / segmentation preprocessing |
| [`extract_yolo_grabcut.py`](../../scripts/05_dataset_preprocessing/extract_yolo_grabcut.py) | `extract_yolo_grabcut.py` | extract_yolo_grabcut.py | Dataset / crop / segmentation preprocessing |
| [`filter_crops.py`](../../scripts/05_dataset_preprocessing/filter_crops.py) | `filter_crops.py` | filter_crops.py | Dataset / crop / segmentation preprocessing |
| [`filter_quality_crops.py`](../../scripts/05_dataset_preprocessing/filter_quality_crops.py) | `filter_quality_crops.py` | filter_quality_crops.py | Dataset / crop / segmentation preprocessing |
| [`generate_seg_labels.py`](../../scripts/05_dataset_preprocessing/generate_seg_labels.py) | `generate_seg_labels.py` | generate_seg_labels.py | Dataset / crop / segmentation preprocessing |
| [`instance_segmentation_yolo.py`](../../scripts/05_dataset_preprocessing/instance_segmentation_yolo.py) | `instance_segmentation_yolo.py` | Instance Segmentation with YOLO | Dataset / crop / segmentation preprocessing |
| [`integrate_k78_mean_border.py`](../../scripts/05_dataset_preprocessing/integrate_k78_mean_border.py) | `integrate_k78_mean_border.py` | from pathlib import Path | Preprocessing helper |
| [`label_dataset.py`](../../scripts/05_dataset_preprocessing/label_dataset.py) | `label_dataset.py` | label_dataset.py  (VS Code / local version) | Dataset / crop / segmentation preprocessing |
| [`label_reserved_tracks.py`](../../scripts/05_dataset_preprocessing/label_reserved_tracks.py) | `label_reserved_tracks.py` | from pathlib import Path | Dataset / crop / segmentation preprocessing |
| [`label_tracks.py`](../../scripts/05_dataset_preprocessing/label_tracks.py) | `label_tracks.py` | Interactive manual labeling of YOLO/ByteTrack proposals. | Dataset / crop / segmentation preprocessing |
| [`prelabel_tracks_yolo.py`](../../scripts/05_dataset_preprocessing/prelabel_tracks_yolo.py) | `prelabel_tracks_yolo.py` | YOLO + ByteTrack assisted ground-truth preparation. | Dataset / crop / segmentation preprocessing |
| [`prepare_roboflow_adaptation.py`](../../scripts/05_dataset_preprocessing/prepare_roboflow_adaptation.py) | `prepare_roboflow_adaptation.py` | Prepare 240 YOLOv8 images for fast correction in Roboflow. | Dataset / crop / segmentation preprocessing |
| [`resize_dataset.py`](../../scripts/05_dataset_preprocessing/resize_dataset.py) | `resize_dataset.py` | resize_dataset.py | Dataset / crop / segmentation preprocessing |
| [`run_crop_strategy_development.sh`](../../scripts/05_dataset_preprocessing/run_crop_strategy_development.sh) | `run_crop_strategy_development.sh` | cd /home/ahmed/two-stage-efficientad-inspection | Dataset / crop / segmentation preprocessing |
| [`crop_dataset.py`](../../scripts/05_dataset_preprocessing/scripts/crop_dataset.py) | `scripts/crop_dataset.py` | crop_dataset.py | Dataset / crop / segmentation preprocessing |
| [`generate_csv.py`](../../scripts/05_dataset_preprocessing/scripts/generate_csv.py) | `scripts/generate_csv.py` | generate_csv.py  –  CSV generator for EfficientAD retraining | Dataset / crop / segmentation preprocessing |
| [`generate_csv_video_split.py`](../../scripts/05_dataset_preprocessing/scripts/generate_csv_video_split.py) | `scripts/generate_csv_video_split.py` | generate_csv_video_split.py | Dataset / crop / segmentation preprocessing |
| [`split_bg2_conditions.py`](../../scripts/05_dataset_preprocessing/split_bg2_conditions.py) | `split_bg2_conditions.py` | split_bg2_conditions.py | Dataset / crop / segmentation preprocessing |
| [`split_dataset.py`](../../scripts/05_dataset_preprocessing/split_dataset.py) | `split_dataset.py` | Split a flat YOLO export (images/ + labels/ + classes.txt) into train/val. | Dataset / crop / segmentation preprocessing |
