# Research and experiment families

- **Baseline / optimized inspection:** `video_streaming_detector.py`, `video_streaming_detector_fast_v2.py`.
- **Lower-resolution EfficientAD:** `retrain_lowres.py`, `evaluate_lowres_pipeline.py`.
- **Width-reduced (Thin):** `retrain_thin.py`, `evaluate_thin_pipeline.py`.
- **Pruning:** `prune_finetune.py`, `evaluate_pruned_pipeline.py`.
- **Quantization-aware training:** `train_qat.py`, `evaluate_qat_pipeline.py`.
- **Temporal decisions and calibration:** `calibrate_ema_thresholds.py`, `analyze_lowres192_temporal_rules.py`.
- **YOLO comparison:** `compare_old_new_yolo_fair.py`, `compare_old_new_yolo_calibration.py`.
- **Background adaptation:** `build_multibg_fast_dataset_approved.py`, `train_lowres192_multibg_fast.sh`.
- **Inference speed:** `benchmark_variants_realtime_v2.py`, `benchmark_video_fps.py`.

## How to reproduce responsibly

1. Freeze model versions and calibration settings **before** using held-out test data.
2. Distinguish frame-, crop-, track- and physical-piece-level evaluation.
3. Record detector checkpoints, EfficientAD checkpoints, resize policy, crop backend and model input size.
4. Record EMA alpha, threshold, minimum track samples, inactivity and early streaks.
5. Do not compare FPS across mismatched hardware, precision, frame counts or preprocessing settings.

No datasets, predictions, results or weights are distributed here.
