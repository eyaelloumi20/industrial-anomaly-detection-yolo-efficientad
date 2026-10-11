# Export audit

Date: 2026-10-11
Python: 191 | Shell: 37 | actual source files: 228

Original scripts copied unchanged into `pipeline/`, `utils/`, and `scripts/`. Relative symlinks in `code/` maintain the original directory layout. Source workspace unchanged.

No media, datasets, checkpoints, evaluation results, JSON/CSV metrics or logs included.

## Flagged files for manual review

### possible hardcoded absolute paths — 107

- `compare_pipeline_methods_batched.py`
- `compare_pipeline_methods_batched1.py`
- `pipeline/single_photo_inference.py`
- `realtime_inference.py`
- `single_photo_inference_seg.py`
- `video_streaming_detector.py`
- `batch_image_processor_yolo.py`
- `export_normal_track_review.py`
- `make_final_ema_qualitative_track16.sh`
- `run_export_normal_track_review.sh`
- `run_oldyolo_final_unseen_lowres192.sh`
- `export_onnx.py`
- `retrain_lowres.py`
- `retrain_thin.py`
- `run_all_posttrain_multibg.sh`
- `run_fast_augmented_baseline_and_lowres.sh`
- `run_fast_augmented_baseline_and_lowres_before_skip.sh`
- `scripts/retrain_efficientad.py`
- `scripts/retrain_efficientad2.py`
- `scripts/train_yolo.py`
- `train_lowres192_multibg_fast.sh`
- `batch_image_processor.py`
- `build_gum_dataset.py`
- `build_multibg_fast_dataset_approved.py`
- `dump_all_frames.py`
- `extract_gum_pieces.py`
- `filter_crops.py`
- `generate_seg_labels.py`
- `instance_segmentation_yolo.py`
- `integrate_k78_mean_border.py`
- `label_dataset.py`
- `prepare_roboflow_adaptation.py`
- `run_crop_strategy_development.sh`
- `scripts/generate_csv.py`
- `scripts/generate_csv_video_split.py`
- `split_bg2_conditions.py`
- `split_dataset.py`
- `calibrate_baseline_ema.py`
- `calibrate_ema_thresholds.py`
- `calibrate_threshold_all.py`
- `calibrate_threshold_all1.py`
- `calibrate_threshold_all_clahe.py`
- `calibrate_threshold_all_fp16.py`
- `calibrate_threshold_all_og.py`
- `calibrate_thresholds.py`
- `calibrate_v3_k78_lowres192.py`
- `compare_old_new_yolo_calibration.py`
- `run_calibrate_v3_k78_lowres192.sh`
- `run_calibrated_variants.sh`
- `run_heldout_calibrated_eval.sh`
- `run_lowres192_calibration.sh`
- `run_newyolo_recalibrated_final_test.sh`
- `run_original_calibration_new_yolo.sh`
- `run_v3_annotated_with_threshold.sh`
- `runs/detect/runs/detect/yolo_runs/compare_old_new_yolo_calibration.py`
- `select_operating_thresholds.py`
- `video_threshold_calibration.py`
- `evaluate_heldout_scores.py`
- `evaluate_lowres_pipeline.py`
- `evaluate_pruned_pipeline.py`
- `evaluate_qat_pipeline.py`
- `evaluate_thin_pipeline.py`
- `pipeline/evaluate_live.py`
- `pipeline/evaluate_pipeline.py`
- `prepare_reserved_gt_newyolo.py`
- `profile_forward.py`
- `build_condition_reference.py`
- `combined_gum_inspection_pipeline_compare_all.py`
- `combined_gum_inspection_pipeline_compare_all_og.py`
- `compare_pipeline_methods.py`
- `compare_pipeline_methods_auto_conditional.py`
- `compare_pipeline_methods_clahe.py`
- `compare_pipeline_methods_reel.py`
- `compare_pipeline_methods_t.py`
- `compare_variants.py`
- `compare_variants_on_videos.py`
- `compare_variants_on_videos_fixed.py`
- `compare_variants_realtime.py`
- `compare_variants_realtime_v2.py`
- `run_current_crop_background_suppression.sh`
- `run_current_vs_k78_domain_shift.sh`
- `run_mean_suppression_grid.sh`
- `ckpt_cropped_real_4/live_compare_app.py`
- `live_compare_app.py`
- `run_v3_annotated_preview.sh`
- `inspect_multibg_fast.sh`
- `install_k78_mean_border.sh`
- `run_all_options.sh`
- `run_all_step4_videos.sh`
- `run_all_variants.sh`
- `run_all_videos.sh`
- `run_aug70k_lowres192_complete_test.sh`
- `run_complete_old_vs_new_comparison.sh`
- `run_FG.sh`
- `run_FG_min10.sh`
- `run_final_12_multibg_lowres192.sh`
- `run_final_unseen_lowres192.sh`
- `run_multibg_fast_approved.sh`
- `search_all_videos_for_fig22.sh`
- `diagnose_heldout_tracks.py`
- `smoke_test_k78.sh`
- `test_all_ema_alphas.sh`
- `test_all_ema_alphas_fixed.sh`
- `test_clipped_pick_single_frame.sh`
- `utils/test_crop_sizes.py`
- `utils/test_sizing.py`
- `c.py`

### potentially destructive commands — 10

- `make_final_ema_qualitative_track16.sh`
- `run_fast_augmented_baseline_and_lowres.sh`
- `run_fast_augmented_baseline_and_lowres_before_skip.sh`
- `train_lowres192_multibg_fast.sh`
- `run_aug70k_lowres192_complete_test.sh`
- `run_final_12_multibg_lowres192.sh`
- `search_all_videos_for_fig22.sh`
- `test_all_ema_alphas.sh`
- `test_all_ema_alphas_fixed.sh`
- `test_clipped_pick_single_frame.sh`

## Python AST check

All included `.py` files parsed, but runtime behavior was not tested.

## Needs categorization review (1)

- `c.py`
