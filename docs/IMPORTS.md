# Original-path local imports

Local imports are identified using Python AST only; dynamic imports may be missed.

| Original script | Category | Local imports | CLI main guard |
|---|---|---|---|
| `combined_gum_inspection_pipeline.py` | `01_inspection_pipelines` | `utils.load_efficientad` | yes |
| `combined_gum_inspection_pipeline1.py` | `01_inspection_pipelines` | `utils.load_efficientad` | yes |
| `combined_gum_inspection_pipeline3.py` | `01_inspection_pipelines` | `utils.load_efficientad` | yes |
| `combined_gum_inspection_pipeline_fixed.py` | `01_inspection_pipelines` | `utils.load_efficientad_adaptive` | yes |
| `combined_gum_inspection_pipeline_thin.py` | `01_inspection_pipelines` | `utils.load_efficientad_thin` | yes |
| `compare_pipeline_methods_batched.py` | `01_inspection_pipelines` | `utils.load_efficientad_adaptive, utils.load_efficientad_pruned, utils.load_efficientad_qat, utils.load_efficientad_thin` | yes |
| `compare_pipeline_methods_batched1.py` | `01_inspection_pipelines` | `utils.load_efficientad_adaptive, utils.load_efficientad_pruned, utils.load_efficientad_qat, utils.load_efficientad_thin` | yes |
| `pipeline/pipeline.py` | `01_inspection_pipelines` | `—` | no |
| `pipeline/pipeline1.py` | `01_inspection_pipelines` | `—` | no |
| `pipeline/pipeline_adaptive.py` | `01_inspection_pipelines` | `—` | no |
| `pipeline/single_photo_inference.py` | `01_inspection_pipelines` | `pipeline.pipeline, utils.detection, utils.efficientad_inference, utils.load_efficientad` | no |
| `pipeline/video_inference.py` | `01_inspection_pipelines` | `utils.load_efficientad` | yes |
| `pipeline/video_inference1.py` | `01_inspection_pipelines` | `utils.load_efficientad` | yes |
| `pipeline/video_inferenceNB.py` | `01_inspection_pipelines` | `utils.load_efficientad` | yes |
| `pipeline/video_inferenceNB1.py` | `01_inspection_pipelines` | `utils.load_efficientad` | yes |
| `realtime_inference.py` | `01_inspection_pipelines` | `—` | yes |
| `realtime_inspector.py` | `01_inspection_pipelines` | `compare_pipeline_methods_batched, utils.load_efficientad, utils.models` | yes |
| `run_inference.py` | `01_inspection_pipelines` | `—` | yes |
| `single_photo_inference_seg.py` | `01_inspection_pipelines` | `utils.efficientad_inference, utils.load_efficientad` | yes |
| `video_inference.py` | `01_inspection_pipelines` | `utils.load_efficientad` | yes |
| `video_inference_OG.py` | `01_inspection_pipelines` | `utils.load_efficientad` | yes |
| `video_inference_seg.py` | `01_inspection_pipelines` | `utils.load_efficientad` | yes |
| `video_inference_simple.py` | `01_inspection_pipelines` | `utils.load_efficientad` | yes |
| `video_inference_simple1.py` | `01_inspection_pipelines` | `utils.load_efficientad` | yes |
| `video_inferenceBG.py` | `01_inspection_pipelines` | `utils.load_efficientad` | yes |
| `video_inferenceNB1_noAD.py` | `01_inspection_pipelines` | `—` | yes |
| `video_inferenceNB_fixed.py` | `01_inspection_pipelines` | `utils.load_efficientad` | yes |
| `video_inferenceNB_noAD.py` | `01_inspection_pipelines` | `—` | yes |
| `video_streaming_detector.py` | `01_inspection_pipelines` | `compare_pipeline_methods_batched, utils.models` | yes |
| `video_streaming_detector1.py` | `01_inspection_pipelines` | `compare_pipeline_methods_batched, utils.models` | yes |
| `video_streaming_detector_fast_early_final.py` | `01_inspection_pipelines` | `compare_pipeline_methods_batched, utils.models` | yes |
| `video_streaming_detector_fast_v2.py` | `01_inspection_pipelines` | `compare_pipeline_methods_batched, utils.models` | yes |
| `video_streaming_detector_optimized.py` | `01_inspection_pipelines` | `compare_pipeline_methods_batched, utils.models, video_streaming_detector1` | yes |
| `video_streaming_detector_optimized1.py` | `01_inspection_pipelines` | `compare_pipeline_methods_batched, utils.models, video_streaming_detector1` | yes |
| `video_streaming_detector_optimized2.py` | `01_inspection_pipelines` | `compare_pipeline_methods_batched, utils.models, video_streaming_detector1` | yes |
| `video_streaming_detector_optimized3.py` | `01_inspection_pipelines` | `compare_pipeline_methods_batched, utils.models, video_streaming_detector1` | yes |
| `video_streaming_detector_optimized4.py` | `01_inspection_pipelines` | `compare_pipeline_methods_batched, utils.models` | yes |
| `video_streaming_detector_simple_optimized.py` | `01_inspection_pipelines` | `compare_pipeline_methods_batched, utils.models` | yes |
| `video_streaming_detector_simple_optimized_v2.py` | `01_inspection_pipelines` | `compare_pipeline_methods_batched, utils.models` | yes |
| `video_streaming_detector_simple_optimized_v3.py` | `01_inspection_pipelines` | `compare_pipeline_methods_batched, utils.models` | yes |
| `batch_image_processor_yolo.py` | `02_detection_tracking` | `—` | yes |
| `export_normal_track_review.py` | `02_detection_tracking` | `compare_pipeline_methods_batched` | yes |
| `utils/detection.py` | `02_detection_tracking` | `—` | no |
| `models_adaptive.py` | `03_efficientad_architectures` | `utils.models` | yes |
| `models_thin.py` | `03_efficientad_architectures` | `utils.models` | yes |
| `utils/efficientad_inference.py` | `03_efficientad_architectures` | `—` | no |
| `utils/load_efficientad.py` | `03_efficientad_architectures` | `—` | yes |
| `utils/load_efficientad_adaptive.py` | `03_efficientad_architectures` | `models_adaptive` | yes |
| `utils/load_efficientad_pruned.py` | `03_efficientad_architectures` | `—` | yes |
| `utils/load_efficientad_qat.py` | `03_efficientad_architectures` | `train_qat` | yes |
| `utils/load_efficientad_thin.py` | `03_efficientad_architectures` | `models_thin` | yes |
| `utils/models.py` | `03_efficientad_architectures` | `—` | yes |
| `utils/models_fixed.py` | `03_efficientad_architectures` | `—` | yes |
| `export_onnx.py` | `04_training_optimization` | `compare_pipeline_methods_batched` | yes |
| `prune_finetune.py` | `04_training_optimization` | `utils.data_loader, utils.models` | yes |
| `retrain_lowres.py` | `04_training_optimization` | `models_adaptive, utils.data_loader, utils.models` | yes |
| `retrain_thin.py` | `04_training_optimization` | `models_thin, utils.data_loader, utils.models` | yes |
| `scripts/retrain_efficientad.py` | `04_training_optimization` | `utils.data_loader, utils.models` | yes |
| `scripts/retrain_efficientad2.py` | `04_training_optimization` | `utils.data_loader, utils.models` | yes |
| `scripts/train_yolo.py` | `04_training_optimization` | `—` | yes |
| `scripts/train_yolo11.py` | `04_training_optimization` | `—` | no |
| `train_qat.py` | `04_training_optimization` | `utils.data_loader, utils.models_fixed` | yes |
| `train_yolov8_seg.py` | `04_training_optimization` | `—` | yes |
| `batch_image_processor.py` | `05_dataset_preprocessing` | `—` | yes |
| `build_gum_dataset.py` | `05_dataset_preprocessing` | `pipeline.pipeline, utils.load_efficientad` | yes |
| `build_multibg_fast_dataset_approved.py` | `05_dataset_preprocessing` | `compare_pipeline_methods_batched` | yes |
| `build_step4_dataset.py` | `05_dataset_preprocessing` | `—` | yes |
| `clahe_normalize_crops.py` | `05_dataset_preprocessing` | `—` | yes |
| `crop_dataset_video.py` | `05_dataset_preprocessing` | `—` | yes |
| `crop_dataset_video1.py` | `05_dataset_preprocessing` | `—` | yes |
| `crop_strategy_development.py` | `05_dataset_preprocessing` | `video_streaming_detector_simple_optimized_v2` | yes |
| `dump_all_frames.py` | `05_dataset_preprocessing` | `compare_pipeline_methods_batched, utils.models` | yes |
| `extract_gum_pieces.py` | `05_dataset_preprocessing` | `—` | yes |
| `extract_raw_256_batch.py` | `05_dataset_preprocessing` | `—` | yes |
| `extract_sam2_yolo.py` | `05_dataset_preprocessing` | `—` | yes |
| `extract_segments.py` | `05_dataset_preprocessing` | `—` | yes |
| `extract_segments_images.py` | `05_dataset_preprocessing` | `—` | yes |
| `extract_used_video_frames.py` | `05_dataset_preprocessing` | `—` | yes |
| `extract_validation_medians.py` | `05_dataset_preprocessing` | `—` | no |
| `extract_yolo_grabcut.py` | `05_dataset_preprocessing` | `—` | yes |
| `filter_crops.py` | `05_dataset_preprocessing` | `compare_pipeline_methods_batched, utils.models` | yes |
| `filter_quality_crops.py` | `05_dataset_preprocessing` | `—` | yes |
| `generate_seg_labels.py` | `05_dataset_preprocessing` | `—` | yes |
| `instance_segmentation_yolo.py` | `05_dataset_preprocessing` | `—` | no |
| `integrate_k78_mean_border.py` | `05_dataset_preprocessing` | `—` | no |
| `label_dataset.py` | `05_dataset_preprocessing` | `—` | no |
| `label_reserved_tracks.py` | `05_dataset_preprocessing` | `—` | yes |
| `label_tracks.py` | `05_dataset_preprocessing` | `—` | yes |
| `prelabel_tracks_yolo.py` | `05_dataset_preprocessing` | `—` | yes |
| `prepare_roboflow_adaptation.py` | `05_dataset_preprocessing` | `—` | yes |
| `resize_dataset.py` | `05_dataset_preprocessing` | `—` | yes |
| `scripts/crop_dataset.py` | `05_dataset_preprocessing` | `—` | yes |
| `scripts/generate_csv.py` | `05_dataset_preprocessing` | `—` | no |
| `scripts/generate_csv_video_split.py` | `05_dataset_preprocessing` | `—` | yes |
| `split_bg2_conditions.py` | `05_dataset_preprocessing` | `—` | no |
| `split_dataset.py` | `05_dataset_preprocessing` | `—` | yes |
| `analyze_lowres192_temporal_rules.py` | `06_calibration_temporal` | `—` | yes |
| `analyze_score_percentiles.py` | `06_calibration_temporal` | `—` | yes |
| `calibrate_baseline_ema.py` | `06_calibration_temporal` | `—` | yes |
| `calibrate_bg2_only.py` | `06_calibration_temporal` | `—` | no |
| `calibrate_efficientad_variants_realtime.py` | `06_calibration_temporal` | `compare_pipeline_methods_batched, video_streaming_detector_optimized3` | yes |
| `calibrate_ema_thresholds.py` | `06_calibration_temporal` | `—` | yes |
| `calibrate_lowres192_threshold.py` | `06_calibration_temporal` | `—` | yes |
| `calibrate_lowres192_threshold_fixed.py` | `06_calibration_temporal` | `—` | yes |
| `calibrate_threshold.py` | `06_calibration_temporal` | `utils.load_efficientad` | yes |
| `calibrate_threshold1.py` | `06_calibration_temporal` | `utils.load_efficientad` | yes |
| `calibrate_threshold2.py` | `06_calibration_temporal` | `utils.load_efficientad` | yes |
| `calibrate_threshold3.py` | `06_calibration_temporal` | `utils.load_efficientad` | yes |
| `calibrate_threshold_all.py` | `06_calibration_temporal` | `utils.load_efficientad_adaptive, utils.load_efficientad_pruned, utils.load_efficientad_qat, utils.load_efficientad_thin` | yes |
| `calibrate_threshold_all1.py` | `06_calibration_temporal` | `utils.load_efficientad_adaptive, utils.load_efficientad_pruned, utils.load_efficientad_qat, utils.load_efficientad_thin` | yes |
| `calibrate_threshold_all_clahe.py` | `06_calibration_temporal` | `utils.load_efficientad_adaptive, utils.load_efficientad_pruned, utils.load_efficientad_qat, utils.load_efficientad_thin` | yes |
| `calibrate_threshold_all_fp16.py` | `06_calibration_temporal` | `utils.load_efficientad_adaptive, utils.load_efficientad_pruned, utils.load_efficientad_qat, utils.load_efficientad_thin` | yes |
| `calibrate_threshold_all_og.py` | `06_calibration_temporal` | `utils.load_efficientad_adaptive, utils.load_efficientad_pruned, utils.load_efficientad_qat, utils.load_efficientad_thin` | yes |
| `calibrate_threshold_fixed.py` | `06_calibration_temporal` | `utils.load_efficientad_adaptive` | yes |
| `calibrate_threshold_thin.py` | `06_calibration_temporal` | `utils.load_efficientad_thin` | yes |
| `calibrate_thresholds.py` | `06_calibration_temporal` | `utils.load_efficientad_adaptive, utils.load_efficientad_pruned, utils.load_efficientad_qat, utils.load_efficientad_thin` | yes |
| `calibrate_v3_k78_lowres192.py` | `06_calibration_temporal` | `—` | no |
| `compare_old_new_yolo_calibration.py` | `06_calibration_temporal` | `—` | yes |
| `condition_specific_calibration.py` | `06_calibration_temporal` | `—` | yes |
| `runs/detect/runs/detect/yolo_runs/compare_old_new_yolo_calibration.py` | `06_calibration_temporal` | `—` | yes |
| `select_operating_thresholds.py` | `06_calibration_temporal` | `—` | yes |
| `split_calibration_test.py` | `06_calibration_temporal` | `—` | yes |
| `video_threshold_calibration.py` | `06_calibration_temporal` | `compare_pipeline_methods_batched` | yes |
| `evaluate_all.py` | `07_evaluation_validation` | `—` | yes |
| `evaluate_final_unseen_lowres192.py` | `07_evaluation_validation` | `—` | yes |
| `evaluate_heldout_scores.py` | `07_evaluation_validation` | `—` | no |
| `evaluate_lowres.py` | `07_evaluation_validation` | `models_adaptive, utils.data_loader, utils.models` | yes |
| `evaluate_lowres_pipeline.py` | `07_evaluation_validation` | `pipeline.pipeline_adaptive, utils.data_loader, utils.load_efficientad_adaptive, utils.metrics, utils.visualisation` | yes |
| `evaluate_pruned_pipeline.py` | `07_evaluation_validation` | `pipeline.pipeline, utils.data_loader, utils.load_efficientad_pruned, utils.metrics, utils.visualisation` | yes |
| `evaluate_qat_pipeline.py` | `07_evaluation_validation` | `pipeline.pipeline, utils.data_loader, utils.load_efficientad_qat, utils.metrics, utils.visualisation` | yes |
| `evaluate_thin_pipeline.py` | `07_evaluation_validation` | `pipeline.pipeline, utils.data_loader, utils.load_efficientad_thin, utils.metrics, utils.visualisation` | yes |
| `evaluate_validation_with_manifest.py` | `07_evaluation_validation` | `—` | no |
| `pipeline/evaluate_live.py` | `07_evaluation_validation` | `utils.data_loader, utils.models` | yes |
| `pipeline/evaluate_pipeline.py` | `07_evaluation_validation` | `pipeline.pipeline, utils.data_loader, utils.load_efficientad, utils.metrics, utils.visualisation` | yes |
| `prepare_reserved_gt_newyolo.py` | `07_evaluation_validation` | `—` | yes |
| `video_eval_pipeline.py` | `07_evaluation_validation` | `pipeline.pipeline, utils.load_efficientad` | yes |
| `video_evaluate.py` | `07_evaluation_validation` | `pipeline.pipeline, utils.load_efficientad, utils.metrics, utils.visualisation` | yes |
| `bench_onnx.py` | `08_benchmarking_performance` | `compare_pipeline_methods_batched` | yes |
| `benchmark_efficientad.py` | `08_benchmarking_performance` | `utils.load_efficientad_adaptive, utils.load_efficientad_pruned, utils.load_efficientad_qat, utils.load_efficientad_thin` | yes |
| `benchmark_variants.py` | `08_benchmarking_performance` | `—` | yes |
| `benchmark_variants_fixed.py` | `08_benchmarking_performance` | `utils.models` | yes |
| `benchmark_variants_realtime.py` | `08_benchmarking_performance` | `utils.models` | yes |
| `benchmark_variants_realtime_v2.py` | `08_benchmarking_performance` | `utils.models` | yes |
| `benchmark_video_fps.py` | `08_benchmarking_performance` | `—` | yes |
| `measure_precision_diff.py` | `08_benchmarking_performance` | `—` | yes |
| `profile_forward.py` | `08_benchmarking_performance` | `compare_pipeline_methods_batched, utils.models` | yes |
| `analyze_defects.py` | `09_comparative_experiments` | `—` | yes |
| `analyze_defects_v2.py` | `09_comparative_experiments` | `—` | yes |
| `analyze_step4_factors.py` | `09_comparative_experiments` | `—` | yes |
| `analyze_verified_backgrounds.py` | `09_comparative_experiments` | `—` | yes |
| `bg_removal_compare.py` | `09_comparative_experiments` | `—` | yes |
| `bg_removal_compareAD.py` | `09_comparative_experiments` | `utils.load_efficientad` | yes |
| `build_condition_reference.py` | `09_comparative_experiments` | `—` | no |
| `combined_gum_inspection_pipeline_compare_all.py` | `09_comparative_experiments` | `utils.load_efficientad_adaptive, utils.load_efficientad_pruned, utils.load_efficientad_qat, utils.load_efficientad_thin` | yes |
| `combined_gum_inspection_pipeline_compare_all_og.py` | `09_comparative_experiments` | `utils.load_efficientad_adaptive, utils.load_efficientad_pruned, utils.load_efficientad_qat, utils.load_efficientad_thin` | yes |
| `compare_extraction_methods.py` | `09_comparative_experiments` | `—` | yes |
| `compare_methods.py` | `09_comparative_experiments` | `utils.load_efficientad` | yes |
| `compare_old_new_yolo_fair.py` | `09_comparative_experiments` | `—` | yes |
| `compare_options.py` | `09_comparative_experiments` | `—` | yes |
| `compare_pipeline_methods.py` | `09_comparative_experiments` | `utils.load_efficientad_adaptive, utils.load_efficientad_pruned, utils.load_efficientad_qat, utils.load_efficientad_thin` | yes |
| `compare_pipeline_methods_auto_conditional.py` | `09_comparative_experiments` | `utils.load_efficientad_adaptive, utils.load_efficientad_pruned, utils.load_efficientad_qat, utils.load_efficientad_thin` | yes |
| `compare_pipeline_methods_clahe.py` | `09_comparative_experiments` | `utils.load_efficientad_adaptive, utils.load_efficientad_pruned, utils.load_efficientad_qat, utils.load_efficientad_thin` | yes |
| `compare_pipeline_methods_reel.py` | `09_comparative_experiments` | `utils.load_efficientad_adaptive, utils.load_efficientad_pruned, utils.load_efficientad_qat, utils.load_efficientad_thin` | yes |
| `compare_pipeline_methods_t.py` | `09_comparative_experiments` | `utils.load_efficientad_adaptive, utils.load_efficientad_pruned, utils.load_efficientad_qat, utils.load_efficientad_thin` | yes |
| `compare_variants.py` | `09_comparative_experiments` | `utils.data_loader` | yes |
| `compare_variants_on_videos.py` | `09_comparative_experiments` | `—` | yes |
| `compare_variants_on_videos_fixed.py` | `09_comparative_experiments` | `—` | yes |
| `compare_variants_realtime.py` | `09_comparative_experiments` | `—` | yes |
| `compare_variants_realtime_v2.py` | `09_comparative_experiments` | `—` | yes |
| `cross_ensemble.py` | `09_comparative_experiments` | `—` | yes |
| `current_crop_background_suppression.py` | `09_comparative_experiments` | `video_streaming_detector_simple_optimized_v2` | yes |
| `current_vs_k78_domain_shift.py` | `09_comparative_experiments` | `video_streaming_detector_simple_optimized_v3` | yes |
| `extract_and_compare.py` | `09_comparative_experiments` | `—` | yes |
| `mean_suppression_grid.py` | `09_comparative_experiments` | `video_streaming_detector_simple_optimized_v2` | yes |
| `video_inference_bg_removal.py` | `09_comparative_experiments` | `utils.load_efficientad` | yes |
| `ckpt_cropped_real_4/live_compare_app.py` | `10_visualization_demos` | `utils.load_efficientad_adaptive, utils.load_efficientad_pruned, utils.load_efficientad_thin` | yes |
| `live_compare_app.py` | `10_visualization_demos` | `utils.load_efficientad_adaptive, utils.load_efficientad_pruned, utils.load_efficientad_thin` | yes |
| `make_ema_visuals.py` | `10_visualization_demos` | `—` | yes |
| `utils/visualisation.py` | `10_visualization_demos` | `—` | no |
| `scripts/diagnose.py` | `11_utilities` | `utils.data_loader, utils.models` | yes |
| `scripts/recompute_quantiles copy.py` | `11_utilities` | `utils.data_loader, utils.detection, utils.models` | yes |
| `scripts/recompute_quantiles.py` | `11_utilities` | `—` | yes |
| `utils/__init__.py` | `11_utilities` | `—` | no |
| `utils/data_loader.py` | `11_utilities` | `—` | no |
| `utils/metrics.py` | `11_utilities` | `—` | no |
| `diagnose_heldout_tracks.py` | `13_tests_diagnostics` | `—` | no |
| `get_video_dims.py` | `13_tests_diagnostics` | `—` | yes |
| `test_cropping_options.py` | `13_tests_diagnostics` | `—` | yes |
| `utils/test_crop_sizes.py` | `13_tests_diagnostics` | `utils.detection` | yes |
| `utils/test_sizing.py` | `13_tests_diagnostics` | `utils.detection` | yes |
| `c.py` | `99_uncategorized` | `—` | no |
