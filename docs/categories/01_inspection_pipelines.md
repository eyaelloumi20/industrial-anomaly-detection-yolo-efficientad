# 01 Inspection Pipelines

End-to-end YOLO / ByteTrack / EfficientAD inference

| Script | Original path | Description | Classification reason |
|---|---|---|---|
| [`combined_gum_inspection_pipeline.py`](../../pipeline/01_inspection_pipelines/combined_gum_inspection_pipeline.py) | `combined_gum_inspection_pipeline.py` | combined_gum_inspection_pipeline.py | Inspection or inference pipeline |
| [`combined_gum_inspection_pipeline1.py`](../../pipeline/01_inspection_pipelines/combined_gum_inspection_pipeline1.py) | `combined_gum_inspection_pipeline1.py` | combined_gum_inspection_pipeline.py | Inspection or inference pipeline |
| [`combined_gum_inspection_pipeline3.py`](../../pipeline/01_inspection_pipelines/combined_gum_inspection_pipeline3.py) | `combined_gum_inspection_pipeline3.py` | combined_gum_inspection_pipeline.py | Inspection or inference pipeline |
| [`combined_gum_inspection_pipeline_fixed.py`](../../pipeline/01_inspection_pipelines/combined_gum_inspection_pipeline_fixed.py) | `combined_gum_inspection_pipeline_fixed.py` | combined_gum_inspection_pipeline.py | Inspection or inference pipeline |
| [`combined_gum_inspection_pipeline_thin.py`](../../pipeline/01_inspection_pipelines/combined_gum_inspection_pipeline_thin.py) | `combined_gum_inspection_pipeline_thin.py` | combined_gum_inspection_pipeline.py | Inspection or inference pipeline |
| [`compare_pipeline_methods_batched.py`](../../pipeline/01_inspection_pipelines/compare_pipeline_methods_batched.py) | `compare_pipeline_methods_batched.py` | combined_gum_inspection_pipeline_compare_all.py | Shared batched pipeline implementation |
| [`compare_pipeline_methods_batched1.py`](../../pipeline/01_inspection_pipelines/compare_pipeline_methods_batched1.py) | `compare_pipeline_methods_batched1.py` | combined_gum_inspection_pipeline_compare_all.py | Shared batched pipeline implementation |
| [`pipeline.py`](../../pipeline/01_inspection_pipelines/pipeline/pipeline.py) | `pipeline/pipeline.py` | pipeline.py | Inspection or inference pipeline |
| [`pipeline1.py`](../../pipeline/01_inspection_pipelines/pipeline/pipeline1.py) | `pipeline/pipeline1.py` | pipeline.py | Inspection or inference pipeline |
| [`pipeline_adaptive.py`](../../pipeline/01_inspection_pipelines/pipeline/pipeline_adaptive.py) | `pipeline/pipeline_adaptive.py` | pipeline_adaptive.py | Inspection or inference pipeline |
| [`single_photo_inference.py`](../../pipeline/01_inspection_pipelines/pipeline/single_photo_inference.py) | `pipeline/single_photo_inference.py` | single_photo_inference.py | Inspection or inference pipeline |
| [`video_inference.py`](../../pipeline/01_inspection_pipelines/pipeline/video_inference.py) | `pipeline/video_inference.py` | video_inference_clean.py | Inspection or inference pipeline |
| [`video_inference1.py`](../../pipeline/01_inspection_pipelines/pipeline/video_inference1.py) | `pipeline/video_inference1.py` | video_inference1_clean.py | Inspection or inference pipeline |
| [`video_inferenceNB.py`](../../pipeline/01_inspection_pipelines/pipeline/video_inferenceNB.py) | `pipeline/video_inferenceNB.py` | video_inferenceNB.py  (ByteTrack + async rembg preprocessing) | Inspection or inference pipeline |
| [`video_inferenceNB1.py`](../../pipeline/01_inspection_pipelines/pipeline/video_inferenceNB1.py) | `pipeline/video_inferenceNB1.py` | video_inferenceNB1.py  (DeepOCSORT + async rembg preprocessing) | Inspection or inference pipeline |
| [`realtime_inference.py`](../../pipeline/01_inspection_pipelines/realtime_inference.py) | `realtime_inference.py` | realtime_inference.py | Inspection or inference pipeline |
| [`realtime_inspector.py`](../../pipeline/01_inspection_pipelines/realtime_inspector.py) | `realtime_inspector.py` | Real-time YOLO + EfficientAD inspection with a LAN MJPEG dashboard. | Inspection or inference pipeline |
| [`run_inference.py`](../../pipeline/01_inspection_pipelines/run_inference.py) | `run_inference.py` | Chewing gum inspection pipeline: background removal (rembg/u2net) + YOLOv8 | Inspection or inference pipeline |
| [`single_photo_inference_seg.py`](../../pipeline/01_inspection_pipelines/single_photo_inference_seg.py) | `single_photo_inference_seg.py` | single_photo_inference.py | Inspection or inference pipeline |
| [`video_inference.py`](../../pipeline/01_inspection_pipelines/video_inference.py) | `video_inference.py` | video_inference.py  (smartphone-calibrated, v3) | Inspection or inference pipeline |
| [`video_inference_OG.py`](../../pipeline/01_inspection_pipelines/video_inference_OG.py) | `video_inference_OG.py` | video_inference_clean.py | Inspection or inference pipeline |
| [`video_inference_seg.py`](../../pipeline/01_inspection_pipelines/video_inference_seg.py) | `video_inference_seg.py` | video_inference.py | Inspection or inference pipeline |
| [`video_inference_simple.py`](../../pipeline/01_inspection_pipelines/video_inference_simple.py) | `video_inference_simple.py` | video_inference_simple.py | Inspection or inference pipeline |
| [`video_inference_simple1.py`](../../pipeline/01_inspection_pipelines/video_inference_simple1.py) | `video_inference_simple1.py` | video_inference_simple.py | Inspection or inference pipeline |
| [`video_inferenceBG.py`](../../pipeline/01_inspection_pipelines/video_inferenceBG.py) | `video_inferenceBG.py` | video_inference_clean.py | Inspection or inference pipeline |
| [`video_inferenceNB1_noAD.py`](../../pipeline/01_inspection_pipelines/video_inferenceNB1_noAD.py) | `video_inferenceNB1_noAD.py` | video_inferenceNB1.py  (DeepOCSORT + async rembg preprocessing) | Inspection or inference pipeline |
| [`video_inferenceNB_fixed.py`](../../pipeline/01_inspection_pipelines/video_inferenceNB_fixed.py) | `video_inferenceNB_fixed.py` | video_inferenceNB_fixed.py  (ByteTrack + frame-level rembg) | Inspection or inference pipeline |
| [`video_inferenceNB_noAD.py`](../../pipeline/01_inspection_pipelines/video_inferenceNB_noAD.py) | `video_inferenceNB_noAD.py` | video_inferenceNB.py  (ByteTrack + async rembg preprocessing) | Inspection or inference pipeline |
| [`video_streaming_detector.py`](../../pipeline/01_inspection_pipelines/video_streaming_detector.py) | `video_streaming_detector.py` | video_streaming_detector.py | Inspection or inference pipeline |
| [`video_streaming_detector1.py`](../../pipeline/01_inspection_pipelines/video_streaming_detector1.py) | `video_streaming_detector1.py` | video_streaming_detector1.py | Inspection or inference pipeline |
| [`video_streaming_detector_fast_early_final.py`](../../pipeline/01_inspection_pipelines/video_streaming_detector_fast_early_final.py) | `video_streaming_detector_fast_early_final.py` | video_streaming_detector_fast_early_final.py | Inspection or inference pipeline |
| [`video_streaming_detector_fast_v2.py`](../../pipeline/01_inspection_pipelines/video_streaming_detector_fast_v2.py) | `video_streaming_detector_fast_v2.py` | video_streaming_detector_fast_v2.py | Inspection or inference pipeline |
| [`video_streaming_detector_optimized.py`](../../pipeline/01_inspection_pipelines/video_streaming_detector_optimized.py) | `video_streaming_detector_optimized.py` | Offline YOLO + EfficientAD detector with final-verdict video annotations. | Inspection or inference pipeline |
| [`video_streaming_detector_optimized1.py`](../../pipeline/01_inspection_pipelines/video_streaming_detector_optimized1.py) | `video_streaming_detector_optimized1.py` | Offline YOLO + EfficientAD detector with final-verdict video annotations. | Inspection or inference pipeline |
| [`video_streaming_detector_optimized2.py`](../../pipeline/01_inspection_pipelines/video_streaming_detector_optimized2.py) | `video_streaming_detector_optimized2.py` | Offline YOLO + EfficientAD detector with final-verdict video annotations. | Inspection or inference pipeline |
| [`video_streaming_detector_optimized3.py`](../../pipeline/01_inspection_pipelines/video_streaming_detector_optimized3.py) | `video_streaming_detector_optimized3.py` | Offline YOLO + EfficientAD detector with final-verdict video annotations. | Inspection or inference pipeline |
| [`video_streaming_detector_optimized4.py`](../../pipeline/01_inspection_pipelines/video_streaming_detector_optimized4.py) | `video_streaming_detector_optimized4.py` | video_streaming_detector_optimized4.py | Inspection or inference pipeline |
| [`video_streaming_detector_simple_optimized.py`](../../pipeline/01_inspection_pipelines/video_streaming_detector_simple_optimized.py) | `video_streaming_detector_simple_optimized.py` | video_streaming_detector_simple_optimized.py | Inspection or inference pipeline |
| [`video_streaming_detector_simple_optimized_v2.py`](../../pipeline/01_inspection_pipelines/video_streaming_detector_simple_optimized_v2.py) | `video_streaming_detector_simple_optimized_v2.py` | video_streaming_detector_simple_optimized.py | Inspection or inference pipeline |
| [`video_streaming_detector_simple_optimized_v3.py`](../../pipeline/01_inspection_pipelines/video_streaming_detector_simple_optimized_v3.py) | `video_streaming_detector_simple_optimized_v3.py` | video_streaming_detector_simple_optimized_v3.py | Inspection or inference pipeline |
