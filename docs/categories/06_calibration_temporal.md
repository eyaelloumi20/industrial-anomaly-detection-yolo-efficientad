# 06 Calibration Temporal

Threshold selection, EMA, temporal decisions

| Script | Original path | Description | Classification reason |
|---|---|---|---|
| [`analyze_lowres192_temporal_rules.py`](../../scripts/06_calibration_temporal/analyze_lowres192_temporal_rules.py) | `analyze_lowres192_temporal_rules.py` | Analyze LowRes192 calibration behavior on the 4 calibration videos only. | Calibration / threshold / temporal rule |
| [`analyze_score_percentiles.py`](../../scripts/06_calibration_temporal/analyze_score_percentiles.py) | `analyze_score_percentiles.py` | analyze_score_percentiles.py | Calibration / threshold / temporal rule |
| [`calibrate_baseline_ema.py`](../../scripts/06_calibration_temporal/calibrate_baseline_ema.py) | `calibrate_baseline_ema.py` | from __future__ import annotations | Calibration / threshold / temporal rule |
| [`calibrate_bg2_only.py`](../../scripts/06_calibration_temporal/calibrate_bg2_only.py) | `calibrate_bg2_only.py` | calibrate_bg2_only.py | Calibration / threshold / temporal rule |
| [`calibrate_efficientad_variants_realtime.py`](../../scripts/06_calibration_temporal/calibrate_efficientad_variants_realtime.py) | `calibrate_efficientad_variants_realtime.py` | calibrate_efficientad_variants_realtime.py | Calibration / threshold / temporal rule |
| [`calibrate_ema_thresholds.py`](../../scripts/06_calibration_temporal/calibrate_ema_thresholds.py) | `calibrate_ema_thresholds.py` | calibrate_ema_thresholds.py | Calibration / threshold / temporal rule |
| [`calibrate_lowres192_threshold.py`](../../scripts/06_calibration_temporal/calibrate_lowres192_threshold.py) | `calibrate_lowres192_threshold.py` | Calibrate the LowRes192 EMA threshold from the 4 development videos. | Calibration / threshold / temporal rule |
| [`calibrate_lowres192_threshold_fixed.py`](../../scripts/06_calibration_temporal/calibrate_lowres192_threshold_fixed.py) | `calibrate_lowres192_threshold_fixed.py` | Calibrate the LowRes192 EMA threshold from the 4 development videos. | Calibration / threshold / temporal rule |
| [`calibrate_threshold.py`](../../scripts/06_calibration_temporal/calibrate_threshold.py) | `calibrate_threshold.py` | calibrate_threshold.py | Calibration / threshold / temporal rule |
| [`calibrate_threshold1.py`](../../scripts/06_calibration_temporal/calibrate_threshold1.py) | `calibrate_threshold1.py` | calibrate_threshold.py | Calibration / threshold / temporal rule |
| [`calibrate_threshold2.py`](../../scripts/06_calibration_temporal/calibrate_threshold2.py) | `calibrate_threshold2.py` | calibrate_threshold.py | Calibration / threshold / temporal rule |
| [`calibrate_threshold3.py`](../../scripts/06_calibration_temporal/calibrate_threshold3.py) | `calibrate_threshold3.py` | calibrate_threshold.py | Calibration / threshold / temporal rule |
| [`calibrate_threshold_all.py`](../../scripts/06_calibration_temporal/calibrate_threshold_all.py) | `calibrate_threshold_all.py` | calibrate_threshold_all.py | Calibration / threshold / temporal rule |
| [`calibrate_threshold_all1.py`](../../scripts/06_calibration_temporal/calibrate_threshold_all1.py) | `calibrate_threshold_all1.py` | calibrate_threshold_all.py | Calibration / threshold / temporal rule |
| [`calibrate_threshold_all_clahe.py`](../../scripts/06_calibration_temporal/calibrate_threshold_all_clahe.py) | `calibrate_threshold_all_clahe.py` | calibrate_threshold_all.py | Calibration / threshold / temporal rule |
| [`calibrate_threshold_all_fp16.py`](../../scripts/06_calibration_temporal/calibrate_threshold_all_fp16.py) | `calibrate_threshold_all_fp16.py` | calibrate_threshold_all.py | Calibration / threshold / temporal rule |
| [`calibrate_threshold_all_og.py`](../../scripts/06_calibration_temporal/calibrate_threshold_all_og.py) | `calibrate_threshold_all_og.py` | calibrate_threshold_all.py | Calibration / threshold / temporal rule |
| [`calibrate_threshold_fixed.py`](../../scripts/06_calibration_temporal/calibrate_threshold_fixed.py) | `calibrate_threshold_fixed.py` | calibrate_threshold.py | Calibration / threshold / temporal rule |
| [`calibrate_threshold_thin.py`](../../scripts/06_calibration_temporal/calibrate_threshold_thin.py) | `calibrate_threshold_thin.py` | calibrate_threshold.py | Calibration / threshold / temporal rule |
| [`calibrate_thresholds.py`](../../scripts/06_calibration_temporal/calibrate_thresholds.py) | `calibrate_thresholds.py` | calibrate_threshold_all.py | Calibration / threshold / temporal rule |
| [`calibrate_v3_k78_lowres192.py`](../../scripts/06_calibration_temporal/calibrate_v3_k78_lowres192.py) | `calibrate_v3_k78_lowres192.py` | Original normal-only calibration adapted to v3 + k78 mean-border preprocessing. | Calibration / threshold / temporal rule |
| [`compare_old_new_yolo_calibration.py`](../../scripts/06_calibration_temporal/compare_old_new_yolo_calibration.py) | `compare_old_new_yolo_calibration.py` | Compare the OLD and NEW YOLO detectors on ONLY the 4 calibration videos. | Calibration / threshold / temporal rule |
| [`condition_specific_calibration.py`](../../scripts/06_calibration_temporal/condition_specific_calibration.py) | `condition_specific_calibration.py` | condition_specific_calibration.py | Calibration / threshold / temporal rule |
| [`run_calibrate_v3_k78_lowres192.sh`](../../scripts/06_calibration_temporal/run_calibrate_v3_k78_lowres192.sh) | `run_calibrate_v3_k78_lowres192.sh` | cd /home/ahmed/two-stage-efficientad-inspection | Calibration / threshold / temporal rule |
| [`run_calibrated_variants.sh`](../../scripts/06_calibration_temporal/run_calibrated_variants.sh) | `run_calibrated_variants.sh` | ROOT="/home/ahmed/two-stage-efficientad-inspection" | Calibration / threshold / temporal rule |
| [`run_heldout_calibrated_eval.sh`](../../scripts/06_calibration_temporal/run_heldout_calibrated_eval.sh) | `run_heldout_calibrated_eval.sh` | ROOT="/home/ahmed/two-stage-efficientad-inspection" | Calibration / threshold / temporal rule |
| [`run_lowres192_calibration.sh`](../../scripts/06_calibration_temporal/run_lowres192_calibration.sh) | `run_lowres192_calibration.sh` | cd /home/ahmed/two-stage-efficientad-inspection | Calibration / threshold / temporal rule |
| [`run_newyolo_recalibrated_final_test.sh`](../../scripts/06_calibration_temporal/run_newyolo_recalibrated_final_test.sh) | `run_newyolo_recalibrated_final_test.sh` | cd /home/ahmed/two-stage-efficientad-inspection | Calibration / threshold / temporal rule |
| [`run_original_calibration_new_yolo.sh`](../../scripts/06_calibration_temporal/run_original_calibration_new_yolo.sh) | `run_original_calibration_new_yolo.sh` | ============================================================ | Calibration / threshold / temporal rule |
| [`run_v3_annotated_with_threshold.sh`](../../scripts/06_calibration_temporal/run_v3_annotated_with_threshold.sh) | `run_v3_annotated_with_threshold.sh` | cd /home/ahmed/two-stage-efficientad-inspection | Calibration / threshold / temporal rule |
| [`compare_old_new_yolo_calibration.py`](../../scripts/06_calibration_temporal/runs/detect/runs/detect/yolo_runs/compare_old_new_yolo_calibration.py) | `runs/detect/runs/detect/yolo_runs/compare_old_new_yolo_calibration.py` | Compare the OLD and NEW YOLO detectors on ONLY the 4 calibration videos. | Calibration / threshold / temporal rule |
| [`select_operating_thresholds.py`](../../scripts/06_calibration_temporal/select_operating_thresholds.py) | `select_operating_thresholds.py` | select_operating_thresholds.py | Calibration / threshold / temporal rule |
| [`split_calibration_test.py`](../../scripts/06_calibration_temporal/split_calibration_test.py) | `split_calibration_test.py` | Assign calibration/test videos AFTER ground truth is finished. | Calibration / threshold / temporal rule |
| [`video_threshold_calibration.py`](../../scripts/06_calibration_temporal/video_threshold_calibration.py) | `video_threshold_calibration.py` | video_threshold_calibration.py | Calibration / threshold / temporal rule |
