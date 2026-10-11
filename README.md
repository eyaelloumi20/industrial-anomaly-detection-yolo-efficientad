# Industrial Anomaly Detection with YOLOv8, ByteTrack & EfficientAD

**Source code and reproducibility guide for a two-stage, video-based quality-inspection research project.**

This repository explores automatic inspection of chewing-gum pieces. It combines **YOLOv8** for locating individual pieces, **ByteTrack** for maintaining identities between video frames, and **EfficientAD** for assigning anomaly scores to cropped objects. Temporal aggregation converts multiple frame-level scores into more stable piece-level decisions.

> **Research repository, not a ready-to-deploy production system.** Some scripts are experimental versions, depend on external data and trained student checkpoints, or use paths from the original research workstation. The categorization is generated from script names and requires review. No benchmark result is implied merely by including a script.

## How the inspection system works

```text
Input video / camera frames
          |
          v
 YOLOv8: detect chewing-gum pieces
          |
          v
 ByteTrack: associate detections across frames
          |
          v
 Crop extraction and preprocessing
          |
          v
 EfficientAD: calculate per-crop anomaly scores
          |
          v
 Temporal aggregation (including EMA where configured)
          |
          v
 Thresholding + per-track classification
          |
          v
 Annotated video / inspection summary
```

**Responsibility boundaries:** YOLOv8 localizes pieces; it does not itself decide whether a piece is anomalous in the core two-stage pipeline. ByteTrack supplies consistent track IDs. EfficientAD produces visual anomaly scores, which are interpreted using model-specific calibration and temporal decision rules. Some experiments modify crop processing or compare detection models; these should not be confused with the reference pipeline.

## Repository layout

```text
.
├── pipeline/                      # Main inference and tracking implementations
│   ├── 01_inspection_pipelines/
│   └── 02_detection_tracking/
├── utils/                         # EfficientAD models and shared helpers
│   ├── 03_efficientad_architectures/
│   └── 11_utilities/             # Some helpers may appear in other categories
├── scripts/                       # Training, experiments, and research tools
│   ├── 04_training_optimization/
│   ├── 05_dataset_preprocessing/
│   ├── 06_calibration_temporal/
│   ├── 07_evaluation_validation/
│   ├── 08_benchmarking_performance/
│   ├── 09_comparative_experiments/
│   ├── 10_visualization_demos/
│   ├── 12_workflows/
│   ├── 13_tests_diagnostics/
│   └── 99_uncategorized/
├── code/                          # Symlinks preserving original script paths
├── checkpoints/
│   └── teacher/
│       └── best_teacher.pth      # Original EfficientAD teacher (if permitted to share)
├── docs/
│   ├── ARCHITECTURE.md
│   ├── GETTING_STARTED.md
│   ├── EXPERIMENTS.md
│   ├── SCRIPT_CATALOG.md
│   ├── EXPORT_AUDIT.md
│   └── PUBLICATION_CHECKLIST.md
├── run_from_code.sh               # Compatibility launcher
├── requirements-original.txt      # Original environment snapshot, if exported
└── README.md
```

The export currently contains **228 scripts: 191 Python files and 37 Bash files**, arranged in functional categories. Check `docs/SCRIPT_CATALOG.md` for the generated script index and original locations. The `code/` directory consists of relative symbolic links, not another copy of all scripts, and preserves names expected by existing imports and workflows. Some categories may be located under a different one of the three top-level folders based on their primary purpose; the catalog is authoritative for the actual export.

## Key source-code areas

| Area | Purpose | Representative files |
| --- | --- | --- |
| Video inspection | End-to-end video analysis and visualization | `video_streaming_detector_fast_v2.py`, `video_streaming_detector_simple_optimized_v3.py` |
| Shared inference | Models and crop/scoring utilities | `compare_pipeline_methods_batched.py`, `utils/models.py` |
| Detector experiments | YOLO training, fine-tuning and comparison | `scripts/train_yolo.py`, `compare_old_new_yolo_fair.py` |
| Model optimization | Low-resolution, thin, pruning and QAT experiments | `retrain_lowres.py`, `retrain_thin.py`, `prune_finetune.py`, `train_qat.py` |
| Calibration | Score thresholds and temporal decision studies | `calibrate_ema_thresholds.py`, `analyze_lowres192_temporal_rules.py` |
| Validation | Held-out evaluations and variant comparison | `evaluate_final_unseen_lowres192.py`, `compare_variants_on_videos.py` |
| Performance | FPS and inference profiling | `benchmark_video_fps.py`, `benchmark_variants_realtime_v2.py` |

Names in this table are **original script names**. To find their actual categorized destination, consult `docs/SCRIPT_CATALOG.md`; to run them with compatible import paths, use `run_from_code.sh` as described below.

## Getting started

### 1. Environment

A Linux workstation with a Python environment is recommended. GPU inference and training require an appropriate NVIDIA driver, CUDA-compatible PyTorch installation and model-specific dependencies. The original project used Python packages including PyTorch, Ultralytics, OpenCV and ByteTrack-related tooling.

```bash
cd ~/industrial-anomaly-detection-yolo-efficientad
conda activate inspection   # Only if this environment exists on your machine
```

`requirements-original.txt` is a copy of the research environment's dependency list, **not** a verified cross-platform installer. Review it before installation. Details: [Getting started](docs/GETTING_STARTED.md).

### 2. Inspect the command-line interface

```bash
./run_from_code.sh video_streaming_detector_fast_v2.py --help
```

This launcher changes into the original-path compatibility layout and sets `PYTHONPATH` so imports such as `utils.models` and `compare_pipeline_methods_batched` can resolve. Successful `--help` output confirms only initial imports and argument parsing, not full pipeline correctness.

### 3. Provide local inputs and trained models

**No datasets, recorded videos, or trained EfficientAD student checkpoints are distributed.** To run a complete inspection you must independently supply:

- Input video or camera access;
- Trained YOLO detector weights;
- Matching EfficientAD student/autoencoder checkpoint directory;
- Model-specific threshold-calibration JSON;
- Dependencies compatible with the chosen implementation.

For the `video_streaming_detector_fast_v2.py` implementation, the important arguments include `--video`, `--yolo_det_weights`, `--ckpt_dir`, and `--thresholds_json`. Check the exact supported flags with `--help` before running:

```bash
./run_from_code.sh video_streaming_detector_fast_v2.py \
  --video /path/to/input.mp4 \
  --yolo_det_weights /path/to/yolo/best.pt \
  --ckpt_dir /path/to/efficientad_student_checkpoint \
  --thresholds_json /path/to/calibrated_thresholds.json
```

The paths above are placeholders, **not files included in this repository**.

## Original EfficientAD teacher checkpoint

The one explicitly retained weight file is:

`checkpoints/teacher/best_teacher.pth`

It was copied from the research workspace's `ckpt/best_teacher.pth`. The copied file was SHA-256 verified against that local original:

```text
516992c053a6f64dd5609f60792fbf1a107562afe35552d03d6fce2167e9d548
```

The teacher checkpoint is approximately **11 MB**. In EfficientAD teacher–student training, the teacher serves as a reference network; **it is not by itself a trained anomaly detector for video inference**. Full inference typically needs the compatible trained student and autoencoder weights, preprocessing, and calibration configuration.

**Redistribution notice:** Check the pretrained teacher checkpoint's original source and licensing terms before committing or publicly sharing it. Its inclusion in a local export does not establish redistribution permission.

## Research organization and experimental variants

The project includes scripts to study:

- Baseline EfficientAD at standard crop resolution;
- Reduced-resolution configurations such as **LowRes224** and **LowRes192**;
- Thinner network variants, including **Thin0.75** and **Thin0.50**;
- **Pruned0.30** and quantization-aware training experiments;
- Data/crop processing variants and changes of video background;
- Temporal score smoothing, track-level rules and decision thresholds;
- Old-vs-new YOLO detections and end-to-end effects;
- Processing throughput and FPS trade-offs.

These are distinct experiments with potentially different training data, calibration videos, evaluation subsets and decision rules. **Do not compare numbers across experiments unless their evaluation protocols match.** This repository intentionally excludes generated metrics and results. See [Experiments](docs/EXPERIMENTS.md) for navigation.

## Reproducibility and limitations

- The repository publishes code and methodological documentation, **not** research datasets, held-out labels, logs or output figures.
- Most scripts were written for a particular Linux/CUDA setup; paths and versions may need adaptation.
- Categorization is heuristic. A few scripts may need moving after inspection of their content; moving them requires fixing imports and references.
- Symbolic links provide compatibility only on filesystems and Git checkouts that preserve them.
- Calibration thresholds cannot be transferred automatically between model variants, score definitions or domains.
- Performance depends on hardware, batch size, crop count, detection/tracking behavior and measurement protocol.
- Original training and experiment scripts may not be independently executable without external assets.

## Documentation

Start with [Architecture](docs/ARCHITECTURE.md) and [Getting started](docs/GETTING_STARTED.md). Use [Script catalog](docs/SCRIPT_CATALOG.md) to identify which implementation to read, [Experiments](docs/EXPERIMENTS.md) for the experiment families, and [Publication checklist](docs/PUBLICATION_CHECKLIST.md) before making the repository public.

## Before publishing

1. Verify the teacher checkpoint's redistribution license.
2. Scan scripts for API keys, private tokens, user-specific file paths and confidential content.
3. Review categorization, Python imports, Bash paths and symlink behavior.
4. Confirm that Git is tracking source and documentation only, plus the explicitly approved teacher checkpoint.
5. Initialize Git and review staged changes before pushing.

## License

See the repository's `LICENSE` file for the license copied from the original project **if present**. Check whether it covers all included code; third-party libraries and pretrained weights may have separate licenses.
