# Environment and execution

This repository deliberately omits datasets, result files, media, checkpoints and thresholds JSON. Obtain those assets separately with appropriate authorization, and keep them outside the Git tree.

## Requirements

Use Linux for the compatibility layout, Python, PyTorch/CUDA where relevant, Ultralytics, OpenCV, Pillow and NumPy. `requirements-original.txt` (when present) is an **environment snapshot**; it is not a guaranteed portable dependency lockfile.

## Compatibility entrypoint

From repository root:

```bash
cd code
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python3 video_streaming_detector_fast_v2.py --help
```

From `code/` the unchanged relative paths and imports should behave closest to the historical workspace. However the symlink compatibility layer is **not a guarantee** that dynamically resolved filesystem paths work. Test individual scripts before training or inference.

Example with external assets; illustrative threshold only:

```bash
cd code
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python3 video_streaming_detector_fast_v2.py \
  --video /external/input.mp4 \
  --yolo_det_weights /external/yolo_best.pt \
  --ckpt_dir /external/efficientad_ckpt \
  --thresholds_json /external/video_thresholds.json \
  --tau_track 0.45 \
  --device cuda:0 --fp16 --yolo_fp16 \
  --crop_backend opencv --batch_frames 12 --batch_crops 96 \
  --early_fire --early_consecutive 3 \
  --save_dir /external/inspection_outputs
```

The example threshold **must be replaced** with the threshold calibrated for the precise weights, data and preprocessing. `--tau_track` is separate from the thresholds JSON.

A script being included in the repo does not imply its dependencies and datasets have been bundled. Some scripts contain workstation-specific absolute paths; review the audit before use.
