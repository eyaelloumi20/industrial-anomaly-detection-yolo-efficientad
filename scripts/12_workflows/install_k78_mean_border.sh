#!/usr/bin/env bash
set -euo pipefail

cd /home/ahmed/two-stage-efficientad-inspection

python integrate_k78_mean_border.py

echo
echo "Checking syntax..."
python -m py_compile video_streaming_detector_simple_optimized_v3.py

echo "Checking CLI..."
python video_streaming_detector_simple_optimized_v3.py --help >/dev/null

echo
echo "OK: video_streaming_detector_simple_optimized_v3.py is ready."
