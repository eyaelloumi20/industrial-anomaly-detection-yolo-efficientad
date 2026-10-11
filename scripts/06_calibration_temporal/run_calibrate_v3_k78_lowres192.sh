#!/usr/bin/env bash
set -euo pipefail

cd /home/ahmed/two-stage-efficientad-inspection

python calibrate_v3_k78_lowres192.py

echo
echo "Threshold file:"
cat ./video_calibration_v3_k78/video_thresholds_lowres192_v3_k78_ema.json
