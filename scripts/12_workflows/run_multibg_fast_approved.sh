#!/usr/bin/env bash
set -euo pipefail

cd /home/ahmed/two-stage-efficientad-inspection

echo "======================================================================"
echo "APPROVED-69 FAST MULTI-BACKGROUND EXPERIMENT"
echo "======================================================================"
echo

python build_multibg_fast_dataset_approved.py

echo
echo "Dataset built from user-approved NORMAL tracks."
echo "Starting LowRes192 20k adaptation..."
echo

./train_lowres192_multibg_fast.sh
