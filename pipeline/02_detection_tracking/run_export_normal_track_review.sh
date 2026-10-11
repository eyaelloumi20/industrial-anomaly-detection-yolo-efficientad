#!/usr/bin/env bash
set -euo pipefail

cd /home/ahmed/two-stage-efficientad-inspection

python export_normal_track_review.py

echo
echo "Review folder:"
echo "  ./normal_track_review"
echo
echo "Open overviews:"
xdg-open ./normal_track_review/overview_black.jpg >/dev/null 2>&1 || true
xdg-open ./normal_track_review/overview_metallic.jpg >/dev/null 2>&1 || true
xdg-open ./normal_track_review/overview_white.jpg >/dev/null 2>&1 || true
xdg-open ./normal_track_review/overview_wooden.jpg >/dev/null 2>&1 || true
