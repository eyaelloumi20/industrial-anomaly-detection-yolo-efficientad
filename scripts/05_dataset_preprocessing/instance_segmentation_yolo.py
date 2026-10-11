"""
Instance Segmentation with YOLO
================================
Overview:
    Train an Instance Segmentation model using YOLOv11.
    Dataset: https://universe.roboflow.com/lini-foundation/lini-coral-forms-1.0/dataset/1
    Dataset should be in YOLO instance segmentation format.

Usage:
    pip install ultralytics
    python instance_segmentation_yolo.py
"""

import os
import cv2
import numpy as np
import plotly.express as px
import plotly.io as pio
from ultralytics import YOLO


# ─────────────────────────────────────────────
# Configuration — update these paths
# ─────────────────────────────────────────────

DATA_YAML        = "/home/ahmed/two-stage-efficientad-inspection/dataset10/data.yaml"
TEST_IMAGE_DIR   = "/home/ahmed/two-stage-efficientad-inspection/dataset10/test/images"
TRAIN_OUTPUT_DIR = "/home/ahmed/two-stage-efficientad-inspection/runs/segment/train"
PREDICT_OUTPUT_DIR = "/home/ahmed/two-stage-efficientad-inspection/runs/segment/predict"
BASE_MODEL       = "yolo11n-seg.pt"   # downloaded automatically on first run
EPOCHS           = 100


# ─────────────────────────────────────────────
# Step 1: Train the model
# ─────────────────────────────────────────────

model = YOLO(BASE_MODEL)

model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    project=os.path.dirname(TRAIN_OUTPUT_DIR),
    name=os.path.basename(TRAIN_OUTPUT_DIR),
)


# ─────────────────────────────────────────────
# Step 2: Run inference on test images
# ─────────────────────────────────────────────

best_weights = os.path.join(TRAIN_OUTPUT_DIR, "weights", "best.pt")
model = YOLO(best_weights)

results = model(
    TEST_IMAGE_DIR,
    save=True,
    project=os.path.dirname(PREDICT_OUTPUT_DIR),
    name=os.path.basename(PREDICT_OUTPUT_DIR),
)


# ─────────────────────────────────────────────
# Step 3: Visualize results with coverage stats
# ─────────────────────────────────────────────

pio.renderers.default = "browser"

for result in results:
    orig_path = result.path
    base_name = os.path.basename(orig_path)

    orig_img = cv2.imread(orig_path)
    if orig_img is None:
        print(f"Could not load image {orig_path}. Skipping.")
        continue

    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    h, w = orig_img.shape[:2]
    total_pixels = h * w

    mask = np.zeros((h, w), dtype=np.uint8)

    # Combine all segmentation masks into one
    if result.masks is not None and result.masks.data is not None:
        for m in result.masks.data:
            m_np = m.cpu().numpy() if hasattr(m, "cpu") else m
            m_bin = (m_np > 0.5).astype(np.uint8)
            mask = np.maximum(mask, m_bin)

    mask_pixel_count = int(np.sum(mask))
    coverage = (mask_pixel_count / total_pixels) * 100

    saved_img_path = os.path.join(PREDICT_OUTPUT_DIR, base_name)
    if not os.path.exists(saved_img_path):
        print(f"Saved image for {base_name} not found in {PREDICT_OUTPUT_DIR}.")
        continue

    saved_img = cv2.imread(saved_img_path)
    saved_img = cv2.cvtColor(saved_img, cv2.COLOR_BGR2RGB)

    fig = px.imshow(
        saved_img,
        title=f"{base_name}: {coverage:.2f}% coverage, {mask_pixel_count} pixels"
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.show()