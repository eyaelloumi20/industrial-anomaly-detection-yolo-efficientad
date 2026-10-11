"""
pipeline.py
===========
Two-Stage Detection → EfficientAD Anomaly-Scoring Pipeline
for VisA Chewing-Gum quality inspection.

CRITICAL — preprocessing must match EfficientAD training EXACTLY:
    Training used:  Resize(256,256) → ToTensor()   ← NO ImageNet normalisation
    The channel_mean/std in the checkpoint are teacher FEATURE statistics
    (computed on ImageNet feature maps), NOT pixel-level statistics.
    Never add transforms.Normalize() to the input image.

Pipeline per image:
  Stage 1 — YOLOv8 detects each gum instance → bounding boxes
  Stage 2 — For each crop:
               a. Pad 5% context around the YOLO box
               b. Resize to 256×256 with LANCZOS
               c. Gaussian blur (3×3, σ=0.8)  — removes resize aliasing
               d. Bilateral filter (d=5, σ=15) — smooths, preserves edges
               e. ToTensor()  (values in [0,1], NO further normalisation)
               f. EfficientAD forward pass → anomaly score
  Aggregation — max score across all instances (worst-case decision)
"""

from __future__ import annotations

import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional
from PIL import Image
from torchvision import transforms

from detection import Detector, BBox

IMAGE_SIZE   = 256
OUT_CHANNELS = 384

# Exactly what EfficientAD training used — ToTensor ONLY
_to_tensor = transforms.ToTensor()


# ── crop preprocessing ─────────────────────────────────────────────────────

def preprocess_crop(
    pil_image:   Image.Image,
    box:         BBox,
    target_size: int   = IMAGE_SIZE,
    pad_ratio:   float = 0.05,
    denoise:     bool  = True,
) -> torch.Tensor:
    """
    Crop a YOLO box with context padding, resize to 256×256, denoise.
    Transform: Resize → Gaussian blur → Bilateral filter → ToTensor
    NO ImageNet normalisation (matches EfficientAD training).
    Returns torch.Tensor [3, 256, 256] with values in [0, 1].
    """
    W, H = pil_image.size
    x1, y1, x2, y2 = box

    bw    = x2 - x1
    bh    = y2 - y1
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)
    x1 = max(0, x1 - pad_x);  y1 = max(0, y1 - pad_y)
    x2 = min(W, x2 + pad_x);  y2 = min(H, y2 + pad_y)

    crop = pil_image.crop((x1, y1, x2, y2)).convert('RGB')
    crop = crop.resize((target_size, target_size), Image.LANCZOS)

    if denoise:
        arr = np.array(crop, dtype=np.uint8)
        arr = cv2.GaussianBlur(arr, (3, 3), sigmaX=0.8, sigmaY=0.8)
        arr = cv2.bilateralFilter(arr.astype(np.uint8),
                                  d=5, sigmaColor=15, sigmaSpace=15)
        crop = Image.fromarray(arr)

    return _to_tensor(crop)   # [3, 256, 256], values in [0, 1]


# ── result dataclass ───────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    image_score:   float
    prediction:    str
    anomaly_map:   np.ndarray = field(repr=False)
    crop_scores:   List[float] = field(default_factory=list)
    boxes:         List[BBox]  = field(default_factory=list)
    inference_ms:  float       = 0.0
    label:         Optional[int] = None
    name:          str         = ""
    img_type:      str         = ""

    @property
    def correct(self) -> Optional[bool]:
        if self.label is None:
            return None
        gt = "ANOMALY" if self.label == 1 else "NORMAL"
        return self.prediction == gt


# ── pipeline ───────────────────────────────────────────────────────────────

class InspectionPipeline:
    """
    Parameters
    ----------
    models          : dict from load_efficientad_model()
    device          : "cuda" or "cpu"
    threshold       : binary decision threshold
    detector_path   : YOLOv8 .pt weights path
    conf_threshold  : YOLO confidence threshold (default 0.25)
    iou_threshold   : YOLO NMS IoU threshold (default 0.45)
    use_full_image  : skip YOLO, use whole image as one crop (ablation)
    ratio           : EfficientAD normalisation ratio (default 0.1)
    pad_ratio       : context padding fraction around each YOLO box (0.05)
    denoise         : apply Gaussian+bilateral denoising on crops (True)
    """

    def __init__(
        self,
        models:         dict,
        device:         str   = "cpu",
        threshold:      float = None,
        detector_path:  str   = None,
        conf_threshold: float = 0.25,
        iou_threshold:  float = 0.45,
        use_full_image: bool  = False,
        ratio:          float = 0.1,
        pad_ratio:      float = 0.05,
        denoise:        bool  = True,
    ) -> None:
        self.models    = models
        self.device    = device
        self.threshold = threshold
        self.ratio     = ratio
        self.pad_ratio = pad_ratio
        self.denoise   = denoise

        self.detector = Detector(
            model_path     = detector_path,
            conf_threshold = conf_threshold,
            iou_threshold  = iou_threshold,
            use_full_image = use_full_image,
            device         = device,
        )

    def set_threshold(self, threshold: float) -> None:
        self.threshold = threshold

    def _efficientad_score(self, image_t: torch.Tensor) -> tuple:
        """
        EfficientAD forward pass.
        Input : [1, 3, 256, 256] float32 in [0,1] on device
        Output: anomaly_map [256,256] numpy, image_score float
        """
        with torch.no_grad():
            t_out = self.models['teacher'](image_t)
            s_out = self.models['student'](image_t)
            a_out = self.models['ae'](image_t)

        y_st   = s_out[:, :OUT_CHANNELS,  :, :]
        y_stae = s_out[:, -OUT_CHANNELS:, :, :]

        norm_t = (t_out - self.models['mean']) / (self.models['std'] + 1e-8)
        d_st   = torch.pow(norm_t - y_st,   2)
        d_stae = torch.pow(a_out  - y_stae, 2)

        fm_st   = torch.mean(d_st,   dim=1, keepdim=True)
        fm_stae = torch.mean(d_stae, dim=1, keepdim=True)
        fm_st   = F.interpolate(fm_st,   size=(IMAGE_SIZE, IMAGE_SIZE),
                                mode='bilinear', align_corners=False)
        fm_stae = F.interpolate(fm_stae, size=(IMAGE_SIZE, IMAGE_SIZE),
                                mode='bilinear', align_corners=False)

        norm_mst = (self.ratio * (fm_st   - self.models['qa_st'])) \
                   / (self.models['qb_st'] - self.models['qa_st'] + 1e-8)
        norm_mae = (self.ratio * (fm_stae - self.models['qa_ae'])) \
                   / (self.models['qb_ae'] - self.models['qa_ae'] + 1e-8)

        combined    = 0.5 * norm_mst + 0.5 * norm_mae
        map_np      = combined[0, 0].cpu().numpy()
        image_score = float(np.max(map_np))
        return map_np, image_score

    def run(
        self,
        pil_image: Image.Image,
        label:     int = None,
        name:      str = "",
        img_type:  str = "",
    ) -> PipelineResult:
        """
        Run the two-stage pipeline on one image.

        If YOLO finds no objects above conf_threshold, returns a result
        with image_score=0.0 and prediction="NO_DETECTION".
        The image is NOT scored — no full-image fallback.

        For each detected instance:
          padded → LANCZOS resized → denoised → EfficientAD scored
        Final score = max over all detected instances.
        """
        t0 = time.perf_counter()

        boxes = self.detector.detect(pil_image)

        # No objects detected — skip scoring entirely
        if not boxes:
            return PipelineResult(
                image_score  = 0.0,
                prediction   = "NO_DETECTION",
                anomaly_map  = np.zeros((IMAGE_SIZE, IMAGE_SIZE),
                                        dtype=np.float32),
                crop_scores  = [],
                boxes        = [],
                inference_ms = (time.perf_counter() - t0) * 1000.0,
                label        = label,
                name         = name,
                img_type     = img_type,
            )

        crop_scores  = []
        anomaly_maps = []

        for box in boxes:
            crop_t = preprocess_crop(
                pil_image   = pil_image,
                box         = box,
                target_size = IMAGE_SIZE,
                pad_ratio   = self.pad_ratio,
                denoise     = self.denoise,
            ).unsqueeze(0).to(self.device)

            map_np, score = self._efficientad_score(crop_t)
            crop_scores.append(score)
            anomaly_maps.append(map_np)

        final_score = float(np.max(crop_scores))
        best_idx    = int(np.argmax(crop_scores))
        anomaly_map = anomaly_maps[best_idx]

        t1 = time.perf_counter()
        prediction = ("ANOMALY" if final_score >= (self.threshold or 0.5)
                      else "NORMAL")

        return PipelineResult(
            image_score  = final_score,
            prediction   = prediction,
            anomaly_map  = anomaly_map,
            crop_scores  = crop_scores,
            boxes        = boxes,
            inference_ms = (t1 - t0) * 1000.0,
            label        = label,
            name         = name,
            img_type     = img_type,
        )