"""
pipeline_adaptive.py
=====================
Resolution-aware sibling of pipeline/pipeline.py.

The original InspectionPipeline hardcodes IMAGE_SIZE = 256 in three places:
  1. preprocess_crop's default target_size (crop resize)
  2. _efficientad_score's F.interpolate target size for the anomaly map
  3. the zero-map returned on NO_DETECTION

This version threads an `image_size` parameter through instead, so the same
pipeline logic works against a lowres (192px/224px) checkpoint. Everything
else -- the two-stage YOLO-crop -> pad -> resize -> denoise flow, the FIX 4 /
FIX 5 preprocessing fixes, the PipelineResult dataclass, the max-over-crops
aggregation -- is UNCHANGED from the original, copied verbatim.

If you don't pass `image_size` explicitly, InspectionPipeline reads it from
`models['image_size']` (as returned by load_efficientad_adaptive.py), so in
practice you usually don't need to think about it at all -- load the
checkpoint, build the pipeline, it just matches.
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

IMAGE_SIZE   = 256   # fallback default only -- actual resolution comes from
                      # the loaded checkpoint via models['image_size']
OUT_CHANNELS = 384

_to_tensor = transforms.ToTensor()


def _resize_with_padding(pil_img: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    """Aspect-ratio-preserving resize + black pad to (size x size).
    Identical to pipeline.py's version -- unchanged, just parameterized
    by whatever `size` the caller passes (the checkpoint's own image_size)."""
    ow, oh  = pil_img.size
    scale   = size / max(ow, oh)
    nw, nh  = int(ow * scale), int(oh * scale)
    resized = pil_img.resize((nw, nh), Image.LANCZOS)
    canvas  = Image.new('RGB', (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def preprocess_crop(
    pil_image:   Image.Image,
    box:         BBox,
    target_size: int   = IMAGE_SIZE,
    pad_ratio:   float = 0.05,
    denoise:     bool  = True,
) -> torch.Tensor:
    """Identical to pipeline.py's preprocess_crop -- target_size was already
    a parameter there, this file just makes sure callers always pass the
    checkpoint's actual image_size instead of relying on the 256 default."""
    W, H = pil_image.size
    x1, y1, x2, y2 = box

    bw    = x2 - x1
    bh    = y2 - y1
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)
    x1 = max(0, x1 - pad_x);  y1 = max(0, y1 - pad_y)
    x2 = min(W, x2 + pad_x);  y2 = min(H, y2 + pad_y)

    crop = pil_image.crop((x1, y1, x2, y2)).convert('RGB')
    crop = _resize_with_padding(crop, size=target_size)

    if denoise:
        arr = np.array(crop, dtype=np.uint8)
        arr = cv2.GaussianBlur(arr, (3, 3), sigmaX=0.8, sigmaY=0.8)
        arr = cv2.bilateralFilter(arr, d=5, sigmaColor=15, sigmaSpace=15)
        crop = Image.fromarray(arr)

    return _to_tensor(crop)


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


class InspectionPipeline:
    """
    Same as pipeline.py's InspectionPipeline, with one addition:

    image_size : int, optional
        Resolution to resize crops to and to size the anomaly map at.
        If omitted, read from models['image_size'] (set by
        load_efficientad_adaptive.py); falls back to 256 if the models
        dict doesn't have that key (e.g. loaded with the original,
        non-adaptive loader).
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
        image_size:     int   = None,
    ) -> None:
        self.models     = models
        self.device     = device
        self.threshold  = threshold
        self.ratio      = ratio
        self.pad_ratio  = pad_ratio
        self.denoise    = denoise
        self.image_size = image_size or models.get('image_size', IMAGE_SIZE)

        if use_full_image:
            import warnings
            warnings.warn(
                "use_full_image=True: the model was trained on YOLO crops. "
                "Scoring full images will produce out-of-distribution activations "
                "and near-random AUROC.  Only use this flag if you retrained on "
                "full images.",
                UserWarning, stacklevel=2,
            )

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
        """Same as pipeline.py's version, but interpolates to self.image_size
        instead of the hardcoded module constant."""
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
        fm_st   = F.interpolate(fm_st,   size=(self.image_size, self.image_size),
                                mode='bilinear', align_corners=False)
        fm_stae = F.interpolate(fm_stae, size=(self.image_size, self.image_size),
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
        """Same as pipeline.py's run(), passing self.image_size through to
        preprocess_crop and the NO_DETECTION zero-map instead of the
        hardcoded module constant."""
        t0 = time.perf_counter()

        boxes = self.detector.detect(pil_image)

        if not boxes:
            return PipelineResult(
                image_score  = 0.0,
                prediction   = "NO_DETECTION",
                anomaly_map  = np.zeros((self.image_size, self.image_size), dtype=np.float32),
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
                target_size = self.image_size,
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
