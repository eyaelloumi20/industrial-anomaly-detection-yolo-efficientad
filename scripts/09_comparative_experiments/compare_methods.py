"""
compare_preproc_methods.py
==========================
Multi-object pre-processing comparison pipeline.
NO denoising. Evaluated BEFORE anomaly detection.

What is compared
----------------
Detection stage
  • Your custom YOLO-Detection  weights  (--yolo_det_weights)
  • Your custom YOLO-Segmentation weights (--yolo_seg_weights)
  • YOLOv8n  /  YOLOv8s  (pretrained COCO fallbacks)

Background-removal stage  (applied on crops from best detector)
  • GrabCut  |  Otsu+Morph  |  Watershed  |  U2Net/rembg (optional)

Classical segmentation stage  (full-image)
  • Contour  |  Watershed-Seg  |  SLIC-Superpixel  |  SAM-lite (optional)

EfficientAD anomaly-detection stage
  • Scores every crop from EVERY detection/bg-removal/segmentation method
  • Reports: anomaly score, is-anomaly flag, EfficientAD-only inference time
  • Timing is split into:
      load_ms   — model / weight loading (one-time, reported separately)
      infer_ms  — actual forward-pass inference on the image / crop

HTML report
  • Summary timing table  (load + infer separated, with/without EfficientAD)
  • Per-method cards: annotated image + crops + overlays + anomaly heatmap
  • EfficientAD per-crop anomaly scores and verdict

Usage
-----
  # Minimal (pretrained YOLO fallback, no EfficientAD)
  python compare_preproc_methods.py --image photo.jpg

  # Full run with your trained models + EfficientAD
  python compare_preproc_methods.py \\
      --image            photo.jpg \\
      --yolo_det_weights runs/detect/best.pt \\
      --yolo_seg_weights runs/segment/best.pt \\
      --ckpt_dir         path/to/efficientad_ckpt \\
      --category         chewinggum \\
      --model_size       S \\
      --save_dir         ./comparison_results
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import textwrap
import warnings
import base64
import traceback
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

warnings.filterwarnings("ignore")

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ─────────────────────────────────────────────────────────────────────────────
# Optional heavy deps — imported lazily; script runs without them
# ─────────────────────────────────────────────────────────────────────────────

def _try_import(name: str):
    try:
        import importlib
        return importlib.import_module(name)
    except ImportError:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Detection:
    bbox:     Tuple[int, int, int, int]   # x1 y1 x2 y2 (pixel coords)
    conf:     float
    cls_id:   int
    cls_name: str
    mask:     Optional[np.ndarray] = None  # H×W bool, only from seg model


@dataclass
class AnomalyResult:
    """EfficientAD result for a single crop."""
    crop_idx:    int
    score:       float
    is_anomaly:  bool
    threshold:   float
    infer_ms:    float                      # EfficientAD forward-pass only
    heatmap:     Optional[np.ndarray] = None  # BGR uint8 heatmap overlay


@dataclass
class MethodResult:
    method_name:     str
    category:        str          # 'detection' | 'bg_removal' | 'segmentation'
    load_ms:         float        # weight / model loading time
    infer_ms:        float        # actual inference on the image
    num_objects:     int
    detections:      List[Detection]        = field(default_factory=list)
    annotated_image: Optional[np.ndarray]  = None
    crops:           List[np.ndarray]      = field(default_factory=list)
    crop_overlays:   List[np.ndarray]      = field(default_factory=list)
    masks:           Optional[np.ndarray]  = None   # full label map
    anomaly_results: List[AnomalyResult]   = field(default_factory=list)
    error:           Optional[str]         = None
    extra:           Dict[str, Any]        = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return self.load_ms + self.infer_ms

    @property
    def ead_total_ms(self) -> float:
        return sum(a.infer_ms for a in self.anomaly_results)


# ══════════════════════════════════════════════════════════════════════════════
# Visual helpers
# ══════════════════════════════════════════════════════════════════════════════

PALETTE = [
    (255, 56,  56),  (255, 157, 151), (255, 112,  31), (255, 178,  29),
    (207, 210,  49), ( 72, 249,  10), (146, 204,  23), ( 61, 219, 134),
    ( 26, 147,  52), (  0, 212, 187), ( 44, 153, 168), (  0, 194, 255),
    ( 52,  69, 147), (100, 115, 255), (  0,  24, 236), (132,  56, 255),
    ( 82,   0, 133), (203,  56, 255), (255,  49, 203), (255,  50, 101),
]

def color_for(idx: int) -> Tuple[int, int, int]:
    return PALETTE[int(idx) % len(PALETTE)]


def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def cv_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def _font(size: int = 14) -> ImageFont.FreeTypeFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_boxes_pil(image: Image.Image,
                   detections: List[Detection],
                   anomaly_results: Optional[List[AnomalyResult]] = None) -> Image.Image:
    img  = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    fnt  = _font(13)
    ead_map = {a.crop_idx: a for a in (anomaly_results or [])}
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det.bbox
        col = color_for(det.cls_id)
        for t in range(3):
            draw.rectangle([x1-t, y1-t, x2+t, y2+t], outline=col)
        label = f"{det.cls_name} {det.conf:.2f}"
        if i in ead_map:
            ar = ead_map[i]
            verdict = "ANOMALY" if ar.is_anomaly else "OK"
            label += f"  [{verdict} {ar.score:.3f}]"
        tw = len(label) * 7
        th = 14
        draw.rectangle([x1, y1 - th - 2, x1 + tw, y1], fill=col)
        draw.text((x1 + 2, y1 - th - 1), label, fill=(255, 255, 255), font=fnt)
    return img


def make_crop_overlay(orig_crop: np.ndarray,
                      mask: np.ndarray,
                      color: Tuple[int, int, int] = (0, 200, 80),
                      alpha: float = 0.45) -> np.ndarray:
    overlay = orig_crop.copy()
    m = (mask > 0) if mask.dtype != bool else mask
    overlay[m] = (np.array(color[::-1]) * alpha +
                  orig_crop[m].astype(float) * (1 - alpha)).astype(np.uint8)
    contours, _ = cv2.findContours(m.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color[::-1], 2)
    return overlay


def make_heatmap_overlay(crop_bgr: np.ndarray,
                         anomaly_map: np.ndarray,
                         alpha: float = 0.5) -> np.ndarray:
    """Colour-map a float anomaly map and blend it on the crop."""
    h, w = crop_bgr.shape[:2]
    amap = cv2.resize(anomaly_map, (w, h), interpolation=cv2.INTER_LINEAR)
    mn, mx = amap.min(), amap.max()
    if mx > mn:
        amap = ((amap - mn) / (mx - mn) * 255).astype(np.uint8)
    else:
        amap = np.zeros_like(amap, dtype=np.uint8)
    heat = cv2.applyColorMap(amap, cv2.COLORMAP_JET)
    return cv2.addWeighted(crop_bgr, 1 - alpha, heat, alpha, 0)


def pad_crop(img_bgr: np.ndarray, pad: int = 8) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    canvas = np.zeros((h + 2*pad, w + 2*pad, 3), dtype=np.uint8)
    canvas[pad:pad+h, pad:pad+w] = img_bgr
    return canvas


def resize_keep_aspect(img: np.ndarray, max_side: int = 640) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / max(h, w)
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)


def to_b64_png(arr: np.ndarray, max_side: int = 0) -> str:
    if max_side > 0:
        arr = resize_keep_aspect(arr, max_side)
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def _resize_with_padding(pil_img: Image.Image, size: int = 256) -> Image.Image:
    """Aspect-ratio-preserving resize + black pad — matches EfficientAD training."""
    ow, oh = pil_img.size
    scale  = size / max(ow, oh)
    nw, nh = int(ow * scale), int(oh * scale)
    resized = pil_img.resize((nw, nh), Image.LANCZOS)
    canvas  = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


# ══════════════════════════════════════════════════════════════════════════════
# EfficientAD loader  (lazy, separated load vs infer timing)
# ══════════════════════════════════════════════════════════════════════════════

class EfficientADRunner:
    """
    Wraps load_efficientad_model + per-crop scoring.
    Keeps load_ms separate from per-crop infer_ms.
    """

    def __init__(self):
        self.models    = None
        self.device    = "cpu"
        self.ratio     = 0.1
        self.img_size  = 256
        self.load_ms   = 0.0
        self.threshold = None    # set after calibration or from threshold_file
        self._available = False
        self._error     = ""

    def load(self, ckpt_dir: str, category: str,
             model_size: str, device: str, ratio: float,
             threshold_file: Optional[str] = None) -> bool:
        """Load EfficientAD weights. Returns True on success."""
        try:
            # We import here so the rest of the script works without the package
            _HERE = os.path.dirname(os.path.abspath(__file__))
            if _HERE not in sys.path:
                sys.path.insert(0, _HERE)
            from utils.load_efficientad import load_efficientad_model

            t0 = time.perf_counter()
            self.models  = load_efficientad_model(
                ckpt_dir   = ckpt_dir,
                category   = category,
                model_size = model_size,
                device     = device,
            )
            self.load_ms   = (time.perf_counter() - t0) * 1000
            self.device    = device
            self.ratio     = ratio
            self._available = True

            # threshold
            if threshold_file and os.path.isfile(threshold_file):
                with open(threshold_file) as f:
                    self.threshold = float(f.read().strip())
                print(f"    [EfficientAD] threshold loaded from file: {self.threshold:.6f}")
            else:
                # No calibration set here — use a sentinel; will be flagged in HTML
                self.threshold = None

            print(f"    [EfficientAD] loaded in {self.load_ms:.1f} ms  "
                  f"(device={device}, ratio={ratio})")
            return True

        except Exception as e:
            self._error     = traceback.format_exc(limit=5)
            self._available = False
            print(f"    [EfficientAD] SKIP — {e}")
            return False

    def score_crop(self, crop_bgr: np.ndarray, crop_idx: int) -> AnomalyResult:
        """
        Score a single BGR crop. Returns AnomalyResult.
        infer_ms = forward-pass time only (no loading).
        """
        if not self._available or self.models is None:
            return AnomalyResult(crop_idx, 0.0, False, 0.0, 0.0)

        import torch
        from torchvision import transforms

        tf = transforms.Compose([
            transforms.Lambda(lambda p: _resize_with_padding(p, self.img_size)),
            transforms.ToTensor(),
        ])

        pil_crop = cv_to_pil(crop_bgr)
        tensor   = tf(pil_crop).unsqueeze(0).to(self.device)

        teacher = self.models.get("teacher") or self.models.get("st")
        student = self.models.get("student") or self.models.get("ts")
        ae      = self.models.get("ae")

        if teacher is None or student is None:
            return AnomalyResult(crop_idx, 0.0, False, 0.0, 0.0)

        t0 = time.perf_counter()
        with torch.no_grad():
            t_out = teacher(tensor)
            s_out = student(tensor)
            # EfficientAD student always outputs 2×C channels (C = teacher channels):
            # first half = ST map (teacher imitation), second half = AE imitation.
            # Split unconditionally so shapes always match t_out.
            if isinstance(s_out, (list, tuple)) and len(s_out) >= 2:
                s_st, s_ae = s_out[0], s_out[1]
            else:
                C = s_out.shape[1]
                s_st = s_out[:, : C // 2]   # first 384 → matches teacher
                s_ae = s_out[:, C // 2 :]   # last  384 → matches autoencoder

            map_st = torch.mean((t_out - s_st) ** 2, dim=1, keepdim=True)

            if ae is not None:
                ae_out = ae(tensor)
                map_ae = torch.mean((ae_out - s_ae) ** 2, dim=1, keepdim=True)
                amap   = (map_st + map_ae) / 2.0
            else:
                amap = map_st

            # normalise with ratio quantile (mirrors EfficientAD paper)
            flat = amap.flatten()
            q_lo = torch.quantile(flat, self.ratio)
            q_hi = torch.quantile(flat, 1.0 - self.ratio)
            if q_hi > q_lo:
                amap = (amap - q_lo) / (q_hi - q_lo)
            score = float(amap.max())

        infer_ms = (time.perf_counter() - t0) * 1000

        # build heatmap overlay
        amap_np = amap.squeeze().cpu().numpy()
        h, w    = crop_bgr.shape[:2]
        heatmap = make_heatmap_overlay(crop_bgr, amap_np)

        thr         = self.threshold if self.threshold is not None else 0.5
        is_anomaly  = score >= thr

        return AnomalyResult(
            crop_idx   = crop_idx,
            score      = score,
            is_anomaly = is_anomaly,
            threshold  = thr,
            infer_ms   = infer_ms,
            heatmap    = heatmap,
        )

    def score_crops(self, crops: List[np.ndarray]) -> List[AnomalyResult]:
        return [self.score_crop(c, i) for i, c in enumerate(crops)]


# ══════════════════════════════════════════════════════════════════════════════
# ── DETECTION methods
# ══════════════════════════════════════════════════════════════════════════════

def _yolo_load(weights: str) -> Tuple[Any, float]:
    """Load YOLO model, return (model, load_ms)."""
    from ultralytics import YOLO
    t0 = time.perf_counter()
    model = YOLO(weights)
    return model, (time.perf_counter() - t0) * 1000


def run_yolo_detection(image_bgr: np.ndarray,
                       weights: str,
                       conf: float,
                       iou: float,
                       label: str,
                       ead: "EfficientADRunner") -> MethodResult:
    """Standard YOLO object-detection model."""
    try:
        from ultralytics import YOLO
    except ImportError:
        return MethodResult(label, "detection", 0, 0, 0,
                            error="ultralytics not installed  →  pip install ultralytics")
    try:
        model, load_ms = _yolo_load(weights)
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        names   = model.names

        t0 = time.perf_counter()
        results = model.predict(pil_img, conf=conf, iou=iou, verbose=False, device="cpu")
        infer_ms = (time.perf_counter() - t0) * 1000

        detections: List[Detection] = []
        result = results[0]
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                c  = float(box.conf[0])
                ci = int(box.cls[0])
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2), conf=c,
                    cls_id=ci, cls_name=names.get(ci, str(ci)),
                ))

        crops, crop_overlays = _extract_crops(image_bgr, detections)
        anomaly_results = ead.score_crops(crops) if ead._available else []

        ann_pil = draw_boxes_pil(pil_img, detections, anomaly_results)
        ann_bgr = pil_to_cv(ann_pil)
        _stamp_anomaly_on_overlays(crop_overlays, anomaly_results)

        return MethodResult(
            method_name=label, category="detection",
            load_ms=load_ms, infer_ms=infer_ms,
            num_objects=len(detections),
            detections=detections,
            annotated_image=ann_bgr,
            crops=crops, crop_overlays=crop_overlays,
            anomaly_results=anomaly_results,
        )
    except Exception:
        return MethodResult(label, "detection", 0, 0, 0,
                            error=traceback.format_exc(limit=5))


def run_yolo_segmentation(image_bgr: np.ndarray,
                          weights: str,
                          conf: float,
                          iou: float,
                          label: str,
                          ead: "EfficientADRunner") -> MethodResult:
    """YOLO segmentation model (outputs masks + boxes)."""
    try:
        from ultralytics import YOLO
    except ImportError:
        return MethodResult(label, "detection", 0, 0, 0,
                            error="ultralytics not installed")
    try:
        model, load_ms = _yolo_load(weights)
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        names   = model.names
        h, w    = image_bgr.shape[:2]

        t0 = time.perf_counter()
        results = model.predict(pil_img, conf=conf, iou=iou, verbose=False, device="cpu")
        infer_ms = (time.perf_counter() - t0) * 1000

        detections: List[Detection] = []
        result = results[0]
        if result.boxes is not None:
            masks_data = result.masks.data if result.masks is not None else None
            for idx, box in enumerate(result.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                c  = float(box.conf[0])
                ci = int(box.cls[0])
                seg_mask = None
                if masks_data is not None and idx < len(masks_data):
                    m_np = masks_data[idx].cpu().numpy()
                    seg_mask = cv2.resize(m_np, (w, h)) > 0.5
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2), conf=c,
                    cls_id=ci, cls_name=names.get(ci, str(ci)),
                    mask=seg_mask,
                ))

        # ── crops use per-instance mask where available
        crops, crop_overlays = _extract_crops(image_bgr, detections, use_mask=True)
        anomaly_results = ead.score_crops(crops) if ead._available else []

        ann_pil = draw_boxes_pil(pil_img, detections, anomaly_results)
        ann_bgr = pil_to_cv(ann_pil)

        # additionally paint segmentation masks on annotated image
        colour_seg = np.zeros_like(image_bgr)
        for i, det in enumerate(detections):
            if det.mask is not None:
                col = color_for(det.cls_id)
                colour_seg[det.mask] = col[::-1]
        if colour_seg.any():
            ann_bgr = cv2.addWeighted(ann_bgr, 0.7, colour_seg, 0.3, 0)

        _stamp_anomaly_on_overlays(crop_overlays, anomaly_results)

        return MethodResult(
            method_name=label, category="detection",
            load_ms=load_ms, infer_ms=infer_ms,
            num_objects=len(detections),
            detections=detections,
            annotated_image=ann_bgr,
            crops=crops, crop_overlays=crop_overlays,
            anomaly_results=anomaly_results,
        )
    except Exception:
        return MethodResult(label, "detection", 0, 0, 0,
                            error=traceback.format_exc(limit=5))


# ── shared crop / overlay helpers ─────────────────────────────────────────────

def _extract_crops(image_bgr: np.ndarray,
                   detections: List[Detection],
                   use_mask: bool = False) -> Tuple[List, List]:
    h, w = image_bgr.shape[:2]
    crops, overlays = [], []
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det.bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = image_bgr[y1:y2, x1:x2].copy()
        if crop.size == 0:
            continue
        if use_mask and det.mask is not None:
            crop_mask = det.mask[y1:y2, x1:x2].astype(np.uint8) * 255
        else:
            crop_mask = np.ones(crop.shape[:2], dtype=np.uint8) * 255
        ov  = make_crop_overlay(crop, crop_mask, color_for(det.cls_id))
        lbl = f"{det.cls_name} {det.conf:.2f}"
        cv2.putText(ov, lbl, (4, 16), cv2.FONT_HERSHEY_DUPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
        crops.append(pad_crop(crop))
        overlays.append(pad_crop(ov))
    return crops, overlays


def _stamp_anomaly_on_overlays(overlays: List[np.ndarray],
                                anomaly_results: List[AnomalyResult]) -> None:
    """Write anomaly verdict on bottom of each overlay in-place."""
    for ar in anomaly_results:
        idx = ar.crop_idx
        if idx >= len(overlays):
            continue
        ov = overlays[idx]
        verdict = "ANOMALY" if ar.is_anomaly else "NORMAL"
        col = (0, 0, 220) if ar.is_anomaly else (0, 200, 80)
        h, w = ov.shape[:2]
        cv2.rectangle(ov, (0, h - 20), (w, h), col, -1)
        txt = f"{verdict}  {ar.score:.3f}"
        cv2.putText(ov, txt, (4, h - 5), cv2.FONT_HERSHEY_DUPLEX,
                    0.4, (255, 255, 255), 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
# ── BACKGROUND REMOVAL methods
# ══════════════════════════════════════════════════════════════════════════════

def _run_grabcut(crop_bgr: np.ndarray) -> np.ndarray:
    h, w = crop_bgr.shape[:2]
    mask_gc = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    rect = (max(1, w//10), max(1, h//10),
            w - max(2, w//5), h - max(2, h//5))
    try:
        cv2.grabCut(crop_bgr, mask_gc, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    except Exception:
        return np.ones((h, w), dtype=np.uint8) * 255
    return np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD),
                    255, 0).astype(np.uint8)


def _run_otsu(crop_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k, iterations=1)
    return binary


def _run_watershed_bg(crop_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, sure_bg = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_bg_d = cv2.dilate(sure_bg, k, iterations=3)
    dist = cv2.distanceTransform(sure_bg, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.4 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg_d, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers_ws = cv2.watershed(crop_bgr.copy(), markers)
    binary = np.zeros(gray.shape, dtype=np.uint8)
    binary[markers_ws > 1] = 255
    return binary


def _run_rembg(crop_bgr: np.ndarray) -> np.ndarray:
    rembg_m = _try_import("rembg")
    if rembg_m is None:
        raise ImportError("rembg not installed  →  pip install rembg")
    pil  = cv_to_pil(crop_bgr)
    out  = rembg_m.remove(pil)
    alpha = np.array(out)[:, :, 3]
    return (alpha > 30).astype(np.uint8) * 255


def run_bg_removal(image_bgr: np.ndarray,
                   detections: List[Detection],
                   method_fn,
                   label: str,
                   ead: "EfficientADRunner") -> MethodResult:
    h, w = image_bgr.shape[:2]
    if not detections:
        detections = [Detection((0, 0, w, h), 1.0, 0, "object")]

    crops_out, overlays_out = [], []
    load_ms = 0.0   # these are pure CV algorithms — no model loading

    t0 = time.perf_counter()
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det.bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = image_bgr[y1:y2, x1:x2].copy()
        if crop.size == 0:
            continue
        try:
            mask = method_fn(crop)
        except Exception:
            mask = np.ones(crop.shape[:2], dtype=np.uint8) * 255
        col = color_for(i)
        ov  = make_crop_overlay(crop, mask, col)
        lbl = f"{det.cls_name} {det.conf:.2f}"
        cv2.putText(ov, lbl, (4, 16), cv2.FONT_HERSHEY_DUPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
        crops_out.append(pad_crop(crop))
        overlays_out.append(pad_crop(ov))
    infer_ms = (time.perf_counter() - t0) * 1000

    anomaly_results = ead.score_crops(crops_out) if ead._available else []
    _stamp_anomaly_on_overlays(overlays_out, anomaly_results)

    pil_full = cv_to_pil(image_bgr)
    ann_pil  = draw_boxes_pil(pil_full, detections, anomaly_results)
    ann_bgr  = pil_to_cv(ann_pil)

    return MethodResult(
        method_name=label, category="bg_removal",
        load_ms=load_ms, infer_ms=infer_ms,
        num_objects=len(detections),
        detections=detections,
        annotated_image=ann_bgr,
        crops=crops_out, crop_overlays=overlays_out,
        anomaly_results=anomaly_results,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ── SEGMENTATION methods
# ══════════════════════════════════════════════════════════════════════════════

def _seg_contour(image_bgr: np.ndarray) -> Tuple[np.ndarray, List[Detection]]:
    gray  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 30, 100)
    k     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dil   = cv2.dilate(edges, k, iterations=2)
    contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    label_map = np.zeros(gray.shape, dtype=np.int32)
    dets: List[Detection] = []
    min_area = gray.shape[0] * gray.shape[1] * 0.005
    idx = 0
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(cnt) < min_area:
            continue
        cv2.drawContours(label_map, [cnt], -1, idx + 1, cv2.FILLED)
        x, y, cw, ch = cv2.boundingRect(cnt)
        dets.append(Detection((x, y, x+cw, y+ch), 1.0, idx, f"obj_{idx+1}"))
        idx += 1
        if idx >= 20:
            break
    return label_map.astype(np.uint8), dets


def _seg_watershed(image_bgr: np.ndarray) -> Tuple[np.ndarray, List[Detection]]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=2)
    sure_bg = cv2.dilate(opening, k, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.4 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    n, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers_ws = cv2.watershed(image_bgr.copy(), markers)
    label_map  = np.clip(markers_ws - 1, 0, 255).astype(np.uint8)
    dets: List[Detection] = []
    h, w = image_bgr.shape[:2]
    min_area = h * w * 0.005
    for lbl in range(1, n + 1):
        ys, xs = np.where(markers_ws == lbl + 1)
        if len(xs) < min_area:
            continue
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        dets.append(Detection((int(x1), int(y1), int(x2), int(y2)),
                               1.0, lbl - 1, f"seg_{lbl}"))
    return label_map, dets


def _seg_slic(image_bgr: np.ndarray) -> Tuple[np.ndarray, List[Detection]]:
    skimage_seg = _try_import("skimage.segmentation")
    if skimage_seg is None:
        raise ImportError("scikit-image not installed  →  pip install scikit-image")
    img_rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    segments = skimage_seg.slic(img_rgb, n_segments=60, compactness=15,
                                sigma=1, start_label=1)
    label_map = segments.astype(np.uint8)
    dets: List[Detection] = []
    h, w = image_bgr.shape[:2]
    min_area = h * w * 0.005
    for lbl in np.unique(segments):
        ys, xs = np.where(segments == lbl)
        if len(xs) < min_area:
            continue
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        if (x2-x1)*(y2-y1) < min_area:
            continue
        dets.append(Detection((x1, y1, x2, y2), 1.0, int(lbl), f"sp_{lbl}"))
    return label_map, dets


def _seg_sam_lite(image_bgr: np.ndarray) -> Tuple[np.ndarray, List[Detection]]:
    try:
        from ultralytics import SAM
        model    = SAM("mobile_sam.pt")
        img_rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results  = model(img_rgb, verbose=False)
        label_map = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        dets: List[Detection] = []
        if results and results[0].masks is not None:
            for i, (m, box) in enumerate(zip(results[0].masks.data,
                                              results[0].boxes.xyxy)):
                mask_np = m.cpu().numpy().astype(bool)
                label_map[mask_np] = (i + 1) % 255
                x1, y1, x2, y2 = map(int, box.tolist())
                dets.append(Detection((x1, y1, x2, y2), 1.0, i, f"sam_{i+1}"))
        return label_map, dets
    except Exception as e:
        raise ImportError(f"SAM-lite unavailable: {e}")


def run_segmentation(image_bgr: np.ndarray,
                     seg_fn,
                     label: str,
                     ead: "EfficientADRunner") -> MethodResult:
    load_ms = 0.0
    try:
        t0 = time.perf_counter()
        label_map, dets = seg_fn(image_bgr)
        infer_ms = (time.perf_counter() - t0) * 1000
    except ImportError as e:
        return MethodResult(label, "segmentation", 0, 0, 0, error=str(e))
    except Exception:
        return MethodResult(label, "segmentation", 0, 0, 0,
                            error=traceback.format_exc(limit=4))

    h, w = image_bgr.shape[:2]
    colour_seg = np.zeros((h, w, 3), dtype=np.uint8)
    for lbl_id in np.unique(label_map):
        if lbl_id == 0:
            continue
        col = color_for(int(lbl_id))
        colour_seg[label_map == lbl_id] = col[::-1]

    ann_bgr = cv2.addWeighted(image_bgr, 0.55, colour_seg, 0.45, 0)

    crops_out, overlays_out = [], []
    for i, det in enumerate(dets):
        x1, y1, x2, y2 = det.bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = image_bgr[y1:y2, x1:x2].copy()
        if crop.size == 0:
            continue
        crop_mask = label_map[y1:y2, x1:x2].copy()
        binary    = (crop_mask == det.cls_id).astype(np.uint8) * 255
        ov        = make_crop_overlay(crop, binary, color_for(i))
        cv2.putText(ov, det.cls_name, (4, 16), cv2.FONT_HERSHEY_DUPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
        crops_out.append(pad_crop(crop))
        overlays_out.append(pad_crop(ov))

    anomaly_results = ead.score_crops(crops_out) if ead._available else []
    _stamp_anomaly_on_overlays(overlays_out, anomaly_results)

    pil_ann = draw_boxes_pil(cv_to_pil(ann_bgr), dets, anomaly_results)
    ann_bgr = pil_to_cv(pil_ann)

    return MethodResult(
        method_name=label, category="segmentation",
        load_ms=load_ms, infer_ms=infer_ms,
        num_objects=len(dets),
        detections=dets,
        annotated_image=ann_bgr,
        crops=crops_out, crop_overlays=overlays_out,
        anomaly_results=anomaly_results,
        masks=label_map,
    )


# ══════════════════════════════════════════════════════════════════════════════
# HTML report
# ══════════════════════════════════════════════════════════════════════════════

CATEGORY_COLOURS = {
    "detection":    "#3b82f6",
    "bg_removal":   "#10b981",
    "segmentation": "#8b5cf6",
}
CATEGORY_LABELS = {
    "detection":    "🔍 Detection",
    "bg_removal":   "✂️ BG Removal",
    "segmentation": "🧩 Segmentation",
}


def _crop_anomaly_html(crops: List[np.ndarray],
                       overlays: List[np.ndarray],
                       anomalies: List[AnomalyResult],
                       max_crops: int = 8) -> str:
    if not crops:
        return "<p class='muted'>No crops.</p>"
    ead_map = {a.crop_idx: a for a in anomalies}
    html = "<div class='crop-strip'>"
    for i, (c, o) in enumerate(zip(crops[:max_crops], overlays[:max_crops])):
        ar = ead_map.get(i)
        verdict_html = ""
        hm_html = ""
        if ar:
            verdict = "ANOMALY" if ar.is_anomaly else "NORMAL"
            vcol    = "#ef4444" if ar.is_anomaly else "#22c55e"
            verdict_html = (
                f"<div class='verdict' style='background:{vcol}'>"
                f"{verdict} &nbsp; score={ar.score:.3f} &nbsp; "
                f"thr={ar.threshold:.3f} &nbsp; ead={ar.infer_ms:.1f}ms"
                f"</div>"
            )
            if ar.heatmap is not None:
                hm_html = (
                    f"<div class='crop-label'>EfficientAD heatmap</div>"
                    f"<img src='{to_b64_png(ar.heatmap, 160)}' class='crop-img'>"
                )
        html += (
            f"<div class='crop-pair'>"
            f"  <div class='crop-label'>#{i+1} crop</div>"
            f"  <img src='{to_b64_png(c, 160)}' class='crop-img'>"
            f"  <div class='crop-label'>overlay</div>"
            f"  <img src='{to_b64_png(o, 160)}' class='crop-img'>"
            f"  {hm_html}"
            f"  {verdict_html}"
            f"</div>"
        )
    if len(crops) > max_crops:
        html += f"<div class='more-label'>+ {len(crops)-max_crops} more crops</div>"
    html += "</div>"
    return html


def build_html(results: List[MethodResult],
               image_bgr: np.ndarray,
               image_path: str,
               ead: "EfficientADRunner") -> str:

    orig_b64 = to_b64_png(resize_keep_aspect(image_bgr, 700))
    ead_load_cell = (
        f"<span class='timing'>{ead.load_ms:.1f} ms</span>"
        if ead._available else
        "<span class='muted'>—</span>"
    )
    ead_thr_str = (
        f"{ead.threshold:.6f}" if (ead._available and ead.threshold is not None)
        else "not calibrated"
    )

    # ── summary table
    table_rows = ""
    for r in results:
        cat_col = CATEGORY_COLOURS.get(r.category, "#666")
        cat_lbl = CATEGORY_LABELS.get(r.category, r.category)
        status  = "✅" if r.error is None else "❌"
        n_objs  = r.num_objects if r.error is None else "—"
        load_s  = f"{r.load_ms:.1f}" if r.load_ms > 0 else "—"
        infer_s = f"{r.infer_ms:.1f}" if r.infer_ms > 0 else "—"
        total_s = f"{r.total_ms:.1f}" if r.total_ms > 0 else "—"
        ead_s   = (f"{r.ead_total_ms:.1f}" if r.anomaly_results else "—")
        grand   = (f"{r.total_ms + r.ead_total_ms:.1f}"
                   if r.anomaly_results else total_s)
        anomaly_count = sum(1 for a in r.anomaly_results if a.is_anomaly)
        ead_col = "#ef4444" if anomaly_count > 0 else "#22c55e"
        ead_txt = (f"<span style='color:{ead_col}'>{anomaly_count}/{r.num_objects}</span>"
                   if r.anomaly_results else "<span class='muted'>—</span>")
        table_rows += (
            f"<tr>"
            f"<td><span class='tag' style='background:{cat_col}'>{cat_lbl}</span></td>"
            f"<td><strong>{r.method_name}</strong></td>"
            f"<td class='center'>{status}</td>"
            f"<td class='center'>{n_objs}</td>"
            f"<td class='center mono'>{load_s}</td>"
            f"<td class='center mono'>{infer_s}</td>"
            f"<td class='center mono bold-g'>{total_s}</td>"
            f"<td class='center mono'>{ead_s}</td>"
            f"<td class='center mono bold-g'>{grand}</td>"
            f"<td class='center'>{ead_txt}</td>"
            f"</tr>"
        )

    # ── fastest per category
    fastest_rows = ""
    for cat in ("detection", "bg_removal", "segmentation"):
        cat_res = [r for r in results
                   if r.category == cat and r.error is None and r.infer_ms > 0]
        if not cat_res:
            continue
        best = min(cat_res, key=lambda x: x.infer_ms)
        cat_col = CATEGORY_COLOURS.get(cat, "#555")
        cat_lbl = CATEGORY_LABELS.get(cat, cat)
        fastest_rows += (
            f"<tr>"
            f"<td><span class='tag' style='background:{cat_col}'>{cat_lbl}</span></td>"
            f"<td><strong>{best.method_name}</strong></td>"
            f"<td class='center mono'>{best.load_ms:.1f}</td>"
            f"<td class='center mono bold-g'>{best.infer_ms:.1f}</td>"
            f"<td class='center mono'>{best.total_ms:.1f}</td>"
            f"<td class='center'>{best.num_objects}</td>"
            f"</tr>"
        )

    # ── per-method cards
    cards_html = ""
    prev_cat   = None
    for r in results:
        if r.category != prev_cat:
            cat_col = CATEGORY_COLOURS.get(r.category, "#555")
            cat_lbl = CATEGORY_LABELS.get(r.category, r.category)
            cards_html += (
                f"<div class='section-divider'>"
                f"<span class='section-pill' style='background:{cat_col}'>{cat_lbl}</span>"
                f"</div>"
            )
            prev_cat = r.category

        err_html = ""
        if r.error:
            err_html = (f"<div class='error-box'><pre>{r.error[:800]}</pre></div>")

        ann_html = ""
        if r.annotated_image is not None:
            ann_html = (
                f"<div class='ann-wrap'>"
                f"  <div class='img-label'>Annotated output</div>"
                f"  <img src='{to_b64_png(r.annotated_image, 560)}' class='ann-img'>"
                f"</div>"
            )

        # timing summary inside card
        timing_html = (
            f"<div class='timing-row'>"
            f"  <span class='titem'>Load: <b>{r.load_ms:.1f} ms</b></span>"
            f"  <span class='titem'>Detect: <b class='bold-g'>{r.infer_ms:.1f} ms</b></span>"
            f"  <span class='titem'>Subtotal: <b>{r.total_ms:.1f} ms</b></span>"
        )
        if r.anomaly_results:
            ac = sum(1 for a in r.anomaly_results if a.is_anomaly)
            ead_col2 = "#ef4444" if ac > 0 else "#22c55e"
            timing_html += (
                f"  <span class='titem'>EfficientAD: <b>{r.ead_total_ms:.1f} ms</b></span>"
                f"  <span class='titem'>Total w/ EAD: <b>{r.total_ms+r.ead_total_ms:.1f} ms</b></span>"
                f"  <span class='titem' style='color:{ead_col2}'>Anomalies: <b>{ac}/{r.num_objects}</b></span>"
            )
        timing_html += "</div>"

        crop_html = _crop_anomaly_html(r.crops, r.crop_overlays, r.anomaly_results)

        cat_col = CATEGORY_COLOURS.get(r.category, "#444")
        cards_html += f"""
<div class='method-card' style='border-left:4px solid {cat_col}'>
  <div class='method-header'>
    <span class='method-name'>{r.method_name}</span>
    <span class='method-meta'>{r.num_objects} object(s)</span>
  </div>
  {err_html}
  {timing_html}
  <div class='method-body'>
    {ann_html}
    <div class='crops-section'>
      <div class='img-label'>Crops · overlays · EfficientAD heatmaps</div>
      {crop_html}
    </div>
  </div>
</div>
"""

    html = f"""<!DOCTYPE html>
<html lang='en'>
<head>
<meta charset='UTF-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Pipeline Pre-processing + EfficientAD Comparison</title>
<style>
  :root {{
    --bg:      #0f1117;
    --surface: #1a1d27;
    --border:  #2a2d3e;
    --text:    #e2e8f0;
    --muted:   #64748b;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Segoe UI', system-ui, sans-serif;
    font-size: 14px;
    line-height: 1.6;
  }}
  header {{
    background: linear-gradient(135deg, #1e1b4b 0%, #1a1d27 100%);
    border-bottom: 1px solid var(--border);
    padding: 24px 32px;
  }}
  header h1 {{ font-size: 22px; font-weight: 700; color: #c7d2fe; }}
  header p  {{ color: var(--muted); margin-top: 4px; font-size: 13px; }}
  .container {{ max-width: 1340px; margin: 0 auto; padding: 32px 24px; }}
  h2 {{
    font-size: 13px; font-weight: 600; color: #a5b4fc;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 14px; padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }}
  .orig-section {{ margin-bottom: 36px; }}
  .orig-section img {{ border-radius: 8px; border: 1px solid var(--border); max-width:100%; }}

  /* ead info bar */
  .ead-info {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 28px;
    display: flex; gap: 24px; flex-wrap: wrap;
  }}
  .ead-item {{ font-size: 13px; }}
  .ead-item b {{ color: #a5b4fc; }}

  table {{
    width: 100%; border-collapse: collapse;
    background: var(--surface);
    border-radius: 8px; overflow: hidden;
  }}
  th {{
    background: #1e1b4b; color: #a5b4fc;
    padding: 9px 12px; text-align: left;
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em;
  }}
  td {{ padding: 9px 12px; border-top: 1px solid var(--border); font-size: 13px; }}
  tr:hover td {{ background: rgba(255,255,255,0.02); }}
  .center {{ text-align: center; }}
  .mono {{ font-family: monospace; }}
  .bold-g {{ color: #34d399; font-weight: 700; }}
  .tag {{
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 11px; font-weight: 600; color: #fff;
  }}
  .summary-section, .fastest-section {{ margin-bottom: 36px; }}

  .section-divider {{
    display: flex; align-items: center; margin: 32px 0 16px; gap: 12px;
  }}
  .section-divider::before, .section-divider::after {{
    content: ''; flex: 1; height: 1px; background: var(--border);
  }}
  .section-pill {{
    padding: 4px 14px; border-radius: 999px;
    font-size: 13px; font-weight: 700; color: #fff; white-space: nowrap;
  }}

  .method-card {{
    background: var(--surface); border-radius: 10px;
    margin-bottom: 20px; overflow: hidden;
  }}
  .method-header {{
    display: flex; align-items: baseline; justify-content: space-between;
    padding: 13px 18px;
    background: rgba(255,255,255,0.03); border-bottom: 1px solid var(--border);
  }}
  .method-name {{ font-size: 15px; font-weight: 700; }}
  .method-meta {{ color: var(--muted); font-size: 12px; }}

  .timing-row {{
    display: flex; flex-wrap: wrap; gap: 16px;
    padding: 10px 18px;
    background: rgba(255,255,255,0.015);
    border-bottom: 1px solid var(--border);
    font-size: 12px;
  }}
  .titem {{ color: var(--muted); }}
  .titem b {{ color: var(--text); }}

  .method-body {{
    padding: 16px 18px; display: flex; gap: 16px; flex-wrap: wrap;
  }}
  .ann-wrap {{ flex: 0 0 auto; }}
  .crops-section {{ flex: 1 1 300px; }}
  .img-label {{
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--muted); margin-bottom: 6px;
  }}
  .ann-img {{
    border-radius: 6px; border: 1px solid var(--border);
    max-width: 560px; width: 100%;
  }}
  .crop-strip {{ display: flex; flex-wrap: wrap; gap: 12px; }}
  .crop-pair {{
    display: flex; flex-direction: column; align-items: center; gap: 3px;
  }}
  .crop-img {{
    border-radius: 4px; border: 1px solid var(--border);
    max-width: 150px; height: auto;
  }}
  .crop-label {{
    font-size: 10px; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.05em;
  }}
  .verdict {{
    font-size: 10px; font-weight: 700; color: #fff;
    border-radius: 4px; padding: 2px 6px; margin-top: 2px;
    text-align: center;
  }}
  .more-label {{
    align-self: center; color: var(--muted); font-size: 12px;
    padding: 4px 8px; background: rgba(255,255,255,0.05); border-radius: 4px;
  }}
  .error-box {{
    margin: 10px 18px; padding: 10px 14px;
    background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.3);
    border-radius: 6px; color: #fca5a5; font-size: 12px;
  }}
  .error-box pre {{ white-space: pre-wrap; word-break: break-word; font-family: monospace; }}
  .muted {{ color: var(--muted); font-size: 12px; }}
  footer {{
    margin-top: 48px; padding: 20px 24px;
    border-top: 1px solid var(--border);
    color: var(--muted); font-size: 12px; text-align: center;
  }}
</style>
</head>
<body>
<header>
  <h1>🔬 Pre-processing + EfficientAD Comparison</h1>
  <p>
    Input: <code>{os.path.basename(image_path)}</code>
    &nbsp;·&nbsp; conf=0.75 &nbsp;·&nbsp; NO denoising
    &nbsp;·&nbsp; Stage: before anomaly detection
  </p>
</header>

<div class='container'>

  <div class='orig-section'>
    <h2>Input Image</h2>
    <img src='{orig_b64}' alt='input image'>
  </div>

  <div class='ead-info'>
    <div class='ead-item'>EfficientAD status: <b>{'✅ loaded' if ead._available else '❌ not loaded'}</b></div>
    <div class='ead-item'>Model load time: <b>{ead_load_cell}</b></div>
    <div class='ead-item'>Anomaly threshold: <b>{ead_thr_str}</b></div>
    {'<div class="ead-item" style="color:#fca5a5">Error: ' + ead._error[:120] + '</div>' if ead._error else ''}
  </div>

  <div class='summary-section'>
    <h2>Inference Time &amp; Detection Summary &nbsp;<span class='muted'>(all times in ms)</span></h2>
    <table>
      <thead>
        <tr>
          <th>Category</th><th>Method</th>
          <th class='center'>Status</th>
          <th class='center'>Objects</th>
          <th class='center'>Load ms</th>
          <th class='center'>Infer ms</th>
          <th class='center'>Total ms</th>
          <th class='center'>EAD ms</th>
          <th class='center'>Total+EAD ms</th>
          <th class='center'>Anomalies</th>
        </tr>
      </thead>
      <tbody>{table_rows}</tbody>
    </table>
  </div>

  <div class='fastest-section'>
    <h2>⚡ Fastest Inference per Category</h2>
    <table>
      <thead>
        <tr>
          <th>Category</th><th>Method</th>
          <th class='center'>Load ms</th>
          <th class='center'>Infer ms</th>
          <th class='center'>Total ms</th>
          <th class='center'>Objects</th>
        </tr>
      </thead>
      <tbody>{fastest_rows}</tbody>
    </table>
  </div>

  <div class='methods-section'>
    <h2>Per-Method Results</h2>
    {cards_html}
  </div>

</div>
<footer>
  compare_preproc_methods.py &nbsp;·&nbsp;
  NO denoising &nbsp;·&nbsp;
  EfficientAD scored per crop &nbsp;·&nbsp;
  Load time and inference time reported separately
</footer>
</body>
</html>"""
    return html


# ══════════════════════════════════════════════════════════════════════════════
# Args
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare detection / bg-removal / segmentation + EfficientAD on a multi-object image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples
        --------
          # Minimal run (pretrained YOLO, no EfficientAD)
          python compare_preproc_methods.py --image photo.jpg

          # Full run with your trained models + EfficientAD
          python compare_preproc_methods.py \\
              --image            photo.jpg \\
              --yolo_det_weights runs/detect/best.pt \\
              --yolo_seg_weights runs/segment/best.pt \\
              --ckpt_dir         path/to/efficientad_ckpt \\
              --category         chewinggum \\
              --model_size       S \\
              --threshold_file   ./results/threshold.txt \\
              --save_dir         ./comparison_results
        """),
    )
    # ── input
    p.add_argument("--image", required=True,
                   help="Path to a multi-object image (jpg/png/…)")
    p.add_argument("--save_dir", default="./comparison_results")

    # ── your trained YOLO models
    p.add_argument("--yolo_det_weights", default=None,
                   help="Your trained YOLO DETECTION .pt weights")
    p.add_argument("--yolo_seg_weights", default=None,
                   help="Your trained YOLO SEGMENTATION .pt weights")

    # ── detection params
    p.add_argument("--conf", type=float, default=0.75,
                   help="YOLO confidence threshold (default 0.75)")
    p.add_argument("--iou",  type=float, default=0.45,
                   help="YOLO NMS IoU threshold (default 0.45)")

    # ── EfficientAD
    p.add_argument("--ckpt_dir",       default=None,
                   help="EfficientAD checkpoint directory (omit to skip EfficientAD)")
    p.add_argument("--category",       default="chewinggum",
                   help="EfficientAD category (default: chewinggum)")
    p.add_argument("--model_size",     default="S", choices=["S", "M"],
                   help="EfficientAD PDN size (default: S)")
    p.add_argument("--ratio",          type=float, default=0.1,
                   help="EfficientAD normalisation ratio (default 0.1)")
    p.add_argument("--threshold_file", default=None,
                   help="Path to threshold.txt produced by evaluate_pipeline.py "
                        "(if omitted, 0.5 is used as placeholder)")
    p.add_argument("--device",         default=None,
                   help="cuda or cpu (auto-detected if omitted)")

    # ── optional skips
    p.add_argument("--skip_sam",   action="store_true",
                   help="Skip SAM-lite (requires network download)")
    p.add_argument("--skip_rembg", action="store_true",
                   help="Skip rembg/U2Net background removal")

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    if not os.path.isfile(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        print(f"[ERROR] Could not read image: {args.image}")
        sys.exit(1)

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*65}")
    print(f"  Pre-processing + EfficientAD Comparison  ·  NO DENOISING")
    print(f"{'='*65}")
    print(f"  Image  : {args.image}  ({image_bgr.shape[1]}×{image_bgr.shape[0]})")
    print(f"  Output : {args.save_dir}")
    print(f"  Device : {device}")
    print(f"  conf={args.conf}  iou={args.iou}")
    print()

    os.makedirs(args.save_dir, exist_ok=True)

    # ── Load EfficientAD (once, shared across all methods) ────────────────────
    ead = EfficientADRunner()
    if args.ckpt_dir:
        print("── Loading EfficientAD ──────────────────────────────────────")
        ead.load(
            ckpt_dir       = args.ckpt_dir,
            category       = args.category,
            model_size     = args.model_size,
            device         = device,
            ratio          = args.ratio,
            threshold_file = args.threshold_file,
        )
        if not ead._available:
            print("    ⚠  EfficientAD failed to load — continuing without anomaly scoring")
    else:
        print("── EfficientAD skipped (--ckpt_dir not provided) ───────────")
    print()

    results: List[MethodResult] = []

    # ══════════════════════════════════════════════════════════════════════════
    # 1. DETECTION
    # ══════════════════════════════════════════════════════════════════════════
    print("── [1/3] Detection methods ──────────────────────────────────────")

    # Your custom detection model
    if args.yolo_det_weights:
        print(f"  → Custom YOLO-Det  [{args.yolo_det_weights}]")
        r = run_yolo_detection(image_bgr, args.yolo_det_weights,
                               args.conf, args.iou, "Custom YOLO-Det", ead)
        results.append(r)
        print(f"     load={r.load_ms:.1f}ms  infer={r.infer_ms:.1f}ms  "
              f"objects={r.num_objects}"
              + (f"\n  ⚠ ERROR:\n{r.error}" if r.error else ""))

    # Your custom segmentation model (used as detector)
    if args.yolo_seg_weights:
        print(f"  → Custom YOLO-Seg  [{args.yolo_seg_weights}]")
        r = run_yolo_segmentation(image_bgr, args.yolo_seg_weights,
                                  args.conf, args.iou, "Custom YOLO-Seg", ead)
        results.append(r)
        print(f"     load={r.load_ms:.1f}ms  infer={r.infer_ms:.1f}ms  "
              f"objects={r.num_objects}"
              + (f"\n  ⚠ ERROR:\n{r.error}" if r.error else ""))

    # Pretrained YOLOv8n
    print("  → YOLOv8n  [yolov8n.pt — pretrained COCO]")
    r_nano = run_yolo_detection(image_bgr, "yolov8n.pt",
                                args.conf, args.iou, "YOLOv8n (pretrained)", ead)
    results.append(r_nano)
    print(f"     load={r_nano.load_ms:.1f}ms  infer={r_nano.infer_ms:.1f}ms  "
          f"objects={r_nano.num_objects}"
          + (f"\n  ⚠ ERROR:\n{r_nano.error}" if r_nano.error else ""))

    # Pretrained YOLOv8s
    print("  → YOLOv8s  [yolov8s.pt — pretrained COCO]")
    r_small = run_yolo_detection(image_bgr, "yolov8s.pt",
                                 args.conf, args.iou, "YOLOv8s (pretrained)", ead)
    results.append(r_small)
    print(f"     load={r_small.load_ms:.1f}ms  infer={r_small.infer_ms:.1f}ms  "
          f"objects={r_small.num_objects}"
          + (f"\n  ⚠ ERROR:\n{r_small.error}" if r_small.error else ""))

    # Pick best detections for the bg-removal stage
    det_candidates = [r for r in results
                      if r.category == "detection" and r.error is None]
    best_det = max(det_candidates, key=lambda r: r.num_objects, default=None)
    shared_dets = best_det.detections if best_det else []

    # ══════════════════════════════════════════════════════════════════════════
    # 2. BACKGROUND REMOVAL
    # ══════════════════════════════════════════════════════════════════════════
    print("\n── [2/3] Background removal methods ────────────────────────────")

    bg_methods = [
        ("GrabCut",      _run_grabcut),
        ("Otsu+Morph",   _run_otsu),
        ("Watershed-BG", _run_watershed_bg),
    ]
    if not args.skip_rembg:
        bg_methods.append(("U2Net/rembg", _run_rembg))

    for lbl, fn in bg_methods:
        print(f"  → {lbl}")
        r = run_bg_removal(image_bgr, shared_dets, fn, lbl, ead)
        results.append(r)
        print(f"     load={r.load_ms:.1f}ms  infer={r.infer_ms:.1f}ms  "
              f"crops={r.num_objects}"
              + (f"  ⚠ {r.error[:60]}" if r.error else ""))

    # ══════════════════════════════════════════════════════════════════════════
    # 3. SEGMENTATION
    # ══════════════════════════════════════════════════════════════════════════
    print("\n── [3/3] Segmentation methods ───────────────────────────────────")

    seg_methods: list = [
        ("Contour-Seg",     _seg_contour),
        ("Watershed-Seg",   _seg_watershed),
        ("SLIC-Superpixel", _seg_slic),
    ]
    if not args.skip_sam:
        seg_methods.append(("SAM-lite", _seg_sam_lite))

    for lbl, fn in seg_methods:
        print(f"  → {lbl}")
        r = run_segmentation(image_bgr, fn, lbl, ead)
        results.append(r)
        print(f"     load={r.load_ms:.1f}ms  infer={r.infer_ms:.1f}ms  "
              f"segments={r.num_objects}"
              + (f"  ⚠ {r.error[:60]}" if r.error else ""))

    # ══════════════════════════════════════════════════════════════════════════
    # 4. Save images
    # ══════════════════════════════════════════════════════════════════════════
    print("\n── Saving outputs ───────────────────────────────────────────────")
    img_dir = os.path.join(args.save_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    for r in results:
        safe = (r.method_name.replace("/", "_").replace(" ", "_")
                              .replace("+", "plus").replace("(", "").replace(")", ""))
        if r.annotated_image is not None:
            cv2.imwrite(os.path.join(img_dir, f"{safe}_annotated.png"),
                        r.annotated_image)
        for i, (c, o) in enumerate(zip(r.crops, r.crop_overlays)):
            cv2.imwrite(os.path.join(img_dir, f"{safe}_crop{i+1:02d}.png"), c)
            cv2.imwrite(os.path.join(img_dir, f"{safe}_overlay{i+1:02d}.png"), o)
        for ar in r.anomaly_results:
            if ar.heatmap is not None:
                cv2.imwrite(
                    os.path.join(img_dir, f"{safe}_heatmap{ar.crop_idx+1:02d}.png"),
                    ar.heatmap)

    # ══════════════════════════════════════════════════════════════════════════
    # 5. HTML report
    # ══════════════════════════════════════════════════════════════════════════
    html = build_html(results, image_bgr, args.image, ead)
    report_path = os.path.join(args.save_dir, "comparison_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    # ══════════════════════════════════════════════════════════════════════════
    # 6. Console summary
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  Report : {report_path}")
    print(f"  Images : {img_dir}/")
    if ead._available:
        print(f"  EfficientAD load : {ead.load_ms:.1f} ms  (one-time)")
        print(f"  Anomaly threshold: {ead.threshold}")
    print(f"{'='*65}")
    print()
    hdr = (f"{'Category':<16} {'Method':<26} {'Obj':>4}  "
           f"{'Load':>8}  {'Infer':>8}  {'Total':>8}  "
           f"{'EAD':>8}  {'w/EAD':>8}  {'Anomalies'}")
    print(hdr)
    print("─" * len(hdr))
    for r in results:
        ac   = sum(1 for a in r.anomaly_results if a.is_anomaly)
        flag = f"{ac}/{r.num_objects}" if r.anomaly_results else "—"
        ead_t = f"{r.ead_total_ms:.1f}ms" if r.anomaly_results else "—"
        grand = f"{r.total_ms+r.ead_total_ms:.1f}ms" if r.anomaly_results else f"{r.total_ms:.1f}ms"
        print(
            f"{r.category:<16} {r.method_name:<26} {r.num_objects:>4}  "
            f"{r.load_ms:>7.1f}ms  {r.infer_ms:>7.1f}ms  {r.total_ms:>7.1f}ms  "
            f"{ead_t:>8}  {grand:>8}  {flag}"
            + (f"  ⚠ SKIP" if r.error else "")
        )
    print()


if __name__ == "__main__":
    main()