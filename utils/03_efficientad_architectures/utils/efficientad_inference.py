"""
efficientad_inference.py
=========================
Pure inference engine for EfficientAD.
Accepts a loaded model bundle (from load_efficientad.py) and runs the
complete anomaly scoring pipeline on a single image tensor or a batch.

This module has NO dependency on any dataset loader — it only works with
raw torch.Tensor inputs, making it easy to plug into any upstream detector
or batch loader.

Exported symbols
----------------
    run_inference(image_t, models, ratio=0.1)
        → anomaly_map [B,1,H,W],  image_scores [B]

    run_inference_pil(pil_image, models, device, ratio=0.1)
        → anomaly_map [1,1,256,256],  image_score float
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

# ── constants ────────────────────────────────────────────────────────────────
OUT_CHANNELS = 384
IMAGE_SIZE   = 256

_default_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


# ── core inference ────────────────────────────────────────────────────────────

def run_inference(
    image_t: torch.Tensor,
    models:  dict,
    ratio:   float = 0.1,
) -> tuple:
    """
    Run the full EfficientAD anomaly scoring pipeline on a pre-processed
    image tensor.

    Parameters
    ----------
    image_t : torch.Tensor  shape [B, 3, H, W], float32, values in [0,1].
              Must already be on the same device as `models`.
    models  : dict returned by load_efficientad.load_efficientad_model()
    ratio   : normalisation ratio (default 0.1, as in the paper)

    Returns
    -------
    anomaly_map   : torch.Tensor  [B, 1, IMAGE_SIZE, IMAGE_SIZE]
                    Combined (student-teacher + AE) normalised anomaly map.
    image_scores  : torch.Tensor  [B]
                    Per-image anomaly score = max of anomaly_map.
    """
    with torch.no_grad():
        t_out = models["teacher"](image_t)   # [B, 384, h, w]
        s_out = models["student"](image_t)   # [B, 768, h, w]
        a_out = models["ae"](image_t)        # [B, 384, h, w]

    # ── split student output ─────────────────────────────────────────────────
    y_st   = s_out[:, :OUT_CHANNELS,  :, :]   # first 384 → vs teacher
    y_stae = s_out[:, -OUT_CHANNELS:, :, :]   # last  384 → vs AE

    # ── normalise teacher output ──────────────────────────────────────────────
    norm_t = (t_out - models["mean"]) / (models["std"] + 1e-8)

    # ── squared-difference maps ──────────────────────────────────────────────
    d_st   = torch.pow(norm_t - y_st,   2)   # [B, 384, h, w]
    d_stae = torch.pow(a_out  - y_stae, 2)   # [B, 384, h, w]

    # ── average over channels → [B, 1, h, w] ─────────────────────────────────
    fm_st   = torch.mean(d_st,   dim=1, keepdim=True)
    fm_stae = torch.mean(d_stae, dim=1, keepdim=True)

    # ── upsample to IMAGE_SIZE × IMAGE_SIZE ──────────────────────────────────
    fm_st   = F.interpolate(fm_st,   size=(IMAGE_SIZE, IMAGE_SIZE),
                            mode="bilinear", align_corners=False)
    fm_stae = F.interpolate(fm_stae, size=(IMAGE_SIZE, IMAGE_SIZE),
                            mode="bilinear", align_corners=False)

    # ── normalise with training quantiles ────────────────────────────────────
    norm_mst = (ratio * (fm_st   - models["qa_st"])) \
               / (models["qb_st"] - models["qa_st"] + 1e-8)
    norm_mae = (ratio * (fm_stae - models["qa_ae"])) \
               / (models["qb_ae"] - models["qa_ae"] + 1e-8)

    # ── combine ──────────────────────────────────────────────────────────────
    anomaly_map = 0.5 * norm_mst + 0.5 * norm_mae   # [B, 1, H, W]

    # ── image-level score = spatial maximum ──────────────────────────────────
    B = anomaly_map.shape[0]
    image_scores = anomaly_map.view(B, -1).max(dim=1).values   # [B]

    return anomaly_map, image_scores


def run_inference_pil(
    pil_image: Image.Image,
    models:    dict,
    device:    str   = "cpu",
    ratio:     float = 0.1,
) -> tuple:
    """
    Convenience wrapper: accepts a PIL image, applies standard transforms,
    runs inference, and returns (anomaly_map_tensor, image_score_float).

    Parameters
    ----------
    pil_image : PIL.Image.Image  (any size, will be resized to 256×256)
    models    : dict from load_efficientad_model()
    device    : device string
    ratio     : normalisation ratio

    Returns
    -------
    anomaly_map  : torch.Tensor  [1, 1, 256, 256]  (on CPU)
    image_score  : float
    """
    image_t = _default_transform(pil_image.convert("RGB")) \
                .unsqueeze(0).to(device)

    anomaly_map, image_scores = run_inference(image_t, models, ratio)
    return anomaly_map.cpu(), float(image_scores[0].cpu())


def anomaly_map_to_heatmap(
    anomaly_map_np: np.ndarray,
    original_pil:   Image.Image = None,
    alpha:          float        = 0.5,
) -> Image.Image:
    """
    Convert a 2-D numpy anomaly map to a colour heatmap, optionally
    blended with the original image.

    Parameters
    ----------
    anomaly_map_np : np.ndarray  [H, W]  float
    original_pil   : optional PIL image to blend with
    alpha          : blend weight for the heatmap (0 = original only)

    Returns
    -------
    PIL.Image.Image  (RGB)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mn, mx = anomaly_map_np.min(), anomaly_map_np.max()
    norm   = (anomaly_map_np - mn) / (mx - mn + 1e-8)
    heat   = (plt.cm.hot(norm)[:, :, :3] * 255).astype(np.uint8)
    heat_pil = Image.fromarray(heat).resize((IMAGE_SIZE, IMAGE_SIZE))

    if original_pil is not None:
        orig = original_pil.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        return Image.blend(orig, heat_pil, alpha)
    return heat_pil
