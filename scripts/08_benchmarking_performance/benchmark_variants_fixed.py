"""
benchmark_variants_fixed.py
===========================
Benchmark EfficientAD checkpoint variants at their native input size.

Key fixes:
- FP16-safe ImageNet normalization (prevents float input / half bias mismatch).
- FP16-safe model forward patch for teacher/student/autoencoder.
- Synthetic benchmark tensors are created directly on the target device,
  avoiding the non-writable NumPy warning and excluding host preprocessing.
- Baseline and low-resolution checkpoints use the adaptive loader, matching
  compare_pipeline_methods_batched.py / video_streaming_detector_optimized1.py.
- CUDA synchronization is used around the timed section for correct latency.
"""

import importlib
import math
import sys
import time
import types

import torch
import torch.nn.functional as F
import utils.models as models_module

sys.path.insert(0, ".")

CATEGORY = "chewinggum"
DEVICE = "cuda:0"
FP16 = True
N_CROPS = 200
WARMUP_CROPS = 8
OUT_CHANNELS = 384

VARIANTS = {
    # Keep the same loader family used by the optimized video pipeline.
    "baseline":   ("utils.load_efficientad_adaptive", "ckpt_cropped_multi"),
    "lowres_192": ("utils.load_efficientad_adaptive", "ckpt_lowres_192"),
    "lowres_224": ("utils.load_efficientad_adaptive", "ckpt_lowres_224"),
    "thin_075":   ("utils.load_efficientad_thin",     "ckpt_thin_075"),
    "thin_050":   ("utils.load_efficientad_thin",     "ckpt_thin_050"),
    "pruned_030": ("utils.load_efficientad_pruned",   "ckpt_pruned_030"),
}

_NORM_CACHE = {}


def _dtype_preserving_norm(x: torch.Tensor) -> torch.Tensor:
    """ImageNet normalization without promoting FP16 activations to FP32."""
    key = (x.device.type, x.device.index, x.dtype)
    cached = _NORM_CACHE.get(key)
    if cached is None:
        mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        cached = (mean, std + x.new_tensor(1e-11))
        _NORM_CACHE[key] = cached
    mean, denom = cached
    return (x - mean) / denom


# Patch the project-level normalization used by utils.models.
models_module.imagenet_norm_batch = _dtype_preserving_norm


def _patch_forward_dtype_safe(model, kind: str) -> None:
    """
    Rebind forward so internal normalization keeps the input dtype.
    This is the important fix for:
        RuntimeError: Input type (float) and bias type (c10::Half) should be the same
    """
    if kind in ("teacher", "student"):
        if not hasattr(model, "pdn"):
            # Some serialized/pruned modules may already have a safe custom forward.
            return

        def forward(self, x):
            return self.pdn(_dtype_preserving_norm(x))

    elif kind == "ae":
        if not (hasattr(model, "encoder") and hasattr(model, "decoder")):
            return

        def forward(self, x):
            return self.decoder(self.encoder(_dtype_preserving_norm(x)))
    else:
        raise ValueError(kind)

    model.forward = types.MethodType(forward, model)


def _first_param_dtype(model):
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return None


def prepare_models(models: dict, device: torch.device, dtype: torch.dtype) -> dict:
    """Put all models/statistics on one device/dtype and channels-last layout."""
    for key in ("teacher", "student", "ae"):
        model = models[key].eval().requires_grad_(False)
        model = model.to(device=device)

        if dtype == torch.float16:
            model = model.half()
            _patch_forward_dtype_safe(model, key)
        else:
            model = model.float()

        # channels_last helps Conv2d throughput on CUDA and is safe for 4D CNNs.
        model = model.to(memory_format=torch.channels_last)
        models[key] = model

    # mean/std participate in the feature-domain computation and should follow
    # the model dtype. Quantile stats are kept FP32 for stable final scoring.
    for key in ("mean", "std"):
        models[key] = torch.as_tensor(models[key], device=device, dtype=dtype)

    for key in ("qa_st", "qb_st", "qa_ae", "qb_ae"):
        models[key] = torch.as_tensor(models[key], device=device, dtype=torch.float32)

    return models


@torch.inference_mode()
def score_batch(models: dict, image_size: int, device: torch.device,
                dtype: torch.dtype, n: int = N_CROPS) -> float:
    """
    Return EfficientAD forward/scoring latency in ms per crop.
    Input is already resident on GPU so this measures model work, not PIL/CPU copy.
    """
    # Build directly on target device: no PIL->NumPy warning and no H2D copy.
    t = torch.full(
        (n, 3, image_size, image_size),
        fill_value=128.0 / 255.0,
        device=device,
        dtype=dtype,
    ).contiguous(memory_format=torch.channels_last)

    expected = _first_param_dtype(models["teacher"])
    if expected is not None and t.dtype != expected:
        raise RuntimeError(
            f"Input dtype {t.dtype} != teacher dtype {expected}. "
            "Model/input preparation is inconsistent."
        )

    # Warm-up each component, not only teacher.
    warm_n = min(WARMUP_CROPS, n)
    tw = t[:warm_n]
    _ = models["teacher"](tw)
    _ = models["student"](tw)
    _ = models["ae"](tw)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    t0 = time.perf_counter()

    teacher = models["teacher"](t)
    student = models["student"](t)
    ae = models["ae"](t)

    norm_t = (teacher - models["mean"]) / (models["std"] + 1e-8)
    d_st = (norm_t - student[:, :OUT_CHANNELS]).pow(2).mean(1, keepdim=True)
    d_ae = (ae - student[:, -OUT_CHANNELS:]).pow(2).mean(1, keepdim=True)

    # Interpolate in model precision, then perform threshold-facing
    # normalization in FP32, like the optimized detector.
    fm_st = F.interpolate(
        d_st, size=(image_size, image_size),
        mode="bilinear", align_corners=False
    ).float()
    fm_ae = F.interpolate(
        d_ae, size=(image_size, image_size),
        mode="bilinear", align_corners=False
    ).float()

    qa_st = models["qa_st"]
    qa_ae = models["qa_ae"]
    nm_st = (0.1 * (fm_st - qa_st)) / (models["qb_st"] - qa_st + 1e-8)
    nm_ae = (0.1 * (fm_ae - qa_ae)) / (models["qb_ae"] - qa_ae + 1e-8)
    scores = (0.5 * nm_st + 0.5 * nm_ae).flatten(1).amax(1)

    # Force completion and validate before timing ends.
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    dt = time.perf_counter() - t0

    if not torch.isfinite(scores).all():
        raise RuntimeError("Non-finite EfficientAD scores were produced.")

    return dt * 1000.0 / n


def main():
    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    device = torch.device(DEVICE)
    dtype = torch.float16 if FP16 and device.type == "cuda" else torch.float32

    if device.type == "cuda":
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        # Fixed native sizes per model; benchmark mode can select fast kernels.
        torch.backends.cudnn.benchmark = True

    results = {}

    for name, (modpath, ckpt) in VARIANTS.items():
        try:
            mod = importlib.import_module(modpath)
            models = mod.load_efficientad_model(
                ckpt_dir=ckpt,
                category=CATEGORY,
                model_size="S",
                device=str(device),
            )
        except Exception as exc:
            print(f"  {name:<14} SKIP ({type(exc).__name__}: {exc})")
            continue

        image_size = int(models.get("image_size", 256))
        models = prepare_models(models, device, dtype)

        m_dtype = _first_param_dtype(models["teacher"])
        print(
            f"  {name:<14} img={image_size:>3} "
            f"model_dtype={m_dtype} input_dtype={dtype}"
        )

        try:
            ms = score_batch(models, image_size, device, dtype)
        except Exception as exc:
            print(f"    FAILED: {type(exc).__name__}: {exc}")
            del models
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        results[name] = {
            "ms_per_crop": float(ms),
            "image_size": image_size,
        }
        print(f"    -> {ms:.3f} ms/crop")

        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print()
    baseline_ms = results.get("baseline", {}).get("ms_per_crop")
    if baseline_ms:
        print(f"{'variant':<14} {'ms/crop':>10} {'speedup':>10} {'img_size':>9}")
        print("-" * 48)
        for name, r in results.items():
            speedup = baseline_ms / r["ms_per_crop"]
            print(
                f"{name:<14} {r['ms_per_crop']:>10.3f} "
                f"{speedup:>9.2f}x {r['image_size']:>9}"
            )


if __name__ == "__main__":
    main()
