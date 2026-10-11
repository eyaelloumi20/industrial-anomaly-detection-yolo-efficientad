"""
benchmark_variants_realtime.py
==============================
Real-time oriented EfficientAD benchmark.

Unlike the old 200-crop benchmark, this reports latency for realistic crop
batch sizes and uses the same optimized scoring equation as
video_streaming_detector_optimized2.py:

  teacher + student + AE
  -> native feature maps
  -> normalize/combine once
  -> ONE bilinear interpolation (exact/compatible fast path)
  -> scalar max per crop

Optional --native-score removes the remaining interpolation and is faster,
but requires threshold recalibration before production use.
"""

import argparse
import importlib
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import utils.models as models_module

sys.path.insert(0, ".")

OUT_CHANNELS = 384

def _compile_no_cudagraphs(module):
    """Compile with TorchInductor but explicitly disable CUDA graphs.

    `mode="reduce-overhead"` enables cudagraphs and can overwrite outputs when
    several separately compiled EfficientAD modules consume the same input.
    This helper keeps Inductor fusion/autotuning while avoiding that failure.
    """
    if not hasattr(torch, "compile"):
        return module
    try:
        return torch.compile(
            module,
            backend="inductor",
            fullgraph=False,
            dynamic=False,
            options={"triton.cudagraphs": False},
        )
    except TypeError:
        # Older PyTorch releases may not accept the options key.
        try:
            import torch._inductor.config as inductor_config
            inductor_config.triton.cudagraphs = False
        except Exception:
            pass
        return torch.compile(
            module,
            backend="inductor",
            fullgraph=False,
            dynamic=False,
        )


VARIANTS = {
    "baseline":   ("utils.load_efficientad_adaptive", "ckpt_cropped_multi"),
    "lowres_192": ("utils.load_efficientad_adaptive", "ckpt_lowres_192"),
    "lowres_224": ("utils.load_efficientad_adaptive", "ckpt_lowres_224"),
    "thin_075":   ("utils.load_efficientad_thin",     "ckpt_thin_075"),
    "thin_050":   ("utils.load_efficientad_thin",     "ckpt_thin_050"),
    "pruned_030": ("utils.load_efficientad_pruned",   "ckpt_pruned_030"),
}

_NORM_CACHE = {}


def dtype_norm(x):
    key = (x.device.type, x.device.index, x.dtype)
    if key not in _NORM_CACHE:
        mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        _NORM_CACHE[key] = (mean, std + x.new_tensor(1e-11))
    mean, std = _NORM_CACHE[key]
    return (x - mean) / std


models_module.imagenet_norm_batch = dtype_norm


def patch_forward(model, kind):
    if kind in ("teacher", "student") and hasattr(model, "pdn"):
        def forward(self, x):
            return self.pdn(dtype_norm(x))
        model.forward = types.MethodType(forward, model)
    elif kind == "ae" and hasattr(model, "encoder") and hasattr(model, "decoder"):
        def forward(self, x):
            return self.decoder(self.encoder(dtype_norm(x)))
        model.forward = types.MethodType(forward, model)


def prepare(models, device, dtype, compile_models=False):
    for key in ("teacher", "student", "ae"):
        m = models[key].eval().requires_grad_(False).to(device)
        if dtype == torch.float16:
            m = m.half()
            patch_forward(m, key)
        else:
            m = m.float()
        m = m.to(memory_format=torch.channels_last)
        if compile_models and hasattr(torch, "compile"):
            try:
                m = _compile_no_cudagraphs(m)
            except Exception as exc:
                print(f"    [compile] {key}: {exc}; eager fallback")
        models[key] = m

    for key in ("mean", "std"):
        models[key] = torch.as_tensor(models[key], device=device, dtype=dtype)

    for key in ("qa_st", "qb_st", "qa_ae", "qb_ae"):
        models[key] = torch.as_tensor(models[key], device=device, dtype=torch.float32)
    return models


@torch.inference_mode()
def score_once(models, x, image_size, native_score=False):
    teacher = models["teacher"](x)
    student = models["student"](x)
    ae = models["ae"](x)

    norm_teacher = (teacher - models["mean"]) / (models["std"] + 1e-8)
    d_st = (norm_teacher - student[:, :OUT_CHANNELS]).pow(2).mean(1, keepdim=True)
    d_ae = (ae - student[:, -OUT_CHANNELS:]).pow(2).mean(1, keepdim=True)

    qa_st = models["qa_st"]
    qa_ae = models["qa_ae"]
    nm_st = (0.1 * (d_st.float() - qa_st)) / (models["qb_st"] - qa_st + 1e-8)
    nm_ae = (0.1 * (d_ae.float() - qa_ae)) / (models["qb_ae"] - qa_ae + 1e-8)
    combined = 0.5 * nm_st + 0.5 * nm_ae

    if not native_score:
        combined = F.interpolate(
            combined,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )

    return combined.flatten(1).amax(1)


def benchmark_batch(models, image_size, device, dtype, batch_size,
                    native_score=False, repeats=8):
    x = torch.full(
        (batch_size, 3, image_size, image_size),
        128.0 / 255.0,
        device=device,
        dtype=dtype,
    ).contiguous(memory_format=torch.channels_last)

    # Warmup, including torch.compile graph creation if enabled.
    for _ in range(3):
        _ = score_once(models, x, image_size, native_score)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    samples = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        scores = score_once(models, x, image_size, native_score)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0
        if not torch.isfinite(scores).all():
            raise RuntimeError("non-finite score")
        samples.append(dt)

    # Median is less sensitive to occasional CUDA/OS spikes.
    sec = float(np.median(samples))
    return {
        "batch_ms": sec * 1000.0,
        "ms_per_crop": sec * 1000.0 / batch_size,
        "crops_per_s": batch_size / sec,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--category", default="chewinggum")
    p.add_argument("--model_size", default="S", choices=["S", "M"])
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--no-fp16", dest="fp16", action="store_false")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--native-score", action="store_true")
    p.add_argument("--batch-sizes", type=int, nargs="+",
                   default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--repeats", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    device = torch.device(args.device)
    dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32

    if device.type == "cuda":
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    mode = "NATIVE (recalibrate thresholds)" if args.native_score else "FAST-EXACT (one interpolation)"
    print(f"mode={mode}  dtype={dtype}  compile={args.compile}")
    print()

    all_results = {}

    for name, (loader_path, ckpt) in VARIANTS.items():
        ckpt_path = Path(ckpt)
        if not ckpt_path.is_dir():
            print(f"{name:<14} SKIP: {ckpt} missing")
            continue

        mod = importlib.import_module(loader_path)
        try:
            models = mod.load_efficientad_model(
                ckpt_dir=ckpt,
                category=args.category,
                model_size=args.model_size,
                device=args.device,
            )
            image_size = int(models.get("image_size", 256))
            models = prepare(models, device, dtype, args.compile)
        except Exception as exc:
            print(f"{name:<14} LOAD FAILED: {type(exc).__name__}: {exc}")
            continue

        print(f"{name:<14} image={image_size}")
        per_batch = {}
        for bs in args.batch_sizes:
            try:
                r = benchmark_batch(
                    models, image_size, device, dtype, bs,
                    native_score=args.native_score,
                    repeats=args.repeats,
                )
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    print(f"  batch={bs:>3}: OOM")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    continue
                raise

            per_batch[bs] = r
            print(
                f"  batch={bs:>3} | {r['batch_ms']:>8.2f} ms/batch | "
                f"{r['ms_per_crop']:>7.2f} ms/crop | "
                f"{r['crops_per_s']:>8.1f} crops/s"
            )

        all_results[name] = per_batch
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print()

    print("=" * 86)
    print("For a camera pipeline, focus on batch latency at the number of pieces")
    print("you normally see across one scoring chunk, not the old 200-crop ms/crop.")
    print("=" * 86)


if __name__ == "__main__":
    main()
