"""
profile_forward.py
==================
Isolated profiler for EfficientAD's forward pass.

Loads a checkpoint, feeds random batches of the same shape the streaming
detector uses (256x256 crops, variable N), and profiles:
  - Per-op CUDA time (top 25 by total)
  - Breakdown by network (teacher / student / autoencoder)
  - Latency vs batch size (48 / 96 / 192)

This tells you exactly where the ~300 ms per 96-crop batch goes: is it
conv kernels, is it the AE's upsampling, is it memory bandwidth, is it
kernel launch overhead.

Usage:
    python profile_forward.py \
        --ckpt_dir /home/ahmed/framework/ckpt_cropped_large \
        --category chewinggum --model_size S \
        --batch_sizes 48 96 192 \
        --fp16 --channels_last
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import time
import argparse
import types

import numpy as np
import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ── Same dtype-safe patch as in video_streaming_detector.py ────────────────
import utils.models as _models_module

def _dtype_preserving_norm(x):
    mean = torch.tensor([0.485, 0.456, 0.406],
                        dtype=x.dtype, device=x.device)[None, :, None, None]
    std  = torch.tensor([0.229, 0.224, 0.225],
                        dtype=x.dtype, device=x.device)[None, :, None, None]
    return (x - mean) / (std + 1e-11)

_models_module.imagenet_norm_batch = _dtype_preserving_norm

def _dtype_safe_norm(x):
    mean = torch.tensor([0.485, 0.456, 0.406],
                        dtype=x.dtype, device=x.device)[None, :, None, None]
    std  = torch.tensor([0.229, 0.224, 0.225],
                        dtype=x.dtype, device=x.device)[None, :, None, None]
    return (x - mean) / (std + 1e-11)

def patch_forward_dtype_safe(model, kind):
    if kind in ("teacher", "student"):
        def _f(self, x):
            x = _dtype_safe_norm(x)
            return self.pdn(x)
    else:
        def _f(self, x):
            x = _dtype_safe_norm(x)
            x = self.encoder(x)
            x = self.decoder(x)
            return x
    model.forward = types.MethodType(_f, model)
    return model

from compare_pipeline_methods_batched import load_bundle, OUT_CHANNELS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--category", default="chewinggum")
    p.add_argument("--model_size", default="S", choices=["S", "M"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[48, 96, 192])
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--channels_last", action="store_true")
    return p.parse_args()


def make_input(n, device, dtype, channels_last):
    t = torch.rand(n, 3, 256, 256, device=device)
    if dtype == torch.float16:
        t = t.half()
    if channels_last:
        t = t.to(memory_format=torch.channels_last)
    return t


@torch.no_grad()
def time_one_batch(models, image_size, n, device, dtype, channels_last,
                   warmup, iters):
    """Time N forward passes, return per-batch ms and top op breakdown."""
    t = make_input(n, device, dtype, channels_last)

    # Warmup — lets cudnn pick kernels, and — if benchmark were on —
    # populates the autotune cache
    for _ in range(warmup):
        t_out = models["teacher"](t)
        s_out = models["student"](t)
        a_out = models["ae"](t)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        t_out = models["teacher"](t)
        s_out = models["student"](t)
        a_out = models["ae"](t)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return elapsed_ms / iters


@torch.no_grad()
def profile_one_batch(models, image_size, n, device, dtype, channels_last,
                      warmup, iters):
    """Collect per-op CUDA profile for the forward pass."""
    t = make_input(n, device, dtype, channels_last)

    for _ in range(warmup):
        _ = models["teacher"](t)
        _ = models["student"](t)
        _ = models["ae"](t)

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        for _ in range(iters):
            _ = models["teacher"](t)
            _ = models["student"](t)
            _ = models["ae"](t)
        torch.cuda.synchronize()

    return prof


def main():
    args = parse_args()
    device = args.device

    print(f"\n{'='*72}")
    print(f"  EfficientAD forward-pass profiler")
    print(f"{'='*72}")
    print(f"  ckpt_dir       : {args.ckpt_dir}")
    print(f"  category/size  : {args.category} / PDN-{args.model_size}")
    print(f"  device         : {device}")
    print(f"  fp16           : {args.fp16}")
    print(f"  channels_last  : {args.channels_last}")
    print(f"  batch sizes    : {args.batch_sizes}")
    print(f"  warmup / iters : {args.warmup} / {args.iters}")
    print(f"{'='*72}\n")

    # ── Load models ─────────────────────────────────────────────────────
    models, image_size, _ = load_bundle(
        "baseline", args.ckpt_dir, args.category, args.model_size, device)

    if args.fp16:
        for key in ("teacher", "student", "ae"):
            models[key] = models[key].half()
        for key in ("mean", "std", "qa_st", "qb_st", "qa_ae", "qb_ae"):
            if isinstance(models.get(key), torch.Tensor):
                models[key] = models[key].half()
        for key, kind in (("teacher", "teacher"),
                          ("student", "student"),
                          ("ae", "ae")):
            models[key] = patch_forward_dtype_safe(models[key], kind)

    if args.channels_last:
        for key in ("teacher", "student", "ae"):
            models[key] = models[key].to(memory_format=torch.channels_last)

    dtype = torch.float16 if args.fp16 else torch.float32

    # ── Sweep batch sizes ───────────────────────────────────────────────
    print(f"{'N':>6s}  {'ms/batch':>10s}  {'ms/crop':>10s}  {'crops/s':>10s}")
    print(f"{'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}")
    results = {}
    for n in args.batch_sizes:
        try:
            ms = time_one_batch(models, image_size, n, device, dtype,
                                args.channels_last, args.warmup, args.iters)
            results[n] = ms
            print(f"{n:>6d}  {ms:>10.2f}  {ms/n:>10.3f}  {1000*n/ms:>10.1f}")
        except RuntimeError as e:
            print(f"{n:>6d}  FAILED: {e}")
        torch.cuda.empty_cache()

    # ── Deep profile on the middle batch size ───────────────────────────
    profile_n = args.batch_sizes[len(args.batch_sizes) // 2]
    print(f"\n{'='*72}")
    print(f"  Per-op CUDA profile (N={profile_n}, {args.iters} iters)")
    print(f"{'='*72}")
    prof = profile_one_batch(models, image_size, profile_n, device, dtype,
                             args.channels_last, args.warmup, args.iters)

    table = prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=25)
    print(table)

    # ── Per-network breakdown by summing known op names ─────────────────
    print(f"\n{'='*72}")
    print(f"  Op-class breakdown (N={profile_n})")
    print(f"{'='*72}")
    total_us = sum(e.cuda_time_total for e in prof.key_averages())
    groups = {
        "conv2d":          ["cudnn_convolution", "conv2d", "convolution"],
        "upsample":        ["upsample"],
        "interpolate":     ["interpolate"],
        "relu":            ["relu", "threshold"],
        "avgpool":         ["avg_pool", "pool"],
        "dropout":         ["dropout", "bernoulli"],
        "norm/arith":      ["native_batch_norm", "layer_norm", "sub",
                            "div", "pow", "mean", "add"],
    }
    for label, keys in groups.items():
        s = 0.0
        for e in prof.key_averages():
            name = e.key.lower()
            if any(k in name for k in keys):
                s += e.cuda_time_total
        pct = 100.0 * s / total_us if total_us else 0.0
        print(f"  {label:<16s}  {s/1000:>9.2f} ms   {pct:>5.1f}%")

    print(f"\n  {'TOTAL':<16s}  {total_us/1000:>9.2f} ms   (100.0%)\n")


if __name__ == "__main__":
    main()