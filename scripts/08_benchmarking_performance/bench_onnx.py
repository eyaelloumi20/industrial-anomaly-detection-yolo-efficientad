"""
bench_onnx.py
=============
Compare PyTorch vs ONNX Runtime (CUDA EP) vs ONNX Runtime (TensorRT EP)
on the same synthetic batch.

Reports:
  - PyTorch full pipeline (teacher + student + AE) with the dtype-safe patch
  - Isolated PyTorch per-network times (teacher / student / AE separately)
  - ONNX teacher + student times for each provider
  - (If AE was exported) ONNX AE time as well

This lets you compare ONNX+PyTorch-AE against PyTorch-all apples-to-apples.

Usage:
    python bench_onnx.py --ckpt_dir ./ckpt_192 --image_size 192 --batch 96
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import time
import json
import types
import argparse

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


# ===========================================================================
# ONNX Runtime session builder
# ===========================================================================
def make_session(onnx_path, provider, device_id=0, trt_cache_dir=None):
    import onnxruntime as ort
    providers = []
    if provider == "trt":
        opts = {
            "device_id": device_id,
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_timing_cache_enable": True,
        }
        if trt_cache_dir:
            os.makedirs(trt_cache_dir, exist_ok=True)
            opts["trt_engine_cache_path"] = trt_cache_dir
        providers.append(("TensorrtExecutionProvider", opts))
    providers.append(("CUDAExecutionProvider", {
        "device_id": device_id,
        "cudnn_conv_algo_search": "DEFAULT",
    }))
    providers.append("CPUExecutionProvider")
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = \
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(onnx_path, sess_options=sess_opts,
                                providers=providers)


# ===========================================================================
# ONNX benchmark
# ===========================================================================
def bench(session, n, image_size, iters=30, warmup=5):
    x = np.random.randn(n, 3, image_size, image_size).astype(np.float32)
    inp = session.get_inputs()[0].name
    for _ in range(warmup):
        session.run(None, {inp: x})
    t0 = time.perf_counter()
    for _ in range(iters):
        session.run(None, {inp: x})
    return (time.perf_counter() - t0) * 1000.0 / iters


# ===========================================================================
# PyTorch benchmark with dtype-safe forward
# ===========================================================================
def _safe_norm(x):
    mean = torch.tensor([0.485, 0.456, 0.406],
                        dtype=x.dtype, device=x.device)[None, :, None, None]
    std  = torch.tensor([0.229, 0.224, 0.225],
                        dtype=x.dtype, device=x.device)[None, :, None, None]
    return (x - mean) / (std + 1e-11)


def _patch_forward(model, kind):
    """Rebind forward so it uses a dtype-safe normalization. Needed because
    utils/models.py's imagenet_norm_batch hardcodes fp32 mean/std, which
    promotes fp16 inputs back to fp32 mid-forward."""
    if kind in ("teacher", "student"):
        def _f(self, x):
            return self.pdn(_safe_norm(x))
    else:
        def _f(self, x):
            x = _safe_norm(x)
            return self.decoder(self.encoder(x))
    model.forward = types.MethodType(_f, model)


def bench_pytorch(ckpt_dir, image_size, batch, device="cuda:0",
                  iters=30, warmup=5, which="all"):
    """Reference PyTorch fp16 + channels_last run.
    which in {"all","teacher","student","ae"} selects what to forward."""
    from compare_pipeline_methods_batched import load_bundle

    models, _, _ = load_bundle("baseline", ckpt_dir, "chewinggum", "S", device)

    for key in ("teacher", "student", "ae"):
        models[key] = models[key].half().to(
            memory_format=torch.channels_last).eval()
    for key, kind in (("teacher", "teacher"),
                      ("student", "student"),
                      ("ae", "ae")):
        _patch_forward(models[key], kind)

    x = torch.randn(batch, 3, image_size, image_size, device=device).half()
    x = x.to(memory_format=torch.channels_last)

    def _fwd():
        if which in ("all", "teacher"):
            _ = models["teacher"](x)
        if which in ("all", "student"):
            _ = models["student"](x)
        if which in ("all", "ae"):
            _ = models["ae"](x)

    with torch.no_grad():
        for _ in range(warmup):
            _fwd()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _fwd()
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


# ===========================================================================
# Main
# ===========================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--image_size", type=int, required=True)
    p.add_argument("--batch", type=int, default=96)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--skip_pytorch", action="store_true")
    p.add_argument("--providers", nargs="*", default=["cuda", "trt"],
                   choices=["cuda", "trt", "cpu"],
                   help="Which ONNX Runtime execution providers to benchmark.")
    args = p.parse_args()

    onnx_dir = os.path.join(args.ckpt_dir, "onnx")
    meta_path = os.path.join(onnx_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise SystemExit(f"No meta.json in {onnx_dir}. Run export_onnx.py.")

    with open(meta_path) as f:
        meta = json.load(f)
    ae_exported = bool(meta.get("ae_exported", False))

    paths = {
        "teacher": os.path.join(onnx_dir, "teacher.onnx"),
        "student": os.path.join(onnx_dir, "student.onnx"),
    }
    if ae_exported:
        paths["ae"] = os.path.join(onnx_dir, "ae.onnx")

    for k, v in paths.items():
        if not os.path.exists(v):
            raise SystemExit(f"Missing {v}. Run export_onnx.py first.")

    device_id = int(args.device.split(":")[-1]) if ":" in args.device else 0

    print(f"\n{'='*78}")
    print(f"  ONNX / TensorRT benchmark")
    print(f"{'='*78}")
    print(f"  ckpt_dir       : {args.ckpt_dir}")
    print(f"  image_size     : {args.image_size}")
    print(f"  batch          : {args.batch}")
    print(f"  iters / warmup : {args.iters} / {args.warmup}")
    print(f"  device         : {args.device}")
    print(f"  AE exported    : {ae_exported}")
    print(f"{'='*78}\n")

    # ---- PyTorch reference ------------------------------------------
    if not args.skip_pytorch:
        print(f"  {'-'*74}")
        print(f"  PyTorch (fp16 + channels_last)")
        print(f"  {'-'*74}")

        try:
            ms_all = bench_pytorch(args.ckpt_dir, args.image_size, args.batch,
                                   args.device, args.iters, args.warmup,
                                   which="all")
            cps = 1000.0 * args.batch / ms_all
            print(f"  {'all':<22} total={ms_all:>8.2f} ms   "
                  f"crops/s={cps:>7.1f}")
        except Exception as e:
            print(f"  all                    FAILED: {type(e).__name__}: {e}")

        # Isolated per-network numbers -- these are what we compare to ONNX
        iso = {}
        for which in ("teacher", "student", "ae"):
            try:
                ms = bench_pytorch(args.ckpt_dir, args.image_size, args.batch,
                                   args.device, args.iters, args.warmup,
                                   which=which)
                iso[which] = ms
            except Exception as e:
                iso[which] = None
                print(f"  {which:<22} FAILED: {type(e).__name__}: {e}")

        if iso.get("teacher") is not None and iso.get("student") is not None:
            t_pt = iso["teacher"]
            s_pt = iso["student"]
            a_pt = iso.get("ae") or 0.0
            ts_pt = t_pt + s_pt
            print(f"  {'teacher + student':<22} total={ts_pt:>8.2f} ms")
            if a_pt:
                print(f"  {'  + AE (full pipe)':<22} "
                      f"total={ts_pt + a_pt:>8.2f} ms")

    # ---- ONNX Runtime ------------------------------------------------
    for provider in args.providers:
        print(f"\n  {'-'*74}")
        print(f"  ONNX Runtime   provider={provider}")
        print(f"  {'-'*74}")

        try:
            trt_cache = os.path.join(onnx_dir, "trt_cache")
            sessions = {
                name: make_session(path, provider,
                                   device_id=device_id,
                                   trt_cache_dir=trt_cache)
                for name, path in paths.items()
            }
        except Exception as e:
            print(f"  session init FAILED: {type(e).__name__}: {e}")
            continue

        # Report what ONNX Runtime actually selected (in case TRT silently
        # fell back to CUDA EP)
        active = sessions["teacher"].get_providers()
        print(f"  active EP(s): {active}")

        try:
            t_ms = bench(sessions["teacher"], args.batch, args.image_size,
                         args.iters, args.warmup)
        except Exception as e:
            t_ms = None
            print(f"  teacher FAILED: {type(e).__name__}: {e}")

        try:
            s_ms = bench(sessions["student"], args.batch, args.image_size,
                         args.iters, args.warmup)
        except Exception as e:
            s_ms = None
            print(f"  student FAILED: {type(e).__name__}: {e}")

        a_ms = None
        if ae_exported:
            try:
                a_ms = bench(sessions["ae"], args.batch, args.image_size,
                             args.iters, args.warmup)
            except Exception as e:
                a_ms = None
                print(f"  ae FAILED: {type(e).__name__}: {e}")

        if t_ms is not None:
            print(f"  {'teacher':<22} total={t_ms:>8.2f} ms")
        if s_ms is not None:
            print(f"  {'student':<22} total={s_ms:>8.2f} ms")
        if a_ms is not None:
            print(f"  {'ae':<22} total={a_ms:>8.2f} ms")

        if t_ms is not None and s_ms is not None:
            ts_onnx = t_ms + s_ms
            cps = 1000.0 * args.batch / ts_onnx
            print(f"  {'teacher + student':<22} total={ts_onnx:>8.2f} ms   "
                  f"crops/s={cps:>7.1f}")

    print(f"\n{'='*78}")
    print(f"  How to read this")
    print(f"{'='*78}")
    print(f"  * Compare ONNX 'teacher + student' against PyTorch")
    print(f"    'teacher + student'. If ONNX is faster, the backend is worth")
    print(f"    wiring in.")
    print(f"  * Full pipeline (with PyTorch AE fallback) ≈")
    print(f"    ONNX teacher + student + PyTorch AE.")
    print(f"  * If TRT build fails or falls back, the 'active EP(s)' line")
    print(f"    will show CUDAExecutionProvider -- that's the CUDA-EP number.")
    print()


if __name__ == "__main__":
    main()