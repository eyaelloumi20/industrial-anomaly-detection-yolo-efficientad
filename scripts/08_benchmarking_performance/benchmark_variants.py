"""
benchmark_variants.py
=====================
Measure per-crop inference latency for each trained variant, at its native
image size, on the current GPU.
"""
import sys
import time
import importlib
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, ".")

CATEGORY = "chewinggum"
DEVICE   = "cuda:0"
FP16     = True
N_CROPS  = 200

VARIANTS = {
    "baseline":   ("utils.load_efficientad",          "ckpt_cropped_multi"),
    "lowres_192": ("utils.load_efficientad_adaptive", "ckpt_lowres_192"),
    "lowres_224": ("utils.load_efficientad_adaptive", "ckpt_lowres_224"),
    "thin_075":   ("utils.load_efficientad_thin",     "ckpt_thin_075"),
    "thin_050":   ("utils.load_efficientad_thin",     "ckpt_thin_050"),
    "pruned_030": ("utils.load_efficientad_pruned",   "ckpt_pruned_030"),
}

OUT_CHANNELS = 384


def _to_half_if_needed(t, dtype):
    if dtype == torch.float16:
        return t.half()
    return t.float()


@torch.inference_mode()
def score_batch(models, image_size, dtype, n=N_CROPS):
    # Build input
    dummy = Image.new("RGB", (image_size, image_size), (128, 128, 128))
    arr = np.asarray(dummy, dtype=np.uint8)
    base = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    base = base.div_(255.0)
    base = base.to(DEVICE)
    base = _to_half_if_needed(base, dtype)
    t = base.expand(n, -1, -1, -1).contiguous(
        memory_format=torch.channels_last)

    assert t.dtype == next(models["teacher"].parameters()).dtype, (
        f"Input dtype {t.dtype} != model dtype "
        f"{next(models['teacher'].parameters()).dtype}"
    )

    _ = models["teacher"](t[:2])
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    teacher = models["teacher"](t)
    student = models["student"](t)
    ae = models["ae"](t)
    norm_t = (teacher - models["mean"]) / (models["std"] + 1e-8)
    d_st = (norm_t - student[:, :OUT_CHANNELS]).pow(2).mean(1, keepdim=True)
    d_ae = (ae - student[:, -OUT_CHANNELS:]).pow(2).mean(1, keepdim=True)
    fm_st = torch.nn.functional.interpolate(
        d_st, size=(image_size, image_size), mode="bilinear", align_corners=False)
    fm_ae = torch.nn.functional.interpolate(
        d_ae, size=(image_size, image_size), mode="bilinear", align_corners=False)
    nm_st = (0.1 * (fm_st - models["qa_st"])) / (models["qb_st"] - models["qa_st"] + 1e-8)
    nm_ae = (0.1 * (fm_ae - models["qa_ae"])) / (models["qb_ae"] - models["qa_ae"] + 1e-8)
    _ = (0.5 * nm_st + 0.5 * nm_ae).flatten(1).amax(1)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return dt * 1000 / n


def prepare_models(models, dtype):
    for key in ("teacher", "student", "ae"):
        models[key] = models[key].eval()
        if dtype == torch.float16:
            models[key] = models[key].half()
        models[key] = models[key].to(memory_format=torch.channels_last)
    for key in ("mean", "std", "qa_st", "qb_st", "qa_ae", "qb_ae"):
        models[key] = torch.as_tensor(
            models[key], device=DEVICE, dtype=dtype)
    return models


def main():
    results = {}
    dtype = torch.float16 if FP16 else torch.float32

    for name, (modpath, ckpt) in VARIANTS.items():
        try:
            mod = importlib.import_module(modpath)
            models = mod.load_efficientad_model(
                ckpt_dir=ckpt, category=CATEGORY,
                model_size="S", device=DEVICE)
        except Exception as e:
            print(f"  {name:<14}  SKIP ({type(e).__name__}: {e})")
            continue

        image_size = models.get("image_size", 256)
        models = prepare_models(models, dtype)

        m_dtype = next(models["teacher"].parameters()).dtype
        print(f"  {name:<14}  img={image_size:>3}  model_dtype={m_dtype}  "
              f"input_dtype={dtype}")

        try:
            ms = score_batch(models, image_size, dtype)
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}")
            continue

        results[name] = {"ms_per_crop": ms, "image_size": image_size}
        print(f"    → {ms:.2f} ms/crop")

    print()
    baseline_ms = results.get("baseline", {}).get("ms_per_crop")
    if baseline_ms:
        print(f"{'variant':<14} {'ms/crop':>9} {'speedup':>9} {'img_size':>9}")
        print("-" * 45)
        for name, r in results.items():
            speedup = baseline_ms / r["ms_per_crop"]
            print(f"{name:<14} {r['ms_per_crop']:>9.2f} {speedup:>8.2f}x "
                  f"{r['image_size']:>9}")


if __name__ == "__main__":
    main()