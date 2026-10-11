"""
filter_crops.py
===============
Score every crop with EfficientAD, split into good/ and bad/.

The crop is RESIZED to the model's expected input size before scoring,
so this works with both:
    baseline (256×256)  -- crops are already 256×256, no resize
    lowres   (192×192)  -- crops are resized down to 192×192

Usage (baseline)
----------------
 python filter_crops.py \
   --crops-dir ./crops_video_new/crops \
   --ckpt-dir /home/ahmed/framework/ckpt_cropped_large1 \
   --method baseline --category chewinggum --model_size S \
   --threshold 1.335 \
   --out-good ./filtered/good \
   --out-bad  ./filtered/bad \
   --fp16 --device cuda:0

Usage (lowres)
--------------
 python filter_crops.py \
   --crops-dir ./crops_video_new/crops \
   --ckpt-dir ./ckpt_192 \
   --method lowres --category chewinggum --model_size S \
   --threshold 0.883 \
   --out-good ./filtered/good \
   --out-bad  ./filtered/bad \
   --fp16 --device cuda:0
"""
import argparse
import csv
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import utils.models as _models_module


def _dtype_preserving_norm(x):
    mean = torch.tensor([0.485, 0.456, 0.406],
                        dtype=x.dtype, device=x.device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225],
                       dtype=x.dtype, device=x.device)[None, :, None, None]
    return (x - mean) / (std + 1e-11)


_models_module.imagenet_norm_batch = _dtype_preserving_norm


def patch_model_forward_dtype_safe(model, kind):
    if kind in ("teacher", "student"):
        def _f(self, x):
            x = _dtype_preserving_norm(x)
            return self.pdn(x)
    elif kind == "ae":
        def _f(self, x):
            x = _dtype_preserving_norm(x)
            x = self.encoder(x)
            x = self.decoder(x)
            return x
    else:
        raise ValueError(f"Unknown kind '{kind}'")
    model.forward = types.MethodType(_f, model)
    return model


from compare_pipeline_methods_batched import load_bundle, OUT_CHANNELS


@torch.no_grad()
def score_one(models, pil_img, device, image_size, model_dtype, ratio=0.1):
    # --- KEY FIX: resize crop to what the model expects -------------------
    if pil_img.size != (image_size, image_size):
        pil_img = pil_img.resize((image_size, image_size), Image.LANCZOS)

    t = transforms.ToTensor()(pil_img).unsqueeze(0).to(device)
    if t.dtype != model_dtype:
        t = t.to(model_dtype)
    t = t.to(memory_format=torch.channels_last)

    t_out = models["teacher"](t)
    s_out = models["student"](t)
    a_out = models["ae"](t)

    y_st = s_out[:, :OUT_CHANNELS, :, :]
    y_stae = s_out[:, -OUT_CHANNELS:, :, :]

    # Sanity check — raise a clear error if the checkpoint is malformed
    if y_stae.shape[-1] != a_out.shape[-1]:
        raise RuntimeError(
            f"Shape mismatch: student tail {tuple(y_stae.shape)} vs "
            f"AE output {tuple(a_out.shape)}. "
            f"This happens when the crop size doesn't match the model's "
            f"expected input size ({image_size})."
        )

    norm_t = (t_out - models["mean"]) / (models["std"] + 1e-8)
    d_st = torch.pow(norm_t - y_st, 2)
    d_stae = torch.pow(a_out - y_stae, 2)

    fm_st = F.interpolate(torch.mean(d_st, dim=1, keepdim=True),
                          size=(image_size, image_size),
                          mode="bilinear", align_corners=False)
    fm_stae = F.interpolate(torch.mean(d_stae, dim=1, keepdim=True),
                            size=(image_size, image_size),
                            mode="bilinear", align_corners=False)

    fm_st, fm_stae = fm_st.float(), fm_stae.float()
    qa_st = models["qa_st"].float(); qb_st = models["qb_st"].float()
    qa_ae = models["qa_ae"].float(); qb_ae = models["qb_ae"].float()
    nm_st = (ratio * (fm_st - qa_st)) / (qb_st - qa_st + 1e-8)
    nm_ae = (ratio * (fm_stae - qa_ae)) / (qb_ae - qa_ae + 1e-8)
    amap = (0.5 * nm_st + 0.5 * nm_ae)[0, 0].cpu().numpy()
    return float(np.max(amap))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--crops-dir", required=True)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--method", default="baseline",
                   choices=["baseline", "lowres", "thin", "qat", "pruned"])
    p.add_argument("--category", default="chewinggum")
    p.add_argument("--model_size", default="S", choices=["S", "M"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--ratio", type=float, default=0.1)

    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--quantile", type=float, default=None)

    p.add_argument("--out-good", required=True)
    p.add_argument("--out-bad", required=True)
    p.add_argument("--scores-csv", default=None)
    p.add_argument("--move", action="store_true")
    args = p.parse_args()

    crops_dir = Path(args.crops_dir)
    files = sorted(crops_dir.glob("*.png"))
    if not files:
        raise SystemExit(f"No PNGs in {crops_dir}")

    out_good = Path(args.out_good); out_good.mkdir(parents=True, exist_ok=True)
    out_bad  = Path(args.out_bad);  out_bad.mkdir(parents=True, exist_ok=True)
    scores_csv = Path(args.scores_csv) if args.scores_csv \
                 else crops_dir.parent / "scores.csv"

    print(f"[Filter] Loading {args.method} from {args.ckpt_dir}")
    models, image_size, device = load_bundle(
        args.method, args.ckpt_dir, args.category, args.model_size, args.device)

    use_fp16 = args.fp16 and device.startswith("cuda")
    if use_fp16:
        for key in ("teacher", "student", "ae"):
            models[key] = models[key].half()
        for key in ("mean", "std", "qa_st", "qb_st", "qa_ae", "qb_ae"):
            if isinstance(models.get(key), torch.Tensor):
                models[key] = models[key].half()
        for key, kind in (("teacher", "teacher"),
                          ("student", "student"),
                          ("ae", "ae")):
            models[key] = patch_model_forward_dtype_safe(models[key], kind)
    for key in ("teacher", "student", "ae"):
        models[key] = models[key].to(memory_format=torch.channels_last)
    model_dtype = torch.float16 if use_fp16 else torch.float32

    print(f"[Filter] Scoring {len(files)} crops with image_size={image_size}")

    scores = []
    for i, f in enumerate(files):
        img = Image.open(f).convert("RGB")
        s = score_one(models, img, device, image_size, model_dtype, args.ratio)
        scores.append((f, s))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(files)}")

    score_vals = np.array([s for _, s in scores])
    if args.threshold is not None:
        thr = float(args.threshold)
        mode = f"absolute threshold = {thr:.4f}"
    elif args.quantile is not None:
        thr = float(np.quantile(score_vals, 1.0 - args.quantile))
        mode = f"top {args.quantile*100:.1f}% (threshold = {thr:.4f})"
    else:
        raise SystemExit("Pass either --threshold or --quantile")

    print(f"[Filter] Decision rule: {mode}")

    with scores_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "score", "verdict"])
        for path, sc in scores:
            verdict = "bad" if sc >= thr else "good"
            w.writerow([path.name, f"{sc:.6f}", verdict])
    print(f"[Filter] Scores written: {scores_csv}")

    n_good = n_bad = 0
    op = shutil.move if args.move else shutil.copy2
    for path, sc in scores:
        dst = out_bad / path.name if sc >= thr else out_good / path.name
        op(str(path), str(dst))
        if sc >= thr: n_bad += 1
        else:         n_good += 1

    print()
    print(f"  threshold = {thr:.4f}   ({mode})")
    print(f"  good -> {out_good}   {n_good} crops")
    print(f"  bad  -> {out_bad}    {n_bad} crops")
    print()
    print(f"  score distribution:")
    print(f"    min    = {score_vals.min():.4f}")
    print(f"    p50    = {np.quantile(score_vals, 0.50):.4f}")
    print(f"    p90    = {np.quantile(score_vals, 0.90):.4f}")
    print(f"    p99    = {np.quantile(score_vals, 0.99):.4f}")
    print(f"    max    = {score_vals.max():.4f}")


if __name__ == "__main__":
    main()