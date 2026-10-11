"""
compare_variants.py
===================
Evaluate every trained EfficientAD variant on the same val split.
Prints a table: AUROC, inference time per crop, parameter count.
"""
import os
import sys
import time
import importlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from utils.data_loader import get_AD_dataset

DATASET = "/home/ahmed/two-stage-efficientad-inspection/dataset_multi"
CATEGORY = "chewinggum"
DEVICE = "cuda:0"

# variant_name -> (loader_module, ckpt_dir)
VARIANTS = {
    "baseline":   ("utils.load_efficientad",          "ckpt_cropped_multi"),
    "lowres_192": ("utils.load_efficientad_adaptive", "ckpt_lowres_192"),
    "lowres_224": ("utils.load_efficientad_adaptive", "ckpt_lowres_224"),
    "thin_075":   ("utils.load_efficientad_thin",     "ckpt_thin_075"),
    "thin_050":   ("utils.load_efficientad_thin",     "ckpt_thin_050"),
    "pruned_030": ("utils.load_efficientad_pruned",   "ckpt_pruned_030"),
    # "qat_int8":   ("utils.load_efficientad_qat",      "ckpt_qat"),
}


def load_split(phase):
    tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    return get_AD_dataset(
        type='VisA', root=DATASET, transform=tf, gt_transform=tf,
        phase=phase, category=CATEGORY, split_ratio=1.0,
    )


@torch.no_grad()
def score_one_crop(models, image_t, image_size):
    t_out = models["teacher"](image_t)
    s_out = models["student"](image_t)
    a_out = models["ae"](image_t)
    y_st = s_out[:, :384, :, :]
    y_stae = s_out[:, -384:, :, :]
    norm_t = (t_out - models["mean"]) / (models["std"] + 1e-8)
    d_st = torch.pow(norm_t - y_st, 2)
    d_stae = torch.pow(a_out - y_stae, 2)
    fm_st = F.interpolate(torch.mean(d_st, dim=1, keepdim=True),
                          size=(image_size, image_size),
                          mode="bilinear", align_corners=False)
    fm_stae = F.interpolate(torch.mean(d_stae, dim=1, keepdim=True),
                            size=(image_size, image_size),
                            mode="bilinear", align_corners=False)
    nm_st = (0.1 * (fm_st - models["qa_st"])) / (models["qb_st"] - models["qa_st"] + 1e-8)
    nm_ae = (0.1 * (fm_stae - models["qa_ae"])) / (models["qb_ae"] - models["qa_ae"] + 1e-8)
    return float((0.5 * nm_st + 0.5 * nm_ae).max().cpu())


def evaluate_variant(name, loader_module, ckpt_dir):
    if not os.path.isdir(ckpt_dir):
        print(f"  [{name}] SKIP — ckpt_dir {ckpt_dir} missing")
        return None

    print(f"\n[{name}] Loading from {ckpt_dir} ...")
    mod = importlib.import_module(loader_module)
    try:
        models = mod.load_efficientad_model(
            ckpt_dir=ckpt_dir, category=CATEGORY,
            model_size="S", device=DEVICE,
        )
    except Exception as e:
        print(f"  [{name}] LOAD FAILED: {e}")
        return None

    image_size = models.get("image_size", 256)

    # Resize to this variant's native resolution
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    val_ds = get_AD_dataset(
        type='VisA', root=DATASET, transform=tf, gt_transform=tf,
        phase='validation', category=CATEGORY, split_ratio=1.0,
    )

    scores, labels, times = [], [], []
    for i in range(len(val_ds)):
        sample = val_ds[i]
        img = sample['image'].unsqueeze(0).to(DEVICE)
        t0 = time.perf_counter()
        s = score_one_crop(models, img, image_size)
        times.append((time.perf_counter() - t0) * 1000)  # ms
        scores.append(s)
        labels.append(int(sample['label']))

    labels = np.array(labels)
    scores = np.array(scores)
    times = np.array(times)

    auroc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else 0.0
    n_params = sum(p.numel() for m in (models['teacher'], models['student'], models['ae'])
                   for p in m.parameters())

    return {
        "name": name,
        "auroc": auroc,
        "ms_per_crop": float(times.mean()),
        "n_params": n_params,
        "image_size": image_size,
    }


def main():
    results = []
    for name, (mod, ckpt) in VARIANTS.items():
        r = evaluate_variant(name, mod, ckpt)
        if r is not None:
            results.append(r)

    print("\n" + "=" * 90)
    print(f"{'variant':<14} {'AUROC':>8} {'ms/crop':>10} {'params':>14} {'img_size':>9}")
    print("=" * 90)
    baseline = next((r for r in results if r['name'] == 'baseline'), None)
    for r in results:
        auroc_str = f"{r['auroc']:.4f}"
        ms_str = f"{r['ms_per_crop']:.2f}"
        if baseline and r['name'] != 'baseline':
            auroc_str += f" ({r['auroc'] - baseline['auroc']:+.4f})"
            ms_str   += f" ({baseline['ms_per_crop'] / r['ms_per_crop']:.2f}x)"
        print(f"{r['name']:<14} {auroc_str:>8} {ms_str:>10} "
              f"{r['n_params']:>14,} {r['image_size']:>9}")
    print("=" * 90)
    print("\nAUROC column: absolute value, then delta vs baseline in parens")
    print("ms/crop column: absolute, then speedup factor vs baseline in parens")


if __name__ == "__main__":
    main()