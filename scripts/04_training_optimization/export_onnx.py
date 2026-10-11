"""
export_onnx.py
==============
Export Teacher and Student to ONNX. AutoEncoder is optional because the
models_adaptive AE uses AdaptiveAvgPool2d, which PyTorch's ONNX exporter
cannot always trace at non-256 inputs.

Usage:
    # Export teacher + student only (recommended for 192 input)
    python export_onnx.py \
        --ckpt_dir ./ckpt_192 --category chewinggum \
        --model_size S --image_size 192 --skip_ae

    # Try to export all three (works at 256 where no adaptive pooling kicks in)
    python export_onnx.py \
        --ckpt_dir /home/ahmed/framework/ckpt_cropped_large \
        --category chewinggum --model_size S --image_size 256
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import json
import argparse

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from compare_pipeline_methods_batched import load_bundle


# ---------------------------------------------------------------------------
class WrappedTeacher(nn.Module):
    def __init__(self, teacher):
        super().__init__()
        self.pdn = teacher.pdn

    def forward(self, x):
        mean = torch.tensor([0.485, 0.456, 0.406],
                            dtype=x.dtype, device=x.device)[None, :, None, None]
        std  = torch.tensor([0.229, 0.224, 0.225],
                            dtype=x.dtype, device=x.device)[None, :, None, None]
        return self.pdn((x - mean) / (std + 1e-11))


class WrappedStudent(nn.Module):
    def __init__(self, student):
        super().__init__()
        self.pdn = student.pdn

    def forward(self, x):
        mean = torch.tensor([0.485, 0.456, 0.406],
                            dtype=x.dtype, device=x.device)[None, :, None, None]
        std  = torch.tensor([0.229, 0.224, 0.225],
                            dtype=x.dtype, device=x.device)[None, :, None, None]
        return self.pdn((x - mean) / (std + 1e-11))


class WrappedAE(nn.Module):
    def __init__(self, ae):
        super().__init__()
        self.encoder = ae.encoder
        self.decoder = ae.decoder

    def forward(self, x):
        mean = torch.tensor([0.485, 0.456, 0.406],
                            dtype=x.dtype, device=x.device)[None, :, None, None]
        std  = torch.tensor([0.229, 0.224, 0.225],
                            dtype=x.dtype, device=x.device)[None, :, None, None]
        x = (x - mean) / (std + 1e-11)
        return self.decoder(self.encoder(x))


def export_one(model, dummy, path, opset):
    model.eval()
    dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}
    print(f"  -> {path}")
    torch.onnx.export(
        model, dummy, path,
        input_names=["input"], output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        export_params=True,
        verbose=False,
    )
    import onnx
    m = onnx.load(path)
    onnx.checker.check_model(m)
    size_mb = os.path.getsize(path) / 1e6
    print(f"     {size_mb:.1f} MB  OK")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--category", default="chewinggum")
    p.add_argument("--model_size", default="S", choices=["S", "M"])
    p.add_argument("--image_size", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--skip_ae", action="store_true", default=True,
                   help="Skip AE export. Default True because the AE may use "
                        "AdaptiveAvgPool2d which fails at 192. Use --no_skip_ae "
                        "to attempt export anyway.")
    p.add_argument("--no_skip_ae", dest="skip_ae", action="store_false")
    args = p.parse_args()

    out_dir = os.path.join(args.ckpt_dir, "onnx")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[Export] Loading checkpoint from {args.ckpt_dir}")
    models, image_size, device = load_bundle(
        "baseline", args.ckpt_dir, args.category, args.model_size, args.device)
    if args.image_size:
        image_size = args.image_size
    print(f"[Export] image_size = {image_size}")
    print(f"[Export] device     = {device}")

    dummy = torch.randn(1, 3, image_size, image_size, device=device)

    # ---- Teacher ----
    print(f"\n[Export] Teacher")
    export_one(WrappedTeacher(models["teacher"]).to(device), dummy,
               os.path.join(out_dir, "teacher.onnx"), args.opset)

    # ---- Student ----
    print(f"\n[Export] Student")
    export_one(WrappedStudent(models["student"]).to(device), dummy,
               os.path.join(out_dir, "student.onnx"), args.opset)

    # ---- AE (optional) ----
    ae_exported = False
    if args.skip_ae:
        print(f"\n[Export] AutoEncoder -- SKIPPED (--skip_ae). "
              f"AE will run in PyTorch at runtime.")
    else:
        print(f"\n[Export] AutoEncoder -- attempting export")
        try:
            export_one(WrappedAE(models["ae"]).to(device), dummy,
                       os.path.join(out_dir, "ae.onnx"), args.opset)
            ae_exported = True
        except Exception as e:
            print(f"[Export] AE export FAILED: {type(e).__name__}: {e}")
            print(f"[Export] Falling back to PyTorch AE at runtime.")

    # ---- meta.json ----
    meta = {
        "image_size":   image_size,
        "category":     args.category,
        "model_size":   args.model_size,
        "opset":        args.opset,
        "ae_exported":  ae_exported,
    }
    try:
        qpath = os.path.join(args.ckpt_dir,
                             f"{args.category}_quantiles.npy")
        q = np.load(qpath, allow_pickle=True).item()
        meta["mean"]  = np.asarray(q["mean"]).ravel().tolist()
        meta["std"]   = np.asarray(q["std"]).ravel().tolist()
        meta["qa_st"] = float(np.asarray(q["qa_st"]))
        meta["qb_st"] = float(np.asarray(q["qb_st"]))
        meta["qa_ae"] = float(np.asarray(q["qa_ae"]))
        meta["qb_ae"] = float(np.asarray(q["qb_ae"]))
        print(f"[Export] Carried through quantile stats from {qpath}")
    except Exception as e:
        print(f"[Export] WARNING: could not load quantiles: {e}")

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[Export] Done. Files in {out_dir}/")
    print(f"[Export] ae_exported = {ae_exported}")


if __name__ == "__main__":
    main()