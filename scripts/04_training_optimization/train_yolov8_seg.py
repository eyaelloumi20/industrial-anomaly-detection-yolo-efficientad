"""
YOLOv8 Instance Segmentation Training Script
============================================
Trains a YOLOv8-seg model on a custom dataset.

Directory structure expected:
    dataset/
    ├── images/
    │   ├── train/   *.jpg / *.png
    │   └── val/     *.jpg / *.png
    └── labels/
        ├── train/   *.txt  (YOLO polygon format)
        └── val/     *.txt

Label format (one object per line):
    <class_id> x1 y1 x2 y2 ... xn yn   (normalised 0-1 coords)

Usage:
    python train_yolov8_seg.py [--config dataset.yaml] [--model yolov8s-seg.pt]
                               [--epochs 100] [--imgsz 640] [--batch 16]
                               [--name my_run] [--resume]
"""

import argparse
import os
import sys
from pathlib import Path

# ── Optional: silence Ultralytics verbose output ─────────────────────────────
os.environ.setdefault("YOLO_VERBOSE", "True")

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit(
        "[ERROR] ultralytics not found.\n"
        "Install it with:  pip install ultralytics"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Default hyper-parameters  (override via CLI args or edit directly)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULTS = dict(
    # ── Paths ──────────────────────────────────────────────────────────────
    config   = "dataset.yaml",          # YAML describing dataset
    model    = "yolov8s-seg.pt",        # backbone: n / s / m / l / x
    project  = "runs/segment",          # output root
    name     = "train",                 # sub-folder name (auto-incremented)

    # ── Core training ──────────────────────────────────────────────────────
    epochs   = 100,
    imgsz    = 640,                     # square resize; use 256 for small crops
    batch    = 16,                      # -1 = AutoBatch
    device   = "",                      # "" = auto-select GPU/CPU

    # ── Optimiser ──────────────────────────────────────────────────────────
    optimizer = "AdamW",                # SGD | Adam | AdamW | RMSProp
    lr0      = 1e-3,                    # initial learning rate
    lrf      = 0.01,                    # final lr = lr0 * lrf
    momentum = 0.937,
    weight_decay = 5e-4,
    warmup_epochs = 3.0,
    warmup_momentum = 0.8,
    warmup_bias_lr  = 0.1,

    # ── Regularisation / augmentation ──────────────────────────────────────
    dropout  = 0.0,
    label_smoothing = 0.0,
    mixup    = 0.0,
    copy_paste = 0.0,                   # useful for segmentation
    degrees  = 0.0,                     # rotation aug
    translate = 0.1,
    scale    = 0.5,
    shear    = 0.0,
    perspective = 0.0,
    flipud   = 0.0,
    fliplr   = 0.5,
    mosaic   = 1.0,
    close_mosaic = 10,                  # disable mosaic last N epochs

    # ── Evaluation / saving ────────────────────────────────────────────────
    val      = True,
    save     = True,
    save_period = -1,                   # -1 = save best + last only
    patience = 50,                      # early-stopping patience
    workers  = 8,

    # ── Misc ───────────────────────────────────────────────────────────────
    exist_ok = False,                   # overwrite existing run folder
    pretrained = True,                  # start from COCO pretrained weights
    resume   = False,                   # resume from last.pt
    seed     = 42,
    amp      = True,                    # mixed-precision (FP16)
    cache    = False,                   # cache images in RAM ("ram") or disk
    plots    = True,                    # save training curves & confusion matrix
    verbose  = True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset YAML  (written automatically if the file doesn't exist)
# ─────────────────────────────────────────────────────────────────────────────
TEMPLATE_YAML = """\
# Auto-generated dataset config — edit paths and class names as needed.
path: ./dataset          # root directory
train: images/train
val:   images/val
# test: images/test      # optional

nc: 1                    # number of classes
names:
  0: chewinggum          # class names (add more as needed)
"""


def ensure_dataset_yaml(path: str) -> None:
    p = Path(path)
    if not p.exists():
        p.write_text(TEMPLATE_YAML)
        print(f"[INFO] Created template dataset YAML → {p.resolve()}")
        print("       Edit it to point at your data before training.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args(defaults: dict) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a YOLOv8 instance-segmentation model."
    )
    p.add_argument("--config",       default=defaults["config"],
                   help="Path to dataset YAML")
    p.add_argument("--model",        default=defaults["model"],
                   help="Model variant or path to .pt checkpoint  "
                        "[yolov8n-seg | yolov8s-seg | yolov8m-seg | "
                        "yolov8l-seg | yolov8x-seg]")
    p.add_argument("--project",      default=defaults["project"])
    p.add_argument("--name",         default=defaults["name"])
    p.add_argument("--epochs",       type=int,   default=defaults["epochs"])
    p.add_argument("--imgsz",        type=int,   default=defaults["imgsz"])
    p.add_argument("--batch",        type=int,   default=defaults["batch"])
    p.add_argument("--device",       default=defaults["device"],
                   help="GPU id(s) e.g. 0 | 0,1 | cpu")
    p.add_argument("--optimizer",    default=defaults["optimizer"])
    p.add_argument("--lr0",          type=float, default=defaults["lr0"])
    p.add_argument("--workers",      type=int,   default=defaults["workers"])
    p.add_argument("--patience",     type=int,   default=defaults["patience"])
    p.add_argument("--save_period",  type=int,   default=defaults["save_period"])
    p.add_argument("--cache",        default=defaults["cache"],
                   help="Cache images: False | 'ram' | 'disk'")
    p.add_argument("--resume",       action="store_true",
                   help="Resume training from last checkpoint")
    p.add_argument("--exist_ok",     action="store_true",
                   help="Overwrite existing run folder")
    p.add_argument("--no_amp",       action="store_true",
                   help="Disable mixed-precision training")
    p.add_argument("--no_plots",     action="store_true",
                   help="Skip saving training plots")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args(DEFAULTS)

    # ── Guarantee dataset YAML exists ──────────────────────────────────────
    ensure_dataset_yaml(args.config)

    # ── Load model ─────────────────────────────────────────────────────────
    # If --resume, ultralytics expects the path to a last.pt checkpoint.
    if args.resume:
        ckpt = Path(args.project) / args.name / "weights" / "last.pt"
        if not ckpt.exists():
            sys.exit(f"[ERROR] Resume checkpoint not found: {ckpt}")
        print(f"[INFO] Resuming from {ckpt}")
        model = YOLO(str(ckpt))
    else:
        model_name = args.model
        # Allow bare names like "yolov8s" → append "-seg.pt"
        if not model_name.endswith(".pt") and "-seg" not in model_name:
            model_name = model_name.rstrip(".pt") + "-seg.pt"
        print(f"[INFO] Loading model: {model_name}")
        model = YOLO(model_name)

    # ── Build training kwargs ───────────────────────────────────────────────
    train_kwargs = dict(
        data          = args.config,
        epochs        = args.epochs,
        imgsz         = args.imgsz,
        batch         = args.batch,
        device        = args.device or None,
        project       = args.project,
        name          = args.name,
        optimizer     = args.optimizer,
        lr0           = args.lr0,
        lrf           = DEFAULTS["lrf"],
        momentum      = DEFAULTS["momentum"],
        weight_decay  = DEFAULTS["weight_decay"],
        warmup_epochs = DEFAULTS["warmup_epochs"],
        warmup_momentum = DEFAULTS["warmup_momentum"],
        warmup_bias_lr  = DEFAULTS["warmup_bias_lr"],
        label_smoothing = DEFAULTS["label_smoothing"],
        dropout       = DEFAULTS["dropout"],
        mixup         = DEFAULTS["mixup"],
        copy_paste    = DEFAULTS["copy_paste"],
        degrees       = DEFAULTS["degrees"],
        translate     = DEFAULTS["translate"],
        scale         = DEFAULTS["scale"],
        flipud        = DEFAULTS["flipud"],
        fliplr        = DEFAULTS["fliplr"],
        mosaic        = DEFAULTS["mosaic"],
        close_mosaic  = DEFAULTS["close_mosaic"],
        val           = DEFAULTS["val"],
        save          = DEFAULTS["save"],
        save_period   = args.save_period,
        patience      = args.patience,
        workers       = args.workers,
        exist_ok      = args.exist_ok,
        pretrained    = DEFAULTS["pretrained"],
        resume        = args.resume,
        seed          = DEFAULTS["seed"],
        amp           = not args.no_amp,
        cache         = args.cache,
        plots         = not args.no_plots,
        verbose       = DEFAULTS["verbose"],
    )

    # ── Train ──────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  YOLOv8 Instance Segmentation — Training")
    print("═" * 60)
    print(f"  Dataset  : {args.config}")
    print(f"  Model    : {args.model}")
    print(f"  Epochs   : {args.epochs}  |  imgsz: {args.imgsz}  |  batch: {args.batch}")
    print(f"  Output   : {args.project}/{args.name}")
    print("═" * 60 + "\n")

    results = model.train(**train_kwargs)

    # ── Post-training summary ──────────────────────────────────────────────
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print("\n" + "═" * 60)
    print("  Training complete!")
    print(f"  Best weights : {best_weights.resolve()}")
    print(f"  Results dir  : {Path(results.save_dir).resolve()}")
    print("═" * 60)

    # ── Quick validation on best model ─────────────────────────────────────
    if best_weights.exists():
        print("\n[INFO] Running final validation with best weights …")
        best_model = YOLO(str(best_weights))
        val_results = best_model.val(data=args.config, imgsz=args.imgsz,
                                     device=args.device or None)
        print(f"  box  mAP@50   : {val_results.box.map50:.4f}")
        print(f"  box  mAP@50-95: {val_results.box.map:.4f}")
        print(f"  mask mAP@50   : {val_results.seg.map50:.4f}")
        print(f"  mask mAP@50-95: {val_results.seg.map:.4f}")


if __name__ == "__main__":
    main()
