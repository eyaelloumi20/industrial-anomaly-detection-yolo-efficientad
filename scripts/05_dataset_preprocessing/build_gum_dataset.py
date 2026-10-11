"""
build_gum_dataset.py
=====================
Score a folder of pre-cropped chewing-gum images with a trained EfficientAD
model and sort them into the dataset_root/<category>/Data/Images/{Normal,Anomaly}
arborescence, using a calibrated threshold (from evaluate_pipeline.py's
threshold.txt).

Crops are expected to already be 256x256, Otsu-refined, aspect-ratio resized
+ padded (i.e. output of crop_dataset_video.py / crop_dataset_video1.py) —
this script therefore runs the pipeline in --use_full_image mode (no YOLO
re-detection) and, by default, WITHOUT re-denoising (assumes denoising, if
any, was already applied during cropping).

Usage:
    python build_gum_dataset.py \
        --ckpt_dir     /home/ahmed/framework/ckpt_cropped_large \
        --crops_dir    ./video1_crops/crops \
        --dataset_root ./dataset_root \
        --threshold_file ./results_yolo_nodenoise/threshold.txt \
        --model_size S --device cuda

    # Dry run first (no files copied, just prints the split counts):
    python build_gum_dataset.py ... --dry_run

    # Move instead of copy (frees disk, but destroys the source crops folder):
    python build_gum_dataset.py ... --move
"""

import os
import sys
import shutil
import argparse
import warnings
warnings.filterwarnings('ignore')

from PIL import Image
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from utils.load_efficientad import load_efficientad_model
from pipeline.pipeline import InspectionPipeline

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')


def parse_args():
    p = argparse.ArgumentParser(
        description='Auto-label video crops Normal/Anomaly with a trained '
                     'EfficientAD model and build dataset_root arborescence.')
    p.add_argument('--ckpt_dir',      required=True,
                   help='EfficientAD checkpoint directory')
    p.add_argument('--crops_dir',     required=True,
                   help='Folder of pre-cropped images to label '
                        '(e.g. video1_crops/crops)')
    p.add_argument('--dataset_root',  required=True,
                   help='Root output folder — will create '
                        '<root>/<category>/Data/Images/{Normal,Anomaly}')
    p.add_argument('--category',      default='chewinggum')
    p.add_argument('--model_size',    default='S', choices=['S', 'M'])
    p.add_argument('--device',        default=None,
                   help='cuda or cpu (auto-detected if omitted)')
    p.add_argument('--ratio',         type=float, default=0.1,
                   help='EfficientAD normalisation ratio (default 0.1, '
                        'must match training)')

    thr = p.add_mutually_exclusive_group(required=True)
    thr.add_argument('--threshold',      type=float,
                      help='Anomaly-score threshold — score >= threshold -> Anomaly')
    thr.add_argument('--threshold_file', type=str,
                      help='Path to threshold.txt produced by evaluate_pipeline.py')

    p.add_argument('--denoise',       action='store_true',
                   help='Re-apply Gaussian+bilateral denoising before scoring '
                        '(default OFF — assumes crops were already denoised/not '
                        'as desired during cropping)')
    p.add_argument('--move',          action='store_true',
                   help='Move files into dataset_root instead of copying '
                        '(default: copy)')
    p.add_argument('--dry_run',       action='store_true',
                   help='Score and report counts only — do not copy/move files')
    return p.parse_args()


def main(args):
    import torch

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    threshold = args.threshold
    if args.threshold_file:
        with open(args.threshold_file) as f:
            threshold = float(f.read().strip())

    print(f'\n{"="*60}')
    print(f'  Auto-labeling crops — {args.category.upper()}')
    print(f'{"="*60}')
    print(f'  Device      : {device}')
    print(f'  EfficientAD : PDN-{args.model_size}  ckpt={args.ckpt_dir}')
    print(f'  Crops dir   : {args.crops_dir}')
    print(f'  Threshold   : {threshold:.5f}')
    print(f'  Denoising   : {"ON" if args.denoise else "OFF (crops assumed pre-processed)"}')
    print(f'  Mode        : {"DRY RUN (no files written)" if args.dry_run else ("MOVE" if args.move else "COPY")}')
    print()

    # ── 1. Load model + pipeline (use_full_image: crops are already cropped) ──
    models = load_efficientad_model(
        ckpt_dir   = args.ckpt_dir,
        category   = args.category,
        model_size = args.model_size,
        device     = device,
    )
    pipeline = InspectionPipeline(
        models         = models,
        device         = device,
        use_full_image = True,      # skip YOLO — crop is already the full input
        detector_path  = None,
        ratio          = args.ratio,
        pad_ratio      = 0.0,       # no extra padding — already padded when cropped
        denoise        = args.denoise,
    )

    # ── 2. Prepare output dirs ─────────────────────────────────────────────────
    normal_dir  = os.path.join(args.dataset_root, args.category, 'Data', 'Images', 'Normal')
    anomaly_dir = os.path.join(args.dataset_root, args.category, 'Data', 'Images', 'Anomaly')
    mask_dir    = os.path.join(args.dataset_root, args.category, 'Data', 'Masks', 'Anomaly')
    if not args.dry_run:
        os.makedirs(normal_dir,  exist_ok=True)
        os.makedirs(anomaly_dir, exist_ok=True)
        os.makedirs(mask_dir,    exist_ok=True)  # left empty — no GT masks available

    # ── 3. Gather crop files ───────────────────────────────────────────────────
    files = sorted(
        f for f in os.listdir(args.crops_dir)
        if f.lower().endswith(IMAGE_EXTS)
    )
    if not files:
        raise RuntimeError(f'No images found in {args.crops_dir}')
    print(f'[Score] {len(files)} crops found\n')

    # ── 4. Score + sort ─────────────────────────────────────────────────────────
    n_normal = n_anomaly = 0
    scores = []
    op = shutil.move if args.move else shutil.copy2

    for fname in tqdm(files, desc='  Scoring'):
        src_path = os.path.join(args.crops_dir, fname)
        pil_image = Image.open(src_path).convert('RGB')

        result = pipeline.run(
            pil_image = pil_image,
            label     = 0,          # unknown — not used for scoring itself
            name      = os.path.splitext(fname)[0],
            img_type  = 'unlabeled',
        )
        scores.append(result.image_score)

        is_anomaly = result.image_score >= threshold
        dest_dir   = anomaly_dir if is_anomaly else normal_dir
        n_anomaly += int(is_anomaly)
        n_normal  += int(not is_anomaly)

        if not args.dry_run:
            op(src_path, os.path.join(dest_dir, fname))

    # ── 5. Summary ───────────────────────────────────────────────────────────
    import numpy as np
    scores = np.array(scores)
    print(f'\n{"="*60}')
    print(f'  Total crops   : {len(files)}')
    print(f'  -> Normal     : {n_normal}  ({100*n_normal/len(files):.1f}%)')
    print(f'  -> Anomaly    : {n_anomaly}  ({100*n_anomaly/len(files):.1f}%)')
    print(f'  Score range   : [{scores.min():.4f}, {scores.max():.4f}]  '
          f'mean={scores.mean():.4f}  std={scores.std():.4f}')
    if not args.dry_run:
        print(f'  Normal  saved -> {normal_dir}')
        print(f'  Anomaly saved -> {anomaly_dir}')
    print(f'{"="*60}\n')

    if not args.dry_run:
        print('⚠  Reminder: images sorted into Anomaly/ have NO ground-truth mask.')
        print('   Masks/Anomaly/ was created empty — add masks manually if you need')
        print('   pixel-level (AUPRO) evaluation on this new data; image-level')
        print('   Normal/Anomaly labels alone are enough to grow the training set.')


if __name__ == '__main__':
    main(parse_args())
