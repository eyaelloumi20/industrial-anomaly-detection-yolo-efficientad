"""
resize_dataset.py
==================
Bulk-resizes every image in a VisA-style dataset tree to a target
resolution, writing a full copy of the tree to --out_root with images
resized (split_csv/1cls.csv and directory structure are preserved as-is,
since image filenames/paths don't change — only pixel content shrinks).

This is OPTIONAL for retrain_lowres.py — that script already resizes every
image on the fly via transforms.Resize((image_size, image_size)), so
training at 192 or 224 works directly against your existing 256x256
dataset with no pre-processing. Use this script if you want smaller files
on disk and/or slightly less per-step CPU resize overhead during training.

Usage
-----
    python resize_dataset.py \
        --in_root  /path/to/cropped_dataset_256 \
        --out_root /path/to/cropped_dataset_192 \
        --size     192

    python resize_dataset.py \
        --in_root  /path/to/cropped_dataset_256 \
        --out_root /path/to/cropped_dataset_224 \
        --size     224

Then point retrain_lowres.py at the resized copy:
    --dataset_path /path/to/cropped_dataset_192 --image_size 192
"""

import os
import argparse
import shutil
from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--in_root',  required=True, help='Existing 256x256 dataset root')
    p.add_argument('--out_root', required=True, help='Where to write the resized copy')
    p.add_argument('--size',     type=int, required=True, choices=[192, 224, 256],
                   help='Target square resolution')
    p.add_argument('--resample', default='lanczos',
                   choices=['nearest', 'bilinear', 'bicubic', 'lanczos'],
                   help='Downsampling filter. lanczos (default) gives the '
                        'sharpest quality for shrinking; bilinear matches what '
                        'the training script itself uses if you want an exact match.')
    return p.parse_args()


RESAMPLE_MAP = {
    'nearest':  Image.NEAREST,
    'bilinear': Image.BILINEAR,
    'bicubic':  Image.BICUBIC,
    'lanczos':  Image.LANCZOS,
}


def main():
    args = parse_args()
    resample = RESAMPLE_MAP[args.resample]

    if os.path.abspath(args.in_root) == os.path.abspath(args.out_root):
        raise ValueError('--out_root must be different from --in_root '
                          '(this script writes a new copy, it does not resize in place).')

    print(f'[Resize] Scanning {args.in_root} ...')
    all_files = []
    for dirpath, _, filenames in os.walk(args.in_root):
        for fname in filenames:
            all_files.append(os.path.join(dirpath, fname))

    n_images = sum(1 for f in all_files if f.lower().endswith(IMAGE_EXTS))
    print(f'[Resize] Found {len(all_files)} files ({n_images} images) to process.')

    os.makedirs(args.out_root, exist_ok=True)

    for src_path in tqdm(all_files, desc=f'Resizing to {args.size}px'):
        rel_path = os.path.relpath(src_path, args.in_root)
        dst_path = os.path.join(args.out_root, rel_path)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        if src_path.lower().endswith(IMAGE_EXTS):
            with Image.open(src_path) as img:
                img = img.convert('RGB') if img.mode not in ('RGB', 'L') else img
                img = img.resize((args.size, args.size), resample=resample)
                img.save(dst_path)
        else:
            # split_csv/1cls.csv and any other non-image file: copy verbatim.
            shutil.copy2(src_path, dst_path)

    print(f'\n[Resize] Done. Resized dataset written to {args.out_root}')
    print(f'  Use it with: --dataset_path {args.out_root} --image_size {args.size}')


if __name__ == '__main__':
    main()
