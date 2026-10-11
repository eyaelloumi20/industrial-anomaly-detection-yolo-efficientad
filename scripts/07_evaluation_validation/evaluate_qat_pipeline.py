"""
evaluate_qat_pipeline.py
===========================
int8-aware sibling of evaluate_pipeline.py, for checkpoints produced by
train_qat.py (quantized Student/AutoEncoder, fp32 Teacher).

Like the thin-model case, QAT checkpoints train at the ORIGINAL 256px
resolution -- quantization doesn't change spatial dimensions -- so
pipeline.py's InspectionPipeline needs NO changes here either. int8
quantized modules still expose the same eval()/to()/callable-forward
interface and return regular float32 tensors after DeQuantStub, so
_efficientad_score's torch.pow/F.interpolate calls work completely
unmodified.

The only real difference from evaluate_pipeline.py:
    from utils.load_efficientad_qat import load_efficientad_model
    from pipeline.pipeline          import InspectionPipeline   # unmodified
and this script FORCES device='cpu' regardless of --device, because int8
kernels in PyTorch eager mode are CPU-only -- passing --device cuda here
will print a warning and silently run on CPU anyway (load_efficientad_qat.py
enforces this at the loader level too, so you can't get it wrong twice).

This also means the latency numbers you get from this script are the ones
that actually matter for a CPU deployment target -- there's no GPU number
to compare against, unlike the lowres/thin checkpoints.

Everything else is copied verbatim from evaluate_pipeline.py: same output
folder layout, same console format, same utils.metrics / utils.visualisation
calls.

Usage
-----
    python evaluate_qat_pipeline.py \
        --dataset_path /path/to/dataset_root \
        --ckpt_dir     ./ckpt_qat \
        --yolo_weights /path/to/best.pt \
        --category     chewinggum \
        --save_dir     ./results_qat \
        --model_size   S

Note: --ckpt_dir must contain best_teacher.pth -- train_qat.py does not copy
it into --out_dir automatically:
    cp ./ckpt/best_teacher.pth ./ckpt_qat/best_teacher.pth
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from utils.load_efficientad_qat import load_efficientad_model
from pipeline.pipeline      import InspectionPipeline
from utils.data_loader      import get_AD_dataset
from utils.metrics          import compute_all_metrics, print_results, save_quantitative_txt
from utils.visualisation    import (save_roc_curve, save_score_histogram,
                                   save_overlay, save_tiff,
                                   save_per_image_csv, print_per_image_table)
from PIL import ImageDraw, ImageFont

IMAGE_SIZE = 256   # QAT checkpoints always train at the original resolution


# ── args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Two-stage YOLOv8 + EfficientAD evaluation pipeline (int8/QAT checkpoints, CPU only).')
    p.add_argument('--dataset_path',   required=True,
                   help='VisA root or cropped_dataset root '
                        '(must contain split_csv/1cls.csv)')
    p.add_argument('--ckpt_dir',       required=True,
                   help='train_qat.py --out_dir. Must also contain best_teacher.pth '
                        '(copy it in yourself -- train_qat.py does not).')
    p.add_argument('--yolo_weights',   default="/home/ahmed/two-stage-efficientad-inspection/runs/detect/yolo_runs1/chewinggum_finetune/weights/best.pt",
                   help='Fine-tuned YOLOv8 .pt weights '
                        '(ignored when --use_full_image is set)')
    p.add_argument('--category',       default='chewinggum')
    p.add_argument('--save_dir',       default='./results')
    p.add_argument('--model_size',     default='S', choices=['S', 'M'])
    p.add_argument('--device',         default='cpu',
                   help='Ignored -- always forced to cpu for int8. Kept for '
                        'command-line consistency with the other evaluate_*_pipeline.py scripts.')
    p.add_argument('--ratio',          type=float, default=0.1,
                   help='EfficientAD normalisation ratio (default 0.1)')
    p.add_argument('--conf_threshold', type=float, default=0.25,
                   help='YOLO confidence threshold (default 0.25)')
    p.add_argument('--iou_threshold',  type=float, default=0.45,
                   help='YOLO NMS IoU threshold (default 0.45)')
    p.add_argument('--pad_ratio',      type=float, default=0.02,
                   help='Context padding around each YOLO crop (default 0.02, '
                        'matches build_crop_dataset.py)')
    p.add_argument('--no_denoise',     action='store_true',
                   help='Disable Gaussian+bilateral denoising on crops')
    p.add_argument('--use_full_image', action='store_true',
                   help='Skip YOLO — use full image as one crop. '
                        'Use this when evaluating a pre-cropped dataset.')
    p.add_argument('--no_overlays',    action='store_true',
                   help='Skip saving heatmap overlay PNGs (faster)')
    p.add_argument('--no_tiffs',       action='store_true',
                   help='Skip saving float32 TIFF anomaly maps')
    p.add_argument('--n_boot',         type=int, default=500,
                   help='Bootstrap resamples for metric std (default 500)')
    return p.parse_args()


# ── main evaluation ───────────────────────────────────────────────────────────

def evaluate(args) -> dict:
    device = 'cpu'   # int8 eager-mode kernels are CPU-only, no exceptions
    if args.device != 'cpu':
        print(f"NOTE: --device={args.device} ignored, running on cpu (int8 requires it).")

    print(f'\n{"="*60}')
    print(f'  Two-Stage Inspection Pipeline — {args.category.upper()}  (int8/QAT, CPU)')
    print(f'{"="*60}')
    print(f'  Device      : {device}')
    print(f'  EfficientAD : PDN-{args.model_size}  ckpt={args.ckpt_dir}  [int8]')
    if args.use_full_image:
        print(f'  Detector    : FULL-IMAGE (--use_full_image)')
    else:
        print(f'  Detector    : YOLOv8  weights={args.yolo_weights or "yolov8n.pt"}')
    print(f'  Crop padding: {args.pad_ratio * 100:.0f}%')
    print(f'  Denoising   : {"OFF" if args.no_denoise else "ON (Gaussian + Bilateral)"}')
    print(f'  Dataset     : {args.dataset_path}')
    print(f'  Eval split  : validation  (normal + anomaly)')
    print()

    # ── 1. Load EfficientAD (int8, CPU-forced) ────────────────────────────────
    models = load_efficientad_model(
        ckpt_dir   = args.ckpt_dir,
        category   = args.category,
        model_size = args.model_size,
        device     = device,
    )
    print(f'  Backend     : {models.get("backend")}')

    # ── 2. Build pipeline (unmodified -- QAT checkpoints are always 256px) ────
    pipeline = InspectionPipeline(
        models         = models,
        device         = device,
        use_full_image = args.use_full_image,
        detector_path  = args.yolo_weights,
        conf_threshold = args.conf_threshold,
        iou_threshold  = args.iou_threshold,
        ratio          = args.ratio,
        pad_ratio      = args.pad_ratio,
        denoise        = not args.no_denoise,
    )
    print(f'[Pipeline] YOLO active : {pipeline.detector.is_yolo_active}')

    # ── 3. Load dataset ───────────────────────────────────────────────────────
    def _resize_with_padding(pil_img: Image.Image,
                             size: int = IMAGE_SIZE) -> Image.Image:
        """Aspect-ratio-preserving resize + black padding — mirrors build_crop_dataset.py."""
        ow, oh = pil_img.size
        scale  = size / max(ow, oh)
        nw, nh = int(ow * scale), int(oh * scale)
        resized = pil_img.resize((nw, nh), Image.LANCZOS)
        canvas  = Image.new('RGB', (size, size), (0, 0, 0))
        canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
        return canvas

    tf = transforms.Compose([
        transforms.Lambda(_resize_with_padding),
        transforms.ToTensor(),
    ])

    print(f'[Pipeline] Loading {args.category} validation split ...')
    dataset = get_AD_dataset(
        type         = 'VisA',
        root         = args.dataset_path,
        transform    = tf,
        gt_transform = tf,
        phase        = 'validation',   # ← dedicated eval split (normal + anomaly)
        category     = args.category,
    )

    n_normal  = sum(1 for i in range(len(dataset)) if dataset[i]['label'] == 0)
    n_anomaly = sum(1 for i in range(len(dataset)) if dataset[i]['label'] == 1)
    print(f'[Pipeline] {len(dataset)} validation images  '
          f'(normal={n_normal}, anomaly={n_anomaly})')

    # ── 4. Output dirs ────────────────────────────────────────────────────────
    cat_dir     = os.path.join(args.save_dir, args.category)
    overlay_dir = os.path.join(cat_dir, 'overlays')
    map_dir     = os.path.join(cat_dir, 'anomaly_maps')
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(map_dir,     exist_ok=True)

    # ── 5. Inference loop ─────────────────────────────────────────────────────
    print(f'\n[Pipeline] Running inference (CPU) ...')
    y_true, y_score, times_s = [], [], []
    pipeline_results = []
    pil_images = []
    csv_rows = []

    for sample in tqdm(dataset, desc='  Inference'):
        pil_image = Image.fromarray(sample['origin'].astype(np.uint8))

        result = pipeline.run(
            pil_image = pil_image,
            label     = int(sample['label']),
            name      = sample['name'],
            img_type  = sample['type'],
        )

        y_true.append(result.label)
        y_score.append(result.image_score)
        times_s.append(result.inference_ms / 1000.0)
        pipeline_results.append(result)
        pil_images.append(pil_image)

        _ = args.no_overlays  # checked later

        if not args.no_tiffs:
            class_map_dir = os.path.join(map_dir, result.img_type)
            os.makedirs(class_map_dir, exist_ok=True)
            save_tiff(result.anomaly_map,
                      os.path.join(class_map_dir, result.name + '.tiff'))

    # ── 6. Metrics ────────────────────────────────────────────────────────────
    y_true  = np.array(y_true)
    y_score = np.array(y_score)
    times_s = np.array(times_s)

    print(f'\n[Pipeline] Computing metrics ...')
    m = compute_all_metrics(y_true, y_score, times_s, n_boot=args.n_boot)

    def _outcome_folder(gt: int, pred: str) -> str:
        if gt == 1 and pred == 'ANOMALY':  return 'TP'
        if gt == 0 and pred == 'ANOMALY':  return 'FP'
        if gt == 0 and pred == 'NORMAL':   return 'TN'
        return 'FN'

    pipeline.set_threshold(m['threshold'])
    for r in pipeline_results:
        r.prediction = 'ANOMALY' if r.image_score >= m['threshold'] else 'NORMAL'
        outcome = _outcome_folder(r.label, r.prediction)
        csv_rows.append({
            'defect_class': r.img_type,
            'filename':     r.name,
            'gt':           r.label,
            'pred':         1 if r.prediction == 'ANOMALY' else 0,
            'outcome':      outcome,
            'score':        round(r.image_score, 6),
            'correct':      int(r.correct) if r.correct is not None else '',
            'time_ms':      round(r.inference_ms, 4),
        })

    # ── 7. Overlays — organised by TP / FP / TN / FN ─────────────────────────
    def _make_side_by_side(orig: Image.Image,
                           overlay_path: str,
                           outcome: str,
                           score: float,
                           gt: int) -> Image.Image:
        overlay_img = Image.open(overlay_path).convert('RGB')
        h = max(orig.height, overlay_img.height)
        orig_r    = orig.resize((int(orig.width  * h / orig.height),    h), Image.LANCZOS)
        overlay_r = overlay_img.resize((int(overlay_img.width * h / overlay_img.height), h), Image.LANCZOS)

        bar_h  = 24
        total_w = orig_r.width + overlay_r.width
        canvas = Image.new('RGB', (total_w, h + bar_h), (30, 30, 30))
        canvas.paste(orig_r,    (0,             bar_h))
        canvas.paste(overlay_r, (orig_r.width,  bar_h))

        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 14)
        except Exception:
            font = ImageFont.load_default()

        gt_str  = 'ANOMALY' if gt == 1 else 'NORMAL'
        label   = f'  [{outcome}]  GT: {gt_str}   Score: {score:.4f}   | original (left)  overlay (right)'
        colours = {'TP': (80, 200, 80), 'TN': (80, 160, 255),
                   'FP': (255, 180, 0), 'FN': (255, 80,  80)}
        draw.rectangle([0, 0, total_w, bar_h], fill=colours.get(outcome, (60, 60, 60)))
        draw.text((4, 4), label, fill=(0, 0, 0), font=font)
        return canvas

    if not args.no_overlays:
        for folder in ('TP', 'FP', 'TN', 'FN'):
            os.makedirs(os.path.join(overlay_dir, folder), exist_ok=True)

        print(f'\n[Pipeline] Saving overlays (side-by-side, sorted by outcome) ...')
        for result, pil_img in tqdm(zip(pipeline_results, pil_images),
                                    total=len(pipeline_results),
                                    desc='  Overlays'):
            outcome = _outcome_folder(result.label, result.prediction)

            tmp_overlay = os.path.join(
                overlay_dir,
                f'_tmp_{result.img_type}_{result.name}_overlay.png')
            save_overlay(pil_img, result.anomaly_map, tmp_overlay)

            sbs = _make_side_by_side(
                pil_img, tmp_overlay, outcome,
                result.image_score, result.label)

            final_path = os.path.join(
                overlay_dir, outcome,
                f'{result.img_type}_{result.name}.png')
            sbs.save(final_path)
            os.remove(tmp_overlay)

    # ── 8. Save outputs ───────────────────────────────────────────────────────
    print_results(m, category=args.category)
    print_per_image_table(pipeline_results)

    save_roc_curve(
        m['fpr'], m['tpr'], m['auroc'], args.category,
        os.path.join(cat_dir, 'roc_curve.png'))
    save_score_histogram(
        y_score, y_true, m['threshold'], args.category,
        os.path.join(cat_dir, 'score_histogram.png'))
    save_per_image_csv(
        csv_rows,
        os.path.join(cat_dir, 'per_image_results.csv'))

    txt_path = os.path.join(args.save_dir, 'quantitative.txt')
    save_quantitative_txt({args.category: m}, txt_path)

    thr_path = os.path.join(args.save_dir, 'threshold.txt')
    with open(thr_path, 'w') as _f:
        _f.write(f"{m['threshold']:.8f}\n")

    print(f'\n[Pipeline] Results saved to {args.save_dir}/')
    print(f'  Backend          → {models.get("backend")}  (int8, CPU)')
    print(f'  ROC curve        → {os.path.join(cat_dir, "roc_curve.png")}')
    print(f'  Score histogram  → {os.path.join(cat_dir, "score_histogram.png")}')
    print(f'  Per-image CSV    → {os.path.join(cat_dir, "per_image_results.csv")}')
    print(f'  Quantitative txt → {txt_path}')
    print(f'  Threshold file   → {thr_path}  '
          f'(pass to single_photo_inference.py --threshold_file)')
    if not args.no_overlays:
        print(f'  Overlays (side-by-side) →')
        for folder in ('TP', 'FP', 'TN', 'FN'):
            n = len(os.listdir(os.path.join(overlay_dir, folder)))
            print(f'    {folder:2s}  {os.path.join(overlay_dir, folder)}  ({n} images)')

    return m


if __name__ == '__main__':
    evaluate(parse_args())
