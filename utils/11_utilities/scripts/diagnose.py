"""
diagnose.py
===========
Diagnoses why EfficientAD scores are not separating normal vs anomaly.
Prints raw quantile values, raw map statistics, and score distributions
BEFORE and AFTER normalisation.

"""

import os, sys, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from utils.models      import Teacher, Student, AutoEncoder
from utils.data_loader import get_AD_dataset

IMAGE_SIZE   = 256
OUT_CHANNELS = 384


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_dir',      required=True)
    p.add_argument('--dataset_path',  required=True)
    p.add_argument('--category',      default='chewinggum')
    p.add_argument('--model_size',    default='S', choices=['S','M'])
    p.add_argument('--device',        default='cuda')
    p.add_argument('--n_images',      type=int, default=30,
                   help='Number of test images to sample for diagnosis')
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device

    # ── 1. Load quantiles and print them ─────────────────────────────────────
    q_path = os.path.join(args.ckpt_dir, f'{args.category}_quantiles.npy')
    q = np.load(q_path, allow_pickle=True).item()

    print('\n' + '='*60)
    print('  QUANTILE VALUES FROM CHECKPOINT')
    print('='*60)
    for k, v in q.items():
        arr = np.array(v)
        print(f'  {k:<10} shape={arr.shape}  '
              f'min={arr.min():.6f}  max={arr.max():.6f}  '
              f'mean={arr.mean():.6f}')

    qa_st = float(np.array(q['qa_st']).mean())
    qb_st = float(np.array(q['qb_st']).mean())
    qa_ae = float(np.array(q['qa_ae']).mean())
    qb_ae = float(np.array(q['qb_ae']).mean())
    print(f'\n  Effective st  range: [{qa_st:.6f}, {qb_st:.6f}]  '
          f'width={qb_st-qa_st:.6f}')
    print(f'  Effective ae  range: [{qa_ae:.6f}, {qb_ae:.6f}]  '
          f'width={qb_ae-qa_ae:.6f}')

    # ── 2. Load models ────────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('  LOADING MODELS')
    print('='*60)
    teacher = Teacher(args.model_size)
    student = Student(args.model_size)
    ae      = AutoEncoder()

    def _load(model, path):
        model.load_state_dict(
            torch.load(path, map_location=device, weights_only=False))
        model.eval().to(device)

    _load(teacher, os.path.join(args.ckpt_dir, 'best_teacher.pth'))
    _load(student, os.path.join(args.ckpt_dir, f'{args.category}_student.pth'))
    _load(ae,      os.path.join(args.ckpt_dir, f'{args.category}_autoencoder.pth'))

    def _t(v):
        return torch.tensor(np.array(v), dtype=torch.float32, device=device)

    mean  = _t(q['mean'])
    std   = _t(q['std'])
    qa_st_t = _t(q['qa_st'])
    qb_st_t = _t(q['qb_st'])
    qa_ae_t = _t(q['qa_ae'])
    qb_ae_t = _t(q['qb_ae'])

    print('  Models loaded OK')

    # ── 3. Load dataset ───────────────────────────────────────────────────────
    tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    dataset = get_AD_dataset(
        type='VisA', root=args.dataset_path,
        transform=tf, gt_transform=tf,
        phase='test', category=args.category)

    # ── 4. Run on sample images and collect raw stats ─────────────────────────
    print('\n' + '='*60)
    print(f'  RAW SCORE ANALYSIS  (first {args.n_images} images)')
    print('='*60)

    raw_st_normal, raw_st_anomaly   = [], []
    raw_ae_normal, raw_ae_anomaly   = [], []
    norm_st_normal, norm_st_anomaly = [], []
    norm_ae_normal, norm_ae_anomaly = [], []
    final_normal, final_anomaly     = [], []

    for i, sample in enumerate(dataset):
        if i >= args.n_images:
            break

        img_t = sample['image'].unsqueeze(0).to(device)
        label = int(sample['label'])

        with torch.no_grad():
            t_out = teacher(img_t)
            s_out = student(img_t)
            a_out = ae(img_t)

        y_st   = s_out[:, :OUT_CHANNELS,  :, :]
        y_stae = s_out[:, -OUT_CHANNELS:, :, :]

        norm_t = (t_out - mean) / (std + 1e-8)
        d_st   = torch.pow(norm_t - y_st,   2)
        d_stae = torch.pow(a_out  - y_stae, 2)

        fm_st   = torch.mean(d_st,   dim=1, keepdim=True)
        fm_stae = torch.mean(d_stae, dim=1, keepdim=True)
        fm_st   = F.interpolate(fm_st,   size=(IMAGE_SIZE,IMAGE_SIZE),
                                mode='bilinear', align_corners=False)
        fm_stae = F.interpolate(fm_stae, size=(IMAGE_SIZE,IMAGE_SIZE),
                                mode='bilinear', align_corners=False)

        raw_st_val  = float(fm_st.max().cpu())
        raw_ae_val  = float(fm_stae.max().cpu())

        norm_mst = (0.1 * (fm_st   - qa_st_t)) / (qb_st_t - qa_st_t + 1e-8)
        norm_mae = (0.1 * (fm_stae - qa_ae_t)) / (qb_ae_t - qa_ae_t + 1e-8)
        combined = 0.5 * norm_mst + 0.5 * norm_mae
        final_score = float(combined.max().cpu())

        if label == 0:
            raw_st_normal.append(raw_st_val)
            raw_ae_normal.append(raw_ae_val)
            norm_st_normal.append(float(norm_mst.max().cpu()))
            norm_ae_normal.append(float(norm_mae.max().cpu()))
            final_normal.append(final_score)
        else:
            raw_st_anomaly.append(raw_st_val)
            raw_ae_anomaly.append(raw_ae_val)
            norm_st_anomaly.append(float(norm_mst.max().cpu()))
            norm_ae_anomaly.append(float(norm_mae.max().cpu()))
            final_anomaly.append(final_score)

    def stats(lst, name):
        a = np.array(lst)
        if len(a) == 0:
            print(f'  {name}: (no samples)')
            return
        print(f'  {name:<30} n={len(a):3d}  '
              f'min={a.min():.4f}  max={a.max():.4f}  '
              f'mean={a.mean():.4f}  std={a.std():.4f}')

    print('\n  --- Raw map max BEFORE quantile normalisation ---')
    stats(raw_st_normal,  'ST map max  — NORMAL')
    stats(raw_st_anomaly, 'ST map max  — ANOMALY')
    stats(raw_ae_normal,  'AE map max  — NORMAL')
    stats(raw_ae_anomaly, 'AE map max  — ANOMALY')

    print('\n  --- Normalised map max (ratio=0.1) ---')
    stats(norm_st_normal,  'norm ST max — NORMAL')
    stats(norm_st_anomaly, 'norm ST max — ANOMALY')
    stats(norm_ae_normal,  'norm AE max — NORMAL')
    stats(norm_ae_anomaly, 'norm AE max — ANOMALY')

    print('\n  --- Final combined score ---')
    stats(final_normal,  'Final score — NORMAL')
    stats(final_anomaly, 'Final score — ANOMALY')

    # ── 5. Diagnosis ──────────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('  DIAGNOSIS')
    print('='*60)

    if len(final_normal) > 0 and len(final_anomaly) > 0:
        fn = np.array(final_normal)
        fa = np.array(final_anomaly)
        overlap = np.sum(fn > fa.min()) / len(fn)

        if fn.mean() > 0.8:
            print('  ⚠  Normal scores are high (>0.8 mean).')
            print('     Likely cause: quantiles qa/qb were computed on')
            print('     a DIFFERENT dataset split than what is saved.')
            print('     The qb values are too small, pushing all scores high.')
            print()
            print('  FIX: Run recompute_quantiles.py to recalculate quantiles')
            print('       on the training split of the VisA chewinggum data.')
        elif overlap > 0.3:
            print('  ⚠  High score overlap between normal and anomaly.')
            print('     The model may not have converged well for this category.')
        else:
            print('  ✓  Scores look separable. Check threshold calibration.')

    print()
    print(f'  qa_st={qa_st:.6f}  qb_st={qb_st:.6f}')
    print(f'  qa_ae={qa_ae:.6f}  qb_ae={qb_ae:.6f}')
    if len(raw_st_normal) > 0:
        actual_p90 = np.percentile(raw_st_normal, 90)
        actual_p99 = np.percentile(raw_st_normal, 99.5)
        print(f'\n  Actual p90 of raw ST on normal images: {actual_p90:.6f}')
        print(f'  Actual p99.5 of raw ST on normal:      {actual_p99:.6f}')
        print(f'  Saved qa_st:                           {qa_st:.6f}')
        if abs(actual_p90 - qa_st) / (qa_st + 1e-8) > 0.2:
            print('\n  *** MISMATCH DETECTED ***')
            print('  The saved qa_st does not match the actual p90 of normal images.')
            print('  This confirms the quantiles need to be recomputed.')
            print('  Run: python recompute_quantiles.py --ckpt_dir '
                  + args.ckpt_dir +
                  ' --dataset_path ' + args.dataset_path)


if __name__ == '__main__':
    main()
