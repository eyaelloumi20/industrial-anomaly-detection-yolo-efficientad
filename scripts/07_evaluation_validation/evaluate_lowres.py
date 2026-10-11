"""
evaluate_lowres.py
===================
Evaluate a checkpoint produced by retrain_lowres.py (or your original
256px retrain_efficientad.py checkpoint) on the VALIDATION split
(normal + anomaly), reporting:

  1. Overall AUROC
  2. Per-defect-type AUROC breakdown (normal vs. each individual defect
     type) -- this is the number that actually tells you whether a lower
     resolution is hurting SMALL defects specifically, rather than just
     hurting everything a bit.
  3. Optional inference latency benchmark (mean/std ms per image, throughput).
  4. Optional side-by-side comparison against a second checkpoint (e.g.
     your original 256px baseline) -- AUROC delta + speedup in one table.

This script uses AutoEncoderAdaptive (models_adaptive.py) for every
checkpoint it loads, not just the lowres ones. At the original 256px
resolution AutoEncoderAdaptive's encoder/decoder are architecturally
identical to the original AutoEncoder (the adaptive pooling step is a
no-op at 8x8, and the decoder schedule scales to exactly the original
[3,8,15,32,63,127,64]), so it's a safe drop-in for state_dict loading --
meaning this one script works for both your lowres checkpoints and your
original fp32 baseline, so comparisons are apples-to-apples.

Usage
-----
    # Evaluate one checkpoint
    python evaluate_lowres.py \
        --dataset_path /path/to/your_cropped_dataset \
        --ckpt_dir     ./ckpt_192 \
        --category     chewinggum \
        --model_size   S \
        --device       cuda \
        --benchmark

    # Compare against your original 256px checkpoint directly
    python evaluate_lowres.py \
        --dataset_path  /path/to/your_cropped_dataset \
        --ckpt_dir      ./ckpt_192 \
        --compare_ckpt_dir ./ckptSmall \
        --category      chewinggum \
        --model_size    S \
        --device        cuda \
        --benchmark

Notes
-----
- --dataset_path should be your ORIGINAL (256px) dataset, not a pre-resized
  copy -- both this script and retrain_lowres.py resize on the fly to
  whatever the checkpoint's own image_size was trained at, via the
  transform, so you don't need a separate resized validation set.
- If a checkpoint's quantiles.npy doesn't have an 'image_size' key (e.g.
  your original retrain_efficientad.py checkpoints, trained before this
  option existed), this script assumes 256px.
"""

import os
import sys
import argparse
import time
import json
import datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from utils.models import Teacher, Student
from utils.data_loader import get_AD_dataset
from models_adaptive import AutoEncoderAdaptive, probe_output_size

OUT_CHANNELS = 384


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_path',      required=True)
    p.add_argument('--ckpt_dir',          required=True,
                   help='Checkpoint dir with best_teacher.pth, '
                        '{category}_student.pth, {category}_autoencoder.pth, '
                        '{category}_quantiles.npy')
    p.add_argument('--compare_ckpt_dir',  default=None,
                   help='Optional second checkpoint dir to compare against '
                        '(e.g. your original 256px baseline).')
    p.add_argument('--category',          default='chewinggum')
    p.add_argument('--model_size',        default='S', choices=['S', 'M'])
    p.add_argument('--device',            default='cuda')
    p.add_argument('--benchmark',         action='store_true',
                   help='Also measure inference latency (mean/std ms, throughput).')
    p.add_argument('--benchmark_iters',   type=int, default=200)
    p.add_argument('--benchmark_warmup',  type=int, default=20)
    p.add_argument('--save_csv',          default=None,
                   help='Optional path to save a per-image CSV of scores/labels/types.')
    p.add_argument('--log_dir',           default='./eval_logs',
                   help='Directory to write evaluation logs to: a full text transcript '
                        'of this run, and an append-only eval_log.jsonl with one '
                        'structured record per run for tracking results over time. '
                        'Set to "" to disable file logging (console output only).')
    p.add_argument('--run_name',          default=None,
                   help='Optional tag to identify this run in the logs '
                        '(defaults to the ckpt_dir folder name).')
    return p.parse_args()


class Tee:
    """Duplicates everything written to stdout into a log file as well,
    so the full console transcript of a run is preserved on disk."""

    def __init__(self, filepath):
        self.file = open(filepath, 'a')
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def load_bundle(ckpt_dir, category, model_size, device):
    """Load Teacher (fixed arch) + Student + AutoEncoderAdaptive (sized to
    match whatever image_size this checkpoint was trained at) + quantiles."""
    teacher_path   = os.path.join(ckpt_dir, 'best_teacher.pth')
    student_path   = os.path.join(ckpt_dir, f'{category}_student.pth')
    ae_path        = os.path.join(ckpt_dir, f'{category}_autoencoder.pth')
    quantiles_path = os.path.join(ckpt_dir, f'{category}_quantiles.npy')

    for name, path in [('teacher', teacher_path), ('student', student_path),
                        ('autoencoder', ae_path), ('quantiles', quantiles_path)]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Missing {name} checkpoint: {path}')

    quantiles = np.load(quantiles_path, allow_pickle=True).item()
    image_size = int(quantiles.get('image_size', 256))

    teacher = Teacher(model_size)
    teacher.load_state_dict(torch.load(teacher_path, map_location=device, weights_only=False))
    teacher.eval().to(device)
    for p in teacher.parameters():
        p.requires_grad = False

    pdn_out_size = probe_output_size(teacher, image_size, device)

    student = Student(model_size)
    student.load_state_dict(torch.load(student_path, map_location=device, weights_only=False))
    student.eval().to(device)

    ae = AutoEncoderAdaptive(target_size=pdn_out_size)
    ae.load_state_dict(torch.load(ae_path, map_location=device, weights_only=False))
    ae.eval().to(device)

    mean = torch.tensor(np.array(quantiles['mean']), dtype=torch.float32, device=device)
    std  = torch.tensor(np.array(quantiles['std']),  dtype=torch.float32, device=device)

    return {
        'teacher': teacher, 'student': student, 'ae': ae,
        'mean': mean, 'std': std,
        'qa_st': quantiles['qa_st'], 'qb_st': quantiles['qb_st'],
        'qa_ae': quantiles['qa_ae'], 'qb_ae': quantiles['qb_ae'],
        'image_size': image_size,
    }


@torch.no_grad()
def score_dataset(bundle, dataloader, device):
    """Run the full pipeline over a dataloader, returning per-sample scores,
    labels, and defect types (for the per-type breakdown)."""
    teacher, student, ae = bundle['teacher'], bundle['student'], bundle['ae']
    mean, std = bundle['mean'], bundle['std']
    qa_st, qb_st, qa_ae, qb_ae = bundle['qa_st'], bundle['qb_st'], bundle['qa_ae'], bundle['qb_ae']
    image_size = bundle['image_size']

    scores, labels, types = [], [], []

    for sample in dataloader:
        img   = sample['image'].to(device)
        label = sample['label'].item()
        dtype = sample['type'][0] if isinstance(sample['type'], (list, tuple)) else sample['type']

        t_out = teacher(img)
        s_out = student(img)
        a_out = ae(img)

        y_st   = s_out[:, :OUT_CHANNELS,  :, :]
        y_stae = s_out[:, -OUT_CHANNELS:, :, :]
        norm_t = (t_out - mean) / (std + 1e-8)
        d_st   = torch.pow(norm_t - y_st,   2)
        d_stae = torch.pow(a_out  - y_stae, 2)

        fm_st   = F.interpolate(torch.mean(d_st,   dim=1, keepdim=True),
                                size=(image_size, image_size), mode='bilinear', align_corners=False)
        fm_stae = F.interpolate(torch.mean(d_stae, dim=1, keepdim=True),
                                size=(image_size, image_size), mode='bilinear', align_corners=False)

        norm_mst = (0.1 * (fm_st   - qa_st)) / (qb_st - qa_st + 1e-8)
        norm_mae = (0.1 * (fm_stae - qa_ae)) / (qb_ae - qa_ae + 1e-8)
        combined = 0.5 * norm_mst + 0.5 * norm_mae

        scores.append(float(combined.max().cpu()))
        labels.append(label)
        types.append(dtype)

    return np.array(scores), np.array(labels), np.array(types)


def compute_auroc_report(scores, labels, types):
    report = {}
    if labels.sum() == 0 or labels.sum() == len(labels):
        report['overall'] = 0.0
    else:
        report['overall'] = float(roc_auc_score(labels, scores))

    normal_mask     = labels == 0
    normal_scores   = scores[normal_mask]
    per_type        = {}
    for defect_type in sorted(set(types[labels == 1])):
        type_mask = (labels == 1) & (types == defect_type)
        y  = np.concatenate([np.zeros(normal_mask.sum()), np.ones(type_mask.sum())])
        sc = np.concatenate([normal_scores, scores[type_mask]])
        n_pos = type_mask.sum()
        if n_pos == 0 or normal_mask.sum() == 0:
            per_type[defect_type] = (float('nan'), int(n_pos))
            continue
        auc = float(roc_auc_score(y, sc))
        per_type[defect_type] = (auc, int(n_pos))
    report['per_type'] = per_type
    return report


def build_result_record(name, ckpt_dir, image_size, report, bench_stats=None):
    """Structured, JSON-serializable summary of one checkpoint's evaluation
    -- this is what gets appended to eval_log.jsonl."""
    per_type_clean = {
        dtype: {'auroc': (auc if auc == auc else None), 'n': n}
        for dtype, (auc, n) in report['per_type'].items()
    }
    record = {
        'name': name,
        'ckpt_dir': os.path.abspath(ckpt_dir),
        'image_size': image_size,
        'overall_auroc': report['overall'],
        'per_type_auroc': per_type_clean,
    }
    if bench_stats:
        record['latency'] = {
            'mean_ms': bench_stats['mean_ms'],
            'std_ms':  bench_stats['std_ms'],
            'p50_ms':  bench_stats['p50_ms'],
            'p95_ms':  bench_stats['p95_ms'],
            'fps':     bench_stats['fps'],
            'iters':   bench_stats['n'],
        }
    return record


def append_jsonl(log_dir, record):
    path = os.path.join(log_dir, 'eval_log.jsonl')
    with open(path, 'a') as f:
        f.write(json.dumps(record) + '\n')
    return path


def print_report(name, report, image_size):
    print(f'\n=== {name} (image_size={image_size}) ===')
    print(f'  Overall AUROC : {report["overall"]:.4f}')
    if report['per_type']:
        print(f'  Per-defect-type AUROC (normal vs. that type only):')
        for dtype, (auc, n) in sorted(report['per_type'].items(),
                                       key=lambda kv: kv[1][0] if kv[1][0] == kv[1][0] else -1):
            auc_str = f'{auc:.4f}' if auc == auc else 'n/a (0 samples)'
            print(f'    {dtype:<20s} AUROC={auc_str:<18s} n={n}')


@torch.no_grad()
def benchmark_latency(bundle, device, iters, warmup):
    teacher, student, ae = bundle['teacher'], bundle['student'], bundle['ae']
    mean, std = bundle['mean'], bundle['std']
    image_size = bundle['image_size']
    dummy = torch.randn(1, 3, image_size, image_size, device=device)

    def run_once():
        t_out = teacher(dummy)
        s_out = student(dummy)
        a_out = ae(dummy)
        norm_t = (t_out - mean) / (std + 1e-8)
        y_st   = s_out[:, :OUT_CHANNELS,  :, :]
        y_stae = s_out[:, -OUT_CHANNELS:, :, :]
        _ = torch.pow(norm_t - y_st,   2)
        _ = torch.pow(a_out  - y_stae, 2)

    for _ in range(warmup):
        run_once()
    if device == 'cuda' or (hasattr(device, 'type') and device.type == 'cuda'):
        torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        run_once()
        if device == 'cuda' or (hasattr(device, 'type') and device.type == 'cuda'):
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)

    times = np.array(times)
    return {
        'mean_ms': float(times.mean()),
        'std_ms':  float(times.std()),
        'p50_ms':  float(np.percentile(times, 50)),
        'p95_ms':  float(np.percentile(times, 95)),
        'fps':     float(1000.0 / times.mean()),
    }


def print_latency(name, stats):
    print(f'\n=== {name} latency (batch=1, {stats["n"]} iters) ===')
    print(f'  mean: {stats["mean_ms"]:.2f} ms   std: {stats["std_ms"]:.2f} ms')
    print(f'  p50 : {stats["p50_ms"]:.2f} ms   p95: {stats["p95_ms"]:.2f} ms')
    print(f'  throughput: {stats["fps"]:.1f} images/sec')


def _run(args):
    device = args.device

    run_name = args.run_name or os.path.basename(os.path.normpath(args.ckpt_dir))
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    tee = None
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        transcript_path = os.path.join(args.log_dir, f'{run_name}_{timestamp}.log')
        tee = Tee(transcript_path)
        sys.stdout = tee
        print(f'[Eval] Logging full transcript to: {transcript_path}')
        print(f'[Eval] Run: {run_name}   Started: {timestamp}')
        print(f'[Eval] Args: {vars(args)}')

    try:
        print(f'[Eval] Loading checkpoint: {args.ckpt_dir}')
        bundle = load_bundle(args.ckpt_dir, args.category, args.model_size, device)
        print(f'[Eval]   image_size = {bundle["image_size"]}')

        # Build the validation dataset resized to THIS checkpoint's image_size.
        val_tf = transforms.Compose([
            transforms.Resize((bundle['image_size'], bundle['image_size'])),
            transforms.ToTensor(),
        ])
        val_ds = get_AD_dataset(type='VisA', root=args.dataset_path, transform=val_tf,
                                 gt_transform=val_tf, phase='validation', category=args.category)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)
        print(f'[Eval] Validation set: {len(val_ds)} images')

        scores, labels, types = score_dataset(bundle, val_loader, device)
        report = compute_auroc_report(scores, labels, types)
        print_report(os.path.basename(os.path.normpath(args.ckpt_dir)), report, bundle['image_size'])

        if args.save_csv:
            import csv
            with open(args.save_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['score', 'label', 'type'])
                for s, l, t in zip(scores, labels, types):
                    writer.writerow([s, l, t])
            print(f'[Eval] Saved per-image scores to {args.save_csv}')

        bench_main = None
        if args.benchmark:
            bench_main = benchmark_latency(bundle, device, args.benchmark_iters, args.benchmark_warmup)
            bench_main['n'] = args.benchmark_iters
            print_latency(os.path.basename(os.path.normpath(args.ckpt_dir)), bench_main)

        if args.log_dir:
            record = build_result_record(run_name, args.ckpt_dir, bundle['image_size'],
                                         report, bench_main)
            record['timestamp'] = timestamp
            record['category']  = args.category
            record['model_size'] = args.model_size
            jsonl_path = append_jsonl(args.log_dir, record)
            print(f'[Eval] Appended result to {jsonl_path}')

        # ── optional comparison ──────────────────────────────────────────
        if args.compare_ckpt_dir:
            print(f'\n[Eval] Loading comparison checkpoint: {args.compare_ckpt_dir}')
            bundle2 = load_bundle(args.compare_ckpt_dir, args.category, args.model_size, device)
            print(f'[Eval]   image_size = {bundle2["image_size"]}')

            val_tf2 = transforms.Compose([
                transforms.Resize((bundle2['image_size'], bundle2['image_size'])),
                transforms.ToTensor(),
            ])
            val_ds2 = get_AD_dataset(type='VisA', root=args.dataset_path, transform=val_tf2,
                                      gt_transform=val_tf2, phase='validation', category=args.category)
            val_loader2 = DataLoader(val_ds2, batch_size=1, shuffle=False, num_workers=4)

            scores2, labels2, types2 = score_dataset(bundle2, val_loader2, device)
            report2 = compute_auroc_report(scores2, labels2, types2)
            print_report(os.path.basename(os.path.normpath(args.compare_ckpt_dir)), report2, bundle2['image_size'])

            bench_cmp = None
            if args.benchmark:
                bench_cmp = benchmark_latency(bundle2, device, args.benchmark_iters, args.benchmark_warmup)
                bench_cmp['n'] = args.benchmark_iters
                print_latency(os.path.basename(os.path.normpath(args.compare_ckpt_dir)), bench_cmp)

            if args.log_dir:
                cmp_name = os.path.basename(os.path.normpath(args.compare_ckpt_dir))
                cmp_record = build_result_record(cmp_name, args.compare_ckpt_dir,
                                                 bundle2['image_size'], report2, bench_cmp)
                cmp_record['timestamp']  = timestamp
                cmp_record['category']   = args.category
                cmp_record['model_size'] = args.model_size
                cmp_record['compared_against'] = run_name
                append_jsonl(args.log_dir, cmp_record)

            print(f'\n=== Summary: {os.path.basename(os.path.normpath(args.ckpt_dir))} '
                  f'({bundle["image_size"]}px) vs. '
                  f'{os.path.basename(os.path.normpath(args.compare_ckpt_dir))} '
                  f'({bundle2["image_size"]}px) ===')
            print(f'  AUROC        : {report["overall"]:.4f}  vs.  {report2["overall"]:.4f}  '
                  f'(delta = {report["overall"] - report2["overall"]:+.4f})')
            if bench_main and bench_cmp:
                speedup = bench_cmp['mean_ms'] / bench_main['mean_ms']
                print(f'  Latency (ms) : {bench_main["mean_ms"]:.2f}  vs.  {bench_cmp["mean_ms"]:.2f}  '
                      f'({speedup:.2f}x {"faster" if speedup > 1 else "slower"})')

        if tee:
            print(f'\n[Eval] Done. Transcript saved, structured results appended to '
                  f'{os.path.join(args.log_dir, "eval_log.jsonl")}')

    except Exception:
        if tee:
            import traceback
            print('\n[Eval] FAILED with an exception:')
            traceback.print_exc(file=tee)
        raise

    finally:
        # Always restore stdout and close the transcript file, even on error,
        # so a crash doesn't leave the terminal writing into a closed file.
        if tee:
            sys.stdout = tee.stdout
            tee.close()


def main():
    args = parse_args()
    _run(args)


if __name__ == '__main__':
    main()