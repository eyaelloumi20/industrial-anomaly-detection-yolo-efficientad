"""
benchmark_efficientad.py
==========================
Isolated EfficientAD (Teacher+Student+AE forward pass only, no YOLO, no
dataset) latency benchmark that works across ALL FOUR checkpoint types in
this project. Every evaluate_*_pipeline.py script reports full-pipeline
"Inf. time" (YOLO detection + cropping + denoising + EfficientAD combined),
which is dominated by YOLO and dilutes whatever the EfficientAD side saved.
evaluate_lowres.py's --benchmark flag is the only isolated-model-only
number that exists so far, and only for lowres checkpoints.

This script generalizes that to baseline/lowres/thin/qat/pruned, using the
correct loader for each, so you can get a real apples-to-apples comparison
of what each optimization actually did to the EfficientAD forward pass by
itself -- separate from whatever YOLO/pipeline overhead is masking it in
the full-pipeline numbers.

Reports per-stage breakdown (Teacher / Student / AE separately) as well as
combined total, since the combined number is what actually matters for
deployment but the per-stage breakdown is what tells you WHERE the time
goes -- e.g. confirming the Teacher-not-optimized theory directly, instead
of inferring it.

Usage
-----
    # Baseline (original 256px fp32)
    python benchmark_efficientad.py --method baseline --ckpt_dir ./ckpt --device cuda

    # Lowres (192px)
    python benchmark_efficientad.py --method lowres --ckpt_dir ./ckpt_192 --device cuda

    # Thin (width_mult=0.25)
    python benchmark_efficientad.py --method thin --ckpt_dir ./ckpt_thin_025 --device cuda

    # QAT (int8, CPU forced automatically)
    python benchmark_efficientad.py --method qat --ckpt_dir ./ckpt_qat

    # Pruned
    python benchmark_efficientad.py --method pruned --ckpt_dir ./ckpt_pruned_030 --device cuda

    # Run all five back to back and get one comparison table (skips any
    # --ckpt_dir that doesn't exist rather than erroring out)
    python benchmark_efficientad.py --compare_all \
        --baseline_dir ./ckpt --lowres_dir ./ckpt_192 --thin_dir ./ckpt_thin_025 \
        --qat_dir ./ckpt_qat --pruned_dir ./ckpt_pruned_030
"""

import os
import sys
import time
import argparse
import json
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

OUT_CHANNELS = 384


def load_bundle(method: str, ckpt_dir: str, category: str, model_size: str, device: str):
    """Dispatch to the correct loader for this checkpoint type. Returns
    (models_dict, image_size, device_actually_used)."""
    if method in ('baseline', 'lowres'):
        from utils.load_efficientad_adaptive import load_efficientad_model
        models = load_efficientad_model(ckpt_dir, category, model_size, device)
        return models, models.get('image_size', 256), device

    elif method == 'thin':
        from utils.load_efficientad_thin import load_efficientad_model
        models = load_efficientad_model(ckpt_dir, category, model_size, device)
        return models, 256, device   # thin always trains at original resolution

    elif method == 'qat':
        from utils.load_efficientad_qat import load_efficientad_model
        models = load_efficientad_model(ckpt_dir, category, model_size, device)
        return models, 256, 'cpu'    # qat loader forces cpu regardless of request

    elif method == 'pruned':
        from utils.load_efficientad_pruned import load_efficientad_model
        models = load_efficientad_model(ckpt_dir, category, model_size, device)
        return models, 256, device

    else:
        raise ValueError(f"Unknown method '{method}'. "
                         f"Choose from: baseline, lowres, thin, qat, pruned")


@torch.no_grad()
def benchmark(models: dict, image_size: int, device: str, iters: int, warmup: int) -> dict:
    """Times Teacher, Student, AE separately AND the full combined pipeline
    (forward passes + the anomaly-map math), matching what
    evaluate_lowres.py's benchmark_latency measures for the combined case,
    plus a per-stage breakdown that's new here."""
    teacher, student, ae = models['teacher'], models['student'], models['ae']
    mean, std = models['mean'], models['std']

    is_cuda = (device == 'cuda') or (hasattr(device, 'type') and device.type == 'cuda')
    dummy = torch.randn(1, 3, image_size, image_size, device=device)

    def sync():
        if is_cuda:
            torch.cuda.synchronize()

    def time_stage(fn, n):
        times = []
        for _ in range(n):
            start = time.perf_counter()
            fn()
            sync()
            times.append((time.perf_counter() - start) * 1000.0)
        return np.array(times)

    # ── warmup ────────────────────────────────────────────────────────────
    for _ in range(warmup):
        t_out = teacher(dummy)
        s_out = student(dummy)
        a_out = ae(dummy)
    sync()

    # ── per-stage timing ─────────────────────────────────────────────────
    teacher_times = time_stage(lambda: teacher(dummy), iters)
    student_times = time_stage(lambda: student(dummy), iters)
    ae_times      = time_stage(lambda: ae(dummy),      iters)

    # ── combined (full forward + anomaly map, matches deployment cost) ────
    import torch.nn.functional as F

    def full_pass():
        t_out = teacher(dummy)
        s_out = student(dummy)
        a_out = ae(dummy)
        y_st   = s_out[:, :OUT_CHANNELS,  :, :]
        y_stae = s_out[:, -OUT_CHANNELS:, :, :]
        norm_t = (t_out - mean) / (std + 1e-8)
        d_st   = torch.pow(norm_t - y_st,   2)
        d_stae = torch.pow(a_out  - y_stae, 2)
        fm_st  = F.interpolate(torch.mean(d_st,   dim=1, keepdim=True),
                               size=(image_size, image_size), mode='bilinear', align_corners=False)
        fm_stae = F.interpolate(torch.mean(d_stae, dim=1, keepdim=True),
                               size=(image_size, image_size), mode='bilinear', align_corners=False)
        _ = torch.max(0.5 * fm_st + 0.5 * fm_stae)

    combined_times = time_stage(full_pass, iters)

    def stats(times):
        return {
            'mean_ms': float(times.mean()), 'std_ms': float(times.std()),
            'p50_ms': float(np.percentile(times, 50)), 'p95_ms': float(np.percentile(times, 95)),
            'fps': float(1000.0 / times.mean()),
        }

    return {
        'teacher':  stats(teacher_times),
        'student':  stats(student_times),
        'ae':       stats(ae_times),
        'combined': stats(combined_times),
        'device': device, 'image_size': image_size, 'iters': iters,
    }


def print_report(name: str, report: dict):
    print(f'\n=== {name}  (image_size={report["image_size"]}, device={report["device"]}, '
          f'{report["iters"]} iters) ===')
    for stage in ('teacher', 'student', 'ae', 'combined'):
        s = report[stage]
        print(f'  {stage:<10s} mean={s["mean_ms"]:7.2f}ms  std={s["std_ms"]:6.2f}ms  '
              f'p50={s["p50_ms"]:7.2f}ms  p95={s["p95_ms"]:7.2f}ms  fps={s["fps"]:7.1f}')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['baseline', 'lowres', 'thin', 'qat', 'pruned'],
                   help='Which loader to use. Required unless --compare_all.')
    p.add_argument('--ckpt_dir')
    p.add_argument('--category',   default='chewinggum')
    p.add_argument('--model_size', default='S', choices=['S', 'M'])
    p.add_argument('--device',     default='cuda',
                   help='Ignored for --method qat (always forced to cpu).')
    p.add_argument('--iters',      type=int, default=200)
    p.add_argument('--warmup',     type=int, default=20)
    p.add_argument('--log_file',   default='./benchmark_log.jsonl',
                   help='Append results as JSON lines for later comparison. Set to "" to disable.')

    p.add_argument('--compare_all', action='store_true',
                   help='Run baseline/lowres/thin/qat/pruned back to back, skipping any '
                        'checkpoint dir that does not exist.')
    p.add_argument('--baseline_dir', default='./ckpt')
    p.add_argument('--lowres_dir',   default='./ckpt_192')
    p.add_argument('--thin_dir',     default='./ckpt_thin_025')
    p.add_argument('--qat_dir',      default='./ckpt_qat')
    p.add_argument('--pruned_dir',   default='./ckpt_pruned_030')
    return p.parse_args()


def run_one(method, ckpt_dir, args):
    print(f'\n[Benchmark] Loading {method} checkpoint: {ckpt_dir}')
    models, image_size, device = load_bundle(
        method, ckpt_dir, args.category, args.model_size, args.device)
    report = benchmark(models, image_size, device, args.iters, args.warmup)
    print_report(method, report)

    if args.log_file:
        record = {'method': method, 'ckpt_dir': os.path.abspath(ckpt_dir), **report}
        with open(args.log_file, 'a') as f:
            f.write(json.dumps(record) + '\n')

    del models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def main():
    args = parse_args()

    if args.compare_all:
        candidates = [
            ('baseline', args.baseline_dir),
            ('lowres',   args.lowres_dir),
            ('thin',     args.thin_dir),
            ('qat',      args.qat_dir),
            ('pruned',   args.pruned_dir),
        ]
        results = {}
        for method, ckpt_dir in candidates:
            if not os.path.isdir(ckpt_dir):
                print(f'\n[Benchmark] Skipping {method}: {ckpt_dir} does not exist.')
                continue
            try:
                results[method] = run_one(method, ckpt_dir, args)
            except Exception as e:
                print(f'\n[Benchmark] {method} FAILED: {e}')

        if results:
            print(f'\n{"="*78}')
            print(f'  Summary — combined (Teacher+Student+AE+anomaly-map) forward pass')
            print(f'{"="*78}')
            print(f'  {"method":<10s} {"device":<6s} {"image":<7s} {"mean_ms":>9s} {"fps":>8s}')
            for method, r in results.items():
                c = r['combined']
                print(f'  {method:<10s} {r["device"]:<6s} {r["image_size"]:<7d} '
                      f'{c["mean_ms"]:>9.2f} {c["fps"]:>8.1f}')
    else:
        if not args.method or not args.ckpt_dir:
            raise SystemExit('--method and --ckpt_dir are required unless --compare_all is set.')
        run_one(args.method, args.ckpt_dir, args)


if __name__ == '__main__':
    main()
