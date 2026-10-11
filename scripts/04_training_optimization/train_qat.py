"""
train_qat.py
============
OPTION 3 — Quantization-Aware Training (int8) for Student + AutoEncoder.

Only Student and AutoEncoder are quantized. The Teacher is only ever used
at TRAINING time to produce the target feature maps (channel_mean/std and
the per-pixel loss target) — at inference/deployment time you only need
Student + AutoEncoder + the saved quantile/mean/std stats, so it's the
Student/AE pair that benefits from int8 speedups on CPU (via fbgemm) or
ARM (via qnnpack).

How this works
---------------
1. Start from an already-trained fp32 checkpoint (student.pth / autoencoder.pth
   from retrain_efficientad.py or retrain_lowres.py) — QAT fine-tunes an
   existing model, it does not train from scratch.
2. Wrap Student and AutoEncoder with QuantStub/DeQuantStub.
3. Attach a QAT qconfig (`get_default_qat_qconfig`) and call `prepare_qat`,
   which inserts fake-quantize ops so training sees quantization noise.
4. Fine-tune for a modest number of iterations (much shorter than the
   original training run — this is calibration + light recovery, not
   training from scratch) using the *same* losses as the original script,
   with the Teacher still frozen and in fp32 (it's not part of the
   quantized graph).
5. Convert to a true int8 model with `torch.ao.quantization.convert` and
   save it — this is the artifact you deploy.

Caveats
-------
- This wraps modules with QuantStub/DeQuantStub but does NOT fuse
  Conv+ReLU pairs via `torch.ao.quantization.fuse_modules`, because the
  PDN/EncConv/DecConv blocks call `F.relu` functionally rather than using
  `nn.ReLU` submodules. Fusion typically buys extra speedup on top of plain
  int8 — if you want it, refactor the conv+relu calls in models.py into
  `nn.Sequential(conv, nn.ReLU())` pairs first, then add a fuse_modules step
  here before `prepare_qat`.
- fbgemm targets x86 server/desktop CPUs; use qnnpack for ARM/mobile
  deployment (--backend qnnpack).
- Expect a modest accuracy drop from int8 rounding — this script reports
  AUROC before and after quantization so you can see exactly how much.
- After `convert`, the model runs on CPU (`--device cpu` at inference) —
  int8 kernels in PyTorch eager mode are CPU-only.

Usage
-----
    python train_qat.py \
        --dataset_path   /path/to/cropped_dataset \
        --ckpt_dir       ./ckptSmall \
        --out_dir        ./ckpt_qat \
        --teacher_path   ./ckptSmall/best_teacher.pth \
        --category       chewinggum \
        --model_size     S \
        --backend        fbgemm \
        --qat_iterations 3000
"""

import os
import sys
import copy
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from utils.models_fixed import Teacher, Student, AutoEncoder, imagenet_norm_batch
from utils.data_loader import get_AD_dataset, load_infinite

IMAGE_SIZE   = 256
OUT_CHANNELS = 384

torch.backends.cudnn.benchmark = True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_path',   required=True)
    p.add_argument('--ckpt_dir',       required=True,
                   help='Directory with an already fp32-trained '
                        '{category}_student.pth / {category}_autoencoder.pth '
                        '/ {category}_quantiles.npy')
    p.add_argument('--out_dir',        required=True,
                   help='Where to save the QAT-fine-tuned + int8-converted checkpoints')
    p.add_argument('--teacher_path',   required=True)
    p.add_argument('--category',       default='chewinggum')
    p.add_argument('--model_size',     default='S', choices=['S', 'M'])
    p.add_argument('--backend',        default='fbgemm', choices=['fbgemm', 'qnnpack'],
                   help='fbgemm = x86 CPU, qnnpack = ARM/mobile CPU')
    p.add_argument('--qat_iterations', type=int, default=3000,
                   help='Short fine-tune to let the network adapt to quantization noise. '
                        'Much shorter than the original 70k-iteration training run.')
    p.add_argument('--batch_size',     type=int, default=1)
    p.add_argument('--lr',             type=float, default=1e-5,
                   help='Small LR — this is fine-tuning, not training from scratch.')
    p.add_argument('--print_freq',     type=int, default=200)
    p.add_argument('--device',         default='cuda',
                   help='Device for the QAT fine-tune loop (fake-quant ops run fine on GPU). '
                        'The final converted int8 model always runs on CPU.')
    p.add_argument('--seed',           type=int, default=42)
    return p.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class QuantStudentWrapper(nn.Module):
    """Wraps a Student's inner PDN for QAT, quantizing only the conv-heavy
    part -- NOT the imagenet_norm_batch preprocessing.

    Student.forward() normally does `x = imagenet_norm_batch(x); return
    self.pdn(x)`. If we naively wrap the whole Student with QuantStub, the
    QuantStub converts x to an int8 tensor BEFORE imagenet_norm_batch runs,
    and imagenet_norm_batch's floating-point subtract/divide then crashes
    on a quantized tensor ("promoteTypes with quantized numbers is not
    handled"). This wrapper instead does the fp32 normalization first, then
    quantizes, then calls only student.pdn directly (skipping Student's own
    forward, which would otherwise re-run imagenet_norm_batch a second time
    on an already-quantized tensor).
    """

    def __init__(self, student: Student):
        super().__init__()
        self.quant   = torch.ao.quantization.QuantStub()
        self.pdn     = student.pdn   # reuse the same submodule/parameters
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        x = imagenet_norm_batch(x)   # fp32 -- must happen BEFORE quantization
        x = self.quant(x)
        x = self.pdn(x)
        x = self.dequant(x)
        return x


class QuantAEWrapper(nn.Module):
    """Same fix as QuantStudentWrapper, for AutoEncoder (encoder + decoder)."""

    def __init__(self, ae: AutoEncoder):
        super().__init__()
        self.quant   = torch.ao.quantization.QuantStub()
        self.encoder = ae.encoder
        self.decoder = ae.decoder
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        x = imagenet_norm_batch(x)   # fp32 -- must happen BEFORE quantization
        x = self.quant(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.dequant(x)
        return x


def load_fp32_checkpoint(ckpt_dir, category, model_size, device):
    student = Student(model_size)
    ae      = AutoEncoder()
    student.load_state_dict(torch.load(
        os.path.join(ckpt_dir, f'{category}_student.pth'),
        map_location=device, weights_only=False))
    ae.load_state_dict(torch.load(
        os.path.join(ckpt_dir, f'{category}_autoencoder.pth'),
        map_location=device, weights_only=False))
    quantiles = np.load(
        os.path.join(ckpt_dir, f'{category}_quantiles.npy'),
        allow_pickle=True).item()
    return student, ae, quantiles


def compute_quantiles(teacher, student, ae, dataloader, mean, std, device, desc='Quantiles'):
    teacher.eval(); student.eval(); ae.eval()
    xst, xae = [], []
    for sample in tqdm(dataloader, desc=f'  {desc}', leave=False):
        img = sample['image'].to(device)
        with torch.no_grad():
            t_out = teacher(img)
            s_out = student(img)
            a_out = ae(img)

        y_st   = s_out[:, :OUT_CHANNELS,  :, :]
        y_stae = s_out[:, -OUT_CHANNELS:, :, :]
        norm_t = (t_out - mean) / (std + 1e-8)
        d_st   = torch.pow(norm_t - y_st,   2)
        d_stae = torch.pow(a_out  - y_stae, 2)

        fm_st   = F.interpolate(torch.mean(d_st,   dim=1, keepdim=True),
                                size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)
        fm_stae = F.interpolate(torch.mean(d_stae, dim=1, keepdim=True),
                                size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)
        xst.append(fm_st.detach().cpu().numpy())
        xae.append(fm_stae.detach().cpu().numpy())

    qa_st = float(np.percentile(np.concatenate(xst), 90))
    qb_st = float(np.percentile(np.concatenate(xst), 99.5))
    qa_ae = float(np.percentile(np.concatenate(xae), 90))
    qb_ae = float(np.percentile(np.concatenate(xae), 99.5))
    return qa_st, qb_st, qa_ae, qb_ae


def eval_auroc(teacher, student, ae, dataloader, mean, std, qa_st, qb_st, qa_ae, qb_ae, device, desc='AUROC'):
    teacher.eval(); student.eval(); ae.eval()
    scores, labels = [], []
    for sample in tqdm(dataloader, desc=f'  {desc}', leave=False):
        img   = sample['image'].to(device)
        label = sample['label'].item()
        with torch.no_grad():
            t_out = teacher(img)
            s_out = student(img)
            a_out = ae(img)

        y_st   = s_out[:, :OUT_CHANNELS,  :, :]
        y_stae = s_out[:, -OUT_CHANNELS:, :, :]
        norm_t = (t_out - mean) / (std + 1e-8)
        d_st   = torch.pow(norm_t - y_st,   2)
        d_stae = torch.pow(a_out  - y_stae, 2)

        fm_st   = F.interpolate(torch.mean(d_st,   dim=1, keepdim=True),
                                size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)
        fm_stae = F.interpolate(torch.mean(d_stae, dim=1, keepdim=True),
                                size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)

        norm_mst = (0.1 * (fm_st   - qa_st)) / (qb_st - qa_st + 1e-8)
        norm_mae = (0.1 * (fm_stae - qa_ae)) / (qb_ae - qa_ae + 1e-8)
        combined = 0.5 * norm_mst + 0.5 * norm_mae
        scores.append(float(combined.max().cpu()))
        labels.append(label)

    labels = np.array(labels)
    scores = np.array(scores)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.0
    return float(roc_auc_score(labels, scores))


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device

    # ── data ──────────────────────────────────────────────────────────────
    data_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    train_ds = get_AD_dataset(type='VisA', root=args.dataset_path, transform=data_tf,
                               gt_transform=data_tf, phase='train', category=args.category)
    test_ds  = get_AD_dataset(type='VisA', root=args.dataset_path, transform=data_tf,
                               gt_transform=data_tf, phase='test', category=args.category)
    val_ds   = get_AD_dataset(type='VisA', root=args.dataset_path, transform=data_tf,
                               gt_transform=data_tf, phase='validation', category=args.category)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)
    val_loader   = DataLoader(val_ds,  batch_size=1, shuffle=False, num_workers=4)
    train_iter   = load_infinite(train_loader)

    # ── models ────────────────────────────────────────────────────────────
    print(f'[QAT] Loading frozen teacher: {args.teacher_path}')
    teacher = Teacher(args.model_size)
    teacher.load_state_dict(torch.load(args.teacher_path, map_location=device, weights_only=False))
    teacher.eval().to(device)
    for p in teacher.parameters():
        p.requires_grad = False

    print(f'[QAT] Loading fp32 student/autoencoder from {args.ckpt_dir}')
    student_fp32, ae_fp32, quantiles = load_fp32_checkpoint(
        args.ckpt_dir, args.category, args.model_size, device)

    mean = torch.tensor(np.array(quantiles['mean']), dtype=torch.float32, device=device)
    std  = torch.tensor(np.array(quantiles['std']),  dtype=torch.float32, device=device)

    # Baseline AUROC before touching anything, for comparison.
    qa_st, qb_st, qa_ae, qb_ae = quantiles['qa_st'], quantiles['qb_st'], quantiles['qa_ae'], quantiles['qb_ae']
    student_fp32.to(device); ae_fp32.to(device)
    baseline_auroc = eval_auroc(teacher, student_fp32, ae_fp32, val_loader,
                                mean, std, qa_st, qb_st, qa_ae, qb_ae, device,
                                desc='Baseline fp32 AUROC (GPU)')
    print(f'[QAT] Baseline fp32 AUROC(val) = {baseline_auroc:.4f}')

    # ── wrap + prepare for QAT ───────────────────────────────────────────
    torch.backends.quantized.engine = args.backend
    qconfig = torch.ao.quantization.get_default_qat_qconfig(args.backend)

    student_q = QuantStudentWrapper(copy.deepcopy(student_fp32)).to(device)
    ae_q      = QuantAEWrapper(copy.deepcopy(ae_fp32)).to(device)
    student_q.qconfig = qconfig
    ae_q.qconfig      = qconfig

    student_q.train()
    ae_q.train()
    torch.ao.quantization.prepare_qat(student_q, inplace=True)
    torch.ao.quantization.prepare_qat(ae_q, inplace=True)

    optimizer = optim.Adam(list(student_q.parameters()) + list(ae_q.parameters()),
                            lr=args.lr)

    print(f'\n[QAT] Fine-tuning with fake-quant for {args.qat_iterations} iterations '
          f'(backend={args.backend}) ...\n')

    for i in range(args.qat_iterations):
        student_q.train(); ae_q.train()
        sample = next(train_iter)
        image  = sample['image'].to(device)

        with torch.no_grad():
            t_out  = teacher(image)
            norm_t = (t_out - mean) / (std + 1e-8)

        s_out = student_q(image)
        y_st  = s_out[:, :OUT_CHANNELS, :, :]
        dist  = torch.pow(norm_t - y_st, 2)
        dhard = torch.quantile(dist[:8], 0.999)
        L_st  = torch.mean(dist[dist >= dhard])

        aug_idx = random.choice([1, 2, 3])
        coeff   = random.uniform(0.8, 1.2)
        if aug_idx == 1:
            aug = transforms.functional.adjust_brightness(image, coeff)
        elif aug_idx == 2:
            aug = transforms.functional.adjust_contrast(image, coeff)
        else:
            aug = transforms.functional.adjust_saturation(image, coeff)

        with torch.no_grad():
            t_aug      = teacher(aug)
            norm_t_aug = (t_aug - mean) / (std + 1e-8)

        ae_out = ae_q(aug)
        s_aug  = student_q(aug)
        y_stae = s_aug[:, -OUT_CHANNELS:, :, :]

        L_ae   = torch.mean(torch.pow(norm_t_aug - ae_out,  2))
        L_stae = torch.mean(torch.pow(ae_out     - y_stae, 2))

        loss = L_st + L_ae + L_stae
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            print(f'  iter {i:6d}/{args.qat_iterations}  loss={loss.item():.4f}  '
                  f'L_st={L_st.item():.4f}  L_ae={L_ae.item():.4f}')

    # ── convert to true int8 ─────────────────────────────────────────────
    print('\n[QAT] Converting to int8 (CPU) ...')
    student_q.eval().to('cpu')
    ae_q.eval().to('cpu')
    student_int8 = torch.ao.quantization.convert(student_q, inplace=False)
    ae_int8      = torch.ao.quantization.convert(ae_q, inplace=False)
    print('[QAT] Conversion done. Re-calibrating on CPU -- this runs the full test '
          f'({len(test_loader.dataset)} imgs) and validation '
          f'({len(val_loader.dataset)} imgs) splits sequentially on CPU (no GPU for '
          f'int8), which is meaningfully slower than the GPU training above. This can '
          f'take several minutes with a full-width fp32 Teacher in the loop -- the '
          f'progress bars below are the only feedback you will see, please let them finish.')

    # ── re-calibrate quantile thresholds + re-check AUROC on CPU/int8 ───
    teacher_cpu = copy.deepcopy(teacher).to('cpu').eval()
    mean_cpu, std_cpu = mean.cpu(), std.cpu()
    qa_st2, qb_st2, qa_ae2, qb_ae2 = compute_quantiles(
        teacher_cpu, student_int8, ae_int8, test_loader, mean_cpu, std_cpu, 'cpu',
        desc='CPU quantiles (test split)')
    int8_auroc = eval_auroc(teacher_cpu, student_int8, ae_int8, val_loader,
                            mean_cpu, std_cpu, qa_st2, qb_st2, qa_ae2, qb_ae2, 'cpu',
                            desc='CPU AUROC (val split)')

    print(f'\n[QAT] Baseline fp32 AUROC(val) : {baseline_auroc:.4f}')
    print(f'[QAT] int8 AUROC(val)          : {int8_auroc:.4f}  '
          f'(delta = {int8_auroc - baseline_auroc:+.4f})')

    # ── save ──────────────────────────────────────────────────────────────
    torch.save(student_int8.state_dict(),
               os.path.join(args.out_dir, f'{args.category}_student_int8.pth'))
    torch.save(ae_int8.state_dict(),
               os.path.join(args.out_dir, f'{args.category}_autoencoder_int8.pth'))
    # Also save via torch.jit.script for a fully self-contained deployable artifact.
    try:
        scripted_student = torch.jit.script(student_int8)
        scripted_ae      = torch.jit.script(ae_int8)
        scripted_student.save(os.path.join(args.out_dir, f'{args.category}_student_int8.jit'))
        scripted_ae.save(os.path.join(args.out_dir, f'{args.category}_autoencoder_int8.jit'))
        print('[QAT] Saved TorchScript int8 artifacts (.jit) for deployment.')
    except Exception as e:
        print(f'[QAT] WARNING: torch.jit.script failed ({e}); '
              f'state_dicts are still saved, but you will need the '
              f'QuantStudentWrapper/QuantAEWrapper classes + prepare/convert '
              f'steps to reload them.')

    np.save(os.path.join(args.out_dir, f'{args.category}_quantiles.npy'), {
        'qa_st': qa_st2, 'qb_st': qb_st2, 'qa_ae': qa_ae2, 'qb_ae': qb_ae2,
        'mean': mean_cpu.numpy(), 'std': std_cpu.numpy(),
        'backend': args.backend,
    })
    print(f'\n[QAT] Done. Artifacts in {args.out_dir}/ — run on CPU with '
          f'torch.backends.quantized.engine = "{args.backend}".')


if __name__ == '__main__':
    main()