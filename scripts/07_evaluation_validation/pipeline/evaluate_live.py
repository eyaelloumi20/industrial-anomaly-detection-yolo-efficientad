"""
evaluate_live.py
================
Live EfficientAD-only evaluation viewer.

Opens a desktop window and shows each test image one by one with the
anomaly heatmap, image-level score, GT label, prediction, and running
metrics (AUROC, accuracy, TP/TN/FP/FN) all overlaid in the window.

NO YOLO — images are fed directly to EfficientAD at 256×256.

Controls
--------
  SPACE / → / any key  : next image
  ←                    : previous image
  S                     : save current frame as PNG
  Q / ESC               : quit

Usage
-----
    python evaluate_live.py \
        --dataset_path ./cropped_dataset   \
        --ckpt_dir     ./ckpt_paper        \
        --category     chewinggum          \
        --model_size   S                   \
        --device       cuda

    # Slow down to examine each image (wait N ms before auto-advancing):
    python evaluate_live.py ... --auto_advance 2000

    # Start from a specific image index:
    python evaluate_live.py ... --start_idx 10
"""

import os
import sys
import argparse
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from utils.models      import Teacher, Student, AutoEncoder
from utils.data_loader import get_AD_dataset

# ── Constants ─────────────────────────────────────────────────────────────────
IMAGE_SIZE   = 256
OUT_CHANNELS = 384

# ── Window layout sizes ───────────────────────────────────────────────────────
IMG_W, IMG_H  = 256, 256     # left panel: original image
MAP_W, MAP_H  = 256, 256     # centre panel: heatmap overlay
INFO_W        = 420          # right panel: text metrics
PANEL_H       = 700          # total window height (tall enough for all metrics)
PADDING       = 14           # inner padding
TOTAL_W       = IMG_W + MAP_W + INFO_W + PADDING * 4
TOTAL_H       = PANEL_H

# ── Colour palette (BGR) ──────────────────────────────────────────────────────
BG        = (18,  18,  18)
PANEL_BG  = (28,  28,  28)
WHITE     = (240, 240, 240)
GRAY      = (130, 130, 130)
GREEN     = ( 60, 210,  80)
RED       = ( 60,  60, 230)
YELLOW    = ( 30, 200, 220)
CYAN      = (200, 200,  40)
ORANGE    = ( 30, 140, 240)
DIVIDER   = ( 50,  50,  50)

FONT       = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def find_teacher(ckpt_dir, category, teacher_path_hint=None):
    """
    Locate the teacher checkpoint. Search order:
      1. Explicit --teacher_path argument (if given and exists)
      2. <ckpt_dir>/best_teacher.pth
      3. <ckpt_dir>/<category>_teacher.pth  or  <category>_teacher_best.pth
      4. Any sibling directory of ckpt_dir that contains best_teacher.pth
         (e.g. the original ckptSmall / ckpt_original next to ckpt_output22)
      5. Any *teacher*.pth anywhere under the framework root (two levels up)
    """
    candidates = []

    if teacher_path_hint:
        candidates.append(teacher_path_hint)

    candidates += [
        os.path.join(ckpt_dir, 'best_teacher.pth'),
        os.path.join(ckpt_dir, f'{category}_teacher.pth'),
        os.path.join(ckpt_dir, f'{category}_teacher_best.pth'),
    ]

    # Search sibling directories of ckpt_dir
    parent = os.path.dirname(os.path.abspath(ckpt_dir))
    for sibling in os.listdir(parent):
        sib_path = os.path.join(parent, sibling)
        if os.path.isdir(sib_path) and sib_path != os.path.abspath(ckpt_dir):
            candidates.append(os.path.join(sib_path, 'best_teacher.pth'))
            candidates.append(os.path.join(sib_path, f'{category}_teacher.pth'))

    # Broad search two levels up (covers /home/ahmed/framework/ckptSmall etc.)
    root = os.path.dirname(parent)
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if 'teacher' in fn.lower() and fn.endswith('.pth'):
                candidates.append(os.path.join(dirpath, fn))

    for c in candidates:
        if c and os.path.isfile(c):
            return c

    raise FileNotFoundError(
        f'Could not find a teacher checkpoint anywhere near "{ckpt_dir}".\n'
        f'Pass the explicit path with --teacher_path /path/to/best_teacher.pth')


def load_models(ckpt_dir, category, model_size, device, teacher_path_hint=None):
    """Load teacher (frozen), student, autoencoder and quantile stats."""
    teacher = Teacher(model_size)
    student = Student(model_size)
    ae      = AutoEncoder()

    def _load(name, model, explicit_path=None):
        # If an explicit path is given, use it directly
        if explicit_path and os.path.isfile(explicit_path):
            model.load_state_dict(
                torch.load(explicit_path, map_location=device,
                           weights_only=False))
            print(f'  Loaded {name}: {explicit_path}')
            return
        # Otherwise search ckpt_dir with several naming conventions
        for suffix in ['', '_best', '_final', '_last']:
            path = os.path.join(ckpt_dir, f'{category}_{name}{suffix}.pth')
            if os.path.isfile(path):
                model.load_state_dict(
                    torch.load(path, map_location=device, weights_only=False))
                print(f'  Loaded {name}: {path}')
                return
        raise FileNotFoundError(
            f'Cannot find checkpoint for "{name}" in {ckpt_dir}')

    # Teacher — may live in a completely different directory
    teacher_file = find_teacher(ckpt_dir, category, teacher_path_hint)
    print(f'  Found teacher: {teacher_file}')
    _load('teacher', teacher, explicit_path=teacher_file)
    _load('student', student)
    _load('autoencoder', ae)

    teacher.eval().to(device)
    student.eval().to(device)
    ae.eval().to(device)
    for p in teacher.parameters():
        p.requires_grad = False

    # Load quantile stats
    for suffix in ['', '_best', '_final']:
        q_path = os.path.join(ckpt_dir, f'{category}_quantiles{suffix}.npy')
        if os.path.isfile(q_path):
            q = np.load(q_path, allow_pickle=True).item()
            print(f'  Loaded quantiles: {q_path}')
            break
    else:
        raise FileNotFoundError(
            f'Cannot find quantiles .npy in {ckpt_dir}')

    mu    = torch.tensor(q['mean'], dtype=torch.float32).to(device)
    sigma = torch.tensor(q['std'],  dtype=torch.float32).to(device)
    return teacher, student, ae, mu, sigma, q


# ─────────────────────────────────────────────────────────────────────────────
# Inference — Algorithm 2 from the paper
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer(image_tensor, teacher, student, ae, mu, sigma, q):
    """
    Run one image through EfficientAD (Algorithm 2).
    Returns (anomaly_map_np [H,W], image_score float).
    """
    t_out  = teacher(image_tensor)
    s_out  = student(image_tensor)
    a_out  = ae(image_tensor)

    norm_t = (t_out - mu) / (sigma + 1e-8)
    y_st   = s_out[:, :OUT_CHANNELS,  :, :]
    y_stae = s_out[:, -OUT_CHANNELS:, :, :]

    d_st   = torch.pow(norm_t - y_st,   2)
    d_stae = torch.pow(a_out  - y_stae, 2)

    m_st   = d_st.mean(dim=1, keepdim=True)
    m_ae   = d_stae.mean(dim=1, keepdim=True)

    m_st = F.interpolate(m_st, size=(IMAGE_SIZE, IMAGE_SIZE),
                         mode='bilinear', align_corners=False)
    m_ae = F.interpolate(m_ae, size=(IMAGE_SIZE, IMAGE_SIZE),
                         mode='bilinear', align_corners=False)

    qa_st, qb_st = float(q['qa_st']), float(q['qb_st'])
    qa_ae, qb_ae = float(q['qa_ae']), float(q['qb_ae'])

    m_st_norm = 0.1 * (m_st - qa_st) / (qb_st - qa_st + 1e-8)
    m_ae_norm = 0.1 * (m_ae - qa_ae) / (qb_ae - qa_ae + 1e-8)

    combined = 0.5 * m_st_norm + 0.5 * m_ae_norm
    score    = float(combined.max().cpu())

    amap = combined.squeeze().cpu().numpy()
    return amap, score


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap helpers
# ─────────────────────────────────────────────────────────────────────────────

def amap_to_heatmap(amap, alpha=0.55):
    """Convert float anomaly map to JET BGR heatmap uint8."""
    mn, mx = amap.min(), amap.max()
    if mx - mn < 1e-8:
        norm = np.zeros_like(amap)
    else:
        norm = (amap - mn) / (mx - mn)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return heat


def overlay_heatmap(pil_img, amap, alpha=0.55):
    """Blend heatmap onto the original image."""
    img_bgr = cv2.cvtColor(np.array(pil_img.resize((MAP_W, MAP_H))),
                            cv2.COLOR_RGB2BGR)
    heat     = amap_to_heatmap(amap)
    blended  = cv2.addWeighted(img_bgr, 1 - alpha, heat, alpha, 0)
    return blended


# ─────────────────────────────────────────────────────────────────────────────
# Running metric state
# ─────────────────────────────────────────────────────────────────────────────

class MetricState:
    def __init__(self):
        self.labels  = []
        self.scores  = []
        self.tp = self.tn = self.fp = self.fn = 0
        self.threshold = 0.5

    def update(self, label, score, threshold):
        self.labels.append(label)
        self.scores.append(score)
        pred = int(score >= threshold)
        if label == 1 and pred == 1: self.tp += 1
        elif label == 0 and pred == 0: self.tn += 1
        elif label == 0 and pred == 1: self.fp += 1
        else: self.fn += 1

    @property
    def auroc(self):
        if len(set(self.labels)) < 2:
            return float('nan')
        return roc_auc_score(self.labels, self.scores)

    @property
    def accuracy(self):
        n = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / n if n else 0.0

    @property
    def precision(self):
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self):
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self):
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Frame builder
# ─────────────────────────────────────────────────────────────────────────────

def put_text(frame, text, x, y, color=WHITE, scale=0.52,
             font=FONT, thickness=1):
    cv2.putText(frame, text, (x, y), font, scale, color, thickness,
                cv2.LINE_AA)


def draw_bar(frame, x, y, w, h, value, color, bg=(50, 50, 50)):
    """Draw a horizontal progress bar (value in [0,1])."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), bg, -1)
    fill = int(np.clip(value, 0, 1) * w)
    if fill > 0:
        cv2.rectangle(frame, (x, y), (x + fill, y + h), color, -1)


def build_frame(pil_img, amap, score, gt_label, sample_name,
                sample_type, idx, total, ms,
                state: MetricState, threshold: float,
                countdown_ms: float = 10000,
                progress: float = 0.0) -> np.ndarray:
    """Build the full display frame."""
    frame = np.full((TOTAL_H, TOTAL_W, 3), BG, dtype=np.uint8)

    # ── header bar ───────────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (TOTAL_W, 40), (35, 35, 35), -1)
    put_text(frame,
             f'EfficientAD Live Evaluation  —  {sample_type}/{sample_name}',
             PADDING, 26, CYAN, scale=0.58, thickness=1)
    put_text(frame, f'Image {idx+1} / {total}',
             TOTAL_W - 150, 26, GRAY, scale=0.50)

    y0 = 52   # content start y

    # ── left panel: original image ────────────────────────────────────────────
    x0 = PADDING
    img_bgr = cv2.cvtColor(
        np.array(pil_img.resize((IMG_W, IMG_H))), cv2.COLOR_RGB2BGR)
    frame[y0:y0+IMG_H, x0:x0+IMG_W] = img_bgr

    # GT label badge
    gt_text  = 'ANOMALY' if gt_label == 1 else 'NORMAL'
    gt_color = RED if gt_label == 1 else GREEN
    cv2.rectangle(frame, (x0, y0), (x0 + IMG_W, y0 + 24), gt_color, -1)
    put_text(frame, f'GT: {gt_text}', x0 + 6, y0 + 17,
             (255, 255, 255), scale=0.52, thickness=1)

    put_text(frame, 'Original Image',
             x0, y0 + IMG_H + 20, GRAY, scale=0.45)

    # ── centre panel: heatmap overlay ─────────────────────────────────────────
    x1 = x0 + IMG_W + PADDING
    overlay = overlay_heatmap(pil_img, amap)
    frame[y0:y0+MAP_H, x1:x1+MAP_W] = overlay

    # prediction badge
    pred      = score >= threshold
    pred_text = 'ANOMALY' if pred else 'NORMAL'
    pred_col  = RED if pred else GREEN
    correct   = (int(pred) == gt_label)
    badge_col = GREEN if correct else (0, 0, 180)
    cv2.rectangle(frame, (x1, y0), (x1 + MAP_W, y0 + 24), badge_col, -1)
    put_text(frame, f'PRED: {pred_text}  {"✓" if correct else "✗"}',
             x1 + 6, y0 + 17, (255, 255, 255), scale=0.52, thickness=1)

    put_text(frame, 'Anomaly Heatmap (JET overlay)',
             x1, y0 + MAP_H + 20, GRAY, scale=0.45)

    # ── score bar below panels ────────────────────────────────────────────────
    bar_y = y0 + MAP_H + 36
    put_text(frame, f'Score: {score:.5f}   Threshold: {threshold:.5f}',
             PADDING, bar_y, WHITE, scale=0.50)
    bar_y += 14
    draw_bar(frame, PADDING, bar_y, IMG_W + PADDING + MAP_W, 10,
             np.clip(score / max(threshold * 2, 1e-6), 0, 1),
             RED if pred else GREEN)
    # threshold marker
    tx = PADDING + int(np.clip(threshold / max(threshold * 2, 1e-6), 0, 1)
                       * (IMG_W + PADDING + MAP_W))
    cv2.line(frame, (tx, bar_y - 2), (tx, bar_y + 12), YELLOW, 2)

    put_text(frame, f'Inference: {ms:.1f} ms',
             PADDING, bar_y + 28, GRAY, scale=0.43)

    # ── right panel: running metrics ──────────────────────────────────────────
    x2  = x1 + MAP_W + PADDING
    ry  = y0 + 10

    def mline(label, val, color=WHITE, scale=0.52):
        nonlocal ry
        put_text(frame, label, x2, ry, GRAY, scale=0.43)
        put_text(frame, val,   x2 + 160, ry, color, scale=scale)
        ry += 22

    def divider():
        nonlocal ry
        cv2.line(frame, (x2, ry), (x2 + INFO_W - PADDING, ry), DIVIDER, 1)
        ry += 10

    put_text(frame, 'RUNNING METRICS', x2, ry, CYAN,
             scale=0.55, thickness=1)
    ry += 30
    divider()

    n_seen = len(state.labels)
    auroc  = state.auroc
    mline('Images seen', f'{n_seen} / {total}')
    mline('AUROC',
          f'{auroc:.4f}' if not np.isnan(auroc) else 'need both classes',
          YELLOW if not np.isnan(auroc) else GRAY)
    mline('Accuracy',    f'{state.accuracy * 100:.2f} %',   WHITE)
    mline('Precision',   f'{state.precision * 100:.2f} %',  WHITE)
    mline('Recall',      f'{state.recall * 100:.2f} %',     WHITE)
    mline('F1 Score',    f'{state.f1:.4f}',                 ORANGE)

    divider()
    put_text(frame, 'CONFUSION', x2, ry, CYAN, scale=0.48)
    ry += 22

    # 2×2 confusion matrix mini-grid
    cx, cy, cw, ch = x2, ry, 90, 30
    labels_cm  = [['TP', state.tp], ['FP', state.fp],
                  ['FN', state.fn], ['TN', state.tn]]
    positions  = [(0, 0), (1, 0), (0, 1), (1, 1)]
    cm_colors  = [GREEN, RED, RED, GREEN]
    for (ci, cj), (lbl, val), col in zip(positions, labels_cm, cm_colors):
        bx, by = cx + ci * (cw + 4), cy + cj * (ch + 4)
        cv2.rectangle(frame, (bx, by), (bx + cw, by + ch), (40,40,40), -1)
        cv2.rectangle(frame, (bx, by), (bx + cw, by + ch), col, 1)
        put_text(frame, f'{lbl}: {val}', bx + 6, by + 20,
                 col, scale=0.44)
    ry += 2 * ch + 20

    divider()
    put_text(frame, 'THIS IMAGE', x2, ry, CYAN, scale=0.48)
    ry += 22
    mline('GT Label',      gt_text,          gt_color)
    mline('Prediction',    pred_text,         pred_col)
    mline('Score',         f'{score:.6f}',    WHITE)
    mline('Inference time',f'{ms:.2f} ms',    ORANGE)
    mline('Correct',       '✓ YES' if correct else '✗ NO',
          GREEN if correct else RED)

    divider()
    # colour bar legend — clamp ry so it never overflows the frame
    ry = min(ry, TOTAL_H - 50)
    bar_leg = np.zeros((14, INFO_W - PADDING * 2, 3), dtype=np.uint8)
    for xi in range(bar_leg.shape[1]):
        v  = int(xi / bar_leg.shape[1] * 255)
        bar_leg[:, xi] = cv2.applyColorMap(
            np.array([[v]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
    frame[ry:ry+14, x2:x2 + INFO_W - PADDING * 2] = bar_leg
    put_text(frame, 'Low',  x2,                   ry + 26, GRAY, scale=0.38)
    put_text(frame, 'High', x2 + INFO_W - 70,     ry + 26, GRAY, scale=0.38)
    put_text(frame, 'Anomaly score scale',
             x2 + 60, ry + 26, GRAY, scale=0.38)

    # ── footer: countdown bar + controls ─────────────────────────────────────
    cv2.rectangle(frame, (0, TOTAL_H - 36), (TOTAL_W, TOTAL_H),
                  (28, 28, 28), -1)

    # countdown progress bar (fills left→right as time runs out, turns red)
    bar_fill  = int(progress * TOTAL_W)
    bar_color = (30, 200, 80) if progress < 0.6 else \
                (30, 140, 240) if progress < 0.85 else (60, 60, 220)
    cv2.rectangle(frame, (0, TOTAL_H - 36), (bar_fill, TOTAL_H - 22),
                  bar_color, -1)

    secs_left = countdown_ms / 1000.0
    put_text(frame,
             f'Next in {secs_left:.1f}s  |  SPACE/→: next   ←: prev   S: save   Q/ESC: quit',
             PADDING, TOTAL_H - 8, GRAY, scale=0.40, font=FONT_SMALL)

    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(results, threshold, final_state, save_dir, category):
    """
    Write two CSV files to save_dir:
      1. per_image_results.csv  — one row per test image
      2. summary_metrics.csv    — overall evaluation metrics
    """
    import csv
    os.makedirs(save_dir, exist_ok=True)

    # ── 1. Per-image CSV ──────────────────────────────────────────────────────
    per_image_path = os.path.join(save_dir, 'per_image_results.csv')
    per_image_fields = [
        'index', 'defect_class', 'filename',
        'gt_label', 'gt_label_name',
        'anomaly_score', 'threshold',
        'prediction', 'correct',
        'tp', 'tn', 'fp', 'fn',          # running confusion counts
        'running_accuracy',
        'running_auroc',
        'inference_ms',
    ]

    running = MetricState()
    running.threshold = threshold
    rows = []
    for i, r in enumerate(results):
        running.update(r['label'], r['score'], threshold)
        pred      = int(r['score'] >= threshold)
        correct   = int(pred == r['label'])
        auroc_run = running.auroc
        rows.append({
            'index':           i,
            'defect_class':    r['type'],
            'filename':        r['name'],
            'gt_label':        r['label'],
            'gt_label_name':   'anomaly' if r['label'] == 1 else 'normal',
            'anomaly_score':   round(r['score'], 8),
            'threshold':       round(threshold, 8),
            'prediction':      'anomaly' if pred == 1 else 'normal',
            'correct':         correct,
            'tp':              running.tp,
            'tn':              running.tn,
            'fp':              running.fp,
            'fn':              running.fn,
            'running_accuracy': round(running.accuracy * 100, 4),
            'running_auroc':   round(auroc_run, 6) if not np.isnan(auroc_run) else 'N/A',
            'inference_ms':    round(r['ms'], 4),
        })

    with open(per_image_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=per_image_fields)
        writer.writeheader()
        writer.writerows(rows)

    # ── 2. Summary CSV ────────────────────────────────────────────────────────
    summary_path = os.path.join(save_dir, 'summary_metrics.csv')
    auroc = final_state.auroc
    total = len(results)
    avg_ms = np.mean([r['ms'] for r in results])
    min_ms = np.min( [r['ms'] for r in results])
    max_ms = np.max( [r['ms'] for r in results])

    # Score distribution stats
    scores = np.array([r['score'] for r in results])
    normal_scores  = scores[[r['label'] == 0 for r in results]]
    anomaly_scores = scores[[r['label'] == 1 for r in results]]

    summary_rows = [
        ('category',              category),
        ('total_images',          total),
        ('normal_images',         int((np.array([r['label'] for r in results]) == 0).sum())),
        ('anomaly_images',        int((np.array([r['label'] for r in results]) == 1).sum())),
        ('threshold',             round(threshold, 8)),
        ('auroc',                 round(auroc, 6) if not np.isnan(auroc) else 'N/A'),
        ('accuracy_%',            round(final_state.accuracy  * 100, 4)),
        ('precision_%',           round(final_state.precision * 100, 4)),
        ('recall_%',              round(final_state.recall    * 100, 4)),
        ('f1_score',              round(final_state.f1, 6)),
        ('tp',                    final_state.tp),
        ('tn',                    final_state.tn),
        ('fp',                    final_state.fp),
        ('fn',                    final_state.fn),
        ('score_mean_all',        round(float(scores.mean()), 6)),
        ('score_std_all',         round(float(scores.std()),  6)),
        ('score_min_all',         round(float(scores.min()),  6)),
        ('score_max_all',         round(float(scores.max()),  6)),
        ('score_mean_normal',     round(float(normal_scores.mean()),  6) if len(normal_scores)  else 'N/A'),
        ('score_mean_anomaly',    round(float(anomaly_scores.mean()), 6) if len(anomaly_scores) else 'N/A'),
        ('score_p50_normal',      round(float(np.median(normal_scores)),  6) if len(normal_scores)  else 'N/A'),
        ('score_p50_anomaly',     round(float(np.median(anomaly_scores)), 6) if len(anomaly_scores) else 'N/A'),
        ('score_p95_normal',      round(float(np.percentile(normal_scores,  95)), 6) if len(normal_scores)  else 'N/A'),
        ('score_p5_anomaly',      round(float(np.percentile(anomaly_scores,  5)), 6) if len(anomaly_scores) else 'N/A'),
        ('inference_ms_avg',      round(avg_ms, 4)),
        ('inference_ms_min',      round(min_ms, 4)),
        ('inference_ms_max',      round(max_ms, 4)),
        ('total_inference_s',     round(sum(r['ms'] for r in results) / 1000, 4)),
    ]

    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        writer.writerows(summary_rows)

    print(f'  Per-image CSV  → {per_image_path}')
    print(f'  Summary CSV    → {summary_path}')
    return per_image_path, summary_path


def parse_args():
    p = argparse.ArgumentParser(
        description='Live EfficientAD-only evaluation viewer.')
    p.add_argument('--dataset_path', required=True)
    p.add_argument('--ckpt_dir',     required=True)
    p.add_argument('--teacher_path', default=None,
                   help='Path to best_teacher.pth. If omitted the script '
                        'searches ckpt_dir and common sibling directories '
                        'automatically.')
    p.add_argument('--category',     default='chewinggum')
    p.add_argument('--model_size',   default='S', choices=['S', 'M'])
    p.add_argument('--device',       default=None)
    p.add_argument('--threshold',    type=float, default=None,
                   help='Fixed decision threshold. If omitted, estimated '
                        'from the median of all scores (updated live).')
    p.add_argument('--threshold_file', default=None,
                   help='Path to a file containing a single float threshold '
                        '(written by evaluate_pipeline.py).')
    p.add_argument('--auto_advance', type=int, default=10000,
                   help='Auto-advance to next image after N ms (default 10000).')
    p.add_argument('--start_idx',   type=int, default=0,
                   help='Start from this image index.')
    p.add_argument('--save_dir',    default='./live_frames',
                   help='Directory to save frames when pressing S.')
    return p.parse_args()


def main():
    args   = parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'\n[Live] EfficientAD Live Evaluation Viewer')
    print(f'  Category  : {args.category}')
    print(f'  Model     : PDN-{args.model_size}')
    print(f'  Ckpt dir  : {args.ckpt_dir}')
    print(f'  Device    : {device}\n')

    # ── Load models ───────────────────────────────────────────────────────────
    teacher, student, ae, mu, sigma, q = load_models(
        args.ckpt_dir, args.category, args.model_size, device,
        teacher_path_hint=args.teacher_path)

    # ── Resolve threshold ─────────────────────────────────────────────────────
    fixed_threshold = None
    if args.threshold is not None:
        fixed_threshold = args.threshold
        print(f'[Live] Using fixed threshold: {fixed_threshold:.6f}')
    elif args.threshold_file and os.path.isfile(args.threshold_file):
        fixed_threshold = float(open(args.threshold_file).read().strip())
        print(f'[Live] Loaded threshold from file: {fixed_threshold:.6f}')

    # ── Dataset ───────────────────────────────────────────────────────────────
    tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    dataset = get_AD_dataset(
        type='VisA', root=args.dataset_path,
        transform=tf, gt_transform=tf,
        phase='test', category=args.category)
    total = len(dataset)
    print(f'[Live] {total} test images found.\n')
    print('[Live] Window opening — controls: SPACE/→ next  ←  prev  '
          'S save  Q quit\n')

    # ── Pre-run all inferences (fast, avoids per-frame GPU delay in viewer) ──
    print('[Live] Running inference on all images ...')
    results = []
    t_start = time.time()
    for i, sample in enumerate(dataset):
        img_tensor = sample['image'].unsqueeze(0).to(device)
        t0   = time.perf_counter()
        amap, score = infer(img_tensor, teacher, student, ae, mu, sigma, q)
        ms   = (time.perf_counter() - t0) * 1000.0
        pil  = Image.fromarray(sample['origin'].astype(np.uint8)) \
               if 'origin' in sample \
               else transforms.ToPILImage()(sample['image'])
        results.append({
            'pil':   pil,
            'amap':  amap,
            'score': score,
            'label': int(sample['label']),
            'name':  sample.get('name', str(i)),
            'type':  sample.get('type', 'test'),
            'ms':    ms,
        })
        if (i + 1) % 20 == 0 or (i + 1) == total:
            print(f'  {i+1}/{total}  last_score={score:.4f}  {ms:.1f}ms')

    elapsed = time.time() - t_start
    print(f'[Live] Inference complete: {total} images in {elapsed:.1f}s '
          f'({elapsed/total*1000:.1f} ms/img avg)\n')

    # ── Compute threshold if not fixed ────────────────────────────────────────
    all_scores = [r['score'] for r in results]
    if fixed_threshold is None:
        # Use Youden's J on all scores as a simple auto-threshold
        from sklearn.metrics import roc_curve
        all_labels = [r['label'] for r in results]
        if len(set(all_labels)) == 2:
            fpr, tpr, thrs = roc_curve(all_labels, all_scores)
            j_idx = np.argmax(tpr - fpr)
            fixed_threshold = float(thrs[j_idx])
        else:
            fixed_threshold = float(np.median(all_scores))
        print(f'[Live] Auto threshold (Youden J): {fixed_threshold:.6f}')

    # ── Build running metric state ────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    state = MetricState()
    state.threshold = fixed_threshold

    # Pre-fill state up to start_idx
    start = max(0, min(args.start_idx, total - 1))
    for i in range(start):
        r = results[i]
        state.update(r['label'], r['score'], fixed_threshold)

    # ── OpenCV window ─────────────────────────────────────────────────────────
    WIN = 'EfficientAD Live Evaluation'
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, TOTAL_W, TOTAL_H)

    idx = start
    while True:
        r = results[idx]

        # Update state for this image (avoid double-counting on revisit)
        if idx >= len(state.labels):
            state.update(r['label'], r['score'], fixed_threshold)

        auto_ms   = args.auto_advance          # ms per image (default 5000)
        deadline  = time.perf_counter() + auto_ms / 1000.0

        while True:
            remaining_ms = max(0.0, (deadline - time.perf_counter()) * 1000)
            progress     = 1.0 - remaining_ms / auto_ms   # 0→1 as time runs out

            frame = build_frame(
                pil_img      = r['pil'],
                amap         = r['amap'],
                score        = r['score'],
                gt_label     = r['label'],
                sample_name  = r['name'],
                sample_type  = r['type'],
                idx          = idx,
                total        = total,
                ms           = r['ms'],
                state        = state,
                threshold    = fixed_threshold,
                countdown_ms = remaining_ms,
                progress     = progress,
            )

            cv2.imshow(WIN, frame)
            key = cv2.waitKey(0) & 0xFF          # poll every 30 ms


            # ── key handling ──────────────────────────────────────────────────
            if key in (ord('q'), 27):             # Q / ESC → quit
                idx = -1
                break
            elif key in (ord(' '), 83, 39, 255) or remaining_ms <= 0:
                # SPACE / → / right-arrow / timeout → advance
                if idx < total - 1:
                    idx += 1
                else:
                    idx = -1   # finished all images
                break
            elif key in (81, 37):                 # ← / left-arrow → back
                if idx > 0:
                    state = MetricState()
                    state.threshold = fixed_threshold
                    for i in range(idx - 1):
                        ri = results[i]
                        state.update(ri['label'], ri['score'], fixed_threshold)
                    idx -= 1
                break
            elif key == ord('s'):                 # S → save frame
                fname = os.path.join(
                    args.save_dir,
                    f'{r["type"]}_{r["name"]}_frame.png')
                cv2.imwrite(fname, frame)
                print(f'  [Save] {fname}')

            # window closed manually
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                idx = -1
                break

        if idx < 0:
            break

    cv2.destroyAllWindows()

    # ── Final summary ─────────────────────────────────────────────────────────
    # Compute on ALL images regardless of where we stopped
    final_state = MetricState()
    final_state.threshold = fixed_threshold
    for r in results:
        final_state.update(r['label'], r['score'], fixed_threshold)

    auroc = final_state.auroc
    print(f'\n{"="*50}')
    print(f'  FINAL RESULTS  ({args.category})')
    print(f'{"="*50}')
    print(f'  Total images : {total}')
    print(f'  Threshold    : {fixed_threshold:.6f}')
    print(f'  AUROC        : {auroc:.4f}' if not np.isnan(auroc) else '  AUROC: N/A')
    print(f'  Accuracy     : {final_state.accuracy * 100:.2f} %')
    print(f'  Precision    : {final_state.precision * 100:.2f} %')
    print(f'  Recall       : {final_state.recall * 100:.2f} %')
    print(f'  F1 Score     : {final_state.f1:.4f}')
    print(f'  TP={final_state.tp}  TN={final_state.tn}  '
          f'FP={final_state.fp}  FN={final_state.fn}')
    print(f'{"="*50}')

    # ── Save CSV reports ──────────────────────────────────────────────────────
    print(f'\n[Live] Saving CSV reports ...')
    save_csv(results, fixed_threshold, final_state, args.save_dir, args.category)
    print()


if __name__ == '__main__':
    main()
