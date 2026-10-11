"""
visualisation.py
================
Plotting and image-saving utilities for the inspection pipeline.

Produces:
    - ROC curve (dark theme, matches existing eval scripts)
    - Score distribution histogram
    - Anomaly-map heatmap overlays  (PNG)
    - TIFF raw anomaly maps

All functions are stateless and write files directly.

Exported symbols
----------------
    save_roc_curve(fpr, tpr, auroc, category, path)
    save_score_histogram(scores, labels, threshold, category, path)
    save_overlay(pil_image, anomaly_map_np, path, alpha=0.5)
    save_tiff(anomaly_map_np, path)
    save_per_image_csv(rows, path)
"""

from __future__ import annotations
import os
import csv
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tifffile


# ── ROC curve ─────────────────────────────────────────────────────────────────

def save_roc_curve(
    fpr:      np.ndarray,
    tpr:      np.ndarray,
    auroc:    float,
    category: str,
    path:     str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.plot(fpr, tpr, color="#e94560", lw=2, label=f"AUROC = {auroc:.4f}")
    ax.plot([0, 1], [0, 1], color="#555", lw=1, linestyle="--")
    ax.set_xlim([0, 1]);  ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", color="#eaeaea")
    ax.set_ylabel("True Positive Rate",  color="#eaeaea")
    ax.set_title(f"ROC — {category}", color="#eaeaea", fontsize=13)
    ax.tick_params(colors="#eaeaea")
    for s in ax.spines.values():
        s.set_edgecolor("#444")
    ax.legend(facecolor="#2a2a4a", edgecolor="#555", labelcolor="#eaeaea")
    plt.tight_layout()
    plt.savefig(path, dpi=120, facecolor="#1a1a2e")
    plt.close()


# ── score histogram ───────────────────────────────────────────────────────────

def save_score_histogram(
    scores:    np.ndarray,
    labels:    np.ndarray,
    threshold: float,
    category:  str,
    path:      str,
) -> None:
    normal  = scores[labels == 0]
    anomaly = scores[labels == 1]
    bins    = np.linspace(scores.min(), scores.max(), 40)

    fig, ax = plt.subplots(figsize=(7, 4), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.hist(normal,  bins=bins, color="#3498db", alpha=0.7, label="Normal")
    ax.hist(anomaly, bins=bins, color="#e74c3c", alpha=0.7, label="Anomaly")
    ax.axvline(threshold, color="#f1c40f", lw=2, linestyle="--",
               label=f"Threshold = {threshold:.4f}")
    ax.set_xlabel("Anomaly Score", color="#eaeaea")
    ax.set_ylabel("Count",         color="#eaeaea")
    ax.set_title(f"Score Distribution — {category}", color="#eaeaea", fontsize=13)
    ax.tick_params(colors="#eaeaea")
    for s in ax.spines.values():
        s.set_edgecolor("#444")
    ax.legend(facecolor="#2a2a4a", edgecolor="#555", labelcolor="#eaeaea")
    plt.tight_layout()
    plt.savefig(path, dpi=120, facecolor="#1a1a2e")
    plt.close()


# ── heatmap overlay ───────────────────────────────────────────────────────────

def save_overlay(
    pil_image:      Image.Image,
    anomaly_map_np: np.ndarray,
    path:           str,
    alpha:          float = 0.5,
    size:           int   = 256,
) -> None:
    """Save a blended heatmap overlay as PNG."""
    mn, mx = anomaly_map_np.min(), anomaly_map_np.max()
    norm   = (anomaly_map_np - mn) / (mx - mn + 1e-8)
    heat   = (plt.cm.hot(norm)[:, :, :3] * 255).astype(np.uint8)
    heat_pil = Image.fromarray(heat).resize((size, size))
    orig     = pil_image.convert("RGB").resize((size, size))
    blend    = Image.blend(orig, heat_pil, alpha)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    blend.save(path)


# ── TIFF anomaly map ──────────────────────────────────────────────────────────

def save_tiff(anomaly_map_np: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tifffile.imwrite(path, anomaly_map_np.astype(np.float32))


# ── per-image CSV ─────────────────────────────────────────────────────────────

def save_per_image_csv(rows: list, path: str) -> None:
    """
    rows : list of dicts with keys:
        defect_class, filename, gt, pred, score, correct, time_ms
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = ["defect_class", "filename", "gt", "pred",
                  "score", "correct", "time_ms"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ── console per-image table ───────────────────────────────────────────────────

def print_per_image_table(
    results: list,   # list of PipelineResult
) -> None:
    print(f"\n  {'#':>4}  {'type/name':<42}  {'GT':>7}  {'Pred':>7}  "
          f"{'Score':>8}  {'ms':>7}  {'✓/✗':>4}")
    print(f"  {'-' * 85}")
    for i, r in enumerate(results):
        gt_s = "ANOMALY" if r.label == 1 else "NORMAL"
        mark = "✓" if r.correct else "✗"
        print(f"  {i:>4}  {r.img_type + '/' + r.name:<42}  "
              f"{gt_s:>7}  {r.prediction:>7}  "
              f"{r.image_score:>8.4f}  {r.inference_ms:>7.2f}  {mark:>4}")