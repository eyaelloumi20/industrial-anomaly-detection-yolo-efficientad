#!/usr/bin/env python3
"""
compare_extraction_methods.py
------------------------------
Compare les 3 méthodes d'extraction de masque YOLO sur une seule image
et sauvegarde les crops en 256×256 et 512×512 (padding-aware, sans déformation).

Méthodes comparées :
  1. raw       — fillPoly direct (polygone brut)
  2. morph     — fillPoly + érosion + GaussianBlur (nettoyage morphologique)
  3. grabcut   — YOLO-det bbox + GrabCut seedé par le polygone YOLO-seg

Usage:
    python compare_extraction_methods.py --image photo.jpg \
        --seg-model chewinggum_seg_v22.pt \
        [--det-model chewinggum_finetune2.pt] \
        --output ./comparison \
        [--conf 0.25] [--iou 0.45]

Sorties (par détection, par taille) :
    <output>/
      det000_raw_256.png
      det000_raw_512.png
      det000_morph_256.png
      det000_morph_512.png
      det000_grabcut_256.png
      det000_grabcut_512.png
      ...
      comparison_grid.png   ← grille visuelle côte à côte (sur fond blanc)
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────
# Resize avec padding (letterbox) — préserve le ratio, fond transparent
# ─────────────────────────────────────────────────────────────

def resize_with_padding(img_rgba: np.ndarray, target_size: int) -> np.ndarray:
    """
    Redimensionne un crop RGBA dans un carré target_size×target_size
    en préservant le ratio d'aspect (padding transparent autour).

    Args:
        img_rgba    : image RGBA (H, W, 4)
        target_size : côté du carré de sortie (256 ou 512)

    Returns:
        image RGBA (target_size, target_size, 4)
    """
    h, w = img_rgba.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_size, target_size, 4), dtype=np.uint8)

    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    new_w = max(1, new_w)
    new_h = max(1, new_h)

    resized = cv2.resize(img_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_size, target_size, 4), dtype=np.uint8)
    pad_top = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return canvas


# ─────────────────────────────────────────────────────────────
# Méthode 1 : RAW — polygone brut sans traitement
# ─────────────────────────────────────────────────────────────

def method_raw(frame_bgr: np.ndarray, polygon: np.ndarray) -> np.ndarray | None:
    pts = polygon.astype(np.int32)
    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()

    H, W = frame_bgr.shape[:2]
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(W - 1, x_max), min(H - 1, y_max)

    crop_bgr = frame_bgr[y_min:y_max + 1, x_min:x_max + 1]
    if crop_bgr.size == 0:
        return None

    pts_s = pts - np.array([x_min, y_min])
    h_c, w_c = y_max - y_min + 1, x_max - x_min + 1
    mask = np.zeros((h_c, w_c), dtype=np.uint8)
    cv2.fillPoly(mask, [pts_s.reshape(-1, 1, 2)], 255)

    rgba = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask
    return rgba


# ─────────────────────────────────────────────────────────────
# Méthode 2 : MORPH — érosion + GaussianBlur (Option C)
# ─────────────────────────────────────────────────────────────

def method_morph(frame_bgr: np.ndarray, polygon: np.ndarray,
                 erode_iters: int = 1, blur_ksize: int = 3) -> np.ndarray | None:
    pts = polygon.astype(np.int32)
    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()

    H, W = frame_bgr.shape[:2]
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(W - 1, x_max), min(H - 1, y_max)

    crop_bgr = frame_bgr[y_min:y_max + 1, x_min:x_max + 1]
    if crop_bgr.size == 0:
        return None

    pts_s = pts - np.array([x_min, y_min])
    h_c, w_c = y_max - y_min + 1, x_max - x_min + 1
    mask = np.zeros((h_c, w_c), dtype=np.uint8)
    cv2.fillPoly(mask, [pts_s.reshape(-1, 1, 2)], 255)

    # Érosion pour retirer la frange de bord
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=erode_iters)

    # Feathering léger
    if blur_ksize > 1:
        mask = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 1)

    rgba = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask
    return rgba


# ─────────────────────────────────────────────────────────────
# Méthode 3 : GRABCUT — bbox YOLO-det + seed polygone YOLO-seg
# ─────────────────────────────────────────────────────────────

def method_grabcut(frame_bgr: np.ndarray,
                   det_box: np.ndarray,
                   seg_polygon: np.ndarray,
                   gc_iters: int = 5) -> np.ndarray | None:
    """
    Args:
        det_box     : [x1, y1, x2, y2] en pixels (bbox détection)
        seg_polygon : (N, 2) coords pixel du polygone segmentation
    """
    H, W = frame_bgr.shape[:2]

    # Bbox de travail depuis det_box
    x1, y1, x2, y2 = det_box.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W - 1, x2), min(H - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    rect = (x1, y1, x2 - x1, y2 - y1)  # (x, y, w, h) pour GrabCut

    # Masque GrabCut : GC_BGD par défaut
    gc_mask = np.full((H, W), cv2.GC_BGD, dtype=np.uint8)

    # Région probable FG = bbox
    gc_mask[y1:y2, x1:x2] = cv2.GC_PR_BGD

    # FG certain = intérieur du polygone YOLO-seg
    pts = seg_polygon.astype(np.int32)
    poly_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(poly_mask, [pts.reshape(-1, 1, 2)], 255)
    gc_mask[poly_mask == 255] = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(frame_bgr, gc_mask, rect,
                    bgd_model, fgd_model,
                    gc_iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        # Repli : utiliser seulement le polygone brut
        return method_raw(frame_bgr, seg_polygon)

    # FG = GC_FGD | GC_PR_FGD
    fg_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
                       255, 0).astype(np.uint8)

    # Crop sur l'union bbox det + polygone seg
    all_x = np.concatenate([pts[:, 0], [x1, x2]])
    all_y = np.concatenate([pts[:, 1], [y1, y2]])
    cx1, cy1 = max(0, all_x.min()), max(0, all_y.min())
    cx2, cy2 = min(W - 1, all_x.max()), min(H - 1, all_y.max())

    crop_bgr = frame_bgr[cy1:cy2 + 1, cx1:cx2 + 1]
    crop_mask = fg_mask[cy1:cy2 + 1, cx1:cx2 + 1]

    if crop_bgr.size == 0:
        return None

    rgba = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = crop_mask
    return rgba


# ─────────────────────────────────────────────────────────────
# Grille de comparaison visuelle
# ─────────────────────────────────────────────────────────────

def make_comparison_grid(results: list[dict], target_size: int = 256) -> np.ndarray:
    """
    Crée une grille PNG (fond blanc) :
        colonnes : méthodes (raw | morph | grabcut)
        lignes   : détections

    Chaque cellule = crop RGBA composité sur fond blanc, 256×256.
    """
    methods = ["raw", "morph", "grabcut"]
    n_dets = len(results)
    if n_dets == 0:
        return None

    cell = target_size
    margin = 4
    label_h = 20
    header_w = 60  # largeur de la colonne "det N°"

    grid_h = n_dets * (cell + margin) + label_h + margin
    grid_w = header_w + len(methods) * (cell + margin)

    grid = np.full((grid_h, grid_w, 3), 240, dtype=np.uint8)  # fond gris clair

    # En-têtes colonnes
    font = cv2.FONT_HERSHEY_SIMPLEX
    for col, m in enumerate(methods):
        cx = header_w + col * (cell + margin) + cell // 2 - 30
        cv2.putText(grid, m, (cx, label_h - 4), font, 0.5, (50, 50, 50), 1)

    for row, det in enumerate(results):
        row_y = label_h + margin + row * (cell + margin)

        # Étiquette ligne
        cv2.putText(grid, f"det{det['idx']:02d}", (4, row_y + cell // 2),
                    font, 0.45, (80, 80, 80), 1)

        for col, m in enumerate(methods):
            cx = header_w + col * (cell + margin)
            rgba = det.get(m)
            if rgba is None:
                continue

            # Resize avec padding
            thumb = resize_with_padding(rgba, cell)

            # Composite sur fond blanc
            alpha = thumb[:, :, 3:4] / 255.0
            rgb = thumb[:, :, :3][:, :, ::-1]  # BGRA → RGB → BGR
            white = np.ones((cell, cell, 3), dtype=np.uint8) * 255
            composited = (rgb * alpha + white * (1 - alpha)).astype(np.uint8)

            grid[row_y:row_y + cell, cx:cx + cell] = composited

    return grid


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare 3 méthodes d'extraction YOLO sur une image"
    )
    p.add_argument("--image", required=True, help="Image d'entrée (JPG/PNG)")
    p.add_argument("--seg-model", required=True,
                   help="Modèle YOLO segmentation (.pt)")
    p.add_argument("--det-model", default=None,
                   help="Modèle YOLO détection (.pt) pour GrabCut "
                        "(optionnel — utilise seg bbox si absent)")
    p.add_argument("--output", default="./comparison",
                   help="Dossier de sortie (défaut: ./comparison)")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--device", default="",
                   help="Device: '' auto, 'cpu', '0' GPU0")
    p.add_argument("--erode-iters", type=int, default=1,
                   help="Itérations d'érosion pour méthode morph (défaut: 1)")
    p.add_argument("--blur-ksize", type=int, default=3,
                   help="Taille kernel GaussianBlur méthode morph (défaut: 3, 0=désactivé)")
    return p.parse_args()


def main():
    args = parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"[ERREUR] Image introuvable: {img_path}")
        sys.exit(1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"[ERREUR] Impossible de lire: {img_path}")
        sys.exit(1)

    print(f"[INFO] Image: {frame.shape[1]}×{frame.shape[0]} px")

    # ── Chargement modèles ──────────────────────────────────
    print(f"[INFO] Modèle segmentation: {args.seg_model}")
    seg_model = YOLO(args.seg_model)

    det_model = None
    if args.det_model:
        print(f"[INFO] Modèle détection: {args.det_model}")
        det_model = YOLO(args.det_model)
    else:
        print("[INFO] Pas de modèle détection — GrabCut utilisera les bbox du modèle seg")

    device = args.device if args.device else None

    # ── Inférence seg ───────────────────────────────────────
    seg_results = seg_model.predict(
        source=frame, conf=args.conf, iou=args.iou,
        device=device, verbose=False
    )
    seg_res = seg_results[0]

    if seg_res.masks is None or len(seg_res.masks.xy) == 0:
        print("[WARN] Aucune détection de segmentation sur cette image.")
        sys.exit(0)

    seg_polygons = seg_res.masks.xy
    seg_boxes    = seg_res.boxes.xyxy.cpu().numpy()  # (N, 4)
    seg_cls      = seg_res.boxes.cls.cpu().numpy().astype(int)
    seg_confs    = seg_res.boxes.conf.cpu().numpy()
    names        = seg_res.names

    # ── Inférence det (optionnel) ───────────────────────────
    det_boxes = None
    if det_model is not None:
        det_results = det_model.predict(
            source=frame, conf=args.conf, iou=args.iou,
            device=device, verbose=False
        )
        det_res = det_results[0]
        if len(det_res.boxes) > 0:
            det_boxes = det_res.boxes.xyxy.cpu().numpy()

    print(f"[INFO] {len(seg_polygons)} détection(s) trouvée(s)")
    print("-" * 60)

    results_for_grid = []

    for idx, (polygon, box, cls_id, conf_val) in enumerate(
        zip(seg_polygons, seg_boxes, seg_cls, seg_confs)
    ):
        if len(polygon) < 3:
            continue

        class_name = names[cls_id]
        tag = f"det{idx:03d}_{class_name}_c{conf_val:.2f}"

        print(f"  [{idx}] {class_name} conf={conf_val:.3f} "
              f"bbox=[{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]")

        # Bbox pour GrabCut : det_model si dispo, sinon seg bbox
        gc_box = box.copy()
        if det_boxes is not None and idx < len(det_boxes):
            gc_box = det_boxes[idx]

        # ── 3 méthodes ──────────────────────────────────────
        raw_rgba     = method_raw(frame, polygon)
        morph_rgba   = method_morph(frame, polygon,
                                    erode_iters=args.erode_iters,
                                    blur_ksize=args.blur_ksize)
        gc_rgba      = method_grabcut(frame, gc_box, polygon)

        det_result = {"idx": idx, "raw": raw_rgba, "morph": morph_rgba, "grabcut": gc_rgba}
        results_for_grid.append(det_result)

        # ── Sauvegarde 256 et 512 ───────────────────────────
        for method_name, rgba in [("raw", raw_rgba),
                                   ("morph", morph_rgba),
                                   ("grabcut", gc_rgba)]:
            if rgba is None:
                print(f"    [{method_name}] aucun crop produit")
                continue
            for size in [256, 512]:
                resized = resize_with_padding(rgba, size)
                fname = f"{tag}_{method_name}_{size}.png"
                save_path = out_dir / fname
                cv2.imwrite(str(save_path), resized)
                print(f"    [{method_name}] {size}px → {fname}")

    # ── Grille de comparaison ────────────────────────────────
    grid = make_comparison_grid(results_for_grid, target_size=256)
    if grid is not None:
        grid_path = out_dir / "comparison_grid.png"
        cv2.imwrite(str(grid_path), grid)
        print(f"\n[INFO] Grille de comparaison → {grid_path.resolve()}")

    print("-" * 60)
    print(f"[DONE] Sorties dans: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
