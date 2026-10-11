#!/usr/bin/env python3
"""
extract_segments_images.py
---------------------------
Extrait tous les polygones de segmentation d'instance YOLOv11 depuis un dossier
d'images et sauvegarde chaque crop en PNG 256×256 avec fond transparent.

Usage:
    python extract_segments_images.py --images ./photos --model best.pt --output ./crops
    python extract_segments_images.py --images ./photos --model best.pt --output ./crops --conf 0.5
    python extract_segments_images.py --images ./photos --model best.pt --output ./crops --size 128
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO11 seg → crop PNG transparent redimensionné (dossier images)"
    )
    parser.add_argument("--images",  required=True, help="Dossier d'images d'entrée")
    parser.add_argument("--model",   required=True, help="Modèle YOLO11 seg (.pt)")
    parser.add_argument("--output",  default="./output_segments",
                        help="Dossier de sortie (défaut: ./output_segments)")
    parser.add_argument("--conf",    type=float, default=0.25,
                        help="Seuil de confiance (défaut: 0.25)")
    parser.add_argument("--iou",     type=float, default=0.45,
                        help="Seuil IoU NMS (défaut: 0.45)")
    parser.add_argument("--size",    type=int, default=256,
                        help="Taille de sortie en pixels (défaut: 256 → 256×256)")
    parser.add_argument("--device",  default="",
                        help="Device: '' auto, 'cpu', '0' GPU 0 (défaut: auto)")
    return parser.parse_args()


def mask_to_polygon_crop_resized(
    image_bgr: np.ndarray,
    mask_xy: np.ndarray,
    out_size: int
) -> np.ndarray | None:
    """
    1. Découpe la bounding-box du polygone dans image_bgr.
    2. Crée un masque binaire exact du polygone.
    3. Met le fond à transparent (alpha=0).
    4. Redimensionne le crop RGBA à out_size × out_size (INTER_AREA pour réduire,
       INTER_CUBIC pour agrandir) sans changer les pixels du sujet.

    Returns: numpy array RGBA (out_size, out_size, 4) ou None si crop vide.
    """
    pts = mask_xy.astype(np.int32)

    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()

    H, W = image_bgr.shape[:2]
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(W - 1, x_max)
    y_max = min(H - 1, y_max)

    crop_bgr = image_bgr[y_min:y_max + 1, x_min:x_max + 1]
    if crop_bgr.size == 0:
        return None

    # Masque binaire dans l'espace du crop
    pts_shifted = pts - np.array([x_min, y_min])
    h_crop = y_max - y_min + 1
    w_crop = x_max - x_min + 1
    binary_mask = np.zeros((h_crop, w_crop), dtype=np.uint8)
    cv2.fillPoly(binary_mask, [pts_shifted.reshape(-1, 1, 2)], 255)

    # BGR → BGRA, applique le masque comme canal alpha
    crop_rgba = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)
    crop_rgba[:, :, 3] = binary_mask

    # Redimensionner à out_size × out_size
    # INTER_AREA = meilleure qualité pour réduction, INTER_CUBIC pour agrandissement
    crop_h, crop_w = crop_rgba.shape[:2]
    interp = cv2.INTER_AREA if (crop_w > out_size or crop_h > out_size) else cv2.INTER_CUBIC
    resized = cv2.resize(crop_rgba, (out_size, out_size), interpolation=interp)

    return resized


def process_one_image(model, image_bgr, source_name, output_dir, conf, iou, device, out_size):
    """Inférence YOLO + rescaling masques + sauvegarde crops."""
    orig_h, orig_w = image_bgr.shape[:2]

    results = model.predict(
        source=image_bgr,
        conf=conf,
        iou=iou,
        device=device if device else None,
        verbose=False,
    )
    result = results[0]

    if result.masks is None or len(result.masks.xy) == 0:
        return 0

    # Recaler les polygones depuis la résolution interne YOLO → résolution originale
    model_h, model_w = result.masks.shape[1], result.masks.shape[2]
    scale_x = orig_w / model_w
    scale_y = orig_h / model_h

    polygons = []
    for poly in result.masks.xy:
        scaled = poly.copy().astype(np.float32)
        scaled[:, 0] *= scale_x
        scaled[:, 1] *= scale_y
        polygons.append(scaled)

    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    confs     = result.boxes.conf.cpu().numpy()
    names     = result.names

    saved = 0
    for det_idx, (polygon, cls_id, conf_val) in enumerate(zip(polygons, class_ids, confs)):
        if len(polygon) < 3:
            continue

        crop_rgba = mask_to_polygon_crop_resized(image_bgr, polygon, out_size)
        if crop_rgba is None:
            continue

        class_name = names[cls_id]
        filename = f"{source_name}_det{det_idx:03d}_{class_name}_conf{conf_val:.2f}.png"
        cv2.imwrite(str(output_dir / filename), crop_rgba)
        saved += 1

    return saved


def main():
    args = parse_args()

    images_dir = Path(args.images)
    if not images_dir.is_dir():
        print(f"[ERREUR] Dossier introuvable: {images_dir}")
        sys.exit(1)

    image_files = sorted([
        p for p in images_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ])
    if not image_files:
        print(f"[ERREUR] Aucune image trouvée dans: {images_dir}")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Chargement du modèle : {args.model}")
    model = YOLO(args.model)

    print(f"[INFO] {len(image_files)} images trouvées")
    print(f"[INFO] Taille de sortie     : {args.size}×{args.size} px")
    print(f"[INFO] Confiance            : {args.conf} | IoU : {args.iou}")
    print(f"[INFO] Sortie               : {output_dir.resolve()}")
    print("-" * 60)

    saved_total = 0
    for i, img_path in enumerate(image_files, 1):
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"  [SKIP] Impossible de lire : {img_path.name}")
            continue

        n = process_one_image(
            model, image_bgr, img_path.stem,
            output_dir, args.conf, args.iou, args.device, args.size
        )
        saved_total += n
        print(f"  [{i:04d}/{len(image_files)}] {img_path.name:<30} → {n} segment(s)")

    print("-" * 60)
    print(f"[DONE] {saved_total} crops {args.size}×{args.size} sauvegardés → {output_dir.resolve()}")


if __name__ == "__main__":
    main()
