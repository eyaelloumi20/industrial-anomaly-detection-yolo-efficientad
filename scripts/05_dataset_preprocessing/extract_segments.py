#!/usr/bin/env python3
"""
extract_segments.py
--------------------
Extrait tous les polygones de segmentation d'instance YOLOv11 depuis une vidéo
et sauvegarde chaque crop dans une image PNG avec fond transparent.

Usage:
    python extract_segments.py --video input.mp4 --model yolo11n-seg.pt --output ./output
    python extract_segments.py --video input.mp4 --model yolo11n-seg.pt --output ./output --conf 0.5
    python extract_segments.py --video input.mp4 --model yolo11n-seg.pt --output ./output --every-n-frames 5
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extrait les polygones YOLO11 segmentation d'une vidéo → PNG transparent"
    )
    parser.add_argument(
        "--video", required=True,
        help="Chemin vers la vidéo d'entrée"
    )
    parser.add_argument(
        "--model", required=True,
        help="Chemin vers le modèle YOLO11 seg (.pt), ex: yolo11n-seg.pt"
    )
    parser.add_argument(
        "--output", default="./output_segments",
        help="Dossier de sortie pour les PNG (défaut: ./output_segments)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Seuil de confiance (défaut: 0.25)"
    )
    parser.add_argument(
        "--iou", type=float, default=0.45,
        help="Seuil IoU pour NMS (défaut: 0.45)"
    )
    parser.add_argument(
        "--every-n-frames", type=int, default=1,
        help="Traiter 1 frame sur N (défaut: 1 = toutes les frames)"
    )
    parser.add_argument(
        "--device", default="",
        help="Device: '' = auto, 'cpu', '0' = GPU 0 (défaut: auto)"
    )
    return parser.parse_args()


def mask_to_polygon_crop(frame_bgr: np.ndarray, mask_xy: np.ndarray) -> np.ndarray:
    """
    Découpe le polygone exact (mask_xy) depuis frame_bgr et retourne
    une image RGBA avec fond transparent.

    Args:
        frame_bgr : frame OpenCV (H, W, 3)
        mask_xy   : tableau (N, 2) de coordonnées pixel du polygone

    Returns:
        crop RGBA (h_crop, w_crop, 4) avec transparence hors polygone
    """
    pts = mask_xy.astype(np.int32)

    # Bounding box du polygone
    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()

    # Clamp aux dimensions de la frame
    H, W = frame_bgr.shape[:2]
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(W - 1, x_max)
    y_max = min(H - 1, y_max)

    # Crop BGR
    crop_bgr = frame_bgr[y_min:y_max + 1, x_min:x_max + 1]

    if crop_bgr.size == 0:
        return None

    # Construire le masque sur la bbox
    pts_shifted = pts - np.array([x_min, y_min])
    h_crop = y_max - y_min + 1
    w_crop = x_max - x_min + 1
    binary_mask = np.zeros((h_crop, w_crop), dtype=np.uint8)
    cv2.fillPoly(binary_mask, [pts_shifted.reshape(-1, 1, 2)], 255)

    # Convertir BGR → RGBA
    crop_rgba = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)
    # Canal alpha = masque binaire (0 = transparent, 255 = opaque)
    crop_rgba[:, :, 3] = binary_mask

    return crop_rgba


def main():
    args = parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[ERREUR] Vidéo introuvable: {video_path}")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Charger le modèle
    print(f"[INFO] Chargement du modèle: {args.model}")
    model = YOLO(args.model)

    # Ouvrir la vidéo
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERREUR] Impossible d'ouvrir la vidéo: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Vidéo: {total_frames} frames @ {fps:.1f} FPS")
    print(f"[INFO] Traitement: 1 frame sur {args.every_n_frames}")
    print(f"[INFO] Confiance: {args.conf} | IoU: {args.iou}")
    print(f"[INFO] Sortie: {output_dir.resolve()}")
    print("-" * 60)

    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Sauter les frames selon --every-n-frames
        if (frame_idx - 1) % args.every_n_frames != 0:
            continue

        # Inférence YOLO
        results = model.predict(
            source=frame,
            conf=args.conf,
            iou=args.iou,
            device=args.device if args.device else None,
            verbose=False,
        )

        result = results[0]

        # Vérifier qu'il y a des masques
        if result.masks is None or len(result.masks.xy) == 0:
            continue

        polygons = result.masks.xy          # liste de (N_pts, 2)
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        names = result.names

        for det_idx, (polygon, cls_id, conf_val) in enumerate(
            zip(polygons, class_ids, confs)
        ):
            if len(polygon) < 3:
                continue  # polygone dégénéré

            crop_rgba = mask_to_polygon_crop(frame, polygon)
            if crop_rgba is None:
                continue

            class_name = names[cls_id]
            filename = (
                f"frame{frame_idx:06d}"
                f"_det{det_idx:03d}"
                f"_{class_name}"
                f"_conf{conf_val:.2f}"
                ".png"
            )
            save_path = output_dir / filename
            cv2.imwrite(str(save_path), crop_rgba)
            saved_count += 1

        if frame_idx % 100 == 0:
            progress = frame_idx / total_frames * 100 if total_frames > 0 else 0
            print(f"  Frame {frame_idx}/{total_frames} ({progress:.1f}%) "
                  f"— {saved_count} segments sauvegardés")

    cap.release()
    print("-" * 60)
    print(f"[DONE] {saved_count} segments extraits → {output_dir.resolve()}")


if __name__ == "__main__":
    main()
