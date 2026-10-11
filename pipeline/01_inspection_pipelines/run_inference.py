#!/usr/bin/env python3
"""
Chewing gum inspection pipeline: background removal (rembg/u2net) + YOLOv8
detection/segmentation + unknown-item flagging.

Usage:
    python run_inference.py --input video.mp4 --weights best.pt --output out.mp4

Run setup.sh first to install matching dependencies for your GPU/CUDA version.
"""
import argparse
import time

import cv2
import numpy as np
from ultralytics import YOLO
from rembg import remove, new_session

COLOR_GUM = (0, 200, 0)       # green - known chewing gum
COLOR_UNKNOWN = (0, 0, 255)   # red - unknown foreground object


def build_foreground_mask(frame_bgr, session, alpha_thresh=30):
    """Run background removal on a single BGR frame and return a binary
    foreground mask (uint8, 0/255) the same size as the frame."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = remove(rgb, session=session)  # RGBA numpy array
    alpha = result[:, :, 3]
    mask = np.where(alpha > alpha_thresh, 255, 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def build_covered_mask(frame_shape, result, task):
    """Build a binary mask of everything YOLO already explained (gum pieces)."""
    h, w = frame_shape[:2]
    covered = np.zeros((h, w), dtype=np.uint8)

    if task == "segment" and result.masks is not None:
        for seg_mask in result.masks.data.cpu().numpy():
            m = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            covered = np.maximum(covered, (m > 0.5).astype(np.uint8) * 255)
    elif result.boxes is not None:
        for box in result.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(covered, (x1, y1), (x2, y2), 255, thickness=-1)

    return covered


def draw_label(frame, x1, y1, text, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y1 = max(y1, th + 6)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)


def process_video(input_path, output_path, weights, task="segment",
                   conf=0.4, device=0, alpha_thresh=30, min_unknown_area=250,
                   rembg_model="u2net"):

    model = YOLO(weights)

    import onnxruntime as ort
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    print(f"rembg onnxruntime providers in use: {providers}")
    session = new_session(rembg_model, providers=providers)  # try "u2netp" for a lighter/faster model

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_idx = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        fg_mask = build_foreground_mask(frame, session, alpha_thresh)
        result = model.predict(frame, conf=conf, device=device, verbose=False)[0]
        covered_mask = build_covered_mask(frame.shape, result, task)
        unknown_mask = cv2.bitwise_and(fg_mask, cv2.bitwise_not(covered_mask))

        annotated = frame.copy()
        names = result.names
        if result.boxes is not None:
            for box, cls_id, score in zip(result.boxes.xyxy.cpu().numpy(),
                                           result.boxes.cls.cpu().numpy(),
                                           result.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = box.astype(int)
                label = f"{names[int(cls_id)]} {score:.2f}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_GUM, 2)
                draw_label(annotated, x1, y1, label, COLOR_GUM)

        contours, _ = cv2.findContours(unknown_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_unknown_area:
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            cv2.rectangle(annotated, (x, y), (x + cw, y + ch), COLOR_UNKNOWN, 2)
            draw_label(annotated, x, y, "unknown item", COLOR_UNKNOWN)

        writer.write(annotated)

        if frame_idx % 25 == 0 or frame_idx == total_frames:
            elapsed = time.time() - t0
            fps_proc = frame_idx / elapsed if elapsed > 0 else 0
            print(f"[{frame_idx}/{total_frames}] {fps_proc:.1f} fps processed")

    cap.release()
    writer.release()
    print(f"Done. Saved annotated video to {output_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Bg-removal + YOLO unknown-item flagging")
    p.add_argument("--input", required=True, help="Path to input video")
    p.add_argument("--output", default="annotated_output.mp4", help="Path to output video")
    p.add_argument("--weights", required=True, help="Path to YOLO .pt weights")
    p.add_argument("--task", choices=["segment", "detect"], default="segment")
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--device", default=0, help="0 for GPU, 'cpu' for CPU")
    p.add_argument("--alpha-thresh", type=int, default=30)
    p.add_argument("--min-unknown-area", type=int, default=250)
    p.add_argument("--rembg-model", default="u2net",
                   help="u2net, u2netp, silueta, isnet-general-use")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = args.device
    if str(device).isdigit():
        device = int(device)

    process_video(
        args.input,
        args.output,
        args.weights,
        task=args.task,
        conf=args.conf,
        device=device,
        alpha_thresh=args.alpha_thresh,
        min_unknown_area=args.min_unknown_area,
        rembg_model=args.rembg_model,
    )