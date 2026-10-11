"""
realtime_inference.py
---------------------
Run a YOLOv8 segmentation model on a video and display + save the annotated output.

Usage:
    python realtime_inference.py --model /home/ahmed/two-stage-efficientad-inspection/sam_yolo_pipeline/runs/segment/runs/segment/chewinggum_seg_v1/weights/best.pt --video video_ch4.mp4

Controls (live window):
    Q / ESC  → quit
    SPACE    → pause / resume
    S        → save screenshot

Requirements:
    pip install ultralytics opencv-python numpy
"""

import argparse
import time
import os
import cv2
import numpy as np
from ultralytics import YOLO


MASK_ALPHA = 0.40
MASK_COLOR = (130, 100, 255)
BOX_COLOR  = (255, 80,  80)
TEXT_COLOR = (255, 255, 255)
FONT       = cv2.FONT_HERSHEY_SIMPLEX


def draw_detections(frame, result, conf_thresh):
    if result.masks is None:
        return frame, 0
    overlay   = frame.copy()
    polygons  = result.masks.xy
    confs     = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    names     = result.names
    count     = 0
    for i, poly in enumerate(polygons):
        c = float(confs[i])
        if c < conf_thresh or len(poly) < 3:
            continue
        cls_name = names.get(int(class_ids[i]), "?")
        pts      = poly.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], MASK_COLOR)
        cv2.polylines(frame, [pts], isClosed=True, color=BOX_COLOR, thickness=2)
        x, y = int(pts[:, 0, 0].min()), int(pts[:, 0, 1].min())
        label = f"{cls_name} {c:.2f}"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, 1)
        cv2.rectangle(frame, (x, y - th - 6), (x + tw + 4, y), BOX_COLOR, -1)
        cv2.putText(frame, label, (x + 2, y - 3), FONT, 0.55, TEXT_COLOR, 1, cv2.LINE_AA)
        count += 1
    cv2.addWeighted(overlay, MASK_ALPHA, frame, 1 - MASK_ALPHA, 0, frame)
    return frame, count


def draw_hud(frame, fps, count, conf, paused):
    h, w = frame.shape[:2]
    lines = [f"FPS : {fps:5.1f}", f"Gum : {count}", f"Conf: {conf:.2f}"]
    if paused:
        lines.append("PAUSED")
    x0, y0 = w - 160, 12
    for j, line in enumerate(lines):
        color = (0, 80, 255) if line == "PAUSED" else (220, 220, 220)
        cv2.putText(frame, line, (x0, y0 + j * 22), FONT, 0.6, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (x0, y0 + j * 22), FONT, 0.6, color,   1, cv2.LINE_AA)


def run(model_path, video_path, conf, iou, frame_skip, width, output_path, screenshot_dir):
    model = YOLO(model_path)
    src   = int(video_path) if video_path.isdigit() else video_path
    cap   = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_path}")

    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Final frame size (after optional resize)
    if width > 0:
        out_w = width
        out_h = int(src_h * width / src_w)
    else:
        out_w, out_h = src_w, src_h

    # ── Video writer ──────────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, src_fps, (out_w, out_h))
    print(f"Source : {video_path}  ({total} frames @ {src_fps:.1f} fps)")
    print(f"Output : {output_path}  ({out_w}x{out_h})")
    print(f"Controls: Q/ESC=quit  SPACE=pause  S=screenshot")

    os.makedirs(screenshot_dir, exist_ok=True)
    paused       = False
    frame_idx    = 0
    screenshot_n = 0
    fps_display  = 0.0
    t_prev       = time.perf_counter()
    frame_draw   = None
    last_count   = 0

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('s') and frame_draw is not None:
            shot = os.path.join(screenshot_dir, f"shot_{screenshot_n:04d}.jpg")
            cv2.imwrite(shot, frame_draw)
            screenshot_n += 1
            print(f"Screenshot: {shot}")

        if paused:
            if frame_draw is not None:
                cv2.imshow("YOLOv8  [Q=quit | SPACE=pause | S=screenshot]", frame_draw)
            continue

        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        if width > 0:
            frame = cv2.resize(frame, (out_w, out_h))

        if frame_idx % frame_skip != 0:
            # Write original (unannotated) frame to keep video in sync
            writer.write(frame)
            frame_idx += 1
            continue

        # ── Inference ─────────────────────────────────────────────────────────
        results = model(frame, conf=conf, iou=iou, verbose=False)
        frame_draw, last_count = draw_detections(frame.copy(), results[0], conf)

        t_now       = time.perf_counter()
        instant_fps = 1.0 / max(t_now - t_prev, 1e-6)
        fps_display = 0.8 * fps_display + 0.2 * instant_fps
        t_prev      = t_now

        draw_hud(frame_draw, fps_display, last_count, conf, paused=False)

        # Write to output video
        writer.write(frame_draw)

        # Show live
        cv2.imshow("YOLOv8  [Q=quit | SPACE=pause | S=screenshot]", frame_draw)

        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx}/{total}  FPS={fps_display:.1f}  detections={last_count}")

        frame_idx += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"\nDone! Annotated video saved to: {output_path}")


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8-seg real-time inference + video output")
    p.add_argument("--model",      required=True)
    p.add_argument("--video",      required=True)
    p.add_argument("--output",     default="output_annotated.mp4",
                   help="Path for the saved output video (default: output_annotated.mp4)")
    p.add_argument("--conf",       type=float, default=0.25)
    p.add_argument("--iou",        type=float, default=0.45)
    p.add_argument("--frame_skip", type=int,   default=1,
                   help="Run inference every Nth frame (default: 1)")
    p.add_argument("--width",      type=int,   default=0,
                   help="Resize width before inference (0=original)")
    p.add_argument("--screenshot_dir", default="screenshots")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        model_path     = args.model,
        video_path     = args.video,
        conf           = args.conf,
        iou            = args.iou,
        frame_skip     = args.frame_skip,
        width          = args.width,
        output_path    = args.output,
        screenshot_dir = args.screenshot_dir,
    )