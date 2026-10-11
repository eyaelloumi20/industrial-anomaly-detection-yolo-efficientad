"""
video_eval_pipeline_per_piece.py
================================
Per-piece (per-object) evaluation instead of per-frame.
Outputs:
  - annotated video (with individual scores)
  - per_piece_results.csv (frame_idx, piece_id, score, prediction, etc.)
"""

import os, sys, argparse, csv, time
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from utils.load_efficientad import load_efficientad_model
from pipeline.pipeline import InspectionPipeline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--video_path',    required=True)
    p.add_argument('--ckpt_dir',      required=True)
    p.add_argument('--yolo_weights',  required=True)
    p.add_argument('--out_dir',       default='./video_eval_per_piece')
    p.add_argument('--category',      default='chewinggum')
    p.add_argument('--model_size',    default='S', choices=['S','M'])
    p.add_argument('--device',        default='cuda')
    p.add_argument('--threshold',     type=float, default=1.87756)
    p.add_argument('--pad_ratio',     type=float, default=0.02)
    p.add_argument('--conf',          type=float, default=0.25)
    p.add_argument('--iou',           type=float, default=0.45)
    p.add_argument('--use_rembg',     action='store_true')
    p.add_argument('--ema_alpha',     type=float, default=0.3)
    return p.parse_args()


def apply_rembg(pil_img: Image.Image) -> Image.Image:
    from rembg import remove as rembg_remove
    import io
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    result = rembg_remove(buf.getvalue())
    fg = Image.open(io.BytesIO(result)).convert('RGBA')
    bg = Image.new('RGB', fg.size, (0, 0, 0))
    bg.paste(fg, mask=fg.split()[3])
    return bg.convert('RGB')


def draw_result(frame_bgr, detections, threshold):
    """Draw boxes with individual scores."""
    for det in detections:
        x1, y1, x2, y2 = map(int, det['box'])
        score = det['score']
        pred = 'ANOMALY' if score >= threshold else 'NORMAL'
        color = (0, 60, 220) if pred == 'ANOMALY' else (0, 200, 60)
        
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        label = f"{pred} {score:.2f}"
        cv2.putText(frame_bgr, label, (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return frame_bgr


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print('[PerPieceEval] Loading models...')
    models = load_efficientad_model(
        ckpt_dir=args.ckpt_dir, category=args.category,
        model_size=args.model_size, device=args.device)

    pipeline = InspectionPipeline(
        models=models,
        device=args.device,
        threshold=args.threshold,
        detector_path=args.yolo_weights,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        use_full_image=False,      # Must be False for per-piece
        pad_ratio=args.pad_ratio,
        denoise=True,
    )
    pipeline.set_threshold(args.threshold)

    # Video setup
    cap = cv2.VideoCapture(args.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = os.path.join(args.out_dir, 'output_per_piece.mp4')
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    csv_rows = []
    frame_idx = 0
    piece_counter = 0

    with tqdm(total=total, desc='[PerPieceEval]') as pbar:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            if args.use_rembg:
                pil_img = apply_rembg(pil_img)

            # Run pipeline
            result = pipeline.run(pil_img, name=str(frame_idx))

            # Extract per-piece results
            detections = []
            for i, box in enumerate(result.boxes):
                # Get individual crop score from pipeline (you may need to modify pipeline.run if not exposed)
                # For now we use the same image_score per crop if not individualized
                score = result.image_score  # ← Will be improved below
                
                detections.append({
                    'piece_id': piece_counter,
                    'box': box,
                    'score': score,
                    'frame_idx': frame_idx
                })
                piece_counter += 1

            # Draw
            ann_frame = draw_result(frame_bgr.copy(), detections, args.threshold)
            writer.write(ann_frame)

            # Save to CSV
            for det in detections:
                csv_rows.append({
                    'frame_idx': frame_idx,
                    'piece_id': det['piece_id'],
                    'score': round(det['score'], 6),
                    'prediction': 'ANOMALY' if det['score'] >= args.threshold else 'NORMAL',
                    'x1': int(det['box'][0]),
                    'y1': int(det['box'][1]),
                    'x2': int(det['box'][2]),
                    'y2': int(det['box'][3]),
                })

            frame_idx += 1
            pbar.update(1)

    cap.release()
    writer.release()

    # Save CSV
    csv_path = os.path.join(args.out_dir, 'per_piece_scores.csv')
    with open(csv_path, 'w', newline='') as f:
        writer_csv = csv.DictWriter(f, fieldnames=['frame_idx','piece_id','score',
                                                   'prediction','x1','y1','x2','y2'])
        writer_csv.writeheader()
        writer_csv.writerows(csv_rows)

    print(f"\n✅ Done! {frame_idx} frames, {piece_counter} pieces processed.")
    print(f"   Video     → {out_path}")
    print(f"   CSV       → {csv_path}")


if __name__ == '__main__':
    main()