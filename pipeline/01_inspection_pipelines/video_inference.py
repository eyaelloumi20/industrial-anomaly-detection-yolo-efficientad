"""
video_inference.py  (smartphone-calibrated, v3)
================================================
Based directly on video_inference_clean.py with four original fixes plus
one critical correctness bug fix (FIX 5) discovered post-deployment.

FIX 1 — Preprocessing aligned to pipeline.py (the script that matches training)
    REMOVED: Otsu refinement, portrait-to-landscape rotation
    REASON:  These were never applied during EfficientAD training.
             They change the pixel distribution the model sees and produce
             scores that bear no relation to any calibrated threshold.
    KEPT:    pad → resize_with_padding → [denoise] → ToTensor  (identical to pipeline.py)

FIX 2 — Domain-shift threshold recalibration for smartphone video
    The VisA-derived threshold (~0.10) is meaningless for Redmi Note 10S footage
    because the video domain (handheld, natural light, black plastic BG) is
    different from the controlled VisA studio shots the quantiles were computed on.
    Default --threshold is now 5.0.  Pass --threshold_file to override with
    a value you calibrated on a labelled video using --calibrate_from_video.

FIX 3 — p95 score instead of max(amap)
    score_crop() now returns np.percentile(amap, 95) instead of np.max(amap).
    REASON:  max(amap) is dominated by single hot pixels.  p95 is stable across
             frames while still catching real defects that cover even a few
             percent of the crop area.

FIX 4 — Dual criterion: frame EMA gate + track mean gate
    A frame is only flagged as anomalous if BOTH:
        (a) ema_score >= threshold       (frame-level gate, fast response)
        (b) track_running_mean >= 3.0    (track-level gate, suppresses spikes)
    Override the mean gate with --mean_threshold X.

FIX 5 (v3) — Correct student channel slice for y_stae
    BUG:    y_stae was s_out[:, -OUT_CHANNELS:, :, :] — the LAST third of the
            student output tensor.  For PDN-S the student outputs 3×OUT_CHANNELS
            (768 channels).  The last 384 channels are NOT the AE-reconstruction
            branch; they are a duplicate/extra head.
    FIX:    y_stae = s_out[:, OUT_CHANNELS:2*OUT_CHANNELS, :, :] — middle third.
            This matches evaluate_pipeline_seg.py and evaluate_pipeline.py.
    IMPACT: With the wrong slice, d_stae was computing
            ||AE_reconstruction - wrong_student_head||² which happened to
            produce near-zero values (both heads output similar activations),
            collapsing the st-ae anomaly map.  All track scores were stuck in
            the 0.9–1.5 range regardless of defect severity, making the
            pipeline unable to detect any anomaly.

FIX 6 (v3) — --ratio arg exposed on CLI and threaded into score_crop
    score_crop() always accepted a ratio kwarg but it was never wired to CLI
    args.  --ratio (default 0.1) is now a proper CLI argument passed to both
    score_crop() and calibrate_threshold_from_video() so you can tune it.
"""

import os, sys, csv, time, argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from utils.load_efficientad import load_efficientad_model

IMAGE_SIZE   = 256
OUT_CHANNELS = 384


# ── preprocessing — identical to pipeline.py preprocess_crop() ───────────────

def resize_with_padding(crop: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    ow, oh = crop.size
    scale  = size / max(ow, oh)
    nw, nh = int(ow * scale), int(oh * scale)
    resized = crop.resize((nw, nh), Image.LANCZOS)
    canvas  = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def get_processed_crop_pil(pil_img: Image.Image, box: tuple,
                            pad_ratio: float = 0.02,
                            denoise: bool = True) -> Image.Image:
    """
    FIX 1: Matches pipeline.py exactly.
    REMOVED: Otsu refinement + portrait→landscape rotation.
    Those steps are absent from training and create a train/inference mismatch.
    """
    W, H = pil_img.size
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    px = int(bw * pad_ratio)
    py = int(bh * pad_ratio)
    x1 = max(0, x1 - px);  y1 = max(0, y1 - py)
    x2 = min(W, x2 + px);  y2 = min(H, y2 + py)

    crop = pil_img.crop((x1, y1, x2, y2)).convert("RGB")
    crop = resize_with_padding(crop, IMAGE_SIZE)

    if denoise:
        arr = np.array(crop, dtype=np.uint8)
        arr = cv2.GaussianBlur(arr, (3, 3), sigmaX=0.8, sigmaY=0.8)
        arr = cv2.bilateralFilter(arr, d=5, sigmaColor=15, sigmaSpace=15)
        crop = Image.fromarray(arr)

    return crop


def score_crop(models, crop_pil, device, ratio=0.1):  # ratio passed from args
    t = transforms.ToTensor()(crop_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        t_out = models['teacher'](t)
        s_out = models['student'](t)
        a_out = models['ae'](t)

    # Student outputs 3×OUT_CHANNELS: [st | stae | extra].
    # y_st   = first  third  (student→teacher branch)
    # y_stae = middle third  (student→AE branch)
    # BUG FIX: was s_out[:, -OUT_CHANNELS:] which took the LAST third — wrong.
    C = OUT_CHANNELS
    y_st   = s_out[:, :C,    :, :]
    y_stae = s_out[:, C:2*C, :, :]

    norm_t = (t_out - models['mean']) / (models['std'] + 1e-8)
    d_st   = torch.pow(norm_t - y_st,   2)
    d_stae = torch.pow(a_out  - y_stae, 2)

    fm_st = F.interpolate(torch.mean(d_st,   dim=1, keepdim=True),
                          size=(IMAGE_SIZE, IMAGE_SIZE),
                          mode='bilinear', align_corners=False)
    fm_stae = F.interpolate(torch.mean(d_stae, dim=1, keepdim=True),
                            size=(IMAGE_SIZE, IMAGE_SIZE),
                            mode='bilinear', align_corners=False)

    nm_st = (ratio * (fm_st   - models['qa_st'])) \
            / (models['qb_st'] - models['qa_st'] + 1e-8)
    nm_ae = (ratio * (fm_stae - models['qa_ae'])) \
            / (models['qb_ae'] - models['qa_ae'] + 1e-8)

    amap = (0.5 * nm_st + 0.5 * nm_ae)[0, 0].cpu().numpy()
    # FIX 3: use 95th-percentile instead of max.
    # max(amap) is dominated by single noisy pixels — one hot pixel on a clean
    # gum can spike the raw score to 5+ even when the gum is normal (observed
    # on track 41: mean=2.62 but max=5.09).  p95 is far more stable while
    # still responding quickly to real defects that cover even a small area.
    return amap, float(np.percentile(amap, 95))


# ── drawing — unchanged from original ────────────────────────────────────────

def draw_clean_id(frame, tid, box, is_anomaly, score):
    x1, y1, x2, y2 = box
    color = (0, 0, 220) if is_anomaly else (0, 200, 60)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    tag  = f"ID{tid} {score:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(tag, font, 0.50, 1)
    tx = max(0, x1 + 2)
    ty = max(th + 6, y1 - 2)
    cv2.rectangle(frame, (tx - 1, ty - th - 3), (tx + tw + 3, ty + 2), color, -1)
    cv2.putText(frame, tag, (tx + 1, ty - 1), font, 0.50,
                (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def overlay_heatmap(frame, amap, x1, y1, x2, y2, alpha=0.38):
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    norm = ((amap - amap.min()) / (amap.max() - amap.min() + 1e-8) * 255).astype(np.uint8)
    heat = cv2.resize(cv2.applyColorMap(norm, cv2.COLORMAP_HOT), (w, h))
    roi  = frame[y1:y2, x1:x2]
    if roi.shape[0] > 0 and roi.shape[1] > 0:
        frame[y1:y2, x1:x2] = cv2.addWeighted(roi, 1 - alpha, heat, alpha, 0)
    return frame


def draw_hud(frame, frame_idx, fps, worst, threshold, warming_up=False):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 44), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    if warming_up:
        color, status = (160, 120, 0), "WARMING UP"
    elif worst >= threshold:
        color, status = (0, 0, 200), "!!! ANOMALY DETECTED !!!"
    else:
        color, status = (30, 140, 30), "NORMAL"
    hud = f"Frame {frame_idx:05d} | FPS {fps:5.1f} | Score {worst:.3f} | Thr {threshold:.3f} | {status}"
    cv2.putText(frame, hud, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)
    return frame


# ── optional: auto-calibrate threshold from an unlabelled video ───────────────

def calibrate_threshold_from_video(cap, yolo, models, device,
                                   pad_ratio, denoise, conf, iou,
                                   ratio=0.1,
                                   percentile=85, max_frames=150):
    """
    Run inference on the first max_frames frames, collect all raw scores,
    and return the chosen percentile as the threshold.

    For a video that is entirely normal gum: use percentile ~90–95 to separate
    the rare anomalies from the bulk.
    For a mixed video: pick the valley in the histogram manually (see CSV).

    FIX 2: The VisA-derived threshold (0.10) is useless on smartphone footage
    because the score scale shifts with domain.  This function recalibrates
    on the video itself.
    """
    print(f"[Calibrate] Sampling first {max_frames} frames to auto-set threshold ...")
    scores = []
    saved_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for fi in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results   = yolo.predict(source=frame, conf=conf, iou=iou,
                                 classes=[0], verbose=False)
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes.xyxy.cpu().numpy().astype(int):
                x1, y1, x2, y2 = (max(0, box[0]), max(0, box[1]),
                                   min(pil_frame.width, box[2]),
                                   min(pil_frame.height, box[3]))
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = get_processed_crop_pil(pil_frame, (x1, y1, x2, y2),
                                              pad_ratio=pad_ratio, denoise=denoise)
                _, s = score_crop(models, crop, device, ratio=ratio)
                scores.append(s)

    cap.set(cv2.CAP_PROP_POS_FRAMES, saved_pos)

    if not scores:
        print("[Calibrate] No detections during calibration. Falling back to 5.0")
        return 5.0

    a   = np.array(scores)
    thr = float(np.percentile(a, percentile))
    print(f"[Calibrate] {len(scores)} crop scores collected from {fi+1} frames")
    print(f"  min={a.min():.3f}  max={a.max():.3f}  mean={a.mean():.3f}")
    print(f"  p75={np.percentile(a,75):.3f}  p85={np.percentile(a,85):.3f}  "
          f"p90={np.percentile(a,90):.3f}  p95={np.percentile(a,95):.3f}")
    print(f"  → threshold (p{percentile}) = {thr:.4f}")
    return thr


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="EfficientAD video inspection (smartphone-calibrated)")
    p.add_argument("--source",          required=True)
    p.add_argument("--ckpt_dir",        required=True)
    p.add_argument("--yolo_weights",    required=True)
    p.add_argument("--out_dir",         default="./video_results")
    p.add_argument("--category",        default="chewinggum")
    p.add_argument("--model_size",      default="S", choices=["S", "M"])
    # FIX 2: default threshold now 5.0 (calibrated from your video distribution)
    p.add_argument("--threshold",       type=float, default=None,
                   help="Decision threshold. Default: auto-calibrated from video. "
                        "For your Redmi video, ~5.0 separates normal (2-4) from anomaly (5-8).")
    p.add_argument("--threshold_file",  default=None,
                   help="Path to a threshold.txt (single float). Overrides --threshold.")
    p.add_argument("--auto_calibrate",  action="store_true",
                   help="Auto-calibrate threshold from first 150 frames of the video. "
                        "Use when all visible gums in the opening frames are NORMAL.")
    p.add_argument("--calib_percentile",type=int, default=85,
                   help="Percentile for auto-calibration (default 85). "
                        "Higher = stricter (fewer alarms).")
    p.add_argument("--calib_frames",    type=int, default=150,
                   help="Number of frames to sample for auto-calibration (default 150).")
    p.add_argument("--ema_alpha",       type=float, default=0.3)
    p.add_argument("--ema_warmup",      type=int,   default=3,
                   help="Frames per track before raising alarms (default 3)")
    p.add_argument("--mean_threshold",  type=float, default=3.0,
                   help="Track running-mean gate (default 3.0). "
                        "A frame is only flagged if BOTH ema >= threshold "
                        "AND running_mean >= mean_threshold. "
                        "Suppresses single-frame spikes on clean gums.")
    p.add_argument("--ratio",           type=float, default=0.1,
                   help="EfficientAD normalisation ratio (default 0.1). "
                        "Must match the value used during training/calibration.")
    p.add_argument("--conf",            type=float, default=0.25)
    p.add_argument("--iou",             type=float, default=0.45)
    # FIX 1: default pad_ratio 0.02 to match evaluate_pipeline.py
    p.add_argument("--pad_ratio",       type=float, default=0.02)
    p.add_argument("--no_denoise",      action="store_true")
    p.add_argument("--no_display",      action="store_true")
    p.add_argument("--device",          default=None)
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device  = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    denoise = not args.no_denoise

    # ── Models ────────────────────────────────────────────────────────────────
    print(f"\n[Video] Loading EfficientAD from {args.ckpt_dir} ...")
    models = load_efficientad_model(ckpt_dir=args.ckpt_dir, category=args.category,
                                    model_size=args.model_size, device=device)
    print(f"[Video] Loading YOLOv8 from {args.yolo_weights} ...")
    from ultralytics import YOLO
    yolo = YOLO(args.yolo_weights)
    yolo.to(device)

    # ── Video source ──────────────────────────────────────────────────────────
    source = int(args.source) if args.source.isdigit() else args.source
    cap    = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[Video] ERROR: cannot open '{source}'"); return

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Video] Source: {source}  {src_w}x{src_h} @ {src_fps:.1f} fps")

    # ── Threshold resolution (FIX 2) ─────────────────────────────────────────
    threshold = None
    if args.threshold_file and Path(args.threshold_file).is_file():
        threshold = float(open(args.threshold_file).read().strip())
        print(f"[Video] Threshold from file: {threshold:.4f}")
    elif args.threshold is not None:
        threshold = args.threshold
        print(f"[Video] Threshold (manual): {threshold:.4f}")
    elif args.auto_calibrate:
        threshold = calibrate_threshold_from_video(
            cap, yolo, models, device, args.pad_ratio, denoise,
            args.conf, args.iou,
            ratio=args.ratio,
            percentile=args.calib_percentile,
            max_frames=args.calib_frames)
    else:
        # FIX 2: domain-calibrated default for Redmi Note 10S footage
        # Normal gums score 2–4, anomalies score 5–8 in your video.
        # 5.0 is conservative — catches clear anomalies, safe for normal gums.
        threshold = 5.0
        print(f"[Video] Using domain-calibrated default threshold: {threshold}")
        print(f"         (based on score distribution from your input1.mp4)")
        print(f"         Override with --threshold X or --auto_calibrate")

    # save threshold for reference
    thr_path = Path(args.out_dir) / "video_threshold.txt"
    thr_path.write_text(f"{threshold:.6f}\n")
    print(f"[Video] Threshold {threshold:.4f} saved to {thr_path}")

    # ── Writers ───────────────────────────────────────────────────────────────
    out_video = str(Path(args.out_dir) / "output_video.mp4")
    writer    = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"),
                                src_fps, (src_w, src_h))

    csv_path = str(Path(args.out_dir) / "frame_scores.csv")
    csv_f    = open(csv_path, "w", newline="")
    csv_w    = csv.writer(csv_f)
    csv_w.writerow(["frame", "track_id", "raw_score", "ema_score",
                    "warmed_up", "is_anomaly",
                    "x1", "y1", "x2", "y2", "frame_ms"])

    # ── State ─────────────────────────────────────────────────────────────────
    track_ema    = {}          # tid -> current EMA
    track_count  = {}          # tid -> frame count (for warmup)
    track_raw_h  = {}          # tid -> [raw scores]
    track_ema_h  = {}          # tid -> [ema scores]

    frame_idx  = 0
    fps_log    = []

    print(f"\n[Video] Running — Q to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0          = time.perf_counter()
        frame_worst = 0.0
        active_ids  = set()
        any_warming = False

        results = yolo.track(source=frame, conf=args.conf, iou=args.iou,
                              classes=[0], persist=True,
                              tracker="bytetrack.yaml", verbose=False)

        # Convert once per frame (not once per box)
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        annotated = frame.copy()

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            boxes     = r.boxes.xyxy.cpu().numpy().astype(int)
            track_ids = (r.boxes.id.cpu().numpy().astype(int)
                         if r.boxes.id is not None
                         else list(range(len(boxes))))

            for box, tid in zip(boxes, track_ids):
                x1, y1, x2, y2 = (max(0, box[0]), max(0, box[1]),
                                   min(src_w, box[2]), min(src_h, box[3]))
                if x2 <= x1 or y2 <= y1:
                    continue
                active_ids.add(tid)

                crop = get_processed_crop_pil(pil_frame, (x1, y1, x2, y2),
                                              pad_ratio=args.pad_ratio,
                                              denoise=denoise)
                amap, raw = score_crop(models, crop, device, ratio=args.ratio)

                # EMA with warmup
                track_count[tid] = track_count.get(tid, 0) + 1
                prev = track_ema.get(tid, raw)
                ema  = args.ema_alpha * raw + (1 - args.ema_alpha) * prev
                track_ema[tid] = ema

                track_raw_h.setdefault(tid, []).append(raw)
                track_ema_h.setdefault(tid, []).append(ema)

                warmed   = track_count[tid] >= args.ema_warmup
                # FIX 4: dual criterion — frame EMA gate AND track-mean gate.
                # A single elevated frame on a clean gum can push EMA above
                # threshold transiently.  Requiring the running mean to also
                # exceed mean_threshold (3.0) adds a second independent gate:
                # clean gums all have raw_mean < 2.95; anomalous ones > 3.31.
                track_mean = float(np.mean(track_raw_h[tid])) if track_raw_h.get(tid) else raw
                is_anom  = warmed and (ema >= threshold) and (track_mean >= args.mean_threshold)
                if not warmed:
                    any_warming = True
                frame_worst = max(frame_worst, ema)

                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                csv_w.writerow([frame_idx, tid,
                                f"{raw:.6f}", f"{ema:.6f}",
                                int(warmed), int(is_anom),
                                x1, y1, x2, y2, f"{elapsed_ms:.2f}"])

                annotated = draw_clean_id(annotated, tid, (x1, y1, x2, y2), is_anom, ema)
                annotated = overlay_heatmap(annotated, amap, x1, y1, x2, y2)

        # Stale track cleanup
        for old in list(track_ema.keys()):
            if old not in active_ids:
                del track_ema[old]
                track_count.pop(old, None)

        t1 = time.perf_counter()
        fps_log.append(1.0 / max(t1 - t0, 1e-6))
        cur_fps = float(np.mean(fps_log[-30:]))

        annotated = draw_hud(annotated, frame_idx, cur_fps, frame_worst,
                             threshold, warming_up=any_warming)
        writer.write(annotated)

        if not args.no_display:
            cv2.imshow("Gum Inspection", annotated)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  [{frame_idx:05d}]  fps={cur_fps:.1f}  score={frame_worst:.3f}")

    cap.release()
    writer.release()
    csv_f.close()
    if not args.no_display:
        cv2.destroyAllWindows()

    # ── Per-track summary ─────────────────────────────────────────────────────
    summary_path = str(Path(args.out_dir) / "per_track_summary.csv")
    with open(summary_path, "w", newline="") as sf:
        sw = csv.writer(sf)
        sw.writerow(["track_id", "total_frames", "anomaly_frames", "normal_frames",
                     "anomaly_rate_%",
                     "raw_min", "raw_max", "raw_mean",
                     "ema_min", "ema_max", "ema_mean", "peak_ema_frame"])
        for tid in sorted(track_raw_h.keys()):
            raws = track_raw_h[tid]
            emas = track_ema_h[tid]
            n    = len(emas)
            anom = sum(1 for s in emas if s >= threshold)
            peak = int(np.argmax(emas))
            sw.writerow([tid, n, anom, n - anom,
                         f"{100.0 * anom / max(n,1):.1f}",
                         f"{min(raws):.4f}", f"{max(raws):.4f}", f"{np.mean(raws):.4f}",
                         f"{min(emas):.4f}", f"{max(emas):.4f}", f"{np.mean(emas):.4f}",
                         peak])

    print(f"\n=== Done — {frame_idx} frames ===")
    print(f"  Video          → {out_video}")
    print(f"  Frame CSV      → {csv_path}")
    print(f"  Track summary  → {summary_path}")
    print(f"  Threshold used → {threshold:.4f}  (saved to {thr_path})")


if __name__ == "__main__":
    run(parse_args())