"""
dump_all_frames.py
==================
Process every video in a folder, and for each video save:
  - frames/            all annotated full frames
  - tracks/            one folder per tracked object (piece):
                         crop_XXXXX.png    crops of this piece at each frame
                         montage.png       all crops in a grid
                         scores.png        score/ema vs frame
                         scores.csv        per-frame score + box geometry
                         summary.json      frame range, max score, padding stats
  - suspicious_tracks.png   top-N most anomalous tracks
  - records.jsonl      per-frame scores/alarms
  - summary.json       video-level totals

NEW in this version
-------------------
scores.csv now includes:
    box_w, box_h, aspect_ratio, padding_frac
summary.json now includes:
    mean_padding_frac, max_padding_frac, mean_aspect_ratio,
    mean_box_w, mean_box_h

These let evaluate_all.py filter out crops that are heavily padded
(aspect ratio far from 1:1), which the montages showed inflate scores
purely from the black padding, not from any real defect.

Usage:
    python dump_all_frames.py \
        --video_dir /home/ahmed/two-stage-efficientad-inspection/video_test \
        --yolo_det_weights ./runs/.../best.pt \
        --method lowres --ckpt_dir ./ckpt_192 \
        --thresholds_json ./video_calibration/video_thresholds_lowres.json \
        --category chewinggum --model_size S \
        --clip_margin 5 --min_box_frac 0.5 --fp16 \
        --save_dir ./tracks_v3
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import csv
import json
import time
import types
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from torchvision import transforms

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


# ===========================================================================
# dtype-safe patch
# ===========================================================================
import utils.models as _models_module


def _dtype_preserving_norm(x):
    mean = torch.tensor([0.485, 0.456, 0.406],
                        dtype=x.dtype, device=x.device)[None, :, None, None]
    std  = torch.tensor([0.229, 0.224, 0.225],
                        dtype=x.dtype, device=x.device)[None, :, None, None]
    return (x - mean) / (std + 1e-11)


_models_module.imagenet_norm_batch = _dtype_preserving_norm


def patch_forward_dtype_safe(model, kind):
    if kind in ("teacher", "student"):
        def _f(self, x):
            x = _dtype_preserving_norm(x)
            return self.pdn(x)
    else:
        def _f(self, x):
            x = _dtype_preserving_norm(x)
            x = self.encoder(x)
            x = self.decoder(x)
            return x
    model.forward = types.MethodType(_f, model)
    return model


from compare_pipeline_methods_batched import (
    load_bundle, get_crop, run_yolo_once,
    draw_unknown_contours, OUT_CHANNELS,
)


# ===========================================================================
# Clipped-box filter
# ===========================================================================
def classify_boxes(frame_dets, src_w, src_h, clip_margin, min_box_frac):
    flags = []
    if not frame_dets:
        return flags
    areas = np.asarray(
        [(x2 - x1) * (y2 - y1) for _, x1, y1, x2, y2 in frame_dets],
        dtype=np.float64)
    use_size_test = (min_box_frac > 0) and (len(frame_dets) >= 3)
    med = float(np.median(areas)) if use_size_test else 0.0
    for i, (_tid, x1, y1, x2, y2) in enumerate(frame_dets):
        edge = (x1 <= clip_margin or
                y1 <= clip_margin or
                x2 >= src_w - clip_margin or
                y2 >= src_h - clip_margin)
        size = use_size_test and (areas[i] < min_box_frac * med)
        flags.append(bool(edge or size))
    return flags


# ===========================================================================
# Scoring helper
# ===========================================================================
@torch.no_grad()
def score_batch(models, crops_pil, device, image_size, model_dtype, ratio):
    if not crops_pil:
        return []
    tensors = [transforms.ToTensor()(c) for c in crops_pil]
    t = torch.stack(tensors, dim=0).to(device)
    if t.dtype != model_dtype:
        t = t.to(model_dtype)
    t = t.to(memory_format=torch.channels_last)

    t_out = models["teacher"](t)
    s_out = models["student"](t)
    a_out = models["ae"](t)

    y_st   = s_out[:, :OUT_CHANNELS,  :, :]
    y_stae = s_out[:, -OUT_CHANNELS:, :, :]

    norm_t = (t_out - models["mean"]) / (models["std"] + 1e-8)
    d_st   = torch.pow(norm_t - y_st,   2)
    d_stae = torch.pow(a_out  - y_stae, 2)

    fm_st   = F.interpolate(torch.mean(d_st,   dim=1, keepdim=True),
                            size=(image_size, image_size),
                            mode="bilinear", align_corners=False)
    fm_stae = F.interpolate(torch.mean(d_stae, dim=1, keepdim=True),
                            size=(image_size, image_size),
                            mode="bilinear", align_corners=False)
    fm_st, fm_stae = fm_st.float(), fm_stae.float()
    qa_st = models["qa_st"].float(); qb_st = models["qb_st"].float()
    qa_ae = models["qa_ae"].float(); qb_ae = models["qb_ae"].float()

    nm_st = (ratio * (fm_st   - qa_st)) / (qb_st - qa_st + 1e-8)
    nm_ae = (ratio * (fm_stae - qa_ae)) / (qb_ae - qa_ae + 1e-8)
    amap  = (0.5 * nm_st + 0.5 * nm_ae)[:, 0].cpu().numpy()
    return [float(np.max(amap[i])) for i in range(amap.shape[0])]


# ===========================================================================
# Image/montage helpers
# ===========================================================================
def save_img(img, path, as_jpg=False, jpg_quality=85):
    if as_jpg:
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
    else:
        cv2.imwrite(path, img)


def build_montage(pil_crops, labels, tile=160, cols=6,
                  bg_color=(32, 32, 32)):
    n = len(pil_crops)
    if n == 0:
        return np.full((tile, tile, 3), bg_color, dtype=np.uint8)
    rows = (n + cols - 1) // cols
    canvas = np.full((rows * (tile + 26), cols * tile, 3),
                     bg_color, dtype=np.uint8)
    for i, (crop_pil, label) in enumerate(zip(pil_crops, labels)):
        r, c = i // cols, i % cols
        thumb = crop_pil.resize((tile, tile), Image.LANCZOS)
        arr = cv2.cvtColor(np.array(thumb), cv2.COLOR_RGB2BGR)
        y0, x0 = r * (tile + 26), c * tile
        canvas[y0:y0 + tile, x0:x0 + tile] = arr
        cv2.putText(canvas, label, (x0 + 4, y0 + tile + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (230, 230, 230), 1, cv2.LINE_AA)
    return canvas


def build_track_score_plot(frames, scores, emas, tau_p, tau_f, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(8, 2.5), facecolor="#111")
    ax.set_facecolor("#111")
    ax.plot(frames, scores, lw=1.2, label="raw score", color="#4ea1ff")
    ax.plot(frames, emas,   lw=1.6, label="EMA",       color="#ffa64e")
    ax.axhline(tau_p, color="#888", ls="--", lw=0.8,
               label=f"tau_p = {tau_p:.2f}")
    ax.axhline(tau_f, color="#f66", ls="--", lw=0.8,
               label=f"tau_f = {tau_f:.2f}")
    ax.set_xlabel("frame", color="#ddd")
    ax.set_ylabel("score", color="#ddd")
    ax.tick_params(colors="#aaa")
    for s in ax.spines.values():
        s.set_color("#444")
    ax.legend(loc="upper right", fontsize=7,
              facecolor="#222", edgecolor="#444", labelcolor="#ddd")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, facecolor="#111")
    plt.close(fig)


# ===========================================================================
# CLI
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video_dir", required=True)
    p.add_argument("--yolo_det_weights", required=True)
    p.add_argument("--method", required=True,
                   choices=["baseline", "lowres", "thin", "qat", "pruned"])
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--thresholds_json", required=True)
    p.add_argument("--category", default="chewinggum")
    p.add_argument("--model_size", default="S", choices=["S", "M"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ratio",     type=float, default=0.1)
    p.add_argument("--conf",      type=float, default=0.25)
    p.add_argument("--iou",       type=float, default=0.45)
    p.add_argument("--pad_ratio", type=float, default=0.02)
    p.add_argument("--denoise",   action="store_true")
    p.add_argument("--fp16",      action="store_true")
    p.add_argument("--ema_alpha", type=float, default=0.3)
    p.add_argument("--persist_K", type=int, default=3)
    p.add_argument("--clear_M",   type=int, default=5)
    p.add_argument("--gamma",     type=float, default=0.5)

    p.add_argument("--clip_margin",  type=int,   default=5)
    p.add_argument("--min_box_frac", type=float, default=0.5)

    p.add_argument("--max_frames", type=int, default=0)

    p.add_argument("--jpg", action="store_true",
                   help="Save .jpg (smaller) instead of .png")
    p.add_argument("--jpg_quality", type=int, default=85)
    p.add_argument("--save_scale",  type=float, default=1.0)

    p.add_argument("--top_tracks", type=int, default=20)
    p.add_argument("--no_tracks", action="store_true",
                   help="Skip the per-track dump (only save per-frame PNGs).")

    p.add_argument("--no_detect_unknown", action="store_true")
    p.add_argument("--otsu_max_area_frac", type=float, default=0.25)
    p.add_argument("--otsu_min_unknown_area", type=int, default=20)

    p.add_argument("--save_dir", required=True)
    return p.parse_args()


# ===========================================================================
# Per-video processing
# ===========================================================================
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".webm"}


def process_video(video_path, models, image_size, device, model_dtype,
                  args, save_root):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir     = os.path.join(save_root, video_name)
    frames_dir  = os.path.join(out_dir, "frames")
    tracks_dir  = os.path.join(out_dir, "tracks")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(tracks_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  {video_name}")
    print(f"{'='*70}")

    detections, src_w, src_h, src_fps, n_frames, frame_cache = run_yolo_once(
        video_path, args.yolo_det_weights, args.conf, args.iou,
        device=args.device,
        max_frames=(args.max_frames or None),
        detect_unknown=not args.no_detect_unknown,
        otsu_max_area_frac=args.otsu_max_area_frac,
        cache_frames=True)

    print(f"  {n_frames} frames, {src_w}x{src_h}")

    with open(args.thresholds_json) as f:
        thr = json.load(f)
    tau_p = float(thr["tau_p"])
    tau_f = float(thr.get("tau_f", thr["tau_p"]))

    ema_hist = {}
    alarm = persist = clear = 0
    records = []
    n_anomalous_frames = 0

    track_records = {}
    track_crop_cache = {}

    ext = ".jpg" if args.jpg else ".png"
    t0 = time.perf_counter()

    for frame_idx in range(n_frames):
        frame_bgr = frame_cache.bgr_frames[frame_idx]
        pil_frame = frame_cache.pil_frames[frame_idx]
        frame_dets = detections.get(frame_idx, [])

        annot = frame_bgr.copy()
        mask = frame_cache.unknown_masks[frame_idx] \
               if frame_cache.unknown_masks else None
        if not args.no_detect_unknown and mask is not None:
            draw_unknown_contours(annot, mask, args.otsu_min_unknown_area)

        clipped = classify_boxes(
            frame_dets, src_w, src_h, args.clip_margin, args.min_box_frac)

        pil_list = []
        meta_list = []
        for k, ((tid, x1, y1, x2, y2), is_clip) in enumerate(
                zip(frame_dets, clipped)):
            if is_clip:
                continue
            crop = get_crop(pil_frame, (x1, y1, x2, y2),
                            image_size, args.pad_ratio, args.denoise)
            pil_list.append(crop)
            meta_list.append((k, tid, (x1, y1, x2, y2)))

        scores = score_batch(models, pil_list, device, image_size,
                             model_dtype, args.ratio)

        emas = [None] * len(frame_dets)
        for (k, tid, box), crop, s in zip(meta_list, pil_list, scores):
            prev = ema_hist.get(tid, s)
            e = args.ema_alpha * s + (1 - args.ema_alpha) * prev
            ema_hist[tid] = e
            emas[k] = e

            if not args.no_tracks:
                track_dir = os.path.join(tracks_dir, f"track_{tid:06d}")
                os.makedirs(track_dir, exist_ok=True)
                crop_path = os.path.join(track_dir,
                                         f"crop_{frame_idx:05d}{ext}")
                save_img(np.array(crop)[:, :, ::-1], crop_path,
                         as_jpg=args.jpg, jpg_quality=args.jpg_quality)
                track_crop_cache[(tid, frame_idx)] = crop_path

            track_records.setdefault(tid, []).append({
                "frame":   frame_idx,
                "box":     list(box),
                "score":   float(s),
                "ema":     float(e),
                "clipped": False,
            })

        for k, ((tid, x1, y1, x2, y2), is_clip) in enumerate(
                zip(frame_dets, clipped)):
            if not is_clip:
                continue
            track_records.setdefault(tid, []).append({
                "frame":   frame_idx,
                "box":     [x1, y1, x2, y2],
                "score":   None,
                "ema":     None,
                "clipped": True,
            })

        valid_emas = [e for e in emas if e is not None]
        frame_max = float(max(valid_emas)) if valid_emas else 0.0
        raw_alarm = (frame_max >= tau_f)

        if raw_alarm:
            persist += 1; clear = 0
        else:
            clear += 1; persist = 0

        just_started = just_ended = False
        if alarm == 0 and persist >= args.persist_K:
            alarm = 1; just_started = True
        elif (alarm == 1
              and clear >= args.clear_M
              and frame_max < args.gamma * tau_f):
            alarm = 0; just_ended = True

        if alarm:
            n_anomalous_frames += 1

        for k, ((tid, x1, y1, x2, y2), e) in enumerate(zip(frame_dets, emas)):
            if e is None:
                cv2.rectangle(annot, (x1, y1), (x2, y2),
                              (128, 128, 128), 2)
                label = f"ID{tid} CLIPPED"
                color = (128, 128, 128)
            else:
                verdict = "ANOMALY" if e >= tau_p else "NORMAL"
                color = (0, 0, 255) if verdict == "ANOMALY" else (0, 200, 0)
                cv2.rectangle(annot, (x1, y1), (x2, y2), color, 2)
                label = f"ID{tid} {verdict} {e:.2f}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            tx, ty = max(0, x1 + 3), max(th + 6, y1 + 3)
            cv2.rectangle(annot, (tx - 2, ty - th - 4),
                          (tx + tw + 4, ty + 2), color, -1)
            cv2.putText(annot, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)

        banner_color = (0, 0, 255) if alarm else (0, 200, 0)
        banner = (f"{video_name} | frame {frame_idx} | "
                  f"max={frame_max:.3f} tau_f={tau_f:.3f} | alarm={alarm}")
        cv2.putText(annot, banner, (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, banner_color, 2)

        if args.save_scale != 1.0:
            annot = cv2.resize(annot, None,
                               fx=args.save_scale, fy=args.save_scale,
                               interpolation=cv2.INTER_AREA)

        save_img(annot,
                 os.path.join(frames_dir, f"frame_{frame_idx:05d}{ext}"),
                 as_jpg=args.jpg, jpg_quality=args.jpg_quality)

        records.append({
            "frame": frame_idx,
            "frame_max": float(frame_max),
            "n_crops_scored": len(valid_emas),
            "n_clipped": int(sum(clipped)),
            "n_dets": len(frame_dets),
            "raw_alarm": bool(raw_alarm),
            "alarm": int(alarm),
            "just_started": just_started,
            "just_ended": just_ended,
        })

        if just_started:
            print(f"  >>> ALARM START at frame {frame_idx} "
                  f"(frame_max={frame_max:.3f})")
        if just_ended:
            print(f"  <<< ALARM END   at frame {frame_idx}")

        if (frame_idx + 1) % 50 == 0:
            dt = time.perf_counter() - t0
            print(f"    frame {frame_idx + 1}/{n_frames}  "
                  f"({(frame_idx + 1) / dt:.1f} fps)")

    # ── Per-track montages, score CSVs, and summaries ────────────────────
    track_summaries = []
    if not args.no_tracks and track_records:
        print(f"\n  Building per-track montages ({len(track_records)} tracks)...")
        for tid, recs in track_records.items():
            track_dir = os.path.join(tracks_dir, f"track_{tid:06d}")
            os.makedirs(track_dir, exist_ok=True)

            # --- Write per-frame score trajectory as CSV -------------------
            csv_path = os.path.join(track_dir, "scores.csv")
            with open(csv_path, "w", newline="") as cf:
                cw = csv.writer(cf)
                cw.writerow(["frame", "score", "ema",
                             "box_x1", "box_y1", "box_x2", "box_y2",
                             "clipped",
                             "box_w", "box_h", "aspect_ratio",
                             "padding_frac"])
                for r in recs:
                    x1, y1, x2, y2 = r["box"]
                    bw = max(1, int(x2 - x1))
                    bh = max(1, int(y2 - y1))
                    ar = max(bw, bh) / min(bw, bh)
                    pfrac = 1.0 - min(bw, bh) / max(bw, bh)
                    cw.writerow([
                        r["frame"],
                        "" if r["score"] is None else f"{r['score']:.6f}",
                        "" if r["ema"]   is None else f"{r['ema']:.6f}",
                        x1, y1, x2, y2,
                        1 if r.get("clipped") else 0,
                        bw, bh, f"{ar:.4f}", f"{pfrac:.4f}",
                    ])

            # --- Montage of this track's non-clipped crops ----------------
            vis = [r for r in recs if not r["clipped"]]
            if not vis:
                continue

            crops = []
            labels = []
            for r in vis:
                p = track_crop_cache.get((tid, r["frame"]))
                if p is None or not os.path.exists(p):
                    continue
                img = cv2.imread(p, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                crops.append(
                    Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                labels.append(f"f{r['frame']} s={r['score']:.2f}")

            if crops:
                montage = build_montage(crops, labels, tile=150, cols=6)
                cv2.imwrite(os.path.join(track_dir, "montage.png"), montage)

            # --- Score-vs-time plot --------------------------------------
            frames_ = [r["frame"] for r in vis]
            scores_ = [r["score"] for r in vis]
            emas_   = [r["ema"]   for r in vis]
            build_track_score_plot(frames_, scores_, emas_, tau_p, tau_f,
                                   os.path.join(track_dir, "scores.png"))

            # --- Per-track aggregate box statistics ----------------------
            valid_boxes = [(r["box"][2] - r["box"][0],
                            r["box"][3] - r["box"][1])
                           for r in vis]
            if valid_boxes:
                ws = [max(1, int(w)) for w, _ in valid_boxes]
                hs = [max(1, int(h)) for _, h in valid_boxes]
                ars = [max(w, h) / min(w, h) for w, h in valid_boxes]
                pfrs = [1 - min(w, h) / max(w, h) for w, h in valid_boxes]
                mean_padding_frac = float(np.mean(pfrs))
                max_padding_frac  = float(np.max(pfrs))
                mean_aspect_ratio = float(np.mean(ars))
                mean_box_w = float(np.mean(ws))
                mean_box_h = float(np.mean(hs))
            else:
                mean_padding_frac = max_padding_frac = 0.0
                mean_aspect_ratio = 1.0
                mean_box_w = mean_box_h = 0.0

            max_score = float(np.max(scores_)) if scores_ else 0.0
            max_ema   = float(np.max(emas_))   if emas_   else 0.0
            first_f   = int(min(frames_)) if frames_ else -1
            last_f    = int(max(frames_)) if frames_ else -1
            n_frames_track = len(vis)

            summary = {
                "track_id":      int(tid),
                "n_frames_seen": n_frames_track,
                "first_frame":   first_f,
                "last_frame":    last_f,
                "max_score":     max_score,
                "max_ema":       max_ema,
                "any_above_tau_p": bool(max_ema >= tau_p),
                "any_above_tau_f": bool(max_ema >= tau_f),
                "n_clipped_frames": sum(1 for r in recs if r["clipped"]),
                "mean_padding_frac":  mean_padding_frac,
                "max_padding_frac":   max_padding_frac,
                "mean_aspect_ratio":  mean_aspect_ratio,
                "mean_box_w":         mean_box_w,
                "mean_box_h":         mean_box_h,
            }
            with open(os.path.join(track_dir, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2)

            track_summaries.append(summary)

        # ---- Top-level "suspicious tracks" overview --------------------
        track_summaries.sort(key=lambda s: s["max_ema"], reverse=True)
        top = track_summaries[:args.top_tracks]

        if top:
            rows = []
            row_labels = []
            for s in top:
                tid = s["track_id"]
                track_dir = os.path.join(tracks_dir, f"track_{tid:06d}")
                mp = os.path.join(track_dir, "montage.png")
                if not os.path.exists(mp):
                    continue
                m = cv2.imread(mp, cv2.IMREAD_COLOR)
                if m is None:
                    continue
                h, w = m.shape[:2]
                target_w = 900
                scale = target_w / max(w, 1)
                m = cv2.resize(m, (target_w, int(h * scale)),
                               interpolation=cv2.INTER_AREA)
                rows.append(m)
                row_labels.append(
                    f"track {tid:06d}  max_ema={s['max_ema']:.2f}  "
                    f"pad_mean={s.get('mean_padding_frac', 0):.3f}  "
                    f"frames={s['first_frame']}-{s['last_frame']}")

            if rows:
                pad = 40
                total_h = sum(r.shape[0] + pad for r in rows)
                canvas = np.full((total_h, rows[0].shape[1], 3),
                                 (24, 24, 24), dtype=np.uint8)
                y = 0
                for r, lbl in zip(rows, row_labels):
                    h = r.shape[0]
                    canvas[y:y + h, :r.shape[1]] = r
                    cv2.putText(canvas, lbl, (6, y + h + 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (230, 230, 230), 1, cv2.LINE_AA)
                    y += h + pad
                cv2.imwrite(
                    os.path.join(out_dir, "suspicious_tracks.png"), canvas)
                print(f"  Suspicious tracks: {out_dir}/suspicious_tracks.png")

    # ---- Video-level summary -------------------------------------------
    with open(os.path.join(out_dir, "records.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    video_summary = {
        "video":              video_path,
        "n_frames":           n_frames,
        "n_anomalous_frames": n_anomalous_frames,
        "n_alarms_started":   sum(1 for r in records if r["just_started"]),
        "n_tracks_total":     len(track_records),
        "n_tracks_above_tau_p": sum(
            1 for s in track_summaries if s["any_above_tau_p"]),
        "n_tracks_above_tau_f": sum(
            1 for s in track_summaries if s["any_above_tau_f"]),
        "tau_p": tau_p,
        "tau_f": tau_f,
        "frames_dir": frames_dir,
        "tracks_dir": tracks_dir,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(video_summary, f, indent=2)

    print(f"  Saved {n_frames} frames -> {frames_dir}/")
    print(f"  Saved {len(track_summaries)} tracks -> {tracks_dir}/")
    print(f"  Alarms started: {video_summary['n_alarms_started']}, "
          f"frames in alarm: {n_anomalous_frames}")

    del frame_cache, detections
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return video_summary


# ===========================================================================
# Main
# ===========================================================================
def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    paths = sorted(
        p for p in os.listdir(args.video_dir)
        if os.path.splitext(p)[1].lower() in VIDEO_EXTS
    )
    if not paths:
        raise SystemExit(f"No videos found in {args.video_dir}")

    print(f"{'='*70}")
    print(f"  Dumping {len(paths)} videos")
    print(f"{'='*70}")
    for p in paths:
        print(f"   {p}")
    print()

    print(f"[Frames] Loading {args.method} models...")
    models, image_size, device = load_bundle(
        args.method, args.ckpt_dir, args.category, args.model_size, args.device)
    use_fp16 = args.fp16 and device.startswith("cuda")
    if use_fp16:
        for key in ("teacher", "student", "ae"):
            models[key] = models[key].half()
        for key in ("mean", "std", "qa_st", "qb_st", "qa_ae", "qb_ae"):
            if isinstance(models.get(key), torch.Tensor):
                models[key] = models[key].half()
        for key, kind in (("teacher", "teacher"),
                          ("student", "student"),
                          ("ae", "ae")):
            models[key] = patch_forward_dtype_safe(models[key], kind)
        for key in ("teacher", "student", "ae"):
            models[key] = models[key].to(memory_format=torch.channels_last)
    model_dtype = torch.float16 if use_fp16 else torch.float32

    dummy = Image.new("RGB", (image_size, image_size))
    _ = score_batch(models, [dummy] * 8, device, image_size,
                    model_dtype, args.ratio)
    print(f"[Frames] Warm-up done.\n")

    summaries = []
    t_start = time.perf_counter()
    for i, name in enumerate(paths, 1):
        video_path = os.path.join(args.video_dir, name)
        print(f"\n[{i}/{len(paths)}] {name}")
        try:
            s = process_video(video_path, models, image_size, device,
                              model_dtype, args, args.save_dir)
            s["video_name"] = name
            summaries.append(s)
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()

    grand = {
        "n_videos": len(paths),
        "total_seconds": round(time.perf_counter() - t_start, 1),
        "videos": summaries,
        "method": args.method,
        "save_dir": os.path.abspath(args.save_dir),
    }
    with open(os.path.join(args.save_dir, "_all_summaries.json"), "w") as f:
        json.dump(grand, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  Done. {len(paths)} videos in {grand['total_seconds']}s")
    print(f"{'='*70}")
    print(f"  {'video':<24} {'frames':>7} {'alarm_fr':>9} "
          f"{'alarms':>7} {'tracks':>7} {'susp':>6}")
    for s in summaries:
        print(f"  {s['video_name']:<24} {s['n_frames']:>7} "
              f"{s['n_anomalous_frames']:>9} {s['n_alarms_started']:>7} "
              f"{s['n_tracks_total']:>7} {s['n_tracks_above_tau_p']:>6}")
    print(f"\n  Output: {grand['save_dir']}/")


if __name__ == "__main__":
    main()