#!/usr/bin/env python3
"""
Analyze acquisition/background conditions for the VERIFIED videos listed in
used_video_frames/used_video_inventory.csv.

This avoids the duplicate-name ambiguity already resolved for video1/video2/video3.
"""

from pathlib import Path
import argparse, csv, json, math, shutil, zipfile
import cv2
import numpy as np

def border_mask(h, w, frac=0.15):
    bh = max(1, int(h * frac))
    bw = max(1, int(w * frac))
    m = np.zeros((h, w), np.uint8)
    m[:bh, :] = 1
    m[-bh:, :] = 1
    m[:, :bw] = 1
    m[:, -bw:] = 1
    return m.astype(bool)

def entropy_u8(v):
    h = np.bincount(v.ravel(), minlength=256).astype(np.float64)
    p = h / max(h.sum(), 1.0)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def frame_metrics(frame, frac=0.15):
    h, w = frame.shape[:2]
    mask = border_mask(h, w, frac)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lum = lab[:, :, 0][mask]
    sat = hsv[:, :, 1][mask]
    gray_bg = gray[mask]

    p05, p95 = np.percentile(lum, [5, 95])
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    edges = cv2.Canny(gray, 80, 160)

    b, g, r = cv2.split(frame)

    return {
        "luminance_mean": float(np.mean(lum)),
        "luminance_std": float(np.std(lum)),
        "dynamic_range": float(p95 - p05),
        "saturation_mean": float(np.mean(sat)),
        "entropy": entropy_u8(gray_bg.astype(np.uint8)),
        "texture_laplacian_var": float(np.var(lap[mask])),
        "edge_density": float(np.mean(edges[mask] > 0)),
        "mean_R": float(np.mean(r[mask])),
        "mean_G": float(np.mean(g[mask])),
        "mean_B": float(np.mean(b[mask])),
    }

def sample_video(path, n_samples=24, frac=0.15):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")

    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    if n <= 0:
        cap.release()
        raise RuntimeError(f"No readable frames: {path}")

    idxs = np.unique(np.linspace(0, n - 1, min(n_samples, n)).round().astype(int))

    rows = []
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        m = frame_metrics(frame, frac)
        m["frame_index"] = int(idx)
        rows.append(m)
        frames.append(frame)

    cap.release()
    if not rows:
        raise RuntimeError(f"No frames sampled from {path}")

    # Representative sampled frame = closest to multivariate median.
    X = np.array([
        [
            r["luminance_mean"],
            r["luminance_std"],
            r["saturation_mean"],
            np.log1p(r["texture_laplacian_var"]),
            r["entropy"],
        ]
        for r in rows
    ], float)

    med = np.median(X, axis=0)
    scale = np.std(X, axis=0)
    scale[scale < 1e-9] = 1.0
    d = np.linalg.norm((X - med) / scale, axis=1)
    k = int(np.argmin(d))

    keys = [
        "luminance_mean", "luminance_std", "dynamic_range",
        "saturation_mean", "entropy", "texture_laplacian_var",
        "edge_density", "mean_R", "mean_G", "mean_B"
    ]
    agg = {key: float(np.mean([r[key] for r in rows])) for key in keys}
    agg["luminance_temporal_std"] = float(
        np.std([r["luminance_mean"] for r in rows])
    )
    agg["representative_frame_index"] = int(rows[k]["frame_index"])
    agg["sampled_frames"] = len(rows)
    agg["fps"] = fps
    agg["frame_count"] = n

    return agg, frames[k]

def tertile_classes(values, low, mid, high):
    a = np.asarray(values, float)
    if len(a) >= 3:
        q1, q2 = np.quantile(a, [1/3, 2/3])
    else:
        q1, q2 = float(a.min()), float(a.max())

    labels = []
    for v in a:
        if v <= q1:
            labels.append(low)
        elif v >= q2:
            labels.append(high)
        else:
            labels.append(mid)
    return labels, (float(q1), float(q2))

def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def make_contact_sheet(records, frames_dir, out_path, cols=4):
    tile_w, tile_h = 520, 370
    tiles = []

    for r in records:
        img = cv2.imread(str(frames_dir / r["representative_frame"]))
        if img is None:
            continue

        tile = np.full((tile_h, tile_w, 3), 255, np.uint8)
        max_img_h = 265
        scale = min((tile_w - 20) / img.shape[1], max_img_h / img.shape[0])
        rw = max(1, int(img.shape[1] * scale))
        rh = max(1, int(img.shape[0] * scale))
        small = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_AREA)
        x = (tile_w - rw) // 2
        y = 5 + (max_img_h - rh) // 2
        tile[y:y+rh, x:x+rw] = small

        lines = [
            r["video"][:62],
            f"Crops={r['crops']}  Split={r['split']}",
            f"L={float(r['luminance_mean']):.1f} ({r['luminosity_class']})   "
            f"C={float(r['luminance_std']):.1f} ({r['contrast_class']})",
            f"Sat={float(r['saturation_mean']):.1f} ({r['saturation_class']})   "
            f"Texture={r['texture_class']}",
            f"Lighting={r['lighting_stability_class']}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(
                tile, line, (10, 287 + i * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45 if i else 0.48,
                (0, 0, 0), 1, cv2.LINE_AA
            )
        tiles.append(tile)

    rows_n = math.ceil(len(tiles) / cols)
    sheet = np.full((rows_n * tile_h, cols * tile_w, 3), 255, np.uint8)
    for i, t in enumerate(tiles):
        rr, cc = divmod(i, cols)
        sheet[rr*tile_h:(rr+1)*tile_h, cc*tile_w:(cc+1)*tile_w] = t

    cv2.imwrite(str(out_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 94])

def make_plots(records, out_dir):
    import matplotlib.pyplot as plt

    names = [r["video"] for r in records]
    lum = [float(r["luminance_mean"]) for r in records]
    con = [float(r["luminance_std"]) for r in records]
    sat = [float(r["saturation_mean"]) for r in records]
    tex = [float(r["texture_laplacian_var"]) for r in records]

    fig, ax = plt.subplots(figsize=(max(12, len(names)*0.35), 5.5))
    ax.bar(range(len(names)), lum)
    ax.set_ylabel("Mean background luminance (0-255)")
    ax.set_xlabel("Source video")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=75, ha="right", fontsize=7)
    ax.set_title("Background luminance by source video")
    fig.tight_layout()
    fig.savefig(out_dir / "luminance_by_video.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    sizes = np.clip(np.sqrt(np.asarray(tex)+1)*4, 25, 250)
    ax.scatter(lum, con, s=sizes)
    ax.set_xlabel("Mean background luminance (0-255)")
    ax.set_ylabel("Luminance standard deviation")
    ax.set_title("Background luminance and contrast")
    for x, y, name in zip(lum, con, names):
        ax.annotate(name, (x, y), fontsize=6, xytext=(3, 2), textcoords="offset points")
    fig.tight_layout()
    fig.savefig(out_dir / "luminance_vs_contrast.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(lum, sat, s=60)
    ax.set_xlabel("Mean background luminance (0-255)")
    ax.set_ylabel("Mean saturation (0-255)")
    ax.set_title("Background luminance and saturation")
    fig.tight_layout()
    fig.savefig(out_dir / "luminance_vs_saturation.png", dpi=180)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument(
        "--inventory",
        default="used_video_frames/used_video_inventory.csv"
    )
    ap.add_argument("--out", default="background_metrics_verified")
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--border-fraction", type=float, default=0.15)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    inv_path = (root / args.inventory).resolve()
    out_dir = (root / args.out).resolve()
    frames_dir = out_dir / "representative_frames"

    if out_dir.exists():
        shutil.rmtree(out_dir)
    frames_dir.mkdir(parents=True)

    with open(inv_path, "r", encoding="utf-8-sig", newline="") as f:
        inventory = list(csv.DictReader(f))

    records = []
    failures = []

    for i, row in enumerate(inventory, 1):
        video_path = root / row["canonical_path"]
        print(f"[{i:02d}/{len(inventory)}] {row['video']} -> {row['canonical_path']}")

        try:
            agg, frame = sample_video(
                video_path,
                n_samples=args.samples,
                frac=args.border_fraction
            )
        except Exception as e:
            failures.append({
                "video": row["video"],
                "canonical_path": row["canonical_path"],
                "error": str(e),
            })
            print("   ERROR:", e)
            continue

        frame_name = f"{i:02d}_{Path(row['representative_frame']).stem}.jpg"
        cv2.imwrite(
            str(frames_dir / frame_name),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        )

        rec = {
            "video": row["video"],
            "canonical_path": row["canonical_path"],
            "crops": row["crops"],
            "split": row["split"],
            "labels": row["labels"],
            "representative_frame": frame_name,
        }
        rec.update(agg)
        records.append(rec)

    lum_labels, lum_thr = tertile_classes(
        [r["luminance_mean"] for r in records],
        "dark", "medium", "bright"
    )
    con_labels, con_thr = tertile_classes(
        [r["luminance_std"] for r in records],
        "low", "medium", "high"
    )
    sat_labels, sat_thr = tertile_classes(
        [r["saturation_mean"] for r in records],
        "low", "medium", "high"
    )
    tex_labels, tex_thr = tertile_classes(
        [r["texture_laplacian_var"] for r in records],
        "smooth", "medium", "textured"
    )
    temp_labels, temp_thr = tertile_classes(
        [r["luminance_temporal_std"] for r in records],
        "stable", "moderate", "variable"
    )

    for i, r in enumerate(records):
        r["luminosity_class"] = lum_labels[i]
        r["contrast_class"] = con_labels[i]
        r["saturation_class"] = sat_labels[i]
        r["texture_class"] = tex_labels[i]
        r["lighting_stability_class"] = temp_labels[i]

    write_csv(out_dir / "background_metrics.csv", records)
    write_csv(out_dir / "failures.csv", failures)

    make_contact_sheet(
        records, frames_dir, out_dir / "background_contact_sheet.jpg"
    )
    make_plots(records, out_dir)

    summary = {
        "videos_analyzed": len(records),
        "failures": len(failures),
        "samples_per_video_requested": args.samples,
        "classification": "relative tertiles over the verified acquisition set",
        "thresholds": {
            "luminance": lum_thr,
            "contrast": con_thr,
            "saturation": sat_thr,
            "texture_laplacian_var": tex_thr,
            "temporal_luminance_std": temp_thr,
        }
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    lines = [
        "# Verified background analysis",
        "",
        f"Videos analyzed: **{len(records)}**",
        f"Failures: **{len(failures)}**",
        "",
        "Metrics are averaged over sampled frames from each verified source video.",
        "The outer 15% of each frame is used as a background proxy.",
        "The qualitative classes are relative to this dataset (tertiles).",
        "",
        "Use the raw numeric values in `background_metrics.csv` for the report.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")

    zip_path = out_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out_dir.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(out_dir.parent))

    print()
    print(f"Analyzed: {len(records)} videos")
    print(f"Failures: {len(failures)}")
    print(f"ZIP created: {zip_path}")

if __name__ == "__main__":
    main()
