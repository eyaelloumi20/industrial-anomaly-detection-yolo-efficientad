#!/usr/bin/env python3
from pathlib import Path
from collections import defaultdict, Counter
import argparse, csv, hashlib, math, re, shutil, zipfile

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}

# Search only source/acquisition folders, not result folders containing annotated videos.
DEFAULT_SOURCE_DIRS = [
    "precision_test/original",
    "precision_test/video",
    "dataset_step4",
    "validation_videos",
    "videos_gum",
    "videos_mycategory",
    "video_test",
    "videos_N",
]

def sha256(path, block=1024*1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def find_videos(root, source_dirs):
    by_stem = defaultdict(list)
    for d in source_dirs:
        base = root / d
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                by_stem[p.stem].append(p)
    return by_stem

def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("CSV is empty.")
    cols = set(rows[0])
    image_col = next((c for c in ["image", "image_path", "path", "filepath", "file"] if c in cols), None)
    if image_col is None:
        raise RuntimeError(f"No image column found. Columns: {sorted(cols)}")
    return rows, image_col

def match_crop(crop, stems):
    s = Path(crop).stem
    cand = [v for v in stems if s == v or s.startswith(v+"_") or s.startswith(v+"-")]
    return max(cand, key=len) if cand else None

def choose_canonical(paths):
    # Preference for obvious original/source folders.
    priority = [
        "precision_test/original",
        "validation_videos",
        "videos_gum",
        "dataset_step4",
        "videos_mycategory",
        "video_test",
        "videos_N",
        "precision_test/video",
    ]
    def rank(p):
        ps = str(p)
        for i, token in enumerate(priority):
            if token in ps:
                return i
        return 999
    return sorted(paths, key=lambda p: (rank(p), len(str(p))))[0]

def extract_mid_frame(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    idx = max(0, n // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read middle frame from {video_path}")
    return frame, idx, n, fps, w, h

def safe(s):
    return re.sub(r"[^A-Za-z0-9._()-]+", "_", s).strip("_")

def make_contact_sheet(items, out_path, cols=4):
    tile_w, tile_h = 480, 330
    tiles = []
    for rec in items:
        img = cv2.imread(str(rec["frame_path"]))
        if img is None:
            continue
        tile = np.full((tile_h, tile_w, 3), 255, np.uint8)
        scale = min((tile_w-20)/img.shape[1], 250/img.shape[0])
        rw, rh = max(1,int(img.shape[1]*scale)), max(1,int(img.shape[0]*scale))
        small = cv2.resize(img, (rw,rh), interpolation=cv2.INTER_AREA)
        x, y = (tile_w-rw)//2, 5+(250-rh)//2
        tile[y:y+rh, x:x+rw] = small
        title = rec["video"][:55]
        sub = f'{rec["split"]} | crops={rec["crops"]}'
        cv2.putText(tile, title, (10, 282), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(tile, sub, (10, 308), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,0,0), 1, cv2.LINE_AA)
        tiles.append(tile)

    rows = math.ceil(len(tiles)/cols)
    sheet = np.full((rows*tile_h, cols*tile_w, 3), 255, np.uint8)
    for i,t in enumerate(tiles):
        r,c = divmod(i,cols)
        sheet[r*tile_h:(r+1)*tile_h, c*tile_w:(c+1)*tile_w] = t
    cv2.imwrite(str(out_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 94])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--csv", default="dataset_multi/split_csv/1cls.csv")
    ap.add_argument("--out", default="used_video_frames")
    ap.add_argument("--source-dir", action="append", dest="source_dirs")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    csv_path = (root / args.csv).resolve()
    out = (root / args.out).resolve()
    frames_dir = out / "frames"

    if out.exists():
        shutil.rmtree(out)
    frames_dir.mkdir(parents=True)

    source_dirs = args.source_dirs or DEFAULT_SOURCE_DIRS
    video_by_stem = find_videos(root, source_dirs)
    rows, image_col = read_csv(csv_path)
    stems = sorted(video_by_stem, key=len, reverse=True)

    use = defaultdict(lambda: {"total":0, "splits":Counter(), "labels":Counter()})
    unknown = []
    for row in rows:
        stem = match_crop(row[image_col], stems)
        if stem is None:
            unknown.append(row[image_col])
            continue
        use[stem]["total"] += 1
        use[stem]["splits"][row.get("split","unknown") or "unknown"] += 1
        use[stem]["labels"][row.get("label","unknown") or "unknown"] += 1

    inventory = []
    contact = []

    for i, stem in enumerate(sorted(use), 1):
        paths = video_by_stem[stem]
        canonical = choose_canonical(paths)

        # Hash every duplicate copy of this source name.
        hashes = [(p, sha256(p)) for p in paths]
        same_content = len({h for _,h in hashes}) == 1

        frame, frame_idx, n, fps, w, h = extract_mid_frame(canonical)
        frame_name = f"{i:02d}_{safe(stem)}.jpg"
        frame_path = frames_dir / frame_name
        cv2.imwrite(str(frame_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        split = "+".join(sorted(use[stem]["splits"]))
        label_text = ", ".join(f"{k}={v}" for k,v in sorted(use[stem]["labels"].items()))
        copies = " | ".join(str(p.relative_to(root)) for p in paths)

        rec = {
            "index": i,
            "video": stem,
            "canonical_path": str(canonical.relative_to(root)),
            "all_copies": copies,
            "duplicate_copies_identical": "YES" if same_content else "NO",
            "crops": use[stem]["total"],
            "split": split,
            "labels": label_text,
            "frame_index": frame_idx,
            "frame_count": n,
            "fps": round(fps,3),
            "width": w,
            "height": h,
            "representative_frame": f"frames/{frame_name}",
        }
        inventory.append(rec)
        contact.append({**rec, "frame_path": frame_path})

    fields = list(inventory[0].keys())
    with open(out/"used_video_inventory.csv","w",encoding="utf-8",newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(inventory)

    (out/"unknown_crops.txt").write_text("\n".join(unknown), encoding="utf-8")
    make_contact_sheet(contact, out/"contact_sheet.jpg")

    summary = []
    summary.append(f"Used source videos: {len(inventory)}")
    summary.append(f"CSV crop rows: {len(rows)}")
    summary.append(f"Unmatched crop rows: {len(unknown)}")
    summary.append(f"Duplicate-name sources with non-identical copies: {sum(r['duplicate_copies_identical']=='NO' for r in inventory if ' | ' in r['all_copies'])}")
    summary.append("")
    summary.append("Review the contact sheet first. If it looks correct, use these 49 canonical videos for background metrics.")
    (out/"README.txt").write_text("\n".join(summary), encoding="utf-8")

    zip_path = out.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED) as zf:
        for p in out.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(out.parent))

    print("\n".join(summary))
    print(f"ZIP created: {zip_path}")

if __name__ == "__main__":
    main()
