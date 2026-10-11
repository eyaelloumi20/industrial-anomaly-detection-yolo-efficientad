"""
live_compare_app.py
====================
Live version of compare_pipeline_methods.py: instead of reading a saved
video and comparing methods offline, this pulls a live MJPEG stream from
a phone (e.g. the "IP Webcam" Android app) and shows baseline / lowres /
thin / pruned side-by-side in a browser, updating in real time.

QAT is intentionally excluded -- it's CPU-forced and meaningfully slower,
which isn't a fair or pleasant fit for a live feed alongside three GPU
methods. Drop it back in by adding 'qat' to LIVE_METHODS and importing
its loader (utils.load_efficientad_qat) the same way as the others.

Setup
-----
1. On the phone: install "IP Webcam" (Pavel Khlebovich), same WiFi as
   this machine, tap "Start server". Note the IP:port it shows
   (usually http://<phone-ip>:8080).
2. Edit the CONFIG block below (phone URL, yolo weights, ckpt dirs).
3. pip install flask --break-system-packages   (if not already installed)
4. python live_compare_app.py
5. Open http://<workstation-ip>:5000 in a browser (phone or laptop,
   same network).

Performance note
-----------------
Every active method runs its own EfficientAD forward pass on every
detection, every frame -- so cost scales as (#methods x #detections).
If it feels laggy, trim LIVE_METHODS down to 2, or raise --conf to
reduce detections, or lower JPEG_QUALITY / target FPS below.
"""

import os
import sys
import time
import threading
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from flask import Flask, Response, jsonify, render_template_string

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ── CONFIG ──────────────────────────────────────────────────────────────────
PHONE_STREAM_URL = "http://192.168.1.50:8080/video"   # <-- set to your phone's IP Webcam URL
YOLO_WEIGHTS      = "/home/ahmed/two-stage-efficientad-inspection/runs/detect/yolo_runs2/chewinggum_finetune2/weights/best.pt"

CKPT_DIRS = {
    "baseline": "/home/ahmed/framework/ckpt_cropped_large",
    "lowres":   "./ckpt_192",
    "thin":     "./ckpt_thin_025",
    "pruned":   "./ckpt_pruned_030",
}
THRESHOLDS = {   # None -> score-only mode for that method (no red/green verdict box)
    "baseline": None,
    "lowres":   None,
    "thin":     None,
    "pruned":   None,
}
LIVE_METHODS = ["baseline", "lowres", "thin", "pruned"]   # trim this list if it's too slow

CATEGORY   = "chewinggum"
MODEL_SIZE = "S"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
CONF, IOU  = 0.25, 0.45
PAD_RATIO  = 0.02
DENOISE    = False
RATIO      = 0.1
EMA_ALPHA  = 0.3
JPEG_QUALITY = 80

OUT_CHANNELS = 384
IMAGE_SIZE_FALLBACK = 256

METHOD_COLOURS = {   # BGR
    "baseline": (200, 150, 0),
    "lowres":   (0, 200, 200),
    "thin":     (0, 200, 0),
    "pruned":   (0, 100, 255),
}

# ── model loading (same dispatch as compare_pipeline_methods.py, minus qat) ──

def load_bundle(method: str, ckpt_dir: str):
    if method in ("baseline", "lowres"):
        from utils.load_efficientad_adaptive import load_efficientad_model
        models = load_efficientad_model(ckpt_dir, CATEGORY, MODEL_SIZE, DEVICE)
        return models, models.get("image_size", IMAGE_SIZE_FALLBACK), DEVICE
    elif method == "thin":
        from utils.load_efficientad_thin import load_efficientad_model
        models = load_efficientad_model(ckpt_dir, CATEGORY, MODEL_SIZE, DEVICE)
        return models, IMAGE_SIZE_FALLBACK, DEVICE
    elif method == "pruned":
        from utils.load_efficientad_pruned import load_efficientad_model
        models = load_efficientad_model(ckpt_dir, CATEGORY, MODEL_SIZE, DEVICE)
        return models, IMAGE_SIZE_FALLBACK, DEVICE
    else:
        raise ValueError(f"Unsupported live method '{method}' (qat excluded by design)")


def resize_with_padding(crop: Image.Image, size: int) -> Image.Image:
    ow, oh = crop.size
    scale = size / max(ow, oh)
    nw, nh = int(ow * scale), int(oh * scale)
    resized = crop.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def get_crop(pil_img: Image.Image, box: tuple, size: int) -> Image.Image:
    W, H = pil_img.size
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * PAD_RATIO), int(bh * PAD_RATIO)
    x1 = max(0, x1 - px); y1 = max(0, y1 - py)
    x2 = min(W, x2 + px); y2 = min(H, y2 + py)
    crop = pil_img.crop((x1, y1, x2, y2)).convert("RGB")
    crop = resize_with_padding(crop, size)
    if DENOISE:
        arr = np.array(crop, dtype=np.uint8)
        arr = cv2.GaussianBlur(arr, (3, 3), sigmaX=0.8, sigmaY=0.8)
        arr = cv2.bilateralFilter(arr, d=5, sigmaColor=15, sigmaSpace=15)
        crop = Image.fromarray(arr)
    return crop


@torch.no_grad()
def score_crop(models: dict, crop_pil: Image.Image, device: str, image_size: int) -> float:
    t = transforms.ToTensor()(crop_pil).unsqueeze(0).to(device)
    t_out = models["teacher"](t)
    s_out = models["student"](t)
    a_out = models["ae"](t)
    y_st = s_out[:, :OUT_CHANNELS, :, :]
    y_stae = s_out[:, -OUT_CHANNELS:, :, :]
    norm_t = (t_out - models["mean"]) / (models["std"] + 1e-8)
    d_st = torch.pow(norm_t - y_st, 2)
    d_stae = torch.pow(a_out - y_stae, 2)
    fm_st = F.interpolate(torch.mean(d_st, dim=1, keepdim=True),
                           size=(image_size, image_size), mode="bilinear", align_corners=False)
    fm_stae = F.interpolate(torch.mean(d_stae, dim=1, keepdim=True),
                             size=(image_size, image_size), mode="bilinear", align_corners=False)
    nm_st = (RATIO * (fm_st - models["qa_st"])) / (models["qb_st"] - models["qa_st"] + 1e-8)
    nm_ae = (RATIO * (fm_stae - models["qa_ae"])) / (models["qb_ae"] - models["qa_ae"] + 1e-8)
    amap = (0.5 * nm_st + 0.5 * nm_ae)[0, 0].cpu().numpy()
    return float(np.max(amap))


# ── shared state, guarded by a lock ────────────────────────────────────────
_lock = threading.Lock()
_latest_jpeg: Dict[str, bytes] = {m: b"" for m in LIVE_METHODS}
_stats: Dict[str, dict] = {m: {"fps": 0.0, "mean_score": 0.0, "n_det": 0, "n_anom": 0} for m in LIVE_METHODS}
_ema_hist: Dict[str, Dict[int, float]] = {m: {} for m in LIVE_METHODS}


def capture_loop():
    print(f"[live] loading YOLO: {YOLO_WEIGHTS}")
    from ultralytics import YOLO
    yolo = YOLO(YOLO_WEIGHTS)
    yolo.to(DEVICE)

    bundles = {}
    for m in LIVE_METHODS:
        ckpt_dir = CKPT_DIRS.get(m)
        if not ckpt_dir or not os.path.isdir(ckpt_dir):
            print(f"[live] SKIP {m}: no valid --dir ({ckpt_dir})")
            continue
        print(f"[live] loading {m} from {ckpt_dir}")
        models, image_size, device = load_bundle(m, ckpt_dir)
        bundles[m] = (models, image_size, device)
    active = list(bundles.keys())
    if not active:
        raise SystemExit("No checkpoint dirs found for any LIVE_METHODS entry.")
    print(f"[live] active methods: {active}")

    print(f"[live] connecting to phone stream: {PHONE_STREAM_URL}")
    cap = cv2.VideoCapture(PHONE_STREAM_URL)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open {PHONE_STREAM_URL} -- check the phone and workstation "
            f"are on the same WiFi, and that IP Webcam's 'Start server' is running.")

    frame_idx = 0
    fps_t0 = {m: time.perf_counter() for m in active}
    fps_count = {m: 0 for m in active}

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        frame_idx += 1
        src_h, src_w = frame.shape[:2]
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        try:
            results = yolo.track(source=frame, conf=CONF, iou=IOU, classes=[0],
                                  persist=True, tracker="bytetrack.yaml", verbose=False)
        except np.linalg.LinAlgError:
            results = yolo.predict(source=frame, conf=CONF, iou=IOU, classes=[0], verbose=False)

        dets: List[Tuple[int, int, int, int, int]] = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
            ids = (r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None
                   else [100_000 + frame_idx * 1000 + i for i in range(len(xyxy))])
            for box, tid in zip(xyxy, ids):
                x1, y1 = max(0, box[0]), max(0, box[1])
                x2, y2 = min(src_w, box[2]), min(src_h, box[3])
                if x2 > x1 and y2 > y1:
                    dets.append((int(tid), x1, y1, x2, y2))

        for m in active:
            models, image_size, device = bundles[m]
            threshold = THRESHOLDS.get(m)
            colour = METHOD_COLOURS.get(m, (255, 255, 255))
            annotated = frame.copy()
            scores_this_frame = []
            n_anom = 0

            for tid, x1, y1, x2, y2 in dets:
                crop = get_crop(pil_frame, (x1, y1, x2, y2), image_size)
                score = score_crop(models, crop, device, image_size)
                prev = _ema_hist[m].get(tid, score)
                ema = EMA_ALPHA * score + (1 - EMA_ALPHA) * prev
                _ema_hist[m][tid] = ema
                scores_this_frame.append(score)

                verdict = ""
                box_colour = colour
                if threshold is not None:
                    verdict = "ANOMALY" if ema >= threshold else "NORMAL"
                    box_colour = (0, 0, 255) if verdict == "ANOMALY" else (0, 200, 0)
                    if verdict == "ANOMALY":
                        n_anom += 1

                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_colour, 2)
                label = f"ID{tid} {verdict + ' ' if verdict else ''}{ema:.2f}"
                cv2.putText(annotated, label, (x1 + 3, max(15, y1 + 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            fps_count[m] += 1
            elapsed = time.perf_counter() - fps_t0[m]
            if elapsed >= 1.0:
                with _lock:
                    _stats[m]["fps"] = fps_count[m] / elapsed
                fps_count[m] = 0
                fps_t0[m] = time.perf_counter()

            cv2.putText(annotated, f"{m}  |  {len(dets)} dets  |  {_stats[m]['fps']:.1f} fps",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)

            ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            with _lock:
                if ok:
                    _latest_jpeg[m] = buf.tobytes()
                _stats[m]["mean_score"] = float(np.mean(scores_this_frame)) if scores_this_frame else 0.0
                _stats[m]["n_det"] = len(dets)
                _stats[m]["n_anom"] = n_anom


# ── Flask app ───────────────────────────────────────────────────────────────
app = Flask(__name__)

PAGE = """
<!doctype html><html><head><title>Live gum inspection - method comparison</title>
<style>
body{background:#111;color:#eee;font-family:sans-serif;margin:0;padding:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:12px}
.panel{background:#1a1a2e;border-radius:8px;padding:8px}
.panel img{width:100%;border-radius:4px;display:block}
.panel .stats{font-size:13px;color:#aaa;margin-top:6px;display:flex;gap:12px;flex-wrap:wrap}
h1{font-size:18px;font-weight:600}
</style></head><body>
<h1>Live comparison (excluding QAT)</h1>
<div class="grid">
{% for m in methods %}
  <div class="panel">
    <img src="/stream/{{m}}">
    <div class="stats" id="stats-{{m}}">{{m}}</div>
  </div>
{% endfor %}
</div>
<script>
async function poll(){
  const r = await fetch('/stats'); const data = await r.json();
  for (const m in data){
    const el = document.getElementById('stats-'+m);
    if (el) el.textContent = m + ' | fps ' + data[m].fps.toFixed(1)
      + ' | dets ' + data[m].n_det + ' | mean_score ' + data[m].mean_score.toFixed(3)
      + ' | anomalous ' + data[m].n_anom;
  }
}
setInterval(poll, 500); poll();
</script>
</body></html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE, methods=LIVE_METHODS)


def mjpeg_generator(method: str):
    while True:
        with _lock:
            jpeg = _latest_jpeg.get(method, b"")
        if jpeg:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
        time.sleep(0.03)


@app.route("/stream/<method>")
def stream(method):
    if method not in LIVE_METHODS:
        return "unknown method", 404
    return Response(mjpeg_generator(method), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stats")
def stats():
    with _lock:
        return jsonify(_stats)


if __name__ == "__main__":
    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=5000, threaded=True)
