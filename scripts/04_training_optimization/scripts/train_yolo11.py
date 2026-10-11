from ultralytics import YOLO

model = YOLO("yolo11s.pt")
model.train(
    data="/path/to/roboflow_export/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    project="./yolo_runs",
    name="chewinggum_yolo11s",
)