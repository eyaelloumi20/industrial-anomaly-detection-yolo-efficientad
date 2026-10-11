from ultralytics import YOLO
import cv2
model = YOLO("/home/ahmed/two-stage-efficientad-inspection/runs/segment/yolo_runs2/chewinggum_seg_v22/weights/best.pt")
frame = cv2.imread("/home/ahmed/two-stage-efficientad-inspection/project/images/frame_0001.png")
results = model(frame, conf=0.25, verbose=False)
results[0].save("debug_with_masks.jpg")  # draws masks on frame