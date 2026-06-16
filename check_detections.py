import cv2
from ultralytics import YOLO

model = YOLO('yolov8m.pt')
frame = cv2.imread('video_frame.jpg')
results = model(frame, conf=0.3, verbose=False)[0]

print(f"Detections found: {len(results.boxes)}")
for box in results.boxes:
    class_id = int(box.cls[0])
    conf = float(box.conf[0])
    name = model.names[class_id]
    print(f"  → {name} (class {class_id}) confidence: {conf:.2f}")