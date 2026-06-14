# tracker.py
# Day 3: Real-Time Object Tracking with ByteTrack
# Purpose: Detect, name, and track objects across video frames
# Impact: Core of our alert system — knows WHAT moved and WHERE

import cv2
import torch
from ultralytics import YOLO
import supervision as sv
import numpy as np
from collections import defaultdict
import time

# ── Configuration ────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.5      # minimum confidence to count as detection
IOU_THRESHOLD = 0.5             # overlap threshold for NMS
DISAPPEARANCE_THRESHOLD = 90    # frames before object is considered "gone"
MIN_BOX_AREA = 3000             # ignore tiny detections (shadows, noise)

# ── Only track these COCO classes (retail relevant) ──────────────────────────
# This excludes people, faces, and background items
ALLOWED_CLASSES = {
    39, 40, 41, 42, 43, 44, 45,  # bottle, wine glass, cup, fork, knife, spoon, bowl
    46, 47, 48, 49, 50, 51, 52,  # banana, apple, sandwich, orange, broccoli, carrot, hotdog
    53, 54, 55, 56, 57, 58, 59,  # pizza, donut, cake, chair, couch, plant, bed
    60, 61, 62, 63, 64, 73, 74,  # dining table, toilet, tv, laptop, mouse, book, clock
    75, 76, 77, 78, 79           # vase, scissors, teddy bear, hair drier, toothbrush
}

# ── COCO class names we care about in retail ─────────────────────────────────
RETAIL_CLASSES = {
    39: 'bottle',     40: 'wine glass', 41: 'cup',
    42: 'fork',       43: 'knife',      44: 'spoon',
    45: 'bowl',       46: 'banana',     47: 'apple',
    48: 'sandwich',   49: 'orange',     50: 'broccoli',
    51: 'carrot',     52: 'hot dog',    53: 'pizza',
    54: 'donut',      55: 'cake',       63: 'laptop',
    64: 'mouse',      73: 'book',       74: 'clock',
    75: 'vase',       76: 'scissors',   77: 'teddy bear',
}

def get_class_name(class_id, confidence=None):
    """
    Return specific name only if confidence is high enough.
    Otherwise return 'object' to avoid wrong guesses.
    """
    # Only trust the class name if confidence > 70%
    # Below that, the model is guessing — call it 'object'
    if confidence is not None and confidence < 0.70:
        return 'object'
    return RETAIL_CLASSES.get(int(class_id), 'object')

def main():
    print("=" * 50)
    print("RETAIL VISION - OBJECT TRACKER")
    print("=" * 50)

    # ── 1. Load COCO model ────────────────────────────────────────────────
    print("Loading COCO model...")
    coco_model = YOLO('yolov8m.pt')
    print("COCO model loaded!")

    # ── 2. Load our custom retail model ──────────────────────────────────
    print("Loading retail model...")
    retail_model = YOLO('runs/train/retail_v1/weights/best.pt')
    print("Retail model loaded!")

    # ── 3. Set up ByteTrack tracker ───────────────────────────────────────
    tracker = sv.ByteTrack()

    # ── 4. Set up annotators ──────────────────────────────────────────────
    box_annotator = sv.BoundingBoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5)

    # ── 5. Open video source ──────────────────────────────────────────────
    print("\nOpening video source...")
    print("Press 'q' to quit")
    print("=" * 50)

    cap = cv2.VideoCapture(0)  # 0 or 1 for webcam

    if not cap.isOpened():
        print("ERROR: Could not open webcam!")
        return

    # ── 6. Tracking state ─────────────────────────────────────────────────
    object_last_seen = defaultdict(int)
    object_names = {}
    alert_log = []
    frame_count = 0

    # ── 7. Main processing loop ───────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read frame. Exiting...")
            break

        frame_count += 1

        # ── 7a. Run COCO model ────────────────────────────────────────────
        coco_results = coco_model(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False
        )[0]

        # ── 7b. Convert to supervision format ────────────────────────────
        detections = sv.Detections.from_ultralytics(coco_results)

        # ── 7c. Filter tiny detections ────────────────────────────────────
        if len(detections) > 0:
            areas = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * \
                    (detections.xyxy[:, 3] - detections.xyxy[:, 1])
            detections = detections[areas > MIN_BOX_AREA]

        # ── 7d. Filter non-retail classes (removes faces, people) ─────────
        if len(detections) > 0 and detections.class_id is not None:
            mask = np.array([c in ALLOWED_CLASSES for c in detections.class_id])
            detections = detections[mask]

        # ── 7e. Run ByteTrack ─────────────────────────────────────────────
        tracked = tracker.update_with_detections(detections)

        # ── 7f. Update object registry ────────────────────────────────────
        for i, tracker_id in enumerate(tracked.tracker_id):
            if tracker_id is not None:
                object_last_seen[tracker_id] = frame_count
                if tracker_id not in object_names:
                    class_id = tracked.class_id[i] if tracked.class_id is not None else -1
                    conf = tracked.confidence[i] if tracked.confidence is not None else None
                    object_names[tracker_id] = get_class_name(class_id, conf)

        # ── 7g. Check for disappeared objects ─────────────────────────────
        for obj_id, last_frame in list(object_last_seen.items()):
            frames_missing = frame_count - last_frame
            if frames_missing > DISAPPEARANCE_THRESHOLD:
                name = object_names.get(obj_id, 'object')
                alert_msg = f"ALERT: {name} (ID#{obj_id}) removed at frame {frame_count}!"
                print(alert_msg)
                alert_log.append({
                    'id': obj_id,
                    'name': name,
                    'frame': frame_count,
                    'time': time.strftime('%H:%M:%S')
                })
                del object_last_seen[obj_id]

        # ── 7h. Build labels ──────────────────────────────────────────────
        labels = []
        if tracked.tracker_id is not None:
            for i, tracker_id in enumerate(tracked.tracker_id):
                name = object_names.get(tracker_id, 'object')
                conf = tracked.confidence[i] if tracked.confidence is not None else 0
                labels.append(f"{name} ID#{tracker_id} {conf:.2f}")

        # ── 7i. Draw on frame ─────────────────────────────────────────────
        annotated = box_annotator.annotate(
            scene=frame.copy(),
            detections=tracked
        )
        annotated = label_annotator.annotate(
            scene=annotated,
            detections=tracked,
            labels=labels
        )

        # ── 7j. Show stats ────────────────────────────────────────────────
        cv2.putText(annotated, f"Objects: {len(tracked)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
        cv2.putText(annotated, f"Alerts: {len(alert_log)}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)
        cv2.putText(annotated, f"Frame: {frame_count}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 0), 2)

        # ── 7k. Display frame ─────────────────────────────────────────────
        cv2.imshow('Retail Vision - Object Tracker', annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ── 8. Cleanup ────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()

    # ── 9. Final summary ──────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("SESSION SUMMARY")
    print("=" * 50)
    print(f"Total frames processed: {frame_count}")
    print(f"Total alerts triggered: {len(alert_log)}")
    if alert_log:
        print("\nAlert Log:")
        for alert in alert_log:
            print(f"  [{alert['time']}] {alert['name']} ID#{alert['id']} removed")
    print("=" * 50)

if __name__ == '__main__':
    main()