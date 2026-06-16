# fence.py
# Day 4: Virtual Fence & Alert Logic
# Purpose: Define zones on video feed and detect when objects cross boundaries
# Impact: Core theft detection logic — knows WHEN and WHERE an object was taken

import cv2
import torch
from ultralytics import YOLO
import supervision as sv
import numpy as np
from collections import defaultdict
import time

# ── Configuration ────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.5
MIN_BOX_AREA = 5000
DISAPPEARANCE_THRESHOLD = 90
MIN_SHELF_FRAMES = 60    # object must be on shelf for 2 seconds before tracking

# ── Zone definitions (will be set interactively) ──────────────────────────────
# These are default values — we'll let user draw zones on first frame
SHELF_ZONE = np.array([[10, 10], [710, 10], [710, 200], [10, 200]])
EXIT_ZONE = np.array([[10, 250], [710, 250], [710, 400], [10, 400]])

# ── COCO retail classes ───────────────────────────────────────────────────────
ALLOWED_CLASSES = {
    39, 40, 41, 42, 43, 44, 45,
    46, 47, 48, 49, 50, 51, 52,
    53, 54, 55, 56, 57, 58, 59,
    60, 61, 62, 63, 64, 73, 74,
    75, 76, 77, 78, 79
}

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
    if confidence is not None and confidence < 0.70:
        return 'object'
    return RETAIL_CLASSES.get(int(class_id), 'object')

def get_center(xyxy):
    """Get center point of a bounding box"""
    x1, y1, x2, y2 = xyxy
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def point_in_zone(point, zone):
    """Check if a point is inside a polygon zone"""
    return cv2.pointPolygonTest(zone.astype(np.float32), point, False) >= 0

def draw_zones(frame):
    """Draw shelf and exit zones on frame"""
    # Draw shelf zone (green)
    cv2.polylines(frame, [SHELF_ZONE], True, (0, 255, 0), 2)
    cv2.fillPoly(frame.copy(), [SHELF_ZONE], (0, 255, 0))
    cv2.putText(frame, "SHELF ZONE", 
                (SHELF_ZONE[0][0] + 5, SHELF_ZONE[0][1] + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Draw exit zone (red)
    cv2.polylines(frame, [EXIT_ZONE], True, (0, 0, 255), 2)
    cv2.putText(frame, "EXIT ZONE",
                (EXIT_ZONE[0][0] + 5, EXIT_ZONE[0][1] + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame

def main():
    print("=" * 50)
    print("RETAIL VISION - VIRTUAL FENCE")
    print("=" * 50)

    # ── 1. Load models ────────────────────────────────────────────────────
    print("Loading models...")
    coco_model = YOLO('yolov8m.pt')
    retail_model = YOLO('runs/train/retail_v1/weights/best.pt')
    print("Models loaded!")

    # ── 2. Set up tracker and annotators ─────────────────────────────────
    tracker = sv.ByteTrack()
    box_annotator = sv.BoundingBoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5)

    # ── 3. Open video source ──────────────────────────────────────────────
    print("\nOpening video source...")
    print("Controls:")
    print("  'q' → quit")
    print("  's' → take screenshot of current frame")
    print("=" * 50)

    # Using video file for testing
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Could not open camera!")
            return
            
    print("Webcam opened successfully!")

    # ── 4. Tracking state ─────────────────────────────────────────────────
    object_last_seen = defaultdict(int)
    object_names = {}
    object_zones = {}       # tracks which zone each object is in
    alert_log = []
    theft_log = []          # specifically theft alerts
    frame_count = 0
    object_shelf_frames = defaultdict(int)  # counts frames object spent on shelf
    object_seen_on_shelf = set()            # objects that visited the shelf
    already_alerted = set()                 # tracks objects already flagged

    # ── 5. Main loop ──────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        display = frame.copy()

        # ── 5a. Draw zones on frame ───────────────────────────────────────
        # Shelf zone overlay (semi-transparent green)
        overlay = display.copy()
        cv2.fillPoly(overlay, [SHELF_ZONE], (0, 255, 0))
        cv2.addWeighted(overlay, 0.15, display, 0.85, 0, display)
        cv2.polylines(display, [SHELF_ZONE], True, (0, 255, 0), 2)
        cv2.putText(display, "SHELF ZONE",
                    (SHELF_ZONE[0][0] + 5, SHELF_ZONE[0][1] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Exit zone overlay (semi-transparent red)
        overlay2 = display.copy()
        cv2.fillPoly(overlay2, [EXIT_ZONE], (0, 0, 255))
        cv2.addWeighted(overlay2, 0.15, display, 0.85, 0, display)
        cv2.polylines(display, [EXIT_ZONE], True, (0, 0, 255), 2)
        cv2.putText(display, "EXIT ZONE",
                    (EXIT_ZONE[0][0] + 5, EXIT_ZONE[0][1] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # ── 5b. Run detection ─────────────────────────────────────────────
        coco_results = coco_model(
            frame, conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD, verbose=False
        )[0]

        detections = sv.Detections.from_ultralytics(coco_results)

        # Filter tiny detections
        if len(detections) > 0:
            areas = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * \
                    (detections.xyxy[:, 3] - detections.xyxy[:, 1])
            detections = detections[areas > MIN_BOX_AREA]

        # Filter non-retail classes
        if len(detections) > 0 and detections.class_id is not None:
            mask = np.array([c in ALLOWED_CLASSES for c in detections.class_id])
            detections = detections[mask]

        # ── 5c. Run ByteTrack ─────────────────────────────────────────────
        tracked = tracker.update_with_detections(detections)

        # ── 5d. Update object registry and check zones ────────────────────
        for i, tracker_id in enumerate(tracked.tracker_id):
            if tracker_id is None:
                continue

            object_last_seen[tracker_id] = frame_count

            # Get object name
            if tracker_id not in object_names:
                class_id = tracked.class_id[i] if tracked.class_id is not None else -1
                conf = tracked.confidence[i] if tracked.confidence is not None else None
                object_names[tracker_id] = get_class_name(class_id, conf)

            # Get center point of object
            center = get_center(tracked.xyxy[i])

            # Determine which zone object is in
            prev_zone = object_zones.get(tracker_id, 'unknown')

            if point_in_zone(center, SHELF_ZONE):
                current_zone = 'shelf'
            elif point_in_zone(center, EXIT_ZONE):
                current_zone = 'exit'
            else:
                current_zone = 'floor'

            # Track if object was ever seen on shelf
            if current_zone == 'shelf':
                object_shelf_frames[tracker_id] += 1
                # Mark this object as "was on shelf"
                object_seen_on_shelf.add(tracker_id)

            # ── THEFT DETECTION LOGIC ─────────────────────────────────────────────
            # Any object detected in exit zone = THEFT (nothing should be there)
            if current_zone == 'exit':
                if tracker_id not in already_alerted:
                    name = object_names.get(tracker_id, 'object')
                    alert_msg = f"🚨 THEFT ALERT: {name} (ID#{tracker_id}) detected in exit zone!"
                    print(alert_msg)

                    theft_log.append({
                        'id': tracker_id,
                        'name': name,
                        'frame': frame_count,
                        'time': time.strftime('%H:%M:%S')
                    })

                    already_alerted.add(tracker_id)

                cv2.putText(display, "!! THEFT DETECTED !!",
                            (100, 240),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0, 0, 255), 4)

            object_zones[tracker_id] = current_zone


        # ── 5e. Clean up lost objects (no alert — just remove from registry) ──
        for obj_id, last_frame in list(object_last_seen.items()):
            if frame_count - last_frame > DISAPPEARANCE_THRESHOLD:
                del object_last_seen[obj_id]

        # ── 5f. Build labels ──────────────────────────────────────────────
        labels = []
        if tracked.tracker_id is not None:
            for i, tracker_id in enumerate(tracked.tracker_id):
                name = object_names.get(tracker_id, 'object')
                zone = object_zones.get(tracker_id, 'unknown')
                conf = tracked.confidence[i] if tracked.confidence is not None else 0
                labels.append(f"{name} ID#{tracker_id} [{zone}]")

        # ── 5g. Draw boxes and labels ─────────────────────────────────────
        display = box_annotator.annotate(scene=display, detections=tracked)
        display = label_annotator.annotate(
            scene=display, detections=tracked, labels=labels
        )

        # ── 5h. Show stats ────────────────────────────────────────────────
        cv2.putText(display, f"Objects: {len(tracked)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"Alerts: {len(alert_log)}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        cv2.putText(display, f"Thefts: {len(theft_log)}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
        cv2.putText(display, f"Frame: {frame_count}",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # ── 5i. Display ───────────────────────────────────────────────────
        cv2.imshow('Retail Vision - Virtual Fence', display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            screenshot_path = f"screenshot_frame_{frame_count}.jpg"
            cv2.imwrite(screenshot_path, display)
            print(f"Screenshot saved: {screenshot_path}")

    # ── 6. Cleanup ────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()

    # ── 7. Final summary ──────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("SESSION SUMMARY")
    print("=" * 50)
    print(f"Total frames: {frame_count}")
    print(f"Total alerts: {len(alert_log)}")
    print(f"Total thefts detected: {len(theft_log)}")

    if theft_log:
        print("\nTheft Log:")
        for t in theft_log:
            print(f"  [{t['time']}] {t['name']} ID#{t['id']} — STOLEN")

    if alert_log:
        print("\nAlert Log:")
        for a in alert_log:
            print(f"  [{a['time']}] {a['name']} ID#{a['id']} removed")
    print("=" * 50)

if __name__ == '__main__':
    main()