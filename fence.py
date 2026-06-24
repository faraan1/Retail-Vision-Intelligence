# fence.py
# Day 4 & 5: Virtual Fence + Alert Logic + Database Integration
# Purpose: Detect zones, trigger theft alerts, save to PostgreSQL
# Impact: Core of the retail vision system

import cv2
import torch
from ultralytics import YOLO
import supervision as sv
import numpy as np
from collections import defaultdict
import time
import winsound
import threading
from database import init_db, log_detection, log_theft, start_session, end_session, check_product_status, mark_as_stolen

# ── Configuration ────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.5
MIN_BOX_AREA = 5000
DISAPPEARANCE_THRESHOLD = 90
MIN_SHELF_FRAMES = 60

# ── Zone definitions ──────────────────────────────────────────────────────────
SHELF_ZONE = np.array([[10, 10], [710, 10], [710, 200], [10, 200]])
EXIT_ZONE  = np.array([[10, 250], [710, 250], [710, 400], [10, 400]])

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

# ── Alarm Control (global) ────────────────────────────────────────────────────
alarm_active = False
alarm_thread = None
alert_active = False
alert_message = ""

def play_alarm():
    """Plays alarm sound in loop until stopped"""
    global alarm_active
    while alarm_active:
        winsound.Beep(1000, 500)
        winsound.Beep(1500, 500)
        winsound.Beep(1000, 500)

def start_alarm(name, tracker_id):
    """Start alarm in background thread and set alert message"""
    global alarm_active, alarm_thread, alert_active, alert_message
    alert_active = True
    alert_message = f"!! THEFT: {name} ID#{tracker_id} STOLEN !!"
    if not alarm_active:
        alarm_active = True
        alarm_thread = threading.Thread(target=play_alarm, daemon=True)
        alarm_thread.start()

def stop_alarm():
    """Stop alarm and clear alert message"""
    global alarm_active, alert_active, alert_message
    alarm_active = False
    alert_active = False
    alert_message = ""

# ── Helper functions ──────────────────────────────────────────────────────────
def get_class_name(class_id, confidence=None):
    if confidence is not None and confidence < 0.70:
        return 'object'
    return RETAIL_CLASSES.get(int(class_id), 'object')

def get_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def point_in_zone(point, zone):
    return cv2.pointPolygonTest(zone.astype(np.float32), point, False) >= 0

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("RETAIL VISION - VIRTUAL FENCE")
    print("=" * 50)

    # ── 1. Load models ────────────────────────────────────────────────────
    print("Loading models...")
    coco_model   = YOLO('yolov8m.pt')
    retail_model = YOLO('runs/train/retail_v1/weights/best.pt')
    print("Models loaded!")

    # ── 2. Tracker and annotators ─────────────────────────────────────────
    tracker        = sv.ByteTrack()
    box_annotator  = sv.BoundingBoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5)

    # ── 3. Open camera ────────────────────────────────────────────────────
    print("\nOpening video source...")
    print("Controls:")
    print("  'q' → quit")
    print("  's' → screenshot")
    print("  'a' → acknowledge alert")
    print("=" * 50)

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        return
    print("Webcam opened successfully!")

    # ── 4. Database init ──────────────────────────────────────────────────
    init_db()
    session_id = start_session()
    print("✅ Database connected!")
    total_detections = 0

    # ── 5. Tracking state ─────────────────────────────────────────────────
    object_last_seen    = defaultdict(int)
    object_names        = {}
    object_zones        = {}
    alert_log           = []
    theft_log           = []
    frame_count         = 0
    object_shelf_frames = defaultdict(int)
    object_seen_on_shelf = set()
    already_alerted     = set()

    # ── 6. Main loop ──────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        display = frame.copy()

        # ── Draw zones ────────────────────────────────────────────────────
        overlay = display.copy()
        cv2.fillPoly(overlay, [SHELF_ZONE], (0, 255, 0))
        cv2.addWeighted(overlay, 0.15, display, 0.85, 0, display)
        cv2.polylines(display, [SHELF_ZONE], True, (0, 255, 0), 2)
        cv2.putText(display, "SHELF ZONE",
                    (SHELF_ZONE[0][0] + 5, SHELF_ZONE[0][1] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        overlay2 = display.copy()
        cv2.fillPoly(overlay2, [EXIT_ZONE], (0, 0, 255))
        cv2.addWeighted(overlay2, 0.15, display, 0.85, 0, display)
        cv2.polylines(display, [EXIT_ZONE], True, (0, 0, 255), 2)
        cv2.putText(display, "EXIT ZONE",
                    (EXIT_ZONE[0][0] + 5, EXIT_ZONE[0][1] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # ── Run detection ─────────────────────────────────────────────────
        coco_results = coco_model(
            frame, conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD, verbose=False
        )[0]

        detections = sv.Detections.from_ultralytics(coco_results)

        if len(detections) > 0:
            areas = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * \
                    (detections.xyxy[:, 3] - detections.xyxy[:, 1])
            detections = detections[areas > MIN_BOX_AREA]

        if len(detections) > 0 and detections.class_id is not None:
            mask = np.array([c in ALLOWED_CLASSES for c in detections.class_id])
            detections = detections[mask]

        # ── ByteTrack ─────────────────────────────────────────────────────
        tracked = tracker.update_with_detections(detections)

        # ── Update registry and check zones ──────────────────────────────
        for i, tracker_id in enumerate(tracked.tracker_id):
            if tracker_id is None:
                continue

            object_last_seen[tracker_id] = frame_count

            if tracker_id not in object_names:
                class_id = tracked.class_id[i] if tracked.class_id is not None else -1
                conf     = tracked.confidence[i] if tracked.confidence is not None else None
                object_names[tracker_id] = get_class_name(class_id, conf)

            center    = get_center(tracked.xyxy[i])
            prev_zone = object_zones.get(tracker_id, 'unknown')

            if point_in_zone(center, SHELF_ZONE):
                current_zone = 'shelf'
            elif point_in_zone(center, EXIT_ZONE):
                current_zone = 'exit'
            else:
                current_zone = 'floor'

            if current_zone == 'shelf':
                object_shelf_frames[tracker_id] += 1
                object_seen_on_shelf.add(tracker_id)

            # ── THEFT DETECTION ───────────────────────────────────────────
            if current_zone == 'exit' and tracker_id not in already_alerted:
                name   = object_names.get(tracker_id, 'object')
                status = check_product_status(int(tracker_id))

                if status == 'available':
                    print(f"🚨 THEFT ALERT: {name} (ID#{tracker_id}) detected in exit zone!")
                    theft_log.append({
                        'id':   tracker_id,
                        'name': name,
                        'frame': frame_count,
                        'time': time.strftime('%H:%M:%S')
                    })
                    already_alerted.add(tracker_id)
                    mark_as_stolen(int(tracker_id))
                    log_theft(name, tracker_id, frame_count)
                    start_alarm(name, tracker_id)   # ← passes name & id correctly

                elif status == 'sold':
                    print(f"✅ {name} (ID#{tracker_id}) cleared — marked as sold")
                    already_alerted.add(tracker_id)

                else:
                    print(f"ℹ️ {name} (ID#{tracker_id}) not in database — ignored")
                    already_alerted.add(tracker_id)

            object_zones[tracker_id] = current_zone

            # Log detection every 30 frames
            if frame_count % 30 == 0:
                conf = tracked.confidence[i] if tracked.confidence is not None else 0.0
                log_detection(
                    object_names.get(tracker_id, 'object'),
                    int(tracker_id),
                    current_zone,
                    float(conf),
                    frame_count
                )
                total_detections += 1

        # ── Clean up lost objects ─────────────────────────────────────────
        for obj_id, last_frame in list(object_last_seen.items()):
            if frame_count - last_frame > DISAPPEARANCE_THRESHOLD:
                del object_last_seen[obj_id]

        # ── Build labels ──────────────────────────────────────────────────
        labels = []
        if tracked.tracker_id is not None:
            for i, tracker_id in enumerate(tracked.tracker_id):
                name  = object_names.get(tracker_id, 'object')
                zone  = object_zones.get(tracker_id, 'unknown')
                conf  = tracked.confidence[i] if tracked.confidence is not None else 0
                labels.append(f"{name} ID#{tracker_id} [{zone}]")

        # ── Draw boxes and labels ─────────────────────────────────────────
        display = box_annotator.annotate(scene=display, detections=tracked)
        display = label_annotator.annotate(scene=display, detections=tracked, labels=labels)

        # ── Persistent alert banner ───────────────────────────────────────
        if alert_active:
            cv2.rectangle(display, (50, 180), (670, 270), (0, 0, 150), -1)
            cv2.putText(display, alert_message,
                        (60, 225),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (255, 255, 255), 2)
            cv2.putText(display, "Press 'A' to acknowledge",
                        (180, 258),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 255), 2)

        # ── Stats ─────────────────────────────────────────────────────────
        cv2.putText(display, f"Objects: {len(tracked)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"Alerts: {len(alert_log)}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        cv2.putText(display, f"Thefts: {len(theft_log)}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(display, f"Frame: {frame_count}",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # ── Display ───────────────────────────────────────────────────────
        cv2.imshow('Retail Vision - Virtual Fence', display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            path = f"screenshot_frame_{frame_count}.jpg"
            cv2.imwrite(path, display)
            print(f"Screenshot saved: {path}")
        elif key == ord('a') or key == ord('A'):
            stop_alarm()
            print("✅ Alert acknowledged by operator")

    # ── Cleanup ───────────────────────────────────────────────────────────
    stop_alarm()
    cap.release()
    cv2.destroyAllWindows()
    end_session(session_id, frame_count, total_detections, len(theft_log))

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("SESSION SUMMARY")
    print("=" * 50)
    print(f"Total frames:  {frame_count}")
    print(f"Total alerts:  {len(alert_log)}")
    print(f"Total thefts:  {len(theft_log)}")
    if theft_log:
        print("\nTheft Log:")
        for t in theft_log:
            print(f"  [{t['time']}] {t['name']} ID#{t['id']} — STOLEN")
    print("=" * 50)

if __name__ == '__main__':
    main()