# generate_test_video.py
# Creates a synthetic test video simulating theft detection
# A red circle (product) moves from shelf zone to exit zone

import cv2
import numpy as np

# Video settings
WIDTH, HEIGHT = 720, 404
FPS = 30
DURATION = 10  # seconds
OUTPUT = 'theft_test_video.mp4'

# Zone definitions (matching fence.py)
SHELF_ZONE = np.array([[10, 10], [710, 10], [710, 200], [10, 200]])
EXIT_ZONE = np.array([[10, 250], [710, 250], [710, 400], [10, 400]])

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT, fourcc, FPS, (WIDTH, HEIGHT))

total_frames = FPS * DURATION

print(f"Generating {DURATION} second test video...")

for frame_num in range(total_frames):
    # Create background (store-like gray)
    frame = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 200

    # Draw shelf background
    cv2.rectangle(frame, (10, 10), (710, 200), (180, 180, 180), -1)
    cv2.putText(frame, "STORE SHELF", (270, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)

    # Draw floor background
    cv2.rectangle(frame, (10, 200), (710, 404), (160, 160, 160), -1)
    cv2.putText(frame, "STORE FLOOR", (270, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)

    # Draw shelf zone (green outline)
    cv2.polylines(frame, [SHELF_ZONE], True, (0, 255, 0), 2)
    cv2.putText(frame, "SHELF ZONE",
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw exit zone (red outline)
    cv2.polylines(frame, [EXIT_ZONE], True, (0, 0, 255), 2)
    cv2.putText(frame, "EXIT ZONE",
                (15, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # ── Animate the "product" (bottle shaped rectangle) ──────────────────
    progress = frame_num / total_frames

    if progress < 0.3:
        # Phase 1: Object sitting still on shelf (0-3 seconds)
        obj_x, obj_y = 350, 100
        color = (0, 100, 255)  # orange = on shelf
        status = "Bottle sitting on shelf"

    elif progress < 0.6:
        # Phase 2: Object being picked up and moved down (3-6 seconds)
        move_progress = (progress - 0.3) / 0.3
        obj_x = int(350 + move_progress * 50)
        obj_y = int(100 + move_progress * 200)
        color = (0, 165, 255)  # orange = moving
        status = "Bottle being taken..."

    else:
        # Phase 3: Object in exit zone (6-10 seconds)
        obj_x, obj_y = 400, 320
        color = (0, 0, 255)  # red = in exit zone
        status = "!! THEFT DETECTED !!"

    # Draw the product (bottle shape)
    # Body
    cv2.rectangle(frame,
                  (obj_x - 20, obj_y - 40),
                  (obj_x + 20, obj_y + 40),
                  color, -1)
    # Neck
    cv2.rectangle(frame,
                  (obj_x - 8, obj_y - 60),
                  (obj_x + 8, obj_y - 40),
                  color, -1)
    # Label
    cv2.putText(frame, "BTL",
                (obj_x - 15, obj_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Draw person (simple stick figure carrying the bottle)
    if progress >= 0.3:
        person_x = obj_x + 40
        person_y = obj_y - 20
        # Head
        cv2.circle(frame, (person_x, person_y - 30), 15, (50, 50, 150), -1)
        # Body
        cv2.line(frame, (person_x, person_y - 15),
                 (person_x, person_y + 40), (50, 50, 150), 3)
        # Arms
        cv2.line(frame, (person_x, person_y),
                 (person_x - 35, person_y + 10), (50, 50, 150), 3)
        cv2.line(frame, (person_x, person_y),
                 (person_x + 20, person_y + 10), (50, 50, 150), 3)
        # Legs
        cv2.line(frame, (person_x, person_y + 40),
                 (person_x - 15, person_y + 80), (50, 50, 150), 3)
        cv2.line(frame, (person_x, person_y + 40),
                 (person_x + 15, person_y + 80), (50, 50, 150), 3)

    # Status text
    status_color = (0, 0, 255) if "THEFT" in status else (0, 100, 0)
    cv2.putText(frame, status, (10, 390),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # Frame counter
    cv2.putText(frame, f"Frame: {frame_num}/{total_frames}",
                (550, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    out.write(frame)

out.release()
print(f"Video generated: {OUTPUT}")
print(f"Total frames: {total_frames}")
print("Now run: python fence.py (after updating VIDEO_PATH to theft_test_video.mp4)")