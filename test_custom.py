# test_custom.py
# Test custom trained model on live camera
# Purpose: Verify Cola Next and Oreo detection works in real time

import cv2
from ultralytics import YOLO
import supervision as sv

def main():
    print("=" * 50)
    print("CUSTOM PRODUCT DETECTION TEST")
    print("=" * 50)

    # Load custom trained model
    model = YOLO('runs/train/custom_products_v1/weights/best.pt')
    print("Custom model loaded!")
    print(f"Classes: {model.names}")

    # Annotators
    box_annotator   = sv.BoundingBoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5)

    # Open webcam
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        return

    print("Camera opened! Press 'q' to quit")
    print("=" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        results = model(frame, conf=0.6, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # Filter tiny detections (Fixed Indentation Here)
        if len(detections) > 0:
            areas = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * \
                    (detections.xyxy[:, 3] - detections.xyxy[:, 1])
            detections = detections[areas > 5000]

            # Build labels
            labels = []
            for i, class_id in enumerate(detections.class_id):
                name = model.names[class_id]
                conf = detections.confidence[i]
                labels.append(f"{name} {conf:.2f}")
                print(f"Detected: {name} ({conf:.2f})")

            # Draw
            frame = box_annotator.annotate(scene=frame, detections=detections)
            frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

        # Stats
        cv2.putText(frame, f"Detections: {len(detections)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('Custom Product Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()