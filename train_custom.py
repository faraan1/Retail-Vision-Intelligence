# train_custom.py
# Day 8: Custom Product Detection Training
# Purpose: Train YOLOv8 on Cola Next and Oreo dataset
# Impact: Model will now identify specific products by name

from ultralytics import YOLO
import torch
import os

def main():
    print("=" * 50)
    print("CUSTOM PRODUCT DETECTION TRAINING")
    print("=" * 50)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("WARNING: No GPU found!")

    print("=" * 50)

    # Load pretrained YOLOv8m
    model = YOLO('yolov8m.pt')
    print("Model loaded: YOLOv8m")

    # Dataset config
    dataset_yaml = os.path.join(os.getcwd(), 'custom_dataset.yaml')
    print(f"Dataset: {dataset_yaml}")
    print(f"Classes: 34 (33 Cola Next variants + Oreo)")

    # Train
    results = model.train(
        data=dataset_yaml,
        epochs=20,
        imgsz=640,
        batch=8,
        device=0,
        workers=4,
        project='runs/train',
        name='custom_products_v1',
        pretrained=True,
        optimizer='SGD',
        lr0=0.01,
        patience=10,
        save=True,
        plots=True,
        verbose=True
    )

    print("=" * 50)
    print("TRAINING COMPLETE!")
    print("Best model: runs/train/custom_products_v1/weights/best.pt")
    print("=" * 50)

if __name__ == '__main__':
    main()