# train.py
# Day 2: YOLOv8 Training Script
# Purpose: Fine-tune YOLOv8m on SKU-110K retail dataset
# Impact: This produces our core detection model used in ALL subsequent steps

from ultralytics import YOLO
import torch
import os

def main():
    # ── 1. Verify GPU is available ──────────────────────────────────────────
    print("=" * 50)
    print("RETAIL VISION - YOLOv8 TRAINING")
    print("=" * 50)
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("WARNING: No GPU found. Training on CPU will be very slow.")
    
    print("=" * 50)

    # ── 2. Load pretrained YOLOv8m model ────────────────────────────────────
    # 'yolov8m.pt' downloads automatically from Ultralytics on first run
    # Pretrained on COCO dataset - already understands basic visual features
    model = YOLO('yolov8m.pt')
    print("Model loaded: YOLOv8m")

    # ── 3. Define dataset path ───────────────────────────────────────────────
    dataset_yaml = os.path.join(os.getcwd(), 'dataset.yaml')
    print(f"Dataset config: {dataset_yaml}")

    # ── 4. Start Training ────────────────────────────────────────────────────
    results = model.train(
        data=dataset_yaml,        # our custom yaml file
        epochs=50,                # 50 passes through the dataset
        imgsz=640,                # image size (YOLOv8 standard)
        batch=8,                  # 8 images per batch (safe for 6GB VRAM)
        device=0,                 # GPU 0 (your RTX 3060)
        workers=4,                # parallel data loading
        project='runs/train',     # save results here
        name='retail_v1',         # experiment name
        pretrained=True,          # use pretrained weights
        optimizer='SGD',          # SGD optimizer works best for detection
        lr0=0.01,                 # initial learning rate
        momentum=0.937,           # SGD momentum
        weight_decay=0.0005,      # regularization to prevent overfitting
        patience=20,              # stop early if no improvement for 20 epochs
        save=True,                # save best model
        plots=True,               # generate training graphs
        verbose=True              # show detailed progress
    )

    # ── 5. Print results location ────────────────────────────────────────────
    print("=" * 50)
    print("TRAINING COMPLETE!")
    print(f"Best model saved at: runs/train/retail_v1/weights/best.pt")
    print(f"Last model saved at: runs/train/retail_v1/weights/last.pt")
    print("=" * 50)

if __name__ == '__main__':
    main()