# eval.py
# Day 9: Model Evaluation
# Purpose: Generate evaluation metrics for FYP presentation
# Impact: Proves model performance with numbers and graphs

from ultralytics import YOLO
import torch
import time
import cv2
import numpy as np

def main():
    print("=" * 50)
    print("RETAIL VISION - MODEL EVALUATION")
    print("=" * 50)

    # ── 1. Evaluate SKU-110K model ────────────────────────────────────────
    print("\n📊 Evaluating SKU-110K General Model...")
    model_general = YOLO('runs/train/retail_v1/weights/best.pt')

    results_general = model_general.val(
        data='dataset.yaml',
        imgsz=640,
        batch=8,
        device=0,
        plots=True,
        save_json=True,
        project='runs/eval',
        name='sku110k_eval'
    )

    print("\n✅ SKU-110K Model Results:")
    print(f"  mAP50:     {results_general.box.map50:.3f}")
    print(f"  mAP50-95:  {results_general.box.map:.3f}")
    print(f"  Precision: {results_general.box.mp:.3f}")
    print(f"  Recall:    {results_general.box.mr:.3f}")

    # ── 2. Evaluate Custom Product model ──────────────────────────────────
    print("\n📊 Evaluating Custom Product Model...")
    model_custom = YOLO('runs/train/custom_products_v1/weights/best.pt')

    results_custom = model_custom.val(
        data='custom_dataset.yaml',
        imgsz=640,
        batch=8,
        device=0,
        plots=True,
        save_json=True,
        project='runs/eval',
        name='custom_eval'
    )

    print("\n✅ Custom Product Model Results:")
    print(f"  mAP50:     {results_custom.box.map50:.3f}")
    print(f"  mAP50-95:  {results_custom.box.map:.3f}")
    print(f"  Precision: {results_custom.box.mp:.3f}")
    print(f"  Recall:    {results_custom.box.mr:.3f}")

    # ── 3. FPS Benchmark ──────────────────────────────────────────────────
    print("\n⚡ Running FPS Benchmark...")
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)

    # Warmup
    for _ in range(10):
        model_custom(dummy_frame, verbose=False)

    # Benchmark
    times = []
    for _ in range(100):
        start = time.time()
        model_custom(dummy_frame, verbose=False)
        times.append(time.time() - start)

    avg_time = np.mean(times)
    fps = 1.0 / avg_time

    print(f"\n✅ FPS Benchmark Results:")
    print(f"  Average inference time: {avg_time*1000:.1f}ms")
    print(f"  FPS: {fps:.1f}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── 4. Final Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"SKU-110K Model (General Detection):")
    print(f"  mAP50: {results_general.box.map50:.1%}")
    print(f"  Precision: {results_general.box.mp:.1%}")
    print(f"  Recall: {results_general.box.mr:.1%}")
    print(f"\nCustom Product Model:")
    print(f"  mAP50: {results_custom.box.map50:.1%}")
    print(f"  Precision: {results_custom.box.mp:.1%}")
    print(f"  Recall: {results_custom.box.mr:.1%}")
    print(f"\nInference Speed: {fps:.1f} FPS")
    print(f"Plots saved to: runs/eval/")
    print("=" * 50)

if __name__ == '__main__':
    main()