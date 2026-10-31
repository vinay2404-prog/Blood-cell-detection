import time
from pathlib import Path
import numpy as np
from ultralytics import YOLO

def main():
    # 1. Load your trained model weights
    # Automatically checks for 'runs/detect/train/weights/best.pt' or falls back to 'yolo11.pt'
    model_path = Path("runs/detect/train/weights/best.pt")
    if not model_path.exists():
        model_path = Path("yolo11.pt")
    
    print(f"Loading YOLO model from: {model_path}")
    model = YOLO(str(model_path))

    # 2. Run validation evaluation using your custom blood cell data.yaml
    print("\n--- Running Evaluation on Validation Dataset ---")
    try:
        metrics = model.val(data="data.yaml", split="val", verbose=True)
        
        # Extract mean results across classes
        p, r, map50, map95 = metrics.box.mean_results()
        
        print("\n========================================")
        print("      EVALUATION RESULTS SUMMARY        ")
        print("========================================")
        print(f"Overall Precision : {p:.4f}")
        print(f"Overall Recall    : {r:.4f}")
        print(f"Overall mAP@50    : {map50:.4f}")
        print(f"Overall mAP@50-95 : {map95:.4f}")
        print("========================================")
        
    except Exception as e:
        print(f"\n[Warning] Could not complete model.val(): {e}")
        print("Please ensure your 'data.yaml' file exists and points correctly to your dataset paths.")

    # 3. Measure Average Inference Latency (Speed Test)
    print("\n--- Measuring Inference Latency ---")
    dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
    latencies = []

    # Warmup and benchmark runs
    for _ in range(35):
        t0 = time.perf_counter()
        _ = model.predict(dummy_image, verbose=False)
        latencies.append((time.perf_counter() - t0) * 1000)

    avg_latency = np.mean(latencies[5:])  # Skip first 5 warmup runs
    print(f"Average Inference Latency: {avg_latency:.2f} ms / image")

if __name__ == "__main__":
    main()