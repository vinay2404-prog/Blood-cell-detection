"""Error analysis: save annotated images for qualitative inspection."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core
from evaluate_dataset import CLASS_NAMES, LABEL_DIR, discover_images, load_yolo_labels

OUT = ROOT / "evaluation_results" / "error_analysis"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    images = discover_images()
    model = core.load_model(str(core.MODEL_PATH))
    saved = 0
    for image_path in images[:10]:
        image = cv2.imread(str(image_path))
        h, w = image.shape[:2]
        label_path = LABEL_DIR / f"{image_path.stem}.txt"
        gt = load_yolo_labels(label_path, w, h)
        pred = core.run_yolo_detection(model, image, core.YOLO_CONFIG, draw=False)
        vis = image.copy()
        # GT boxes in green
        for g in gt:
            x1, y1, x2, y2 = map(int, g["box"])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Pred boxes in red
        for b in pred["boxes"]:
            x1, y1 = b["x"], b["y"]
            x2, y2 = x1 + b["w"], y1 + b["h"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(vis, f"{b['label']} {b['confidence']:.0f}", (x1, max(15, y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
        out_path = OUT / f"{image_path.stem}_error.png"
        cv2.imwrite(str(out_path), vis)
        saved += 1
    print(f"Saved {saved} error-analysis images to {OUT}")


if __name__ == "__main__":
    main()