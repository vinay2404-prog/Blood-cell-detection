"""Tune the operating confidence threshold on a labeled validation set.

Runs inference once at a low confidence, then evaluates a grid of operating
thresholds using the same greedy matching as evaluate_dataset.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathlib import Path

import numpy as np

import core
from evaluate_dataset import CLASS_NAMES, load_yolo_labels

BASE_DIR = Path(__file__).resolve().parent.parent
VALID_IMAGE_DIR = BASE_DIR / "data" / "valid" / "images"
VALID_LABEL_DIR = BASE_DIR / "data" / "valid" / "labels"


def discover_images():
    return sorted(p for p in VALID_IMAGE_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})

THRESHOLDS_TO_TRY = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
IOU_THRESHOLD = 0.50


def main():
    images = discover_images()
    model = core.load_model(str(core.MODEL_PATH))

    records = []
    for image_path in images:
        image = __import__("cv2").imread(str(image_path))
        h, w = image.shape[:2]
        label_path = VALID_LABEL_DIR / f"{image_path.stem}.txt"
        gt = load_yolo_labels(label_path, w, h)
        pred = core.run_yolo_detection(model, image, core.EVAL_YOLO_CONFIG, draw=False)
        # Convert detection boxes to prediction records at conf 0.001 (already filtered by EVAL config)
        preds = []
        for b in pred["boxes"]:
            preds.append({
                "class_id": list(CLASS_NAMES.keys())[list(CLASS_NAMES.values()).index(b["label"])],
                "confidence": b["confidence"] / 100.0,
                "box": [b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]],
            })
        records.append({"image_id": image_path.name, "ground_truth": gt, "predictions": preds})

    print(f"{'conf':>6} {'precision':>10} {'recall':>10} {'f1':>10}")
    best = None
    for conf in THRESHOLDS_TO_TRY:
        tp_total = fp_total = fn_total = 0
        for rec in records:
            gt_boxes = np.asarray([g["box"] for g in rec["ground_truth"]], dtype=float).reshape(-1, 4)
            gt_classes = np.asarray([g["class_id"] for g in rec["ground_truth"]], dtype=int)
            # reuse evaluate_dataset matching
            def evaluate_image(gt_boxes, gt_classes, predictions):
                pred_boxes, pred_classes, confidences = predictions
                results = {}
                for class_id in CLASS_NAMES:
                    gt_indices = np.where(gt_classes == class_id)[0]
                    pred_indices = np.where(pred_classes == class_id)[0]
                    matched_gt = set()
                    tp = fp = 0
                    ordered = sorted(pred_indices, key=lambda i: confidences[i], reverse=True)
                    for pred_index in ordered:
                        best_iou = 0.0
                        best_gt = None
                        for gt_index in gt_indices:
                            if gt_index in matched_gt:
                                continue
                            iou = core.calculate_iou(pred_boxes[pred_index], gt_boxes[gt_index])
                            if iou > best_iou:
                                best_iou = iou
                                best_gt = gt_index
                        if best_gt is not None and best_iou >= IOU_THRESHOLD:
                            tp += 1
                            matched_gt.add(best_gt)
                        else:
                            fp += 1
                    fn = len(gt_indices) - len(matched_gt)
                    results[class_id] = {"tp": tp, "fp": fp, "fn": fn}
                return results
            # Build predictions filtered at conf
            filtered = [p for p in rec["predictions"] if p["confidence"] >= conf]
            # Reconstruct a fake record for evaluate_image
            fake = {"ground_truth": rec["ground_truth"], "predictions": [
                {"class_id": p["class_id"], "confidence": p["confidence"], "box": p["box"]}
                for p in filtered
            ]}
            # evaluate_image expects (gt_boxes, gt_classes, predictions_tuple)
            boxes = np.asarray([p["box"] for p in filtered], dtype=float).reshape(-1, 4) if filtered else np.zeros((0, 4))
            classes = np.asarray([p["class_id"] for p in filtered], dtype=int)
            confs = np.asarray([p["confidence"] for p in filtered], dtype=float)
            metrics = evaluate_image(gt_boxes, gt_classes, (boxes, classes, confs))
            for cid in CLASS_NAMES:
                tp_total += metrics[cid]["tp"]
                fp_total += metrics[cid]["fp"]
                fn_total += metrics[cid]["fn"]

        precision = tp_total / (tp_total + fp_total) if tp_total + fp_total else 0.0
        recall = tp_total / (tp_total + fn_total) if tp_total + fn_total else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        print(f"{conf:>6.2f} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f}")
        if best is None or f1 > best[1]:
            best = (conf, f1)

    print(f"\nBest threshold by F1: {best[0]:.2f} (F1={best[1]:.4f})")


if __name__ == "__main__":
    main()