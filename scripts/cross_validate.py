"""Cross-validation style stability evaluation on the test set.

Splits the test set into k folds and reports per-fold and aggregate metrics
to quantify evaluation stability. (Model is not retrained; this assesses
metric variance across data subsets.)
"""
from __future__ import annotations
import sys

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core
from evaluate_dataset import CLASS_NAMES, LABEL_DIR, discover_images, load_yolo_labels

K = 5
OPERATING_CONF = 0.50
IOU = 0.50


def match_metrics(gt_boxes, gt_classes, boxes, classes, confs):
    filtered = [(b, c, cf) for b, c, cf in zip(boxes, classes, confs) if cf >= OPERATING_CONF]
    if filtered:
        fb = np.asarray([x[0] for x in filtered]).reshape(-1, 4)
        fc = np.asarray([x[1] for x in filtered])
        fcf = np.asarray([x[2] for x in filtered])
    else:
        fb = np.zeros((0, 4)); fc = np.asarray([], dtype=int); fcf = np.asarray([], dtype=float)
    tp = fp = fn = 0
    for cid in CLASS_NAMES:
        gt_idx = np.where(gt_classes == cid)[0]
        pred_idx = np.where(fc == cid)[0]
        matched = set()
        ordered = sorted(pred_idx, key=lambda i: fcf[i], reverse=True)
        for pi in ordered:
            best_iou = 0.0; best_gt = None
            for gi in gt_idx:
                if gi in matched:
                    continue
                iou = core.calculate_iou(fb[pi], gt_boxes[gi])
                if iou > best_iou:
                    best_iou = iou; best_gt = gi
            if best_gt is not None and best_iou >= IOU:
                tp += 1; matched.add(best_gt)
            else:
                fp += 1
        fn += len(gt_idx) - len(matched)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def main():
    images = discover_images()
    model = core.load_model(str(core.MODEL_PATH))
    kf = KFold(n_splits=K, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(images), 1):
        f1s = []
        for i in test_idx:
            image_path = images[i]
            image = cv2.imread(str(image_path))
            h, w = image.shape[:2]
            label_path = LABEL_DIR / f"{image_path.stem}.txt"
            gt = load_yolo_labels(label_path, w, h)
            gt_boxes = np.asarray([g["box"] for g in gt], dtype=float).reshape(-1, 4)
            gt_classes = np.asarray([g["class_id"] for g in gt], dtype=int)
            pred = core.run_yolo_detection(model, image, core.YOLO_CONFIG, draw=False)
            boxes = np.asarray([[b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]] for b in pred["boxes"]], dtype=float).reshape(-1, 4)
            classes = np.asarray([list(CLASS_NAMES.keys())[list(CLASS_NAMES.values()).index(b["label"])] for b in pred["boxes"]], dtype=int)
            confs = np.asarray([b["confidence"] / 100.0 for b in pred["boxes"]], dtype=float)
            _, _, f1 = match_metrics(gt_boxes, gt_classes, boxes, classes, confs)
            f1s.append(f1)
        fold_results.append({"fold": fold, "mean_f1": float(np.mean(f1s)), "std_f1": float(np.std(f1s)), "n": len(f1s)})

    summary = {
        "k": K,
        "folds": fold_results,
        "overall_mean_f1": float(np.mean([f["mean_f1"] for f in fold_results])),
        "overall_std_f1": float(np.std([f["mean_f1"] for f in fold_results])),
    }
    out = ROOT / "evaluation_results" / "cross_validation_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()