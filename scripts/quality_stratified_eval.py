"""Quality-stratified evaluation.

Groups test images by configured quality bands and reports baseline vs QAPS
metrics per group, supporting robustness analysis for a research paper.
"""
from __future__ import annotations
import sys

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core
from evaluate_dataset import CLASS_NAMES, LABEL_DIR, discover_images, load_yolo_labels

OPERATING_CONF = 0.50
IOU = 0.50


def quality_band(q):
    bands = []
    if q["blur_score"] < core.THRESHOLDS["blur"]:
        bands.append("blurry")
    else:
        bands.append("sharp")
    if q["brightness"] < core.THRESHOLDS["brightness_low"]:
        bands.append("dark")
    elif q["brightness"] > core.THRESHOLDS["brightness_high"]:
        bands.append("bright")
    else:
        bands.append("normal_brightness")
    if q["noise_score"] >= core.THRESHOLDS["noise"]:
        bands.append("noisy")
    else:
        bands.append("clean")
    return "|".join(bands)


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
    groups = {}

    for image_path in images:
        image = cv2.imread(str(image_path))
        h, w = image.shape[:2]
        label_path = LABEL_DIR / f"{image_path.stem}.txt"
        gt = load_yolo_labels(label_path, w, h)
        gt_boxes = np.asarray([g["box"] for g in gt], dtype=float).reshape(-1, 4)
        gt_classes = np.asarray([g["class_id"] for g in gt], dtype=int)
        q = core.assess_image_quality(image)
        band = quality_band(q)

        # baseline
        bpred = core.run_yolo_detection(model, image, core.EVAL_YOLO_CONFIG, draw=False)
        bboxes = np.asarray([[x["x"], x["y"], x["x"] + x["w"], x["y"] + x["h"]] for x in bpred["boxes"]], dtype=float).reshape(-1, 4)
        bclasses = np.asarray([list(CLASS_NAMES.keys())[list(CLASS_NAMES.values()).index(x["label"])] for x in bpred["boxes"]], dtype=int)
        bconfs = np.asarray([x["confidence"] / 100.0 for x in bpred["boxes"]], dtype=float)
        bp, br, bf = match_metrics(gt_boxes, gt_classes, bboxes, bclasses, bconfs)

        # qaps
        processed, _ = core.adaptive_preprocessing(image, q)
        qpred = core.run_yolo_detection(model, processed, core.EVAL_YOLO_CONFIG, draw=False)
        qboxes = np.asarray([[x["x"], x["y"], x["x"] + x["w"], x["y"] + x["h"]] for x in qpred["boxes"]], dtype=float).reshape(-1, 4)
        qclasses = np.asarray([list(CLASS_NAMES.keys())[list(CLASS_NAMES.values()).index(x["label"])] for x in qpred["boxes"]], dtype=int)
        qconfs = np.asarray([x["confidence"] / 100.0 for x in qpred["boxes"]], dtype=float)
        qp, qr, qf = match_metrics(gt_boxes, gt_classes, qboxes, qclasses, qconfs)

        groups.setdefault(band, {"baseline_f1": [], "qaps_f1": [], "n": 0})
        groups[band]["baseline_f1"].append(bf)
        groups[band]["qaps_f1"].append(qf)
        groups[band]["n"] += 1

    summary = {}
    for band, v in groups.items():
        summary[band] = {
            "images": v["n"],
            "baseline_f1": float(np.mean(v["baseline_f1"])),
            "qaps_f1": float(np.mean(v["qaps_f1"])),
            "delta": float(np.mean(v["qaps_f1"]) - np.mean(v["baseline_f1"])),
        }

    out = ROOT / "evaluation_results" / "quality_stratified_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()