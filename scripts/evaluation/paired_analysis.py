
from __future__ import annotations
import sys

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# =========================================================
# 1. CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

JSON_PATH = BASE_DIR / "evaluation_results" / "image_details.json"
IMAGE_DIR = BASE_DIR / "data" / "test" / "images"
LABEL_DIR = BASE_DIR / "data" / "test" / "labels"

OUTPUT_DIR = BASE_DIR / "evaluation_results" / "paired_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50

CLASS_NAMES = {
    0: "Platelets",
    1: "RBC",
    2: "WBC",
}

N_BOOTSTRAP = 5000
RANDOM_SEED = 42


# =========================================================
# 2. VALIDATE INPUT PATHS
# =========================================================

for path in [JSON_PATH, IMAGE_DIR, LABEL_DIR]:
    if not path.exists():
        raise FileNotFoundError(f"Missing path: {path}")

with open(JSON_PATH, "r", encoding="utf-8") as f:
    records = json.load(f)

if not isinstance(records, list):
    raise ValueError("Expected image_details.json to contain a list.")

if not records:
    raise ValueError("No image records found.")

print(f"Loaded {len(records)} image records.")


# =========================================================
# 3. BOUNDING BOX UTILITIES
# =========================================================

def calculate_iou(box_a, box_b):
    """Calculate IoU between two xyxy bounding boxes."""

    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, box_a[2] - box_a[0]) * max(
        0, box_a[3] - box_a[1]
    )

    area_b = max(0, box_b[2] - box_b[0]) * max(
        0, box_b[3] - box_b[1]
    )

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def load_ground_truth(label_path, width, height):
    """
    Read YOLO labels:
    class_id, x_center, y_center, box_width, box_height
    """

    boxes = []
    classes = []

    if not label_path.exists():
        raise FileNotFoundError(
            f"Missing annotation file: {label_path}"
        )

    with open(label_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):

            parts = line.strip().split()

            if not parts:
                continue

            if len(parts) != 5:
                raise ValueError(
                    f"Invalid YOLO label at {label_path}, "
                    f"line {line_number}"
                )

            class_id = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:])

            if class_id not in CLASS_NAMES:
                raise ValueError(
                    f"Unexpected class ID {class_id} in {label_path}"
                )

            x1 = (xc - w / 2) * width
            y1 = (yc - h / 2) * height
            x2 = (xc + w / 2) * width
            y2 = (yc + h / 2) * height

            boxes.append([x1, y1, x2, y2])
            classes.append(class_id)

    return (
        np.asarray(boxes, dtype=float).reshape(-1, 4),
        np.asarray(classes, dtype=int),
    )


def prepare_predictions(predictions, width, height):
    """Filter predictions at the operating confidence threshold."""

    boxes = []
    classes = []
    confidences = []

    for prediction in predictions:

        confidence = float(prediction["confidence"])

        if confidence < CONF_THRESHOLD:
            continue

        class_id = int(prediction["class_id"])

        if class_id not in CLASS_NAMES:
            raise ValueError(f"Unexpected predicted class: {class_id}")

        box = np.asarray(prediction["box"], dtype=float)

        if box.shape != (4,) or not np.all(np.isfinite(box)):
            raise ValueError("Invalid prediction bounding box.")

        # YOLO boxes.xyxy should be in original image pixel coordinates.
        # Clip only tiny floating-point boundary differences.
        if (
            np.any(box < -1.0)
            or box[2] > width + 1.0
            or box[3] > height + 1.0
        ):
            raise ValueError(
                "Prediction coordinates do not match image dimensions. "
                f"Image size: {width}x{height}, box: {box.tolist()}. "
                "Check the coordinate system before continuing."
            )

        box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, height)

        boxes.append(box)
        classes.append(class_id)
        confidences.append(confidence)

    return (
        np.asarray(boxes, dtype=float).reshape(-1, 4),
        np.asarray(classes, dtype=int),
        np.asarray(confidences, dtype=float),
    )


# =========================================================
# 4. CLASS-AWARE ONE-TO-ONE MATCHING
# =========================================================

def evaluate_image(gt_boxes, gt_classes, predictions):
    """
    Greedy confidence-ordered matching.
    Each ground-truth object and prediction can match only once.
    """

    pred_boxes, pred_classes, confidences = predictions

    results = {}

    for class_id in CLASS_NAMES:

        gt_indices = np.where(gt_classes == class_id)[0]
        pred_indices = np.where(pred_classes == class_id)[0]

        matched_gt = set()

        tp = 0
        fp = 0

        # Highest-confidence detections are matched first.
        ordered_predictions = sorted(
            pred_indices,
            key=lambda i: confidences[i],
            reverse=True,
        )

        for pred_index in ordered_predictions:

            best_iou = 0.0
            best_gt_index = None

            for gt_index in gt_indices:

                if gt_index in matched_gt:
                    continue

                iou = calculate_iou(
                    pred_boxes[pred_index],
                    gt_boxes[gt_index],
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_index = gt_index

            if (
                best_gt_index is not None
                and best_iou >= IOU_THRESHOLD
            ):
                tp += 1
                matched_gt.add(best_gt_index)
            else:
                fp += 1

        fn = len(gt_indices) - len(matched_gt)

        results[class_id] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    return results


# =========================================================
# 5. EVALUATE EVERY IMAGE
# =========================================================

image_results = []

for record in records:

    image_id = record["image_id"]

    image_path = IMAGE_DIR / image_id
    label_path = LABEL_DIR / (Path(image_id).stem + ".txt")

    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    height, width = image.shape[:2]

    gt_boxes, gt_classes = load_ground_truth(
        label_path, width, height
    )

    if len(gt_boxes) != int(record["ground_truth_objects"]):
        raise ValueError(
            f"Ground-truth count mismatch for {image_id}: "
            f"JSON={record['ground_truth_objects']}, "
            f"labels={len(gt_boxes)}"
        )

    baseline_predictions = prepare_predictions(
        record["baseline_predictions"], width, height
    )

    qaps_predictions = prepare_predictions(
        record["qaps_predictions"], width, height
    )

    baseline_metrics = evaluate_image(
        gt_boxes, gt_classes, baseline_predictions
    )

    qaps_metrics = evaluate_image(
        gt_boxes, gt_classes, qaps_predictions
    )

    image_results.append({
        "image_id": image_id,
        "baseline": baseline_metrics,
        "qaps": qaps_metrics,
    })

print(f"Successfully evaluated {len(image_results)} images.")


# =========================================================
# 6. AGGREGATE METRICS
# =========================================================

def aggregate_counts(results, method, indices=None):

    if indices is None:
        indices = range(len(results))

    totals = {
        class_id: {"tp": 0, "fp": 0, "fn": 0}
        for class_id in CLASS_NAMES
    }

    for index in indices:

        image_metrics = results[index][method]

        for class_id in CLASS_NAMES:

            for metric in ("tp", "fp", "fn"):
                totals[class_id][metric] += (
                    image_metrics[class_id][metric]
                )

    return totals


def calculate_metrics(counts):

    class_metrics = {}

    precision_values = []
    recall_values = []
    f1_values = []

    for class_id, values in counts.items():

        tp = values["tp"]
        fp = values["fp"]
        fn = values["fn"]

        precision = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / (tp + fn) if tp + fn else np.nan
        f1 = (
            2 * tp / (2 * tp + fp + fn)
            if 2 * tp + fp + fn else np.nan
        )

        class_metrics[class_id] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        }

        if np.isfinite(precision):
            precision_values.append(precision)

        if np.isfinite(recall):
            recall_values.append(recall)

        if np.isfinite(f1):
            f1_values.append(f1)

    return {
        "class_metrics": class_metrics,
        "macro_precision": float(np.mean(precision_values)),
        "macro_recall": float(np.mean(recall_values)),
        "macro_f1": float(np.mean(f1_values)),
    }


baseline_counts = aggregate_counts(image_results, "baseline")
qaps_counts = aggregate_counts(image_results, "qaps")

baseline_summary = calculate_metrics(baseline_counts)
qaps_summary = calculate_metrics(qaps_counts)


# =========================================================
# 7. SAVE FULL-TEST CLASS METRICS
# =========================================================

class_rows = []

for method, summary in [
    ("Baseline YOLO", baseline_summary),
    ("QAPS-YOLO", qaps_summary),
]:

    for class_id, metrics in summary["class_metrics"].items():

        class_rows.append({
            "method": method,
            "class": CLASS_NAMES[class_id],
            **metrics,
        })

pd.DataFrame(class_rows).to_csv(
    OUTPUT_DIR / "paired_class_metrics.csv",
    index=False,
)


# =========================================================
# 8. SAVE PER-IMAGE COUNTS
# =========================================================

image_rows = []

for result in image_results:

    row = {"image_id": result["image_id"]}

    for method in ("baseline", "qaps"):

        for class_id, values in result[method].items():

            prefix = f"{method}_{CLASS_NAMES[class_id]}"

            for metric in ("tp", "fp", "fn"):
                row[f"{prefix}_{metric}"] = values[metric]

    image_rows.append(row)

pd.DataFrame(image_rows).to_csv(
    OUTPUT_DIR / "paired_image_metrics.csv",
    index=False,
)


# =========================================================
# 9. PAIRED IMAGE-LEVEL BOOTSTRAP
# =========================================================

rng = np.random.default_rng(RANDOM_SEED)

bootstrap_metrics = {
    "macro_precision": {"baseline": [], "qaps": []},
    "macro_recall": {"baseline": [], "qaps": []},
    "macro_f1": {"baseline": [], "qaps": []},
}

n_images = len(image_results)

for _ in range(N_BOOTSTRAP):

    # The same sampled images are used for both methods.
    sampled_indices = rng.integers(
        0, n_images, size=n_images
    )

    for method in ("baseline", "qaps"):

        counts = aggregate_counts(
            image_results,
            method,
            sampled_indices,
        )

        metrics = calculate_metrics(counts)

        for metric_name in bootstrap_metrics:
            bootstrap_metrics[metric_name][method].append(
                metrics[metric_name]
            )


# =========================================================
# 10. CONFIDENCE INTERVALS FOR PAIRED DIFFERENCES
# =========================================================

bootstrap_summary = {}

for metric_name, values in bootstrap_metrics.items():

    baseline_values = np.asarray(values["baseline"])
    qaps_values = np.asarray(values["qaps"])

    differences = qaps_values - baseline_values

    bootstrap_summary[metric_name] = {
        "baseline_full_test": baseline_summary[metric_name],
        "qaps_full_test": qaps_summary[metric_name],
        "difference_qaps_minus_baseline": (
            qaps_summary[metric_name]
            - baseline_summary[metric_name]
        ),
        "bootstrap_ci_95": [
            float(np.percentile(differences, 2.5)),
            float(np.percentile(differences, 97.5)),
        ],
        "bootstrap_replicates": N_BOOTSTRAP,
    }


# =========================================================
# 11. SAVE RESULTS
# =========================================================

final_results = {
    "configuration": {
        "images": n_images,
        "confidence_threshold": CONF_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "bootstrap_samples": N_BOOTSTRAP,
        "random_seed": RANDOM_SEED,
        "matching": "class-aware greedy one-to-one",
    },
    "baseline_class_metrics": baseline_summary["class_metrics"],
    "qaps_class_metrics": qaps_summary["class_metrics"],
    "bootstrap_comparison": bootstrap_summary,
}

with open(
    OUTPUT_DIR / "paired_bootstrap_results.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(final_results, f, indent=4, allow_nan=False)


# =========================================================
# 12. DISPLAY RESULTS
# =========================================================

print("\n========== PAIRED ANALYSIS RESULTS ==========")

for metric_name, values in bootstrap_summary.items():

    low, high = values["bootstrap_ci_95"]

    print(f"\n{metric_name}")

    print(
        f"Baseline: "
        f"{values['baseline_full_test']:.6f}"
    )

    print(
        f"QAPS: "
        f"{values['qaps_full_test']:.6f}"
    )

    print(
        f"Difference: "
        f"{values['difference_qaps_minus_baseline']:+.6f}"
    )

    print(
        f"95% paired bootstrap CI: "
        f"[{low:.6f}, {high:.6f}]"
    )

print("\nPaired analysis completed.")
print("Results saved in:", OUTPUT_DIR)
