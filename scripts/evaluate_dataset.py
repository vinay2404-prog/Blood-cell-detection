
from __future__ import annotations
import sys

import csv
import hashlib
import json
import logging
import time

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics import YOLO

from core import (
    CLASS_NAMES as CORE_CLASS_NAMES,
    MODEL_PATH as CORE_MODEL_PATH,
    THRESHOLDS as CORE_THRESHOLDS,
    YOLO_CONFIG as CORE_YOLO_CONFIG,
    adaptive_preprocessing as core_adaptive_preprocessing,
    load_model as core_load_model,
    sha256_file as core_sha256_file,
)

# Keep legacy module-level aliases stable.
CLASS_NAMES = CORE_CLASS_NAMES
MODEL_PATH = CORE_MODEL_PATH
THRESHOLDS = CORE_THRESHOLDS


# =========================================================
# 1. CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

MODEL_PATH = (
    BASE_DIR
    / "runs"
    / "detect"
    / "train"
    / "weights"
    / "best.pt"
)

IMAGE_DIR = BASE_DIR / "data" / "test" / "images"
LABEL_DIR = BASE_DIR / "data" / "test" / "labels"

OUTPUT_DIR = BASE_DIR / "evaluation_results"

CLASS_NAMES = {
    0: "Platelets",
    1: "RBC",
    2: "WBC",
}

YOLO_CONFIG = dict(CORE_YOLO_CONFIG)

OPERATING_CONFIDENCE = 0.50
OPERATING_IOU = 0.50

MAP_IOU_THRESHOLDS = np.arange(
    0.50,
    0.96,
    0.05
)

QAPS_THRESHOLDS = {
    "noise": 8.0,
    "contrast": 35.0,
    "brightness_low": 70.0,
    "brightness_high": 190.0,
}

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("HemoScanEvaluation")


# =========================================================
# 2. JSON SERIALIZATION SAFETY
# =========================================================

def make_json_safe(value):
    """
    Recursively convert NumPy values and non-finite floats
    into values supported by strict JSON serialization.

    NaN and Infinity become None (JSON null).
    """

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())

    if isinstance(value, (float, np.floating)):
        return (
            float(value)
            if np.isfinite(value)
            else None
        )

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, np.bool_):
        return bool(value)

    return value


# =========================================================
# 3. REPRODUCIBILITY
# =========================================================

def sha256_file(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as file:

        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def synchronize_gpu():

    if torch.cuda.is_available():
        torch.cuda.synchronize()


# =========================================================
# 4. QAPS IMAGE QUALITY
# =========================================================

def estimate_noise(image: np.ndarray) -> float:

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    median = cv2.medianBlur(gray, 3)

    residual = (
        gray.astype(np.float32)
        - median.astype(np.float32)
    )

    mad = np.median(
        np.abs(residual - np.median(residual))
    )

    return float(mad / 0.6745)


def assess_image_quality(image: np.ndarray) -> dict:

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return {
        "brightness": float(np.mean(gray)),
        "contrast": float(np.std(gray)),
        "noise": estimate_noise(image),
    }


# =========================================================
# 5. QAPS OPERATIONS
# =========================================================

def apply_median_denoising(image: np.ndarray):

    return cv2.medianBlur(image, 3)


def apply_clahe(image: np.ndarray):

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(l)

    merged = cv2.merge((enhanced, a, b))

    return cv2.cvtColor(
        merged,
        cv2.COLOR_LAB2BGR
    )


def apply_gamma_correction(
    image: np.ndarray,
    gamma: float
):

    gamma = float(
        np.clip(gamma, 0.5, 1.5)
    )

    table = np.array(
        [
            ((i / 255.0) ** gamma) * 255
            for i in range(256)
        ],
        dtype=np.uint8
    )

    return cv2.LUT(image, table)


# =========================================================
# 6. QAPS ADAPTIVE PREPROCESSING
# =========================================================

def adaptive_preprocessing(image: np.ndarray):

    quality = assess_image_quality(image)

    processed = image.copy()

    operations = []

    brightness = quality["brightness"]
    contrast = quality["contrast"]
    noise = quality["noise"]

    # Noise reduction

    if noise >= QAPS_THRESHOLDS["noise"]:

        processed = apply_median_denoising(processed)

        operations.append("Median denoising")

    # Contrast enhancement

    if contrast < QAPS_THRESHOLDS["contrast"]:

        processed = apply_clahe(processed)

        operations.append("CLAHE")

    # Brightness correction

    if (
        brightness < QAPS_THRESHOLDS["brightness_low"]
        or brightness > QAPS_THRESHOLDS["brightness_high"]
    ):

        target_brightness = 120.0

        normalized = np.clip(
            brightness / 255.0,
            0.01,
            0.99
        )

        gamma = (
            np.log(target_brightness / 255.0)
            / np.log(normalized)
        )

        processed = apply_gamma_correction(
            processed,
            gamma
        )

        operations.append("Gamma correction")

    if not operations:
        operations.append("No preprocessing")

    return processed, {
        "quality": quality,
        "operations": operations,
    }


# =========================================================
# 7. LOAD GROUND-TRUTH YOLO LABELS
# =========================================================

def load_yolo_labels(
    label_path: Path,
    image_width: int,
    image_height: int,
) -> list[dict]:
    """
    Read YOLO normalized annotations.

    Expected row:
        class_id x_center y_center width height

    Coordinates are converted to pixel-based xyxy.
    """

    if not label_path.exists():

        raise FileNotFoundError(
            f"Missing ground-truth annotation: {label_path}"
        )

    ground_truth = []

    with label_path.open("r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:

                raise ValueError(
                    f"Invalid annotation in {label_path}, "
                    f"line {line_number}. Expected 5 values."
                )

            class_id = int(float(parts[0]))

            if class_id not in CLASS_NAMES:

                raise ValueError(
                    f"Unknown class ID {class_id} in {label_path}"
                )

            xc, yc, bw, bh = map(
                float,
                parts[1:]
            )

            if not all(
                0.0 <= value <= 1.0
                for value in (xc, yc, bw, bh)
            ):

                raise ValueError(
                    f"Coordinates outside normalized range "
                    f"in {label_path}, line {line_number}."
                )

            if bw <= 0 or bh <= 0:

                raise ValueError(
                    f"Invalid bounding-box size in {label_path}, "
                    f"line {line_number}."
                )

            x1 = (xc - bw / 2) * image_width
            y1 = (yc - bh / 2) * image_height
            x2 = (xc + bw / 2) * image_width
            y2 = (yc + bh / 2) * image_height

            x1 = max(0.0, min(x1, image_width))
            y1 = max(0.0, min(y1, image_height))
            x2 = max(0.0, min(x2, image_width))
            y2 = max(0.0, min(y2, image_height))

            ground_truth.append({
                "class_id": class_id,
                "box": [x1, y1, x2, y2],
            })

    return ground_truth


# =========================================================
# 8. YOLO INFERENCE
# =========================================================

def predict_image(
    model,
    image: np.ndarray,
) -> dict:

    synchronize_gpu()
    start = time.perf_counter()

    result = model.predict(
        source=image,
        conf=YOLO_CONFIG["confidence"],
        iou=YOLO_CONFIG["iou"],
        imgsz=YOLO_CONFIG["image_size"],
        max_det=YOLO_CONFIG["max_det"],
        verbose=False,
    )[0]

    synchronize_gpu()

    inference_time = time.perf_counter() - start

    predictions = []

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0].item())

            if class_id not in CLASS_NAMES:
                continue

            coordinates = box.xyxy[0].cpu().tolist()

            predictions.append({
                "class_id": class_id,
                "confidence": float(box.conf[0].item()),
                "box": [float(value) for value in coordinates],
            })

    return {
        "predictions": predictions,
        "inference_time": inference_time,
    }


# =========================================================
# 9. INTERSECTION OVER UNION
# =========================================================

def calculate_iou(
    box_a: list[float],
    box_b: list[float],
) -> float:

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)

    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(
        0.0,
        intersection_x2 - intersection_x1
    )

    intersection_height = max(
        0.0,
        intersection_y2 - intersection_y1
    )

    intersection = (
        intersection_width * intersection_height
    )

    area_a = (
        max(0.0, ax2 - ax1)
        * max(0.0, ay2 - ay1)
    )

    area_b = (
        max(0.0, bx2 - bx1)
        * max(0.0, by2 - by1)
    )

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# =========================================================
# 10. CLASS-WISE MATCHING
# =========================================================

def match_class_predictions(
    records: list[dict],
    class_id: int,
    iou_threshold: float,
    confidence_threshold: float,
) -> dict:
    """
    Greedily match predictions to ground truth.

    Predictions are sorted by confidence.
    Each ground-truth box can be matched once per image.
    """

    detections = []
    total_ground_truth = 0

    # Build an image lookup to avoid repeatedly searching
    # the complete records list during matching.

    records_by_id = {
        record["image_id"]: record
        for record in records
    }

    for record in records:

        image_id = record["image_id"]

        gt_boxes = [
            item["box"]
            for item in record["ground_truth"]
            if item["class_id"] == class_id
        ]

        total_ground_truth += len(gt_boxes)

        predictions = [
            item
            for item in record["predictions"]
            if (
                item["class_id"] == class_id
                and item["confidence"] >= confidence_threshold
            )
        ]

        for prediction in predictions:

            detections.append({
                "image_id": image_id,
                "confidence": prediction["confidence"],
                "box": prediction["box"],
            })

    detections.sort(
        key=lambda item: item["confidence"],
        reverse=True
    )

    matched_ground_truth = {
        record["image_id"]: set()
        for record in records
    }

    true_positives = []
    false_positives = []

    for detection in detections:

        image_id = detection["image_id"]

        record = records_by_id[image_id]

        gt_boxes = [
            item["box"]
            for item in record["ground_truth"]
            if item["class_id"] == class_id
        ]

        best_iou = 0.0
        best_index = -1

        for index, gt_box in enumerate(gt_boxes):

            if index in matched_ground_truth[image_id]:
                continue

            iou = calculate_iou(
                detection["box"],
                gt_box
            )

            if iou > best_iou:

                best_iou = iou
                best_index = index

        if (
            best_index >= 0
            and best_iou >= iou_threshold
        ):

            true_positives.append(1)
            false_positives.append(0)

            matched_ground_truth[image_id].add(
                best_index
            )

        else:

            true_positives.append(0)
            false_positives.append(1)

    return {
        "tp": np.asarray(true_positives, dtype=np.float64),
        "fp": np.asarray(false_positives, dtype=np.float64),
        "ground_truth_count": total_ground_truth,
        "prediction_count": len(detections),
    }


# =========================================================
# 11. AVERAGE PRECISION
# =========================================================

def calculate_average_precision(
    tp: np.ndarray,
    fp: np.ndarray,
    ground_truth_count: int,
) -> float:
    """
    Calculate interpolated AP using 101 recall points.
    """

    if ground_truth_count == 0:
        return float("nan")

    if len(tp) == 0:
        return 0.0

    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)

    recall = cumulative_tp / ground_truth_count

    precision = cumulative_tp / np.maximum(
        cumulative_tp + cumulative_fp,
        1e-12
    )

    recall_points = np.linspace(0.0, 1.0, 101)

    interpolated_precision = []

    for recall_point in recall_points:

        valid = precision[recall >= recall_point]

        interpolated_precision.append(
            float(np.max(valid))
            if len(valid)
            else 0.0
        )

    return float(
        np.mean(interpolated_precision)
    )


# =========================================================
# 12. CLASS METRICS
# =========================================================

def evaluate_class(
    records: list[dict],
    class_id: int,
) -> dict:

    # Operating metrics at confidence 0.25 and IoU 0.50.

    operating = match_class_predictions(
        records,
        class_id,
        OPERATING_IOU,
        OPERATING_CONFIDENCE,
    )

    tp_count = float(np.sum(operating["tp"]))
    fp_count = float(np.sum(operating["fp"]))
    gt_count = operating["ground_truth_count"]

    fn_count = max(
        0,
        gt_count - int(tp_count)
    )

    precision = (
        tp_count / (tp_count + fp_count)
        if tp_count + fp_count > 0
        else 0.0
    )

    recall = (
        tp_count / gt_count
        if gt_count > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    # AP at IoU 0.50 through 0.95.

    ap_by_iou = {}

    for iou_threshold in MAP_IOU_THRESHOLDS:

        matched = match_class_predictions(
            records,
            class_id,
            float(iou_threshold),
            YOLO_CONFIG["confidence"],
        )

        ap = calculate_average_precision(
            matched["tp"],
            matched["fp"],
            matched["ground_truth_count"],
        )

        ap_by_iou[round(float(iou_threshold), 2)] = ap

    ap50 = ap_by_iou[0.5]

    valid_ap_values = [
        value
        for value in ap_by_iou.values()
        if np.isfinite(value)
    ]

    map_50_95 = (
        float(np.mean(valid_ap_values))
        if valid_ap_values
        else float("nan")
    )

    return {
        "ground_truth": gt_count,
        "predictions_at_confidence": operating["prediction_count"],
        "TP": int(tp_count),
        "FP": int(fp_count),
        "FN": int(fn_count),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "AP50": ap50,
        "AP50_95": map_50_95,
        "AP_by_IoU": ap_by_iou,
    }


# =========================================================
# 13. COMPLETE METHOD EVALUATION
# =========================================================

def evaluate_method(
    records: list[dict],
    prediction_key: str,
) -> dict:

    method_records = []

    for record in records:

        method_records.append({
            "image_id": record["image_id"],
            "ground_truth": record["ground_truth"],
            "predictions": record[prediction_key],
        })

    class_results = {}

    for class_id, class_name in CLASS_NAMES.items():

        class_results[class_name] = evaluate_class(
            method_records,
            class_id
        )

    # Macro average over classes represented in ground truth.

    valid_classes = [
        result
        for result in class_results.values()
        if result["ground_truth"] > 0
    ]

    if not valid_classes:

        raise ValueError(
            "No ground-truth objects were found in the dataset."
        )

    def macro_average(key):

        values = [
            result[key]
            for result in valid_classes
            if np.isfinite(result[key])
        ]

        return (
            float(np.mean(values))
            if values
            else float("nan")
        )

    return {
        "classes": class_results,
        "macro_precision": macro_average("precision"),
        "macro_recall": macro_average("recall"),
        "macro_f1": macro_average("f1"),
        "mAP50": macro_average("AP50"),
        "mAP50_95": macro_average("AP50_95"),
        "number_of_images": len(records),
        "ground_truth_objects": sum(
            result["ground_truth"]
            for result in class_results.values()
        ),
    }


# =========================================================
# 14. DATASET DISCOVERY
# =========================================================

def discover_images() -> list[Path]:

    if not IMAGE_DIR.exists():

        raise FileNotFoundError(
            f"Validation image directory not found: {IMAGE_DIR}"
        )

    if not LABEL_DIR.exists():

        raise FileNotFoundError(
            f"Validation label directory not found: {LABEL_DIR}"
        )

    images = sorted(
        path
        for path in IMAGE_DIR.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    )

    if not images:

        raise ValueError(
            f"No supported images found in {IMAGE_DIR}"
        )

    return images


# =========================================================
# 15. RUN PAIRED DATASET EXPERIMENT
# =========================================================

def run_dataset_experiment():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    images = discover_images()

    model_hash = sha256_file(MODEL_PATH)

    logger.info("Loading existing YOLO model...")

    model = core_load_model(str(MODEL_PATH))

    records = []

    total_images = len(images)

    experiment_start = time.perf_counter()

    for index, image_path in enumerate(images, start=1):

        logger.info(
            "Processing image %s/%s: %s",
            index,
            total_images,
            image_path.name
        )

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR
        )

        if image is None:

            raise ValueError(
                f"Could not read image: {image_path}"
            )

        height, width = image.shape[:2]

        label_path = (
            LABEL_DIR
            / f"{image_path.stem}.txt"
        )

        ground_truth = load_yolo_labels(
            label_path,
            width,
            height
        )

        # Branch A: original image -> YOLO

        baseline_result = predict_image(
            model,
            image
        )

        # Branch B: QAPS -> YOLO

        qaps_image, qaps_metadata = core_adaptive_preprocessing(
            image
        )

        qaps_result = predict_image(
            model,
            qaps_image
        )

        records.append({
            "image_id": image_path.name,
            "ground_truth": ground_truth,
            "baseline_predictions": baseline_result["predictions"],
            "qaps_predictions": qaps_result["predictions"],
            "baseline_inference_time": baseline_result["inference_time"],
            "qaps_inference_time": qaps_result["inference_time"],
            "qaps_metadata": qaps_metadata,
        })

    total_time = time.perf_counter() - experiment_start

    logger.info("Evaluating baseline YOLO...")

    baseline_metrics = evaluate_method(
        records,
        "baseline_predictions"
    )

    logger.info("Evaluating QAPS-YOLO...")

    qaps_metrics = evaluate_method(
        records,
        "qaps_predictions"
    )

    results = {
        "experiment": {
            "name": "Baseline YOLO vs QAPS-YOLO",
            "timestamp": datetime.now().astimezone().isoformat(),
            "experiment_version": "1.0",
            "model_path": str(MODEL_PATH),
            "model_sha256": model_hash,
            "dataset_image_directory": str(IMAGE_DIR),
            "dataset_label_directory": str(LABEL_DIR),
            "image_count": total_images,
            "yolo_config": YOLO_CONFIG,
            "operating_confidence": OPERATING_CONFIDENCE,
            "operating_iou": OPERATING_IOU,
            "map_iou_thresholds": [
                round(float(value), 2)
                for value in MAP_IOU_THRESHOLDS
            ],
            "qaps_thresholds": CORE_THRESHOLDS,
            "total_experiment_time_seconds": total_time,
        },
        "baseline": baseline_metrics,
        "qaps": qaps_metrics,
    }

    # -----------------------------------------------------
    # SAVE FULL JSON RESULTS
    # -----------------------------------------------------

    json_path = OUTPUT_DIR / "evaluation_results.json"

    with json_path.open("w", encoding="utf-8") as file:

        json.dump(
            make_json_safe(results),
            file,
            indent=4,
            allow_nan=False,
        )

    # -----------------------------------------------------
    # SAVE CLASS-WISE CSV COMPARISON
    # -----------------------------------------------------

    csv_path = OUTPUT_DIR / "class_comparison.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Class",
            "Method",
            "Ground Truth",
            "Predictions",
            "TP",
            "FP",
            "FN",
            "Precision",
            "Recall",
            "F1",
            "AP50",
            "AP50_95",
        ])

        for method_name, method_result in [
            ("Baseline YOLO", baseline_metrics),
            ("QAPS-YOLO", qaps_metrics),
        ]:

            for class_name, metrics in method_result["classes"].items():

                writer.writerow([
                    class_name,
                    method_name,
                    metrics["ground_truth"],
                    metrics["predictions_at_confidence"],
                    metrics["TP"],
                    metrics["FP"],
                    metrics["FN"],
                    metrics["precision"],
                    metrics["recall"],
                    metrics["f1"],
                    metrics["AP50"],
                    metrics["AP50_95"],
                ])

    # -----------------------------------------------------
    # SAVE OVERALL COMPARISON CSV
    # -----------------------------------------------------

    overall_path = OUTPUT_DIR / "overall_comparison.csv"

    with overall_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Metric",
            "Baseline YOLO",
            "QAPS-YOLO",
            "Difference",
        ])

        metric_keys = [
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "mAP50",
            "mAP50_95",
        ]

        for key in metric_keys:

            before = baseline_metrics[key]
            after = qaps_metrics[key]

            writer.writerow([
                key,
                before,
                after,
                (
                    after - before
                    if np.isfinite(before) and np.isfinite(after)
                    else ""
                ),
            ])

    # -----------------------------------------------------
    # SAVE PER-IMAGE DETAILS
    # -----------------------------------------------------

    image_details_path = OUTPUT_DIR / "image_details.json"

    image_details = []

    for record in records:

        image_details.append({
            "image_id": record["image_id"],
            "ground_truth_objects": len(record["ground_truth"]),
            "baseline_predictions": record["baseline_predictions"],
            "qaps_predictions": record["qaps_predictions"],
            "baseline_inference_time": record["baseline_inference_time"],
            "qaps_inference_time": record["qaps_inference_time"],
            "qaps_metadata": record["qaps_metadata"],
        })

    with image_details_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            make_json_safe(image_details),
            file,
            indent=4,
            allow_nan=False,
        )

    logger.info("All evaluation files saved successfully.")

    return results


# =========================================================
# 16. PRINT RESULTS
# =========================================================

def print_results(results: dict):

    baseline = results["baseline"]
    qaps = results["qaps"]

    print("\n" + "=" * 65)
    print("HEMOSCAN AI — DATASET-LEVEL EVALUATION")
    print("=" * 65)

    print("\nOVERALL METRICS")

    print(
        f"{'Metric':<20}"
        f"{'Baseline':>14}"
        f"{'QAPS-YOLO':>14}"
        f"{'Difference':>14}"
    )

    print("-" * 65)

    metrics = [
        ("Precision", "macro_precision"),
        ("Recall", "macro_recall"),
        ("F1-score", "macro_f1"),
        ("mAP@50", "mAP50"),
        ("mAP@50:95", "mAP50_95"),
    ]

    for label, key in metrics:

        before = baseline[key]
        after = qaps[key]

        difference = (
            after - before
            if np.isfinite(before) and np.isfinite(after)
            else float("nan")
        )

        print(
            f"{label:<20}"
            f"{before:>14.4f}"
            f"{after:>14.4f}"
            f"{difference:>14.4f}"
        )

    print("\nCLASS-WISE RESULTS")

    for class_name in CLASS_NAMES.values():

        print(f"\n{class_name}")

        for method_name, method_result in [
            ("Baseline", baseline),
            ("QAPS-YOLO", qaps),
        ]:

            metrics = method_result["classes"][class_name]

            print(
                f"{method_name:<12} "
                f"P={metrics['precision']:.4f} "
                f"R={metrics['recall']:.4f} "
                f"F1={metrics['f1']:.4f} "
                f"AP50={metrics['AP50']:.4f} "
                f"AP50:95={metrics['AP50_95']:.4f}"
            )

    print("\nEXPERIMENT DETAILS")

    print(
        "Validation images:",
        results["experiment"]["image_count"]
    )

    print(
        "Ground-truth objects:",
        baseline["ground_truth_objects"]
    )

    print(
        "Total experiment time:",
        round(
            results["experiment"]["total_experiment_time_seconds"],
            3
        ),
        "seconds"
    )

    print("\nFiles saved in:", OUTPUT_DIR)

    print("=" * 65)


# =========================================================
# 17. MAIN
# =========================================================

if __name__ == "__main__":

    try:

        results = run_dataset_experiment()

        print_results(results)

    except Exception as exc:

        logger.exception(
            "Dataset evaluation failed: %s",
            exc
        )

        raise
