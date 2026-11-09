"""Shared core for HemoScan AI.

Single source of truth for configuration, image quality, QAPS preprocessing,
YOLO inference helpers, and evaluation metrics. Both the Streamlit app
(`app.py`) and the dataset evaluation script (`evaluate_dataset.py`) import
from here so the two code paths can never drift apart.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image as PILImage, ImageOps
from ultralytics import YOLO

import os

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("HEMOSCAN_MODEL", BASE_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"))

MAX_UPLOAD_SIZE = 15 * 1024 * 1024
MAX_PIXELS = 25_000_000

CLASS_NAMES = {0: "Platelets", 1: "RBC", 2: "WBC"}
COLORS = {"Platelets": (0, 215, 255), "RBC": (69, 53, 220), "WBC": (255, 144, 30)}

# Inference settings used by the interactive app (operating point).
YOLO_CONFIG = {"confidence": 0.50, "iou": 0.70, "image_size": 640, "max_det": 1000}

# Lower confidence for AP/mAP computation during offline evaluation.
EVAL_YOLO_CONFIG = {"confidence": 0.001, "iou": 0.70, "image_size": 640, "max_det": 1000}

OPERATING_CONFIDENCE = 0.25
OPERATING_IOU = 0.50

THRESHOLDS = {
    "min_side": 256,
    "blur": 50.0,
    "noise": 8.0,
    "contrast": 35.0,
    "brightness_low": 70.0,
    "brightness_high": 190.0,
    "target_brightness": 120.0,
}

ALGORITHM_VERSION = "3.0"
EXPERIMENT_VERSION = "1.1"

# Optional: override defaults from config.yaml if present.
try:
    import yaml as _yaml
    _cfg_path = BASE_DIR / "config" / "config.yaml"
    if not _cfg_path.exists():
        _cfg_path = BASE_DIR / "config.yaml"
    if _cfg_path.exists():
        _cfg = _yaml.safe_load(_cfg_path.read_text(encoding="utf-8")) or {}
        if "inference" in _cfg:
            YOLO_CONFIG["confidence"] = _cfg["inference"].get("confidence", YOLO_CONFIG["confidence"])
            YOLO_CONFIG["iou"] = _cfg["inference"].get("iou", YOLO_CONFIG["iou"])
            YOLO_CONFIG["max_det"] = _cfg["inference"].get("max_det", YOLO_CONFIG["max_det"])
        if "quality_thresholds" in _cfg:
            for k, v in _cfg["quality_thresholds"].items():
                THRESHOLDS[k] = v
except Exception:
    pass

logger = logging.getLogger("HemoScan")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def model_sha256(path_str: str, mtime: float) -> str:
    """Hash a model file (not cached here; callers may cache)."""
    digest = hashlib.sha256()
    with open(path_str, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sync_gpu() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def to_png(rgb: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


def _apply_exif_orientation(raw: bytes) -> np.ndarray | None:
    """Decode with PIL so EXIF orientation is honoured, then convert to BGR."""
    try:
        pil = PILImage.open(BytesIO(raw))
        pil = ImageOps.exif_transpose(pil)
        pil = pil.convert("RGB")
        arr = np.asarray(pil)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def decode_image(raw: bytes) -> np.ndarray:
    img = _apply_exif_orientation(raw)
    if img is None:
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unable to decode the uploaded image.")
    h, w = img.shape[:2]
    if h * w > MAX_PIXELS:
        raise ValueError(
            f"Image is {w}x{h} ({h * w / 1e6:.1f} MP); limit is {MAX_PIXELS / 1e6:.0f} MP."
        )
    return img


def load_model(model_path: str | Path = MODEL_PATH):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"YOLO model not found: {path}")

    model = YOLO(str(path))

    model_names = {int(k): str(v).strip().lower() for k, v in model.names.items()}
    expected = {k: v.lower() for k, v in CLASS_NAMES.items()}
    if model_names != expected:
        raise ValueError(f"Class mismatch. Model has {model.names}, app expects {CLASS_NAMES}.")

    model.predict(
        np.zeros((YOLO_CONFIG["image_size"],) * 2 + (3,), np.uint8),
        imgsz=YOLO_CONFIG["image_size"],
        verbose=False,
    )
    sync_gpu()
    return model


# ---------------- Image quality ----------------
def estimate_noise(gray: np.ndarray) -> float:
    median = cv2.medianBlur(gray, 3)
    residual = gray.astype(np.float32) - median.astype(np.float32)
    mad = np.median(np.abs(residual - np.median(residual)))
    return round(float(mad / 0.6745), 2)


def gray_stats(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray, float(np.mean(gray)), float(np.std(gray))


def assess_image_quality(image: np.ndarray) -> dict:
    h, w = image.shape[:2]
    gray, brightness, contrast = gray_stats(image)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    noise = estimate_noise(gray)
    T = THRESHOLDS

    warnings = []
    if min(h, w) < T["min_side"]:
        warnings.append("Low image resolution detected.")
    if blur < T["blur"]:
        warnings.append("Possible image blur detected.")
    if brightness < T["brightness_low"]:
        warnings.append("Image may be too dark.")
    elif brightness > T["brightness_high"]:
        warnings.append("Image may be overexposed.")
    if contrast < T["contrast"]:
        warnings.append("Low image contrast detected.")
    if noise >= T["noise"]:
        warnings.append("Possible high-frequency noise detected.")

    return {
        "width": w,
        "height": h,
        "blur_score": round(blur, 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "noise_score": noise,
        "warnings": warnings,
        "status": "Review recommended" if warnings else "No configured quality warning",
    }


# ---------------- QAPS ----------------
def apply_clahe(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    gamma = float(np.clip(gamma, 0.5, 1.5))
    table = (((np.arange(256) / 255.0) ** gamma) * 255).astype(np.uint8)
    return cv2.LUT(image, table)


def adaptive_preprocessing(image: np.ndarray, quality: dict | None = None,
                          enable_denoise: bool = True,
                          enable_gamma: bool = True,
                          enable_clahe: bool = True):
    """Order: denoise -> gamma -> CLAHE, recomputing stats between steps.

    Enable flags support ablation studies of individual QAPS components.
    """
    t0 = time.perf_counter()
    T = THRESHOLDS
    if quality is None:
        quality = assess_image_quality(image)

    out = image.copy()
    ops, params = [], {}

    if enable_denoise and quality["noise_score"] >= T["noise"]:
        out = cv2.medianBlur(out, 3)
        ops.append("Median denoising")
        params["denoising"] = {"method": "median", "kernel_size": 3}

    _, brightness, _ = gray_stats(out)
    if enable_gamma and (brightness < T["brightness_low"] or brightness > T["brightness_high"]):
        m = float(np.clip(brightness / 255.0, 0.01, 0.99))
        gamma = float(np.clip(np.log(T["target_brightness"] / 255.0) / np.log(m), 0.5, 1.5))
        out = apply_gamma(out, gamma)
        ops.append("Gamma correction")
        params["gamma"] = {"gamma_value": round(gamma, 3), "target_brightness": T["target_brightness"]}

    _, _, contrast = gray_stats(out)
    if enable_clahe and contrast < T["contrast"]:
        out = apply_clahe(out)
        ops.append("CLAHE")
        params["clahe"] = {"clip_limit": 2.0, "tile_grid_size": [8, 8]}

    applied = bool(ops)
    if not ops:
        ops = ["No preprocessing"]

    return out, {
        "algorithm": "QAPS",
        "algorithm_version": ALGORITHM_VERSION,
        "selected_operations": ops,
        "parameters": params,
        "noise_score": quality["noise_score"],
        "original_brightness": quality["brightness"],
        "original_contrast": quality["contrast"],
        "processing_applied": applied,
        "processing_time": round(time.perf_counter() - t0, 6),
        "thresholds": T.copy(),
        "ablation": {
            "denoise": enable_denoise,
            "gamma": enable_gamma,
            "clahe": enable_clahe,
        },
    }


# ---------------- YOLO ----------------
def run_yolo_detection(model, image: np.ndarray, config: dict = YOLO_CONFIG, draw: bool = True,
                      tta: bool = False) -> dict:
    sync_gpu()
    t0 = time.perf_counter()
    result = model.predict(
        source=image,
        conf=config["confidence"],
        iou=config["iou"],
        imgsz=config["image_size"],
        max_det=config["max_det"],
        augment=tta,
        verbose=False,
    )[0]
    sync_gpu()
    inference_time = time.perf_counter() - t0

    counts = {n: 0 for n in CLASS_NAMES.values()}
    confs, boxes = [], []
    annotated = image.copy() if draw else None
    h, w = image.shape[:2]

    if result.boxes is not None:
        for box in result.boxes:
            cid = int(box.cls[0].item())
            if cid not in CLASS_NAMES:
                continue
            name = CLASS_NAMES[cid]
            conf = float(box.conf[0].item())
            counts[name] += 1
            confs.append(conf * 100)

            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())
            x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
            y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))

            if draw:
                color = COLORS[name]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"{name} {conf * 100:.1f}%", (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

            boxes.append({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1,
                          "label": name, "confidence": round(conf * 100, 2)})

    return {
        "counts": counts,
        "total_detections": sum(counts.values()),
        "confidence_values": confs,
        "mean_confidence": round(float(np.mean(confs)) if confs else 0.0, 2),
        "boxes": boxes,
        "annotated_rgb": cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB) if draw else None,
        "inference_time": round(inference_time, 6),
    }


def compare_experiments(base: dict, qaps: dict) -> dict:
    change = qaps["total_detections"] - base["total_detections"]
    return {
        "baseline_detections": base["total_detections"],
        "qaps_detections": qaps["total_detections"],
        "detection_change": change,
        "detection_change_percent": (
            round(change / base["total_detections"] * 100, 2)
            if base["total_detections"] else None
        ),
        "confidence_change": round(qaps["mean_confidence"] - base["mean_confidence"], 2),
        "inference_time_change": round(qaps["inference_time"] - base["inference_time"], 6),
        "class_comparison": {
            n: {"baseline": base["counts"][n], "qaps": qaps["counts"][n],
                "difference": qaps["counts"][n] - base["counts"][n]}
            for n in CLASS_NAMES.values()
        },
    }


def analyze_image(model, image: np.ndarray, sample_id: str, image_hash: str,
                  model_hash: str, draw: bool = True, config: dict = YOLO_CONFIG,
                  tta: bool = False) -> dict:
    t0 = time.perf_counter()
    now = datetime.now().astimezone()

    quality = assess_image_quality(image)
    baseline = run_yolo_detection(model, image, config, draw, tta=tta)
    pre, preprocessing = adaptive_preprocessing(image, quality)
    qaps = run_yolo_detection(model, pre, config, draw, tta=tta)

    return {
        "analysis_id": now.strftime("%Y%m%d%H%M%S%f"),
        "sample_id": sample_id.strip()[:80] or "Sample not provided",
        "timestamp": now.strftime("%d %b %Y, %H:%M:%S %Z"),
        "image_hash": image_hash,
        "model_hash": model_hash,
        "image_quality": quality,
        "preprocessing": preprocessing,
        "baseline": baseline,
        "qaps": qaps,
        "comparison": compare_experiments(baseline, qaps),
        "original_rgb": cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if draw else None,
        "preprocessed_rgb": cv2.cvtColor(pre, cv2.COLOR_BGR2RGB) if draw else None,
        "processed_rgb": qaps["annotated_rgb"],
        "analysis_time": round(time.perf_counter() - t0, 6),
        "review_status": "Pending human review",
        "model_path": str(MODEL_PATH),
        "yolo_config": config.copy(),
        "experiment_version": EXPERIMENT_VERSION,
    }


def generate_conclusion(a: dict) -> dict:
    c, cmp_ = a["qaps"]["counts"], a["comparison"]
    findings = [
        f"QAPS-YOLO detected {a['qaps']['total_detections']} supported objects: "
        f"{c['RBC']} RBC, {c['WBC']} WBC and {c['Platelets']} platelets.",
        f"Baseline YOLO produced {cmp_['baseline_detections']} detections.",
        f"QAPS-YOLO produced {cmp_['qaps_detections']} detections.",
        f"Detection-count difference: {cmp_['detection_change']:+d}.",
        f"Mean-confidence difference: {cmp_['confidence_change']:+.2f} percentage points.",
    ]
    recs = []
    if a["image_quality"]["warnings"]:
        findings.append("The input image triggered configured quality warnings.")
        recs.append("Review the original image quality before interpreting outputs.")
    recs += [
        "Evaluate both branches against manually annotated ground truth.",
        "Calculate precision, recall, F1-score and mAP.",
        "Repeat experiments across multiple images and quality groups.",
        "Review annotated images with a qualified laboratory professional.",
        "Do not interpret image-level detections as a complete blood count.",
    ]
    return {"title": "Experimental Detection Conclusion",
            "status": "Research results require ground-truth validation",
            "findings": findings, "recommendations": recs}


def calculate_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def csv_rows(a: dict) -> list[dict]:
    rows = []
    for method in ("baseline", "qaps"):
        r = a[method]
        for cls, n in r["counts"].items():
            rows.append({
                "analysis_id": a["analysis_id"], "sample_id": a["sample_id"],
                "timestamp": a["timestamp"], "image_hash": a["image_hash"],
                "model_hash": a["model_hash"], "method": method.upper(),
                "cell_class": cls, "detections": n,
                "total_detections": r["total_detections"],
                "mean_confidence_percent": r["mean_confidence"],
                "inference_time_seconds": r["inference_time"],
                "qaps_processing_time_seconds":
                    a["preprocessing"]["processing_time"] if method == "qaps" else 0,
                "qaps_operations": "; ".join(a["preprocessing"]["selected_operations"]),
            })
    return rows


def rows_to_csv(rows: list[dict]) -> bytes:
    import csv as _csv
    from io import StringIO
    out = StringIO()
    w = _csv.DictWriter(out, fieldnames=[
        "analysis_id", "sample_id", "timestamp", "image_hash", "model_hash", "method",
        "cell_class", "detections", "total_detections", "mean_confidence_percent",
        "inference_time_seconds", "qaps_processing_time_seconds", "qaps_operations",
    ])
    w.writeheader()
    w.writerows(rows)
    return out.getvalue().encode("utf-8")


def create_distribution_chart(a: dict) -> bytes:
    from matplotlib.figure import Figure
    names = ["RBC", "WBC", "Platelets"]
    base = [a["baseline"]["counts"][n] for n in names]
    qaps = [a["qaps"]["counts"][n] for n in names]
    x, width = np.arange(len(names)), 0.35

    fig = Figure(figsize=(8, 4))
    ax = fig.subplots()
    ax.bar(x - width / 2, base, width, label="Baseline YOLO")
    ax.bar(x + width / 2, qaps, width, label="QAPS-YOLO")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Number of detections")
    ax.set_title("Baseline vs QAPS-YOLO")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    return buf.getvalue()


def make_json_safe(value: Any):
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value