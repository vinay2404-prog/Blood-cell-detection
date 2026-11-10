"""Single consolidated test script for the Blood-Cell-Detection project.

Combines all unit and integration tests into one file.
Run with:  python -m pytest tests/test_all.py -q
"""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core


# =====================================================================
# UNIT TESTS
# =====================================================================

def test_calculate_iou_basic():
    assert core.calculate_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert core.calculate_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    iou = core.calculate_iou([0, 0, 10, 10], [5, 5, 15, 15])
    assert 0.13 < iou < 0.15


def test_assess_image_quality_keys():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    q = core.assess_image_quality(img)
    for k in ("width", "height", "blur_score", "brightness", "contrast", "noise_score", "warnings", "status"):
        assert k in q
    assert q["warnings"]


def test_decode_image_pixel_limit():
    big = np.zeros((6000, 6000, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", big)
    try:
        core.decode_image(buf.tobytes())
        assert False, "expected ValueError"
    except ValueError as e:
        assert "MP" in str(e) or "limit" in str(e)


def test_adaptive_preprocessing_returns_ops():
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    q = core.assess_image_quality(img)
    out, meta = core.adaptive_preprocessing(img, q)
    assert out.shape == img.shape
    assert "selected_operations" in meta
    assert isinstance(meta["selected_operations"], list)


def test_csv_rows_shape():
    a = {
        "analysis_id": "1", "sample_id": "s", "timestamp": "t",
        "image_hash": "h", "model_hash": "m",
        "preprocessing": {"processing_time": 0.1, "selected_operations": ["No preprocessing"]},
        "baseline": {"counts": {"Platelets": 1, "RBC": 2, "WBC": 0},
                     "total_detections": 3, "mean_confidence": 50.0, "inference_time": 0.01},
        "qaps": {"counts": {"Platelets": 1, "RBC": 2, "WBC": 0},
                 "total_detections": 3, "mean_confidence": 50.0, "inference_time": 0.01},
    }
    rows = core.csv_rows(a)
    assert len(rows) == 6
    assert rows[0]["method"] in ("BASELINE", "QAPS")


def test_sha256_bytes_deterministic():
    data = b"hello world"
    assert core.sha256_bytes(data) == core.sha256_bytes(data)
    assert len(core.sha256_bytes(data)) == 64


def test_decode_image_invalid_raises():
    try:
        core.decode_image(b"not an image")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_assess_quality_warnings():
    dark = np.zeros((64, 64, 3), dtype=np.uint8)
    q = core.assess_image_quality(dark)
    assert len(q["warnings"]) > 0


def test_compare_experiments():
    base = {"total_detections": 10, "mean_confidence": 80.0, "inference_time": 0.1,
            "counts": {"Platelets": 2, "RBC": 7, "WBC": 1}}
    qaps = {"total_detections": 12, "mean_confidence": 82.0, "inference_time": 0.12,
            "counts": {"Platelets": 3, "RBC": 8, "WBC": 1}}
    cmp_ = core.compare_experiments(base, qaps)
    assert cmp_["detection_change"] == 2
    assert cmp_["confidence_change"] == 2.0
    assert cmp_["class_comparison"]["RBC"]["difference"] == 1


def test_generate_conclusion():
    a = {
        "qaps": {"total_detections": 5, "counts": {"RBC": 3, "WBC": 1, "Platelets": 1}},
        "comparison": {"baseline_detections": 4, "qaps_detections": 5, "detection_change": 1,
                       "confidence_change": 1.5},
        "image_quality": {"warnings": []},
    }
    c = core.generate_conclusion(a)
    assert "findings" in c and "recommendations" in c
    assert len(c["findings"]) >= 4


def test_adaptive_preprocessing_ablation():
    img = np.zeros((64, 64, 3), dtype=np.uint8) + 100
    q = core.assess_image_quality(img)
    out_full, meta_full = core.adaptive_preprocessing(img, q, enable_denoise=True, enable_gamma=True, enable_clahe=True)
    out_none, meta_none = core.adaptive_preprocessing(img, q, enable_denoise=False, enable_gamma=False, enable_clahe=False)
    assert out_full.shape == img.shape
    assert out_none.shape == img.shape
    assert meta_none["ablation"]["denoise"] is False


def test_to_png_roundtrip():
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[:, :, 0] = 255
    data = core.to_png(rgb)
    assert len(data) > 0
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert arr is not None
    assert arr.shape[:2] == (32, 32)


def test_rows_to_csv_header():
    rows = [{"analysis_id": "1", "sample_id": "s", "timestamp": "t", "image_hash": "h",
             "model_hash": "m", "method": "BASELINE", "cell_class": "RBC", "detections": 1,
             "total_detections": 1, "mean_confidence_percent": 50.0,
             "inference_time_seconds": 0.01, "qaps_processing_time_seconds": 0,
             "qaps_operations": "No preprocessing"}]
    data = core.rows_to_csv(rows)
    assert b"analysis_id" in data
    assert b"RBC" in data


# =====================================================================
# INTEGRATION TESTS
# =====================================================================

def test_analyze_image_integration():
    model = core.load_model(str(core.MODEL_PATH))
    img_path = next((ROOT / "data" / "test" / "images").iterdir())
    image = cv2.imread(str(img_path))
    raw = cv2.imencode(".jpg", image)[1].tobytes()
    result = core.analyze_image(model, image, "TEST", core.sha256_bytes(raw),
                                core.model_sha256(str(core.MODEL_PATH), core.MODEL_PATH.stat().st_mtime),
                                draw=False)
    assert "baseline" in result and "qaps" in result
    assert "comparison" in result
    assert result["baseline"]["total_detections"] >= 0
    assert result["qaps"]["total_detections"] >= 0


def test_app_import():
    import app  # noqa: F401


def test_streamlit_ui_smoke():
    import time
    import requests
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py",
                            "--server.headless", "true", "--server.port", "8520"],
                           cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(30):
            try:
                r = requests.get("http://localhost:8520", timeout=2)
                if r.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            raise RuntimeError("Streamlit did not start")
        r = requests.get("http://localhost:8520", timeout=5)
        assert r.status_code == 200
        assert "streamlit" in r.text.lower()
    finally:
        proc.terminate()
        proc.wait(timeout=10)
