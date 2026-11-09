import numpy as np

import core


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
    assert q["warnings"]  # dark/low contrast etc.


def test_decode_image_pixel_limit():
    big = np.zeros((6000, 6000, 3), dtype=np.uint8)
    ok, buf = __import__("cv2").imencode(".png", big)
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
    # Minimal fake analysis dict
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