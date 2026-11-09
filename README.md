# HemoScan AI - Blood Smear Research Analyzer

Research-oriented Streamlit application that detects and counts RBCs, WBCs and
platelets in blood-smear images using a YOLO model, with an optional
Quality-Aware Preprocessing Selection (QAPS) branch for paired comparison.

> **Not a diagnostic tool.** Outputs are model detections, not validated
> blood-cell concentrations or a complete blood count. Do not use for clinical
> decisions.

## Run

```bash
pip install -r requirements.txt
streamlit run main.py
```

## Project layout

- `app.py` - Streamlit UI
- `main.py` - launcher that applies the shadcn skin
- `core.py` - shared logic (quality, QAPS, inference, exports)
- `train.py` - model training
- `evaluate_dataset.py` - offline paired evaluation against ground truth
- `paired_analysis.py`, `statistical_analysis.py` - paired/bootstrap statistics
- `tools/` - utility scripts (threshold tuning, verification, inspection)
- `tests/` - unit tests

## Limitations

- Counts are detections per image, not calibrated to volume.
- Model trained on a limited dataset; precision is modest (~0.6) with
  over-detection of RBCs/platelets.
- QAPS has not shown a statistically significant accuracy improvement.
- Small/unbalanced test set; metrics are unstable for rare classes.
- Not validated for clinical use.