# HemoScan AI — Blood Smear Research Analyzer

Research-oriented Streamlit application that detects and counts RBCs, WBCs, and
platelets in blood-smear images using YOLO, with an optional Quality-Aware
Preprocessing Selection (QAPS) branch for paired comparison.

> **Not a diagnostic tool.** Outputs are model detections, not validated
> blood-cell concentrations or a complete blood count. Do not use for clinical
> decisions.

## Problem
Manual blood cell counting from smear images is time-consuming, subjective, and error-prone. Automated detection can assist, but performance depends heavily on image quality (blur, brightness, contrast, noise). Fixed preprocessing pipelines may help some images while degrading others, and there is limited evidence on whether adaptive preprocessing actually improves detection accuracy.

## Objectives
1. Detect and count RBCs, WBCs, and platelets using YOLO.
2. Assess image quality automatically (blur, brightness, contrast, noise).
3. Adaptively preprocess images using Quality-Aware Preprocessing Selection (QAPS).
4. Compare baseline YOLO vs. QAPS-YOLO on the same image with identical settings.
5. Evaluate rigorously with ground-truth metrics, ablation studies, and statistical significance testing.
6. Provide a research-grade tool with reproducibility, audit logging, and automated reporting.

## Workflow
1. **Data preparation** — blood smear images with YOLO-format labels (Platelets, RBC, WBC).
2. **Model training** — YOLO11n, 100 epochs, early stopping, cosine LR.
3. **Image analysis (app)** — upload → quality assessment → baseline detection → QAPS preprocessing → QAPS detection → comparison → export.
4. **Offline evaluation** — ground-truth matching, precision/recall/F1/mAP, bootstrap CIs, significance tests, ablation, quality-stratified analysis, threshold tuning.
5. **Reporting** — automated research paper PDF with all metrics, tables, and figures.

## Key finding
QAPS did not show a statistically significant improvement over baseline YOLO on the current dataset. Only CLAHE was active; denoising and gamma correction were inactive. This null result defines the research gap for future work.

## Run the app

```bash
pip install -r requirements.txt
streamlit run main.py
```

## Project layout

```
├── app.py                  # Streamlit UI
├── main.py                 # Launcher (applies CSS, runs app.py)
├── core.py                 # Shared logic (quality, QAPS, inference, exports)
├── config/
│   └── config.yaml         # Hyperparameters and thresholds
├── data/                   # train / valid / test (images + YOLO labels)
├── models/                 # Base YOLO model (yolo11n.pt)
├── runs/                   # Trained model weights
├── scripts/
│   ├── training/           # train.py
│   └── evaluation/         # evaluate_dataset, paired_analysis, statistical_analysis
├── tests/
│   └── test_all.py         # Consolidated test script (16 tests)
├── docs/
│   └── MODEL_CARD.md
├── .github/workflows/      # CI
├── Dockerfile              # Container build
└── requirements.txt
```

## Training

```bash
python scripts/training/train.py
```

## Evaluation

```bash
python scripts/evaluation/evaluate_dataset.py
python scripts/evaluation/paired_analysis.py
python scripts/evaluation/statistical_analysis.py
```

## Tests

```bash
# Single consolidated test script
python -m pytest tests/test_all.py -q
```

## Docker

```bash
docker build -t hemoscan .
docker run -p 8501:8501 hemoscan
```

## Limitations
- Counts are detections per image, not calibrated clinical concentrations.
- Model trained on a limited dataset; precision is modest (~0.60–0.75).
- QAPS has not shown a statistically significant accuracy improvement.
- Small/unbalanced test set; metrics are unstable for rare classes.
- Not validated for clinical use.

## License
Research use only.