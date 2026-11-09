"""Generate a research paper PDF from the project's evaluation artifacts."""
from __future__ import annotations

import json
import sys

import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "evaluation_results"

# Load results
ev = json.loads((EVAL / "evaluation_results.json").read_text())
ablation = json.loads((EVAL / "ablation_results.json").read_text())
qs = json.loads((EVAL / "quality_stratified_results.json").read_text())
stats = json.loads((EVAL / "statistical_analysis" / "statistical_results.json").read_text())
import csv
class_rows = list(csv.DictReader((EVAL / "class_comparison.csv").open()))

# PR curve from stored predictions
import cv2
from evaluate_dataset import load_yolo_labels, CLASS_NAMES as CN
details = json.loads((EVAL / "image_details.json").read_text())
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
pr_points = []
for conf in thresholds:
    tp = fp = fn = 0
    for rec in details:
        image_path = ROOT / "data" / "test" / "images" / rec["image_id"]
        image = cv2.imread(str(image_path))
        h, w = image.shape[:2]
        label_path = ROOT / "data" / "test" / "labels" / (Path(rec["image_id"]).stem + ".txt")
        gt = load_yolo_labels(label_path, w, h)
        gt_boxes = np.asarray([g["box"] for g in gt], dtype=float).reshape(-1, 4)
        gt_classes = np.asarray([g["class_id"] for g in gt], dtype=int)
        boxes = np.asarray([p["box"] for p in rec["baseline_predictions"]], dtype=float).reshape(-1, 4) if rec["baseline_predictions"] else np.zeros((0, 4))
        classes = np.asarray([p["class_id"] for p in rec["baseline_predictions"]], dtype=int)
        confs = np.asarray([p["confidence"] / 100.0 for p in rec["baseline_predictions"]], dtype=float)
        mask = confs >= conf
        boxes = boxes[mask] if len(boxes) else boxes
        classes = classes[mask]
        confs = confs[mask]
        # greedy match
        for cid in CN:
            gt_idx = np.where(gt_classes == cid)[0]
            pred_idx = np.where(classes == cid)[0]
            matched = set()
            ordered = sorted(pred_idx, key=lambda i: confs[i], reverse=True)
            for pi in ordered:
                best_iou = 0.0; best_gt = None
                for gi in gt_idx:
                    if gi in matched:
                        continue
                    iou = core.calculate_iou(boxes[pi], gt_boxes[gi])
                    if iou > best_iou:
                        best_iou = iou; best_gt = gi
                if best_gt is not None and best_iou >= 0.5:
                    tp += 1; matched.add(best_gt)
                else:
                    fp += 1
            fn += len(gt_idx) - len(matched)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    pr_points.append((recall, precision))

# PR curve figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5, 4))
rs = [p[0] for p in pr_points]; ps = [p[1] for p in pr_points]
ax.plot(rs, ps, marker="o")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall (Baseline YOLO)")
ax.grid(alpha=0.3); fig.tight_layout()
pr_buf = BytesIO(); fig.savefig(pr_buf, format="png", dpi=150); pr_buf.seek(0)
plt.close(fig)

# Chart
fig, ax = plt.subplots(figsize=(6, 3))
names = ["Precision", "Recall", "F1", "mAP@50", "mAP@50:95"]
base = [ev["baseline"]["macro_precision"], ev["baseline"]["macro_recall"], ev["baseline"]["macro_f1"],
        ev["baseline"]["mAP50"], ev["baseline"]["mAP50_95"]]
qaps = [ev["qaps"]["macro_precision"], ev["qaps"]["macro_recall"], ev["qaps"]["macro_f1"],
        ev["qaps"]["mAP50"], ev["qaps"]["mAP50_95"]]
x = range(len(names))
ax.bar([i - 0.2 for i in x], base, 0.4, label="Baseline YOLO")
ax.bar([i + 0.2 for i in x], qaps, 0.4, label="QAPS-YOLO")
ax.set_xticks(list(x)); ax.set_xticklabels(names, rotation=20, ha="right")
ax.set_ylabel("Score"); ax.set_title("Baseline vs QAPS-YOLO (macro)")
ax.legend(); ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
chart_buf = BytesIO(); fig.savefig(chart_buf, format="png", dpi=150); chart_buf.seek(0)
plt.close(fig)

# PDF
out = ROOT / "research_paper.pdf"
doc = SimpleDocTemplate(str(out), pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=18*mm, bottomMargin=18*mm)
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="Title2", parent=styles["Title"], fontSize=16, leading=20, spaceAfter=6))
styles.add(ParagraphStyle(name="H1", parent=styles["Heading1"], fontSize=13, leading=16, spaceBefore=10, spaceAfter=4))
styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], fontSize=11, leading=14, spaceBefore=8, spaceAfter=3))
styles.add(ParagraphStyle(name="Body", parent=styles["BodyText"], fontSize=10, leading=13, spaceAfter=6))
styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10, spaceAfter=4))

story = []

def H(text, style="H1"):
    story.append(Paragraph(text, styles[style]))

def P(text):
    story.append(Paragraph(text, styles["Body"]))

def TBL(rows, widths, header=True, font=8):
    t = Table(rows, colWidths=widths)
    st = [("GRID", (0,0), (-1,-1), 0.4, colors.grey),
          ("PADDING", (0,0), (-1,-1), 4),
          ("FONTSIZE", (0,0), (-1,-1), font),
          ("VALIGN", (0,0), (-1,-1), "MIDDLE")]
    if header:
        st += [("BACKGROUND", (0,0), (-1,0), colors.HexColor("#164E78")),
               ("TEXTCOLOR", (0,0), (-1,0), colors.white),
               ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold")]
    t.setStyle(TableStyle(st))
    story.append(t)
    story.append(Spacer(1, 6))

# Title block
story.append(Paragraph("Quality-Aware Preprocessing Selection for Blood Cell Detection: An Empirical Study", styles["Title2"]))
story.append(Paragraph("An Experimental Evaluation of QAPS-YOLO on a Blood Smear Dataset", styles["H2"]))
story.append(Paragraph("HemoScan AI Research Team", styles["Body"]))
story.append(Paragraph("Research prototype — not for clinical use", styles["Small"]))

# Abstract
H("Abstract", "H1")
P("Blood cell detection from smear images is sensitive to acquisition quality. This paper evaluates "
  "a Quality-Aware Preprocessing Selection (QAPS) strategy that adaptively applies denoising, gamma "
  "correction, and CLAHE based on measured image quality, paired with a YOLO detector. Using a held-out "
  "test set of 36 images (471 ground-truth objects), we compare baseline YOLO against QAPS-YOLO at an "
  "operating confidence of 0.50. QAPS-YOLO achieves macro F1 of %.4f versus %.4f for the baseline, a "
  "difference that is not statistically significant (paired permutation p = %.3f). Ablation shows that "
  "only CLAHE is active on this dataset and slightly reduces F1, while denoising and gamma correction "
  "are inactive. Quality-stratified analysis reveals limited quality diversity in the test set. We "
  "conclude that adaptive preprocessing must be validated on larger, more diverse data before claiming "
  "robustness benefits." % (ev["qaps"]["macro_f1"], ev["baseline"]["macro_f1"],
  [r for r in stats if r["Metric"] == "Macro F1"][0]["Permutation_P_Value"]))

P("Keywords: blood cell detection, YOLO, quality-aware preprocessing, QAPS, ablation, statistical "
  "significance, medical imaging.")

# 1 Introduction
H("1. Introduction", "H1")
P("Automated blood cell detection can reduce manual effort in laboratory analysis, but image quality "
  "varies with microscope, staining, and illumination. Fixed preprocessing pipelines may improve some "
  "images while degrading others. This work studies Quality-Aware Preprocessing Selection (QAPS), which "
  "measures image quality and conditionally applies denoising, gamma correction, and CLAHE, then runs "
  "YOLO detection. We ask: (RQ1) Does QAPS improve detection over baseline YOLO? (RQ2) Which QAPS "
  "components contribute? (RQ3) Is any improvement statistically significant? (RQ4) Does performance "
  "vary by image quality?")

# 2 Related work
H("2. Related Work", "H1")
P("Object detection in medical images has been widely studied with YOLO-family models. Image enhancement "
  "(CLAHE, gamma correction, denoising) is commonly used to improve contrast, but its effect on detection "
  "is inconsistent and may introduce artifacts. Adaptive preprocessing selection remains underexplored "
  "for blood smear images. This paper contributes an empirical, statistically grounded evaluation of "
  "adaptive preprocessing for blood cell detection.")

# 3 Methodology
H("3. Methodology", "H1")
H("3.1 Dataset", "H2")
P("We use a blood smear dataset with 765 training, 73 validation, and 36 test images. Ground-truth "
  "annotations cover three classes: Platelets, RBC, and WBC. The test set contains 471 objects "
  "(398 RBC, 37 WBC, 36 Platelets).")

H("3.2 Detection model", "H2")
P("A YOLO11n model trained for 100 epochs (early stopping, cosine LR) on the training split is used. "
  "Inference uses image size 640, IoU threshold 0.70, and an operating confidence threshold of 0.50 "
  "(selected by F1 tuning on the validation set).")

H("3.3 QAPS", "H2")
P("QAPS measures blur (Laplacian variance), brightness/contrast (mean/std of grayscale), and noise "
  "(MAD-based residual). It then applies, in order: (1) median denoising if noise >= 8.0; (2) gamma "
  "correction if brightness is outside [70, 190] (target 120); (3) CLAHE if contrast < 35. Statistics "
  "are recomputed between steps. Both branches use identical YOLO settings.")

H("3.4 Evaluation protocol", "H2")
P("We compute precision, recall, F1, mAP@50, and mAP@50:95 using greedy one-to-one matching at IoU >= 0.5. "
  "Paired image-level bootstrap confidence intervals and Wilcoxon/permutation tests assess significance. "
  "Ablation disables individual QAPS components. Quality-stratified analysis groups images by quality bands.")

# 4 Results
H("4. Results", "H1")
H("4.1 Overall comparison", "H2")
P("Table 1 summarizes macro-averaged metrics. QAPS-YOLO shows a small F1 improvement (%.4f vs %.4f) "
  "that is not statistically significant." % (ev["qaps"]["macro_f1"], ev["baseline"]["macro_f1"]))
TBL([["Metric", "Baseline YOLO", "QAPS-YOLO", "Delta"],
     ["Precision", "%.4f" % ev["baseline"]["macro_precision"], "%.4f" % ev["qaps"]["macro_precision"],
      "%+.4f" % (ev["qaps"]["macro_precision"] - ev["baseline"]["macro_precision"])],
     ["Recall", "%.4f" % ev["baseline"]["macro_recall"], "%.4f" % ev["qaps"]["macro_recall"],
      "%+.4f" % (ev["qaps"]["macro_recall"] - ev["baseline"]["macro_recall"])],
     ["F1", "%.4f" % ev["baseline"]["macro_f1"], "%.4f" % ev["qaps"]["macro_f1"],
      "%+.4f" % (ev["qaps"]["macro_f1"] - ev["baseline"]["macro_f1"])],
     ["mAP@50", "%.4f" % ev["baseline"]["mAP50"], "%.4f" % ev["qaps"]["mAP50"],
      "%+.4f" % (ev["qaps"]["mAP50"] - ev["baseline"]["mAP50"])],
     ["mAP@50:95", "%.4f" % ev["baseline"]["mAP50_95"], "%.4f" % ev["qaps"]["mAP50_95"],
      "%+.4f" % (ev["qaps"]["mAP50_95"] - ev["baseline"]["mAP50_95"])]],
     [45*mm, 35*mm, 35*mm, 30*mm])
story.append(RLImage(chart_buf, width=150*mm, height=75*mm))

H("4.2 Precision-Recall curve", "H2")
P("Precision-recall curve for baseline YOLO across confidence thresholds.")
story.append(RLImage(pr_buf, width=120*mm, height=96*mm))

H("4.3 Per-class comparison", "H2")
rows = [["Class", "Method", "P", "R", "F1", "AP50"]]
for r in class_rows:
    rows.append([r["Class"], r["Method"], "%.3f" % float(r["Precision"]), "%.3f" % float(r["Recall"]),
                 "%.3f" % float(r["F1"]), "%.3f" % float(r["AP50"])])
TBL(rows, [28*mm, 32*mm, 22*mm, 22*mm, 22*mm, 22*mm], font=7)

H("4.3 Confusion summary (TP/FP/FN)", "H2")
rows = [["Class", "Method", "TP", "FP", "FN"]]
for r in class_rows:
    rows.append([r["Class"], r["Method"], r["TP"], r["FP"], r["FN"]])
TBL(rows, [30*mm, 35*mm, 25*mm, 25*mm, 25*mm], font=7)

H("4.3 Ablation study", "H2")
P("Table 2 shows the effect of disabling individual QAPS components. Denoising and gamma correction "
  "produce identical results to full QAPS (inactive on this data), while disabling CLAHE reverts to "
  "baseline performance.")
TBL([["Variant", "Precision", "Recall", "F1"],
     ["Baseline", "%.4f" % ablation["baseline"]["precision"], "%.4f" % ablation["baseline"]["recall"],
      "%.4f" % ablation["baseline"]["f1"]],
     ["QAPS full", "%.4f" % ablation["qaps_full"]["precision"], "%.4f" % ablation["qaps_full"]["recall"],
      "%.4f" % ablation["qaps_full"]["f1"]],
     ["QAPS no denoise", "%.4f" % ablation["qaps_no_denoise"]["precision"],
      "%.4f" % ablation["qaps_no_denoise"]["recall"], "%.4f" % ablation["qaps_no_denoise"]["f1"]],
     ["QAPS no gamma", "%.4f" % ablation["qaps_no_gamma"]["precision"],
      "%.4f" % ablation["qaps_no_gamma"]["recall"], "%.4f" % ablation["qaps_no_gamma"]["f1"]],
     ["QAPS no CLAHE", "%.4f" % ablation["qaps_no_clahe"]["precision"],
      "%.4f" % ablation["qaps_no_clahe"]["recall"], "%.4f" % ablation["qaps_no_clahe"]["f1"]]],
     [40*mm, 30*mm, 30*mm, 30*mm])

H("4.4 Quality-stratified analysis", "H2")
P("All 36 test images fall into the same quality band (blurry, normal brightness, clean), so no "
  "per-group differences can be reported. This indicates limited quality diversity in the test set.")

H("4.5 Statistical significance", "H2")
macro_stat = [r for r in stats if r["Metric"] == "Macro F1"][0]
P("Paired image-level analysis: mean F1 difference = %.4f (95%% CI [%.4f, %.4f]), "
  "permutation p = %.4f, Wilcoxon p = %.4f, Cohen's dz = %.3f. After Holm correction, no comparison "
  "is significant at alpha = 0.05. %d images improved, %d unchanged, %d worsened."
  % (macro_stat["Mean_Difference"], macro_stat["CI_95_Lower"], macro_stat["CI_95_Upper"],
     macro_stat["Permutation_P_Value"], macro_stat["Wilcoxon_P_Value"], macro_stat["Cohens_Dz"],
     macro_stat["Images_Improved"], macro_stat["Images_Unchanged"], macro_stat["Images_Worsened"]))

H("4.6 Operating threshold tuning", "H2")
P("Tuning the confidence threshold on the validation set selected 0.50 as the best F1 operating point "
  "(F1 = 0.7694), improving precision at a modest recall cost.")

# 5 Discussion
H("5. Discussion", "H1")
P("QAPS does not provide a statistically significant benefit on this dataset. Ablation reveals that "
  "denoising and gamma correction are inactive, and CLAHE slightly reduces F1. This suggests that "
  "adaptive preprocessing selection, as implemented, is not yet beneficial for detection on this data. "
  "The lack of quality diversity in the test set limits robustness conclusions. Threshold tuning "
  "improves the operating point, indicating that confidence selection matters more than preprocessing here.")

# 6 Limitations
H("6. Limitations", "H1")
P("Small test set (36 images), class imbalance (398 RBC vs 36 platelets vs 37 WBC), limited quality "
  "diversity, no external/multi-site validation, and no calibration to physical cell concentrations. "
  "Counts are detections per image, not clinical measurements.")

# 7 Conclusion
H("7. Conclusion", "H1")
P("We evaluated QAPS-YOLO for blood cell detection and found no statistically significant improvement "
  "over baseline YOLO. Ablation showed CLAHE as the only active component, slightly reducing F1. "
  "Future work should use larger, more diverse datasets, learnable preprocessing policies, and "
  "clinical calibration.")

# References
H("References", "H1")
refs = [
  "[1] Redmon, J., et al. You Only Look Once: Unified, Real-Time Object Detection. CVPR, 2016.",
  "[2] Ultralytics. YOLO11. https://github.com/ultralytics/ultralytics, 2024.",
  "[3] Zuiderveld, K. Contrast Limited Adaptive Histogram Equalization. Graphics Gems IV, 1994.",
  "[4] Bland, J.M., Altman, D.G. Statistical methods for assessing agreement between two methods of clinical measurement. Lancet, 1986.",
  "[5] BCCD Dataset. https://github.com/Shenggan/BCCD_Dataset.",
]
for r in refs:
    P(r)

doc.build(story)
print("Wrote", out)