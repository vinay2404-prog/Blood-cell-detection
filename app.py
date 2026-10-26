from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
import logging

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from ultralytics import YOLO

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, KeepTogether
)

# =========================================================
# CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="HemoScan AI | Blood Smear Analyzer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"
MAX_UPLOAD_SIZE = 15 * 1024 * 1024

CLASS_NAMES = {0: "Platelets", 1: "RBC", 2: "WBC"}
# OpenCV BGR colors
COLORS = {
    "Platelets": (0, 215, 255),
    "RBC": (69, 53, 220),
    "WBC": (255, 144, 30),
}
DISPLAY_COLORS = {
    "Platelets": "#00B7D9",
    "RBC": "#DC3545",
    "WBC": "#F59E0B",
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hemoscan")


# =========================================================
# UI THEME
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root {
  --bg: #09090b; --surface: #111113; --surface2: #18181b;
  --border: #27272a; --text: #fafafa; --muted: #a1a1aa;
}
html, body, [class*="css"] { font-family: Inter, sans-serif; }
.stApp { background: var(--bg); color: var(--text); }
.block-container { max-width: 1450px; padding-top: 2rem; padding-bottom: 3rem; }
header[data-testid="stHeader"] { background: rgba(9,9,11,.85); }
section[data-testid="stSidebar"] { display: none; }
.hero {
  border: 1px solid var(--border); border-radius: 22px; padding: 28px 30px;
  background: radial-gradient(circle at 90% 10%, rgba(34,197,94,.12), transparent 30%), #111113;
  margin-bottom: 22px;
}
.eyebrow { color: #4ade80; font-size: .78rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }
.hero h1 { color: #fafafa; font-size: clamp(2rem, 4vw, 3.2rem); line-height: 1.1; margin: 10px 0; }
.hero p { color: #a1a1aa; max-width: 780px; margin-bottom: 0; }
.section-title { font-size: 1.15rem; font-weight: 700; margin: 1rem 0 .8rem; }
.metric-card {
  background: var(--surface); border: 1px solid var(--border); border-radius: 16px;
  padding: 18px 20px; min-height: 112px;
}
.metric-label { color: #a1a1aa; font-size: .85rem; margin-bottom: 10px; }
.metric-value { color: #fafafa; font-size: 2rem; font-weight: 800; line-height: 1.1; }
.metric-note { color: #71717a; font-size: .78rem; margin-top: 8px; }
.panel {
  background: #111113; border: 1px solid #27272a; border-radius: 16px;
  padding: 18px; margin-bottom: 12px;
}
.small-muted { color: #a1a1aa; font-size: .88rem; }
.notice {
  background: rgba(245,158,11,.08); border: 1px solid rgba(245,158,11,.3);
  color: #fcd34d; border-radius: 12px; padding: 13px 15px; font-size: .88rem;
}
div[data-testid="stFileUploader"] {
  border: 1px dashed #3f3f46; border-radius: 14px; padding: 10px;
  background: #111113;
}
.stButton > button, .stDownloadButton > button {
  border-radius: 10px; font-weight: 600; min-height: 2.8rem;
}
hr { border-color: #27272a; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL LOADING
# =========================================================
@st.cache_resource(show_spinner="Loading the HemoScan AI model...")
def load_model(model_path: str):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file was not found at: {path}. "
            "Place best.pt at runs/detect/train/weights/best.pt beside app.py."
        )
    return YOLO(str(path))


# =========================================================
# IMAGE QUALITY
# =========================================================
def assess_image_quality(image: np.ndarray) -> dict:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    warnings = []

    if width < 256 or height < 256:
        warnings.append("Low image resolution. A higher-resolution image may be needed.")
    if blur_score < 50:
        warnings.append("Possible blur detected. Consider capturing a sharper image.")
    if brightness < 45:
        warnings.append("Image may be too dark.")
    elif brightness > 220:
        warnings.append("Image may be overexposed.")
    if contrast < 20:
        warnings.append("Low image contrast may affect detection quality.")

    return {
        "width": width, "height": height,
        "blur_score": round(blur_score, 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "warnings": warnings,
        "status": "Review recommended" if warnings else "No basic quality warning",
    }


# =========================================================
# ANALYSIS / INFERENCE
# =========================================================
def analyze_image(model, image: np.ndarray, sample_id: str) -> dict:
    quality = assess_image_quality(image)
    result = model.predict(source=image, verbose=False)[0]
    counts = {name: 0 for name in CLASS_NAMES.values()}
    confidence_values = []
    boxes_data = []
    processed = image.copy()
    height, width = processed.shape[:2]

    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            if cls_id not in CLASS_NAMES:
                logger.warning("Unknown class ID from model: %s", cls_id)
                continue

            class_name = CLASS_NAMES[cls_id]
            counts[class_name] += 1
            confidence_values.append(confidence * 100)
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())
            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width - 1))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height - 1))
            color = COLORS[class_name]

            cv2.rectangle(processed, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {confidence * 100:.1f}%"
            cv2.putText(
                processed, label, (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA
            )
            red, green, blue = color[::-1]
            boxes_data.append({
                "x": x1, "y": y1, "w": max(0, x2-x1), "h": max(0, y2-y1),
                "label": class_name, "confidence": round(confidence * 100, 1),
                "color": f"rgb({red}, {green}, {blue})",
            })

    total = sum(counts.values())
    mean_confidence = float(np.mean(confidence_values)) if confidence_values else 0.0
    return {
        "analysis_id": datetime.now().strftime("%Y%m%d%H%M%S"),
        "sample_id": sample_id.strip()[:80] or "Sample not provided",
        "timestamp": datetime.now().astimezone().strftime("%d %b %Y, %H:%M:%S %Z"),
        "counts": counts,
        "boxes": boxes_data,
        "total_detections": total,
        "confidence_values": confidence_values,
        "mean_confidence": round(mean_confidence, 1),
        "image_quality": quality,
        "review_status": "Pending human review",
        "original_rgb": cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        "processed_rgb": cv2.cvtColor(processed, cv2.COLOR_BGR2RGB),
    }


def generate_conclusion(analysis: dict) -> dict:
    counts = analysis.get("counts", {})
    total = analysis.get("total_detections", sum(counts.values()))
    warnings = analysis.get("image_quality", {}).get("warnings", [])
    findings, recommendations = [], []

    if total == 0:
        findings.append(
            "The AI model did not detect any cells in the submitted image. "
            "This does not mean that blood cells are absent."
        )
        recommendations.append(
            "Verify that the image contains a suitable blood smear and repeat the analysis if needed."
        )
    else:
        findings.append(
            f"The AI model detected {total} objects: {counts.get('RBC', 0)} RBC, "
            f"{counts.get('WBC', 0)} WBC, and {counts.get('Platelets', 0)} platelet detections."
        )

    if warnings:
        findings.append(
            f"The image-quality checks raised {len(warnings)} warning(s), which may affect detection reliability."
        )
        recommendations.append("Review the image-quality warnings and consider recapturing the image.")
    else:
        findings.append(
            "No warnings were triggered by the configured basic image-quality checks. "
            "This does not establish clinical quality."
        )

    if analysis.get("confidence_values"):
        findings.append(
            f"Mean model confidence was {analysis.get('mean_confidence', 0):.1f}%. "
            "Confidence is not a measure of diagnostic accuracy."
        )

    recommendations.extend([
        "Have the annotated image and detections reviewed by a qualified laboratory professional.",
        "Do not use these image-level counts alone to diagnose a disease or determine whether a blood count is normal or abnormal.",
    ])
    return {
        "title": "Automated Detection Conclusion",
        "findings": findings,
        "recommendations": recommendations,
        "status": "Requires professional review",
    }


# =========================================================
# CHART + PDF
# =========================================================
def create_chart(counts: dict) -> BytesIO:
    labels = list(counts.keys())
    values = list(counts.values())
    chart_colors = [np.array(COLORS[name][::-1]) / 255.0 for name in labels]
    fig, ax = plt.subplots(figsize=(7, 3.7))
    bars = ax.bar(labels, values, color=chart_colors, edgecolor="white", linewidth=0.8)
    ax.set_title("Detected Cell Distribution", fontsize=14, pad=14)
    ax.set_ylabel("Number of detections")
    ax.set_xlabel("Cell class")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(value),
                ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer


def generate_pdf(analysis: dict) -> bytes:
    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer, pagesize=A4,
        rightMargin=16*mm, leftMargin=16*mm,
        topMargin=15*mm, bottomMargin=15*mm,
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportSmallCustom", parent=styles["BodyText"], fontSize=8, leading=11
    ))
    styles.add(ParagraphStyle(
        name="ReportTitleCustom", parent=styles["Title"], fontSize=20, leading=24,
        textColor=colors.HexColor("#123B63"), alignment=TA_CENTER, spaceAfter=4
    ))
    story = []

    def section(title):
        story.extend([Spacer(1, 9), Paragraph(title, styles["Heading2"]), Spacer(1, 4)])

    def para(text, style="Normal"):
        safe = (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        story.append(Paragraph(safe, styles[style]))

    story.append(Paragraph("HEMOSCAN AI", styles["ReportTitleCustom"]))
    story.append(Paragraph("BLOOD SMEAR IMAGE ANALYSIS REPORT", styles["Heading2"]))
    story.append(Paragraph(
        "AI-assisted research report • Not a diagnostic laboratory result",
        styles["ReportSmallCustom"]
    ))
    story.append(Spacer(1, 12))

    section("1. SPECIMEN AND REPORT DETAILS")
    metadata = [
        ["Sample ID", analysis.get("sample_id", "Not provided")],
        ["Analysis ID", analysis.get("analysis_id", "Not available")],
        ["Report generated", analysis.get("timestamp", "Not available")],
        ["AI model file", MODEL_PATH.name],
        ["Review status", analysis.get("review_status", "Pending human review")],
    ]
    meta_table = Table(metadata, colWidths=[42*mm, 128*mm], hAlign="LEFT")
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF2F8")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#1F2937")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), .45, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(meta_table)

    section("2. ANNOTATED BLOOD SMEAR IMAGE")
    img_buf = BytesIO()
    plt.imsave(img_buf, analysis["processed_rgb"], format="png")
    img_buf.seek(0)
    story.append(RLImage(img_buf, width=165*mm, height=100*mm, kind="proportional"))
    para("Detections and labels shown are model-generated and require review.", "ReportSmallCustom")

    section("3. DETECTION SUMMARY")
    counts = analysis["counts"]
    total = analysis["total_detections"]
    rows = [["Cell class", "Detected objects", "Share of detections"]]
    for name in ("RBC", "WBC", "Platelets"):
        count = int(counts.get(name, 0))
        rows.append([name, str(count), f"{(count/total*100) if total else 0:.1f}%"])
    rows.append(["TOTAL", str(total), "100.0%" if total else "0.0%"])
    count_table = Table(rows, colWidths=[70*mm, 45*mm, 55*mm], repeatRows=1)
    count_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#164E78")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#EAF2F8")),
        ("GRID", (0, 0), (-1, -1), .45, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, colors.HexColor("#F8FAFC")]),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(count_table)

    section("4. MODEL CONFIDENCE")
    if analysis["confidence_values"]:
        para(f"Mean model confidence: {analysis['mean_confidence']:.1f}%")
        para("Confidence is the model's score for its detections; it is not diagnostic accuracy.", "ReportSmallCustom")
    else:
        para("No objects were detected by the model. This does not prove that blood cells are absent.")

    section("5. IMAGE QUALITY REVIEW")
    quality = analysis["image_quality"]
    for line in [
        f"Image dimensions: {quality['width']} × {quality['height']} pixels",
        f"Blur indicator (Laplacian variance): {quality['blur_score']}",
        f"Mean brightness: {quality['brightness']}",
        f"Contrast indicator: {quality['contrast']}",
        f"Quality status: {quality['status']}",
    ]:
        para(line, "ReportSmallCustom")
    if quality["warnings"]:
        para("Quality warnings:", "Heading3")
        for warning in quality["warnings"]:
            para("• " + warning, "ReportSmallCustom")
    else:
        para("No warnings were triggered by the configured basic image-quality checks. This does not establish clinical quality.", "ReportSmallCustom")

    section("6. CELL DISTRIBUTION CHART")
    chart_buffer = create_chart(counts)
    story.append(RLImage(chart_buffer, width=145*mm, height=82*mm, kind="proportional"))

    section("7. AUTOMATED DETECTION CONCLUSION")
    conclusion = generate_conclusion(analysis)
    conclusion_rows = [["Interpretation status", conclusion["status"]]]
    for finding in conclusion["findings"]:
        conclusion_rows.append(["Finding", Paragraph(finding.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), styles["ReportSmallCustom"])])
    for recommendation in conclusion["recommendations"]:
        conclusion_rows.append(["Recommendation", Paragraph(recommendation.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), styles["ReportSmallCustom"])])
    conclusion_table = Table(conclusion_rows, colWidths=[38*mm, 132*mm], hAlign="LEFT")
    conclusion_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF2F8")),
        ("GRID", (0, 0), (-1, -1), .45, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("PADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(conclusion_table)

    section("8. LABORATORY REVIEW")
    review_rows = [
        ["Reviewer name", "________________________________________"],
        ["Signature", "________________________________________"],
        ["Review date", "________________________________________"],
        ["Comments", "________________________________________\n________________________________________"],
    ]
    review_table = Table(review_rows, colWidths=[42*mm, 128*mm], hAlign="LEFT")
    review_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), .45, colors.HexColor("#CBD5E1")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(review_table)

    section("IMPORTANT MEDICAL DISCLAIMER")
    para(
        "This report was generated by an AI-based research and educational prototype. "
        "Counts represent model detections in the uploaded image, not validated blood-cell "
        "concentrations or a complete blood count. Image-quality checks are heuristic. "
        "These results must not be used alone for diagnosis, treatment, or clinical decision-making. "
        "Review by a qualified laboratory professional and appropriate validation are required "
        "before any clinical use.",
        "ReportSmallCustom"
    )
    doc.build(story)
    return pdf_buffer.getvalue()


# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="hero">
  <div class="eyebrow">AI-powered microscopy • Research prototype</div>
  <h1>HemoScan AI</h1>
  <p>Blood smear image analysis with YOLO-powered cell detection, image-quality indicators,
  confidence summaries, and downloadable laboratory-style reports.</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="notice"><strong>Research use only.</strong> This tool counts model detections '
    'in a submitted image. It is not a validated CBC test and must not be used alone for diagnosis '
    'or treatment decisions.</div>',
    unsafe_allow_html=True,
)
st.write("")


# =========================================================
# UPLOAD + RUN
# =========================================================
left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.markdown('<div class="section-title">01 / Upload specimen image</div>', unsafe_allow_html=True)
    sample_id = st.text_input("Sample ID (optional)", placeholder="e.g. SAMPLE-001", max_chars=80)
    uploaded_file = st.file_uploader(
        "Choose a blood smear image",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
        help="Maximum file size: 15 MB. Use a clear microscopy image.",
    )
    if uploaded_file is not None:
        if uploaded_file.size > MAX_UPLOAD_SIZE:
            st.error("Image is too large. Maximum upload size is 15 MB.")
            uploaded_file = None
        else:
            st.image(uploaded_file, caption="Uploaded image preview", use_container_width=True)

    run_analysis = st.button("Run cell detection", type="primary", use_container_width=True, disabled=uploaded_file is None)

with right:
    st.markdown('<div class="section-title">02 / Analysis workflow</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="panel">
      <strong>01 — Image validation</strong><br><span class="small-muted">Decode image and check basic technical quality.</span><br><br>
      <strong>02 — YOLO inference</strong><br><span class="small-muted">Detect the model's Platelets, RBC, and WBC classes.</span><br><br>
      <strong>03 — Results & report</strong><br><span class="small-muted">Review annotated image, counts, confidence, quality checks, and PDF.</span>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Model: runs/detect/train/weights/best.pt")
    st.caption("Maximum upload: 15 MB")


if run_analysis and uploaded_file is not None:
    try:
        raw_bytes = uploaded_file.getvalue()
        if not raw_bytes:
            st.error("The uploaded file is empty.")
        else:
            encoded = np.frombuffer(raw_bytes, np.uint8)
            image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image_bgr is None or image_bgr.size == 0:
                st.error("Invalid or unsupported image. Please upload a valid image.")
            else:
                with st.spinner("Loading model and analyzing image..."):
                    yolo_model = load_model(str(MODEL_PATH))
                    result_data = analyze_image(yolo_model, image_bgr, sample_id)
                    st.session_state["hemoscan_latest"] = result_data
                    st.session_state["hemoscan_pdf"] = generate_pdf(result_data)
                st.success("Analysis completed. Review the model-generated results below.")
    except FileNotFoundError as exc:
        st.error(str(exc))
    except Exception as exc:
        logger.exception("Image analysis failed")
        st.error("Image analysis failed. Check that the model file and dependencies are correct.")
        with st.expander("Technical error details"):
            st.code(str(exc))


# =========================================================
# RESULTS
# =========================================================
analysis = st.session_state.get("hemoscan_latest")
if analysis:
    st.divider()
    st.markdown('<div class="section-title">03 / Detection overview</div>', unsafe_allow_html=True)

    counts = analysis["counts"]
    metrics = st.columns(5, gap="medium")
    metric_items = [
        ("Total detections", analysis["total_detections"], "All supported classes"),
        ("Red blood cells", counts.get("RBC", 0), "RBC detections"),
        ("White blood cells", counts.get("WBC", 0), "WBC detections"),
        ("Platelets", counts.get("Platelets", 0), "Platelet detections"),
        ("Mean confidence", f'{analysis["mean_confidence"]:.1f}%', "Not diagnostic accuracy"),
    ]
    for col, (label, value, note) in zip(metrics, metric_items):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
                unsafe_allow_html=True,
            )

    st.write("")
    img_col, quality_col = st.columns([1.25, 0.75], gap="large")
    with img_col:
        st.markdown('<div class="section-title">Annotated detection image</div>', unsafe_allow_html=True)
        st.image(analysis["processed_rgb"], use_container_width=True)
        image_bytes = BytesIO()
        plt.imsave(image_bytes, analysis["processed_rgb"], format="png")
        st.download_button(
            "Download annotated image",
            data=image_bytes.getvalue(),
            file_name=f"hemoscan_{analysis['analysis_id']}.png",
            mime="image/png",
        )

    with quality_col:
        st.markdown('<div class="section-title">Image quality review</div>', unsafe_allow_html=True)
        quality = analysis["image_quality"]
        st.markdown(
            f'<div class="panel"><div class="metric-label">Quality indicator</div>'
            f'<strong>{quality["status"]}</strong><hr>'
            f'<span class="small-muted">Dimensions</span><br><strong>{quality["width"]} × {quality["height"]} px</strong><br><br>'
            f'<span class="small-muted">Blur indicator</span><br><strong>{quality["blur_score"]}</strong><br><br>'
            f'<span class="small-muted">Brightness / contrast</span><br>'
            f'<strong>{quality["brightness"]} / {quality["contrast"]}</strong></div>',
            unsafe_allow_html=True,
        )
        if quality["warnings"]:
            for warning in quality["warnings"]:
                st.warning(warning)
        else:
            st.info("No warnings triggered by the configured basic checks. This does not establish clinical quality.")

    st.markdown('<div class="section-title">Cell distribution</div>', unsafe_allow_html=True)
    chart_col, table_col = st.columns([1, 1], gap="large")
    with chart_col:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        names = ["RBC", "WBC", "Platelets"]
        vals = [counts.get(name, 0) for name in names]
        chart_cols = [np.array(COLORS[name][::-1]) / 255.0 for name in names]
        bars = ax.bar(names, vals, color=chart_cols, edgecolor="white", linewidth=.8)
        ax.set_ylabel("Detections")
        ax.grid(axis="y", linestyle="--", alpha=.25)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(value),
                    ha="center", va="bottom")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with table_col:
        st.dataframe(
            [{"Cell class": name, "Detections": counts.get(name, 0),
              "Share": f'{(counts.get(name, 0) / analysis["total_detections"] * 100) if analysis["total_detections"] else 0:.1f}%'}
             for name in ("RBC", "WBC", "Platelets")],
            hide_index=True, use_container_width=True,
        )
        st.markdown(f'**Analysis ID:** `{analysis["analysis_id"]}`')
        st.markdown(f'**Sample ID:** `{analysis["sample_id"]}`')
        st.markdown(f'**Analyzed:** {analysis["timestamp"]}')
        st.markdown(f'**Review status:** {analysis["review_status"]}')

    st.markdown('<div class="section-title">Automated detection conclusion</div>', unsafe_allow_html=True)
    conclusion = generate_conclusion(analysis)
    st.markdown(f'<div class="panel"><strong>{conclusion["status"]}</strong></div>', unsafe_allow_html=True)
    for finding in conclusion["findings"]:
        st.write("• " + finding)
    st.markdown("**Recommendations**")
    for recommendation in conclusion["recommendations"]:
        st.write("• " + recommendation)

    st.divider()
    pdf_bytes = st.session_state.get("hemoscan_pdf")
    if pdf_bytes:
        st.download_button(
            "Download laboratory-style PDF report",
            data=pdf_bytes,
            file_name=f"hemoscan_report_{analysis['analysis_id']}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True,
        )

st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("HemoScan AI • AI-assisted blood smear research prototype • Human review required")
