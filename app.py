from flask import (
    Flask, render_template, request, jsonify,
    send_file, url_for
)
from ultralytics import YOLO

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
import uuid
import logging
from datetime import datetime
from pathlib import Path
from threading import Lock
from html import escape

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Image as RLImage, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm


# =====================================================
# APPLICATION CONFIGURATION
# =====================================================

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "static" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = BASE_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"

# Prevent extremely large uploads. Adjust for your dataset.
MAX_UPLOAD_SIZE = 15 * 1024 * 1024
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE

logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

# =====================================================
# YOLO MODEL
# =====================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"YOLO model not found at: {MODEL_PATH}\n"
        "Check that your trained best.pt file exists."
    )

model = YOLO(str(MODEL_PATH))

CLASS_NAMES = {
    0: "Platelets",
    1: "RBC",
    2: "WBC",
}

# OpenCV colors use BGR format.
COLORS = {
    "Platelets": (0, 215, 255),
    "RBC": (69, 53, 220),
    "WBC": (255, 144, 30),
}

# In-memory result index for the current server process.
# For production, use a database and persistent file storage.
analyses = {}
latest_analysis_id = None
analysis_lock = Lock()


# =====================================================
# IMAGE QUALITY ASSESSMENT
# =====================================================

def assess_image_quality(image):
    """
    Basic technical quality checks.
    These are image-quality indicators, NOT clinical assessments.
    """
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    warnings = []

    if width < 256 or height < 256:
        warnings.append(
            "Low image resolution. A higher-resolution image may be needed."
        )

    # Heuristic thresholds; calibrate these for your microscope and dataset.
    if blur_score < 50:
        warnings.append(
            "Possible blur detected. Consider capturing a sharper image."
        )

    if brightness < 45:
        warnings.append(
            "Image may be too dark."
        )
    elif brightness > 220:
        warnings.append(
            "Image may be overexposed."
        )

    if contrast < 20:
        warnings.append(
            "Low image contrast may affect detection quality."
        )

    return {
        "width": width,
        "height": height,
        "blur_score": round(blur_score, 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "warnings": warnings,
        "status": "Review recommended" if warnings else "No basic quality warning",
    }


# =====================================================
# CHART GENERATION
# =====================================================

def generate_chart(counts, chart_path):
    labels = list(counts.keys())
    values = list(counts.values())

    chart_colors = [
        np.array(COLORS[name][::-1]) / 255.0
        for name in labels
    ]

    fig, ax = plt.subplots(figsize=(7, 4))

    bars = ax.bar(
        labels,
        values,
        color=chart_colors,
        edgecolor="white",
        linewidth=0.8,
    )

    ax.set_title("Detected Cell Distribution", fontsize=14, pad=14)
    ax.set_ylabel("Number of detections")
    ax.set_xlabel("Cell class")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(chart_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# =====================================================


def generate_conclusion(analysis):
    """
    Generate a cautious, automated summary of image detections.
    This does not diagnose disease or assess clinical normality.
    """
    counts = analysis.get("counts", {})
    total = analysis.get("total_detections", sum(counts.values()))
    quality = analysis.get("image_quality", {})
    warnings = quality.get("warnings", [])
    confidence_values = analysis.get("confidence_values", [])

    rbc = counts.get("RBC", 0)
    wbc = counts.get("WBC", 0)
    platelets = counts.get("Platelets", 0)

    findings = []
    recommendations = []

    # Overall detection summary
    if total == 0:
        findings.append(
            "The AI model did not detect any cells in the submitted image. "
            "This does not mean that blood cells are absent."
        )
        recommendations.append(
            "Verify that the image contains a suitable blood smear and "
            "repeat the analysis if necessary."
        )
    else:
        findings.append(
            f"The AI model detected {total} objects: "
            f"{rbc} red blood cell (RBC) detections, "
            f"{wbc} white blood cell (WBC) detections, and "
            f"{platelets} platelet detections."
        )

    # Image quality
    if warnings:
        findings.append(
            "The image-quality checks raised "
            f"{len(warnings)} warning(s). These may affect detection "
            "reliability."
        )
        recommendations.append(
            "Review the image-quality warnings and consider recapturing "
            "the image before interpreting the results."
        )
    else:
        findings.append(
            "No warnings were triggered by the configured basic "
            "image-quality checks. This does not establish clinical quality."
        )

    # Model confidence
    if confidence_values:
        mean_confidence = analysis.get("mean_confidence", 0)
        findings.append(
            f"The mean model confidence across detected objects was "
            f"{mean_confidence:.1f}%. Confidence is not a measure of "
            f"diagnostic accuracy."
        )

    recommendations.append(
        "Have the annotated image and detections reviewed by a qualified "
        "laboratory professional."
    )
    recommendations.append(
        "Do not use these image-level counts alone to diagnose a disease "
        "or determine whether a blood count is normal or abnormal."
    )

    return {
        "title": "Automated Detection Conclusion",
        "findings": findings,
        "recommendations": recommendations,
        "status": "Requires professional review",
    }

# PDF REPORT GENERATION
# =====================================================
def generate_pdf(analysis):
    """Create a laboratory-style PDF with detection results and cautious conclusion."""
    pdf_path = analysis["pdf_path"]
    chart_path = analysis["chart_path"]
    processed_path = analysis["processed_path"]
    generate_chart(analysis["counts"], chart_path)

    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4,
        rightMargin=16 * mm, leftMargin=16 * mm,
        topMargin=15 * mm, bottomMargin=15 * mm,
    )
    styles = getSampleStyleSheet()
    styles.add(__import__('reportlab.lib.styles', fromlist=['ParagraphStyle']).ParagraphStyle(
        name="ReportSmall", parent=styles["BodyText"], fontSize=8, leading=11
    ))
    styles.add(__import__('reportlab.lib.styles', fromlist=['ParagraphStyle']).ParagraphStyle(
        name="ReportTitle", parent=styles["Title"], fontSize=20, leading=24,
        textColor=colors.HexColor("#123B63"), alignment=1, spaceAfter=4
    ))
    story = []

    def section(title):
        story.append(Spacer(1, 9))
        story.append(Paragraph(escape(title), styles["Heading2"]))
        story.append(Spacer(1, 4))

    def para(value, style="Normal"):
        story.append(Paragraph(escape(str(value)), styles[style]))

    story.append(Paragraph("HEMOSCAN AI", styles["ReportTitle"]))
    story.append(Paragraph("BLOOD SMEAR IMAGE ANALYSIS REPORT", styles["Heading2"]))
    story.append(Paragraph(
        "AI-assisted research report • Not a diagnostic laboratory result",
        styles["ReportSmall"]
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
    meta_table = Table(metadata, colWidths=[42 * mm, 128 * mm], hAlign="LEFT")
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF2F8")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#1F2937")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(meta_table)

    section("2. ANNOTATED BLOOD SMEAR IMAGE")
    if os.path.exists(processed_path):
        img = RLImage(str(processed_path), width=165 * mm, height=100 * mm, kind="proportional")
        story.append(img)
        para("Detections and labels shown are model-generated and require review.", "ReportSmall")
    else:
        para("The annotated image file was not available.")

    section("3. DETECTION SUMMARY")
    counts = analysis.get("counts", {})
    total = analysis.get("total_detections", sum(counts.values()))
    rows = [["Cell class", "Detected objects", "Share of detections"]]
    for class_name in ("RBC", "WBC", "Platelets"):
        count = int(counts.get(class_name, 0))
        share = (count / total * 100) if total else 0
        rows.append([class_name, str(count), f"{share:.1f}%"])
    rows.append(["TOTAL", str(total), "100.0%" if total else "0.0%"])
    count_table = Table(rows, colWidths=[70 * mm, 45 * mm, 55 * mm], repeatRows=1)
    count_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#164E78")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#EAF2F8")),
        ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, colors.HexColor("#F8FAFC")]),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(count_table)

    section("4. MODEL CONFIDENCE")
    confidence_values = analysis.get("confidence_values", [])
    if confidence_values:
        para(f"Mean model confidence: {analysis.get('mean_confidence', 0):.1f}%")
        para("Confidence is the model's score for its detections; it is not a measure of diagnostic accuracy.", "ReportSmall")
    else:
        para("No objects were detected by the model. This does not prove that blood cells are absent.")

    section("5. IMAGE QUALITY REVIEW")
    quality = analysis.get("image_quality", {})
    quality_lines = [
        f"Image dimensions: {quality.get('width', 'N/A')} × {quality.get('height', 'N/A')} pixels",
        f"Blur indicator (Laplacian variance): {quality.get('blur_score', 'N/A')}",
        f"Mean brightness: {quality.get('brightness', 'N/A')}",
        f"Contrast indicator: {quality.get('contrast', 'N/A')}",
        f"Quality status: {quality.get('status', 'Not available')}",
    ]
    for line in quality_lines:
        para(line, "ReportSmall")
    warnings = quality.get("warnings", [])
    if warnings:
        para("Quality warnings:", "Heading3")
        for warning in warnings:
            para("• " + warning, "ReportSmall")
    else:
        para("No warnings were triggered by the configured basic image-quality checks. This does not establish clinical quality.", "ReportSmall")

    section("6. CELL DISTRIBUTION CHART")
    if os.path.exists(chart_path):
        story.append(RLImage(str(chart_path), width=145 * mm, height=82 * mm, kind="proportional"))

    section("7. AUTOMATED DETECTION CONCLUSION")
    conclusion = generate_conclusion(analysis)
    conclusion_rows = [["Interpretation status", conclusion["status"]]]
    for finding in conclusion["findings"]:
        conclusion_rows.append(["Finding", Paragraph(escape(finding), styles["ReportSmall"])])
    for recommendation in conclusion["recommendations"]:
        conclusion_rows.append(["Recommendation", Paragraph(escape(recommendation), styles["ReportSmall"])])
    conclusion_table = Table(conclusion_rows, colWidths=[38 * mm, 132 * mm], hAlign="LEFT")
    conclusion_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF2F8")),
        ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#CBD5E1")),
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
    review_table = Table(review_rows, colWidths=[42 * mm, 128 * mm], hAlign="LEFT")
    review_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#CBD5E1")),
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
        "ReportSmall"
    )
    doc.build(story)


# HOME PAGE
# =====================================================

@app.route("/")
def index():
    return render_template("index.html")


# =====================================================
# PREDICTION API
# =====================================================

@app.route("/predict", methods=["POST"])
def predict():
    global latest_analysis_id

    # Validate request.
    if "image" not in request.files:
        return jsonify({
            "error": "No image was uploaded. Please choose an image."
        }), 400

    file = request.files["image"]

    if not file or not file.filename:
        return jsonify({"error": "Please select an image file."}), 400

    img_bytes = file.read()

    if not img_bytes:
        return jsonify({"error": "The uploaded file is empty."}), 400

    if len(img_bytes) > MAX_UPLOAD_SIZE:
        return jsonify({
            "error": "Image is too large. Maximum upload size is 15 MB."
        }), 413

    # Decode the file rather than trusting its extension or MIME type.
    np_image = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({
            "error": "Invalid or unsupported image. Upload a valid image."
        }), 400

    if image.size == 0:
        return jsonify({"error": "The image contains no usable pixels."}), 400

    try:
        # Assess technical image quality before inference.
        quality = assess_image_quality(image)

        # Run model on the original decoded image.
        results = model.predict(
            source=image,
            verbose=False,
        )[0]

        counts = {name: 0 for name in CLASS_NAMES.values()}
        boxes_data = []
        confidence_values = []

        # Draw on a copy, preserving the original for quality calculations.
        processed_image = image.copy()
        height, width = processed_image.shape[:2]

        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            # Skip unknown classes safely.
            if cls_id not in CLASS_NAMES:
                app.logger.warning("Model returned unknown class ID: %s", cls_id)
                continue

            class_name = CLASS_NAMES[cls_id]
            counts[class_name] += 1
            confidence_values.append(confidence * 100)

            x1, y1, x2, y2 = map(
                int, box.xyxy[0].cpu().tolist()
            )

            # Clamp coordinates to the image dimensions.
            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width - 1))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height - 1))

            color = COLORS[class_name]

            cv2.rectangle(
                processed_image,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            label = f"{class_name} {confidence * 100:.1f}%"
            label_y = max(20, y1 - 8)

            cv2.putText(
                processed_image,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

            # Convert OpenCV's BGR color to CSS RGB.
            red, green, blue = color[::-1]

            boxes_data.append({
                "x": x1,
                "y": y1,
                "w": max(0, x2 - x1),
                "h": max(0, y2 - y1),
                "label": class_name,
                "confidence": round(confidence * 100, 1),
                "color": f"rgb({red}, {green}, {blue})",
            })

        # Use a unique ID and output paths for every analysis.
        analysis_id = uuid.uuid4().hex
        timestamp = datetime.now().astimezone().strftime(
            "%d %b %Y, %H:%M:%S %Z"
        )

        processed_filename = f"processed_{analysis_id}.jpg"
        chart_filename = f"chart_{analysis_id}.png"
        pdf_filename = f"blood_report_{analysis_id}.pdf"

        processed_path = OUTPUT_DIR / processed_filename
        chart_path = OUTPUT_DIR / chart_filename
        pdf_path = OUTPUT_DIR / pdf_filename

        if not cv2.imwrite(str(processed_path), processed_image):
            raise RuntimeError("Could not save the processed image.")

        sample_id = (
            request.form.get("sample_id", "").strip()[:80]
            or f"SAMPLE-{analysis_id[:8].upper()}"
        )

        total_detections = sum(counts.values())
        mean_confidence = (
            float(np.mean(confidence_values))
            if confidence_values else 0.0
        )

        analysis = {
            "analysis_id": analysis_id,
            "sample_id": sample_id,
            "timestamp": timestamp,
            "counts": counts,
            "boxes": boxes_data,
            "total_detections": total_detections,
            "confidence_values": confidence_values,
            "mean_confidence": round(mean_confidence, 1),
            "image_quality": quality,
            "review_status": "Pending human review",
            "processed_path": str(processed_path),
            "chart_path": str(chart_path),
            "pdf_path": str(pdf_path),
        }

        with analysis_lock:
            analyses[analysis_id] = analysis
            latest_analysis_id = analysis_id

        app.logger.info(
            "Analysis %s completed: %s",
            analysis_id,
            counts,
        )

        return jsonify({
            "success": True,
            "analysis_id": analysis_id,
            "sample_id": sample_id,
            "timestamp": timestamp,
            "counts": counts,
            "total_detections": total_detections,
            "boxes": boxes_data,
            "mean_confidence": round(mean_confidence, 1),
            "image_quality": quality,
            "review_status": analysis["review_status"],
            "image": url_for(
                "static",
                filename=f"output/{processed_filename}",
            ),
            "report_url": url_for(
                "download_report",
                analysis_id=analysis_id,
            ),
        })

        except Exception as e:
        import traceback

        print("========== IMAGE ANALYSIS FAILED ==========", flush=True)
        print(f"Error: {repr(e)}", flush=True)
        traceback.print_exc()

        return jsonify({
            "error": "Image analysis failed",
            "details": str(e)
        }), 500


# =====================================================
# DOWNLOAD REPORT
# =====================================================

@app.route("/download-report")
def download_report():
    requested_id = request.args.get("analysis_id")

    with analysis_lock:
        if requested_id:
            analysis = analyses.get(requested_id)
        else:
            analysis = analyses.get(latest_analysis_id)

    if not analysis:
        return jsonify({
            "error": "No analysis report is available. Analyze an image first."
        }), 404

    try:
        generate_pdf(analysis)

        return send_file(
            analysis["pdf_path"],
            as_attachment=True,
            download_name=f"blood_report_{analysis['analysis_id'][:8]}.pdf",
            mimetype="application/pdf",
        )

    except Exception:
        app.logger.exception("PDF report generation failed")
        return jsonify({
            "error": "Could not generate the PDF report."
        }), 500


# =====================================================
# ANALYSIS DETAILS API
# =====================================================

@app.route("/analysis/<analysis_id>", methods=["GET"])
def get_analysis(analysis_id):
    with analysis_lock:
        analysis = analyses.get(analysis_id)

    if not analysis:
        return jsonify({"error": "Analysis not found."}), 404

    return jsonify({
        "analysis_id": analysis["analysis_id"],
        "sample_id": analysis["sample_id"],
        "timestamp": analysis["timestamp"],
        "counts": analysis["counts"],
        "total_detections": analysis["total_detections"],
        "mean_confidence": analysis["mean_confidence"],
        "image_quality": analysis["image_quality"],
        "review_status": analysis["review_status"],
    })


# =====================================================
# ERROR HANDLERS
# =====================================================

@app.errorhandler(413)
def request_entity_too_large(_error):
    return jsonify({
        "error": "Upload too large. Maximum allowed size is 15 MB."
    }), 413


@app.errorhandler(404)
def page_not_found(_error):
    return jsonify({"error": "The requested resource was not found."}), 404


# =====================================================
# RUN APPLICATION
# =====================================================

if __name__ == "__main__":
    # Keep debug mode off when exposing the app to other machines.
    app.run(host="127.0.0.1", port=5000, debug=False)