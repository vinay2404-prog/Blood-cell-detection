"""HemoScan AI - QAPS-YOLO research prototype.

Streamlit interface. All scientific logic lives in `core.py` so that the app
and the offline evaluation script share one implementation.
"""
from __future__ import annotations

import json
import logging
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

import core
from core import (
    ALGORITHM_VERSION,
    CLASS_NAMES,
    EXPERIMENT_VERSION,
    MAX_UPLOAD_SIZE,
    MODEL_PATH,
    THRESHOLDS,
    YOLO_CONFIG,
    analyze_image,
    create_distribution_chart,
    csv_rows,
    decode_image,
    generate_conclusion,
    load_model,
    model_sha256 as core_model_sha256,
    rows_to_csv,
    sha256_bytes,
    to_png,
)

st.set_page_config(
    page_title="HemoScan AI | Research Analyzer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HemoScan")


@st.cache_data(show_spinner=False)
def model_sha256(path_str: str, mtime: float) -> str:
    return core_model_sha256(path_str, mtime)


def generate_pdf(a: dict, chart_png: bytes, annotated_png: bytes) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=16 * mm, leftMargin=16 * mm,
                            topMargin=15 * mm, bottomMargin=15 * mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SmallReport", parent=styles["BodyText"], fontSize=8, leading=11))
    styles.add(ParagraphStyle(name="CustomTitle", parent=styles["Title"], fontSize=20, leading=24,
                              textColor=colors.HexColor("#123B63"), alignment=TA_CENTER))
    story = []

    def section(title):
        story.extend([Spacer(1, 10), Paragraph(title, styles["Heading2"]), Spacer(1, 5)])

    def para(value, style="Normal"):
        safe = str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        story.append(Paragraph(safe, styles[style]))

    def table(rows, widths, header=False, font=None):
        t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
        style = [("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("PADDING", (0, 0), (-1, -1), 6),
                 ("VALIGN", (0, 0), (-1, -1), "TOP")]
        if header:
            style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#164E78")),
                      ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                      ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
        else:
            style += [("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF2F8")),
                      ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold")]
        if font:
            style.append(("FONTSIZE", (0, 0), (-1, -1), font))
        t.setStyle(TableStyle(style))
        story.append(t)

    story.append(Paragraph("HEMOSCAN AI", styles["CustomTitle"]))
    para("QAPS-YOLO BLOOD SMEAR RESEARCH REPORT", "Heading2")
    para("AI-assisted experimental analysis. Not a diagnostic report.", "SmallReport")

    section("1. Experiment details")
    table([
        ["Sample ID", a["sample_id"]], ["Analysis ID", a["analysis_id"]],
        ["Timestamp", a["timestamp"]], ["Model", Path(a["model_path"]).name],
        ["Algorithm", "QAPS " + ALGORITHM_VERSION],
        ["Experiment version", a["experiment_version"]],
        ["Review status", a["review_status"]],
        ["YOLO config", json.dumps(a["yolo_config"])],
        ["Image SHA-256", a["image_hash"]], ["Model SHA-256", a["model_hash"]],
    ], [45 * mm, 125 * mm], font=7)

    section("2. QAPS-YOLO annotated image")
    story.append(RLImage(BytesIO(annotated_png), width=165 * mm, height=100 * mm, kind="proportional"))

    section("3. Baseline vs QAPS experimental comparison")
    b, q, cmp_ = a["baseline"], a["qaps"], a["comparison"]
    rows = [["Metric", "Baseline YOLO", "QAPS-YOLO"],
            ["Total detections", b["total_detections"], q["total_detections"]],
            ["Mean confidence", f"{b['mean_confidence']:.2f}%", f"{q['mean_confidence']:.2f}%"],
            ["Inference time", f"{b['inference_time']:.5f} s", f"{q['inference_time']:.5f} s"]]
    rows += [[n, b["counts"][n], q["counts"][n]] for n in CLASS_NAMES.values()]
    table(rows, [60 * mm, 55 * mm, 55 * mm], header=True)
    para(f"Detection-count difference: {cmp_['detection_change']:+d}")
    para(f"Mean-confidence difference: {cmp_['confidence_change']:+.2f} percentage points")
    para("Note: count and confidence differences do not establish improved detection accuracy.")

    section("4. Detection distribution comparison")
    story.append(RLImage(BytesIO(chart_png), width=150 * mm, height=80 * mm, kind="proportional"))

    section("5. Image quality assessment")
    qa = a["image_quality"]
    table([["Measurement", "Value"], ["Resolution", f"{qa['width']} x {qa['height']}"],
           ["Blur indicator", qa["blur_score"]], ["Brightness", qa["brightness"]],
           ["Contrast", qa["contrast"]], ["Noise estimate", qa["noise_score"]],
           ["Quality status", qa["status"]]], [70 * mm, 100 * mm], header=True)
    for w in qa["warnings"]:
        para("Warning: " + w, "SmallReport")

    section("6. QAPS preprocessing details")
    pp = a["preprocessing"]
    para("Selected operations: " + ", ".join(pp["selected_operations"]))
    para("Parameters: " + json.dumps(pp["parameters"]))
    para("Thresholds: " + json.dumps(pp["thresholds"]))
    para(f"QAPS processing time: {pp['processing_time']:.5f} seconds")

    section("7. Experimental conclusion")
    concl = generate_conclusion(a)
    para(concl["status"], "Heading3")
    for item in concl["findings"] + concl["recommendations"]:
        para("- " + item, "SmallReport")

    section("8. Laboratory review")
    line = "_" * 32
    table([["Reviewer", line], ["Signature", line], ["Review date", line], ["Comments", line]],
           [45 * mm, 125 * mm])

    section("Important research disclaimer")
    para("This application is an AI-based research prototype. Detection counts represent "
         "model outputs in an image, not validated blood-cell concentrations or a complete "
         "blood count. Increased detections or confidence do not establish improved accuracy. "
         "Ground-truth annotations and quantitative validation are required. This report must "
         "not be used independently for diagnosis, treatment or clinical decisions.",
         "SmallReport")

    doc.build(story)
    return buffer.getvalue()


# =========================================================
# UI
# =========================================================

st.title("🧬 HemoScan AI")
st.subheader("Quality-Aware Blood Smear Image Analyzer")

with st.sidebar:
    st.header("About")
    st.write("Research prototype for blood-cell detection and paired "
             "baseline vs QAPS-YOLO comparison.")
    st.caption(f"Model: {MODEL_PATH.name}")
    st.caption(f"Confidence: {YOLO_CONFIG['confidence']}")
    st.caption(f"IoU: {YOLO_CONFIG['iou']}")
    st.divider()
    st.header("How to use")
    st.markdown("1. Upload a blood smear image\n"
                "2. Click **Run experiment**\n"
                "3. Review counts and quality\n"
                "4. Download PDF/CSV/PNG")
    st.warning("Research use only. Not for clinical diagnosis.")
st.write("A research-oriented blood-cell detection system integrating Quality-Aware "
         "Preprocessing Selection (QAPS) with an existing YOLO object detection model.")
st.warning("Research use only. Model outputs are not validated clinical blood counts "
           "or diagnostic results. Not for clinical diagnosis or treatment decisions.")

tab_single, tab_batch = st.tabs(["Single image", "Batch experiment"])

# ---------------------------------------------------------
# Single image
# ---------------------------------------------------------
with tab_single:
    st.header("1. Upload specimen image")
    left, right = st.columns([1.05, 0.95])

    with left:
        sample_id = st.text_input("Sample ID (optional)", placeholder="SAMPLE-001", max_chars=80,
                                  help="Optional label for the specimen.")
        uploaded = st.file_uploader(
            "Upload blood smear image",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
            help="Supported: PNG, JPG, BMP, TIFF, WEBP (max 15 MB).",
        )
        valid = uploaded is not None and uploaded.size <= MAX_UPLOAD_SIZE
        current_hash = None
        if uploaded is not None:
            if not valid:
                st.error("Maximum upload size is 15 MB.")
            else:
                st.image(uploaded, caption="Original specimen", use_container_width=True)
                current_hash = sha256_bytes(uploaded.getvalue())
                try:
                    _q = core.assess_image_quality(decode_image(uploaded.getvalue()))
                    st.caption("Quality preview")
                    qcols = st.columns(4)
                    qcols[0].metric("Blur", _q["blur_score"])
                    qcols[1].metric("Brightness", _q["brightness"])
                    qcols[2].metric("Contrast", _q["contrast"])
                    qcols[3].metric("Noise", _q["noise_score"])
                    if _q["warnings"]:
                        for w in _q["warnings"]:
                            st.warning(w)
                except Exception:
                    pass
        run_clicked = st.button("Run Baseline + QAPS-YOLO Experiment", type="primary",
                                use_container_width=True, disabled=not valid)

    with right:
        st.subheader("Experimental workflow")
        st.markdown("1. Image quality assessment\n"
                    "2. Baseline YOLO detection\n"
                    "3. Adaptive QAPS preprocessing\n"
                    "4. QAPS-YOLO detection\n"
                    "5. Comparative evaluation\n"
                    "6. Research report generation")
        st.caption(f"Model: {MODEL_PATH.name}")
        st.caption(f"YOLO config: {YOLO_CONFIG}")

    if run_clicked and valid:
        try:
            raw = uploaded.getvalue()
            image = decode_image(raw)
            with st.status("Running experiment...", expanded=True) as status:
                status.write("Loading model...")
                model = load_model(str(MODEL_PATH))
                mhash = model_sha256(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
                status.write("Running baseline + QAPS-YOLO...")
                result = analyze_image(model, image, sample_id, sha256_bytes(raw), mhash)
                status.write("Building chart and report...")
                chart_png = create_distribution_chart(result)
                annotated_png = to_png(result["processed_rgb"])
                st.session_state["hs"] = {
                    "result": result,
                    "chart": chart_png,
                    "annotated": annotated_png,
                    "conclusion": generate_conclusion(result),
                    "pdf": generate_pdf(result, chart_png, annotated_png),
                    "csv": rows_to_csv(csv_rows(result)),
                }
                status.update(label="Experiment completed", state="complete")
            st.toast("Analysis finished successfully")
            try:
                import json as _json
                with (BASE_DIR / "audit_log.jsonl").open("a", encoding="utf-8") as _f:
                    _f.write(_json.dumps({
                        "timestamp": result["timestamp"],
                        "sample_id": result["sample_id"],
                        "image_hash": result["image_hash"],
                        "model_hash": result["model_hash"],
                    }) + "\n")
            except Exception:
                pass
        except (FileNotFoundError, ValueError) as exc:
            st.error(str(exc))
        except Exception as exc:
            logger.exception("Analysis failed")
            st.error("Analysis failed.")
            with st.expander("Technical error details"):
                st.code(str(exc))

    state = st.session_state.get("hs")
    if state and current_hash and state["result"]["image_hash"] == current_hash:
        a = state["result"]
        base, qaps, cmp_ = a["baseline"], a["qaps"], a["comparison"]
        pp, qa = a["preprocessing"], a["image_quality"]

        st.divider()
        st.header("2. Detection overview")
        for col, (label, value) in zip(st.columns(5), [
            ("QAPS total", qaps["total_detections"]), ("RBC", qaps["counts"]["RBC"]),
            ("WBC", qaps["counts"]["WBC"]), ("Platelets", qaps["counts"]["Platelets"]),
            ("Mean confidence", f"{qaps['mean_confidence']:.1f}%"),
        ]):
            col.metric(label, value)

        st.divider()
        st.header("3. Baseline vs QAPS-YOLO comparison")
        st.caption("Both branches use the same model and identical inference settings.")
        c1, c2 = st.columns(2)
        for col, title, r, cap in [
            (c1, "Baseline YOLO", base, "Detection on original image"),
            (c2, "Proposed QAPS-YOLO", qaps, "Detection on QAPS-preprocessed image"),
        ]:
            with col:
                st.subheader(title)
                st.metric("Total detections", r["total_detections"])
                st.metric("Mean confidence", f"{r['mean_confidence']:.2f}%")
                st.metric("Inference time", f"{r['inference_time']:.5f} s")
                st.image(r["annotated_rgb"], caption=cap, use_container_width=True)

        d1, d2, d3 = st.columns(3)
        d1.metric("Detection count difference", f"{cmp_['detection_change']:+d}")
        d2.metric("Confidence difference", f"{cmp_['confidence_change']:+.2f} pp")
        d3.metric("QAPS processing time", f"{pp['processing_time']:.5f} s")
        st.caption("Descriptive differences only, not accuracy improvements.")

        st.subheader("Class-wise detection comparison")
        st.dataframe(pd.DataFrame([
            {"Cell class": n, "Baseline": v["baseline"], "QAPS-YOLO": v["qaps"],
             "Difference": v["difference"]}
            for n, v in cmp_["class_comparison"].items()
        ]), hide_index=True, use_container_width=True,
           column_config={
               "Difference": st.column_config.NumberColumn("Difference", format="%+d"),
               "Baseline": st.column_config.NumberColumn("Baseline"),
               "QAPS-YOLO": st.column_config.NumberColumn("QAPS-YOLO"),
           })

        st.divider()
        st.header("4. QAPS adaptive preprocessing")
        st.write(f"**Version:** {pp['algorithm_version']}")
        st.write("**Selected operations:** " + ", ".join(pp["selected_operations"]))
        m1, m2, m3 = st.columns(3)
        m1.metric("Noise", pp["noise_score"])
        m2.metric("Brightness", pp["original_brightness"])
        m3.metric("Contrast", pp["original_contrast"])
        o, e = st.columns(2)
        with o:
            st.subheader("Original image")
            st.image(a["original_rgb"], use_container_width=True)
        with e:
            st.subheader("QAPS preprocessed image")
            st.image(a["preprocessed_rgb"], use_container_width=True)
        with st.expander("View algorithm parameters"):
            st.json(pp["parameters"])
            st.json(pp["thresholds"])

        st.divider()
        st.header("5. Image quality assessment")
        for col, (label, key) in zip(st.columns(4), [
            ("Blur indicator", "blur_score"), ("Brightness", "brightness"),
            ("Contrast", "contrast"), ("Noise", "noise_score"),
        ]):
            col.metric(label, qa[key])
        st.write(f"**Resolution:** {qa['width']} × {qa['height']}")
        st.write(f"**Quality status:** {qa['status']}")
        if qa["warnings"]:
            for w in qa["warnings"]:
                st.warning(w)
        else:
            st.info("No configured quality warnings were triggered.")

        st.divider()
        st.header("6. Cell distribution")
        st.image(state["chart"], use_container_width=True)



        st.divider()
        st.header("7. Experimental conclusion")
        concl = state["conclusion"]
        st.info(concl["status"])
        st.subheader("Findings")
        for f in concl["findings"]:
            st.write("- " + f)
        st.subheader("Recommendations")
        for r in concl["recommendations"]:
            st.write("- " + r)

        st.divider()
        st.header("8. Download research results")
        x1, x2, x3 = st.columns(3)
        x1.download_button("Download PDF report", state["pdf"],
                           file_name=f"hemoscan_{a['analysis_id']}.pdf",
                           mime="application/pdf", type="primary", use_container_width=True)
        x2.download_button("Download experiment CSV", state["csv"],
                           file_name=f"hemoscan_experiment_{a['analysis_id']}.csv",
                           mime="text/csv", use_container_width=True)
        x3.download_button("Download annotated image", state["annotated"],
                           file_name=f"hemoscan_{a['analysis_id']}.png",
                           mime="image/png", use_container_width=True)

        with st.expander("Research experiment metadata"):
            st.write("**Analysis ID:**", a["analysis_id"])
            st.write("**Sample ID:**", a["sample_id"])
            st.write("**Timestamp:**", a["timestamp"])
            st.write("**Image SHA-256:**", a["image_hash"])
            st.write("**Model SHA-256:**", a["model_hash"])
            st.json(a["yolo_config"])
            st.write("**Total analysis time:**", f"{a['analysis_time']:.5f} seconds")

# ---------------------------------------------------------
# Batch experiment
# ---------------------------------------------------------
with tab_batch:
    st.header("Batch experiment")
    st.caption("Runs baseline and QAPS-YOLO on many images and exports one aggregate CSV "
               "(one row per image per branch per class).")

    files = st.file_uploader(
        "Upload multiple blood smear images",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
        accept_multiple_files=True,
        key="batch_files",
    )

    if st.button("Run batch experiment", type="primary", disabled=not files):
        try:
            model = load_model(str(MODEL_PATH))
            mhash = model_sha256(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
            all_rows, summary, skipped = [], [], []
            bar = st.progress(0.0)
            status = st.status("Processing batch...", expanded=True)

            for i, f in enumerate(files, 1):
                status.write(f"{i}/{len(files)}: {f.name}")
                try:
                    if f.size > MAX_UPLOAD_SIZE:
                        raise ValueError("file larger than 15 MB")
                    raw = f.getvalue()
                    res = analyze_image(model, core.decode_image(raw), Path(f.name).stem,
                                        sha256_bytes(raw), mhash, draw=False)
                    all_rows += csv_rows(res)
                    summary.append({
                        "Image": f.name,
                        "Quality warnings": len(res["image_quality"]["warnings"]),
                        "QAPS operations": ", ".join(res["preprocessing"]["selected_operations"]),
                        "Baseline total": res["baseline"]["total_detections"],
                        "QAPS total": res["qaps"]["total_detections"],
                        "Count diff": res["comparison"]["detection_change"],
                        "Conf diff (pp)": res["comparison"]["confidence_change"],
                    })
                except Exception as exc:
                    skipped.append(f"{f.name}: {exc}")
                bar.progress(i / len(files))

            status.update(label="Batch complete", state="complete")
            st.session_state["hs_batch"] = {
                "summary": pd.DataFrame(summary),
                "csv": rows_to_csv(all_rows) if all_rows else b"",
                "skipped": skipped,
            }
            st.toast("Batch experiment finished")
        except (FileNotFoundError, ValueError) as exc:
            st.error(str(exc))
        except Exception as exc:
            logger.exception("Batch failed")
            st.error(f"Batch failed: {exc}")

    batch = st.session_state.get("hs_batch")
    if batch:
        for msg in batch["skipped"]:
            st.warning("Skipped " + msg)
        df = batch["summary"]
        if not df.empty:
            st.dataframe(df, hide_index=True, use_container_width=True)
            s1, s2, s3 = st.columns(3)
            s1.metric("Images analysed", len(df))
            s2.metric("Mean count difference", f"{df['Count diff'].mean():+.2f}")
            s3.metric("Mean confidence difference", f"{df['Conf diff (pp)'].mean():+.2f} pp")
            st.caption("Counts and confidence are not accuracy. Evaluate against ground "
                       "truth (precision, recall, F1, mAP) before drawing conclusions.")
            st.download_button("Download aggregate CSV", batch["csv"],
                               file_name="hemoscan_batch.csv", mime="text/csv",
                               type="primary")

st.divider()
st.caption("HemoScan AI | QAPS-YOLO Research Prototype | Paired baseline comparison | "
           "Human review required | Not for clinical diagnosis")