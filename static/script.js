/* HemoScan AI frontend: upload, analyze, and present review-oriented results. */
(() => {
  "use strict";

  const input = document.getElementById("imageInput");
  const button = document.getElementById("analyzeBtn");
  const status = document.getElementById("resultStatus");
  const downloadLink = document.getElementById("downloadReportBtn");

  const $ = (id) => document.getElementById(id);
  const safeNumber = (value) => Number.isFinite(Number(value)) ? Number(value) : 0;

  window.uploadImage = async function uploadImage() {
    const file = input?.files?.[0];
    if (!file) {
      alert("Please choose a microscopic blood image first.");
      return;
    }
    if (!file.type.startsWith("image/")) {
      alert("Please select a valid image file.");
      return;
    }
    if (file.size > 15 * 1024 * 1024) {
      alert("The image must be 15 MB or smaller.");
      return;
    }

    const formData = new FormData();
    formData.append("image", file);
    const sampleId = $("sampleId")?.value?.trim();
    if (sampleId) formData.append("sample_id", sampleId);

    button.disabled = true;
    button.innerHTML = '<span>◌</span> Analyzing image…';
    if (status) status.innerHTML = '<span class="status-dot"></span> Processing image';
    downloadLink?.classList.add("disabled");

    try {
      const response = await fetch("/predict", { method: "POST", body: formData });
      const data = await response.json().catch(() => ({}));
      if (!response.ok || !data.success) {
        throw new Error(data.error || "Analysis failed. Check the Flask terminal for details.");
      }
      renderResults(data);
      if (downloadLink) {
        downloadLink.href = data.report_url || `/download-report?analysis_id=${encodeURIComponent(data.analysis_id)}`;
        downloadLink.classList.remove("disabled");
      }
      $("results")?.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (error) {
      console.error(error);
      alert(error.message || "Could not analyze this image.");
      if (status) status.innerHTML = '<span class="status-dot"></span> Analysis failed';
    } finally {
      button.disabled = false;
      button.innerHTML = '<span>✦</span> Analyze Blood Sample <span class="btn-arrow">→</span>';
    }
  };

  function renderResults(data) {
    const counts = data.counts || {};
    const rbc = safeNumber(counts.RBC);
    const wbc = safeNumber(counts.WBC);
    const platelets = safeNumber(counts.Platelets);
    const total = safeNumber(data.total_detections ?? (rbc + wbc + platelets));

    $("rbcCount").textContent = rbc.toLocaleString();
    $("wbcCount").textContent = wbc.toLocaleString();
    $("plateletCount").textContent = platelets.toLocaleString();

    const percentages = total
      ? { rbc: rbc / total * 100, wbc: wbc / total * 100, platelets: platelets / total * 100 }
      : { rbc: 0, wbc: 0, platelets: 0 };
    document.querySelector(".rbc .cell-progress span").style.width = `${percentages.rbc}%`;
    document.querySelector(".wbc .cell-progress span").style.width = `${percentages.wbc}%`;
    document.querySelector(".platelets .cell-progress span").style.width = `${percentages.platelets}%`;

    if (status) status.innerHTML = '<span class="status-dot"></span> Analysis complete';

    const quality = data.image_quality || {};
    const warnings = Array.isArray(quality.warnings) ? quality.warnings : [];
    $("qualityStatus").textContent = warnings.length ? "REVIEW RECOMMENDED" : "NO BASIC WARNINGS";
    $("qualitySummary").textContent =
      `Image: ${quality.width ?? "?"} × ${quality.height ?? "?"} px · ` +
      `Blur indicator: ${quality.blur_score ?? "N/A"} · ` +
      `Brightness: ${quality.brightness ?? "N/A"} · Contrast: ${quality.contrast ?? "N/A"}`;

    const warningList = $("qualityWarnings");
    warningList.replaceChildren();
    if (warnings.length) {
      warnings.forEach((warning) => {
        const li = document.createElement("li");
        li.textContent = warning;
        warningList.appendChild(li);
      });
    } else {
      const li = document.createElement("li");
      li.style.color = "#15803d";
      li.textContent = "No warning was triggered by the configured basic checks. This does not prove clinical image quality.";
      warningList.appendChild(li);
    }

    $("conclusionStatus").textContent = warnings.length ? "CHECK IMAGE QUALITY" : "DETECTIONS SUMMARIZED";
    const summary = total === 0
      ? "The model did not detect any objects in this image. This does not prove that blood cells are absent."
      : `The model detected ${total} objects: ${rbc} RBC detections, ${wbc} WBC detections, and ${platelets} platelet detections. These are image-level detections, not validated blood-cell concentrations.`;
    $("conclusionText").textContent = summary;
    $("nextSteps").textContent = warnings.length
      ? "Next steps: review the image-quality warnings, inspect the annotated image, and verify uncertain detections before finalizing the report."
      : "Next steps: inspect the annotated image and have the detections reviewed by a qualified laboratory professional. Do not use these counts alone to diagnose disease or determine normality.";

    $("reviewStatus").textContent = data.review_status || "PENDING HUMAN REVIEW";
    $("analysisDetails").textContent =
      `Sample ID: ${data.sample_id || "Not provided"} · ` +
      `Total detections: ${total} · ` +
      `Mean model confidence: ${safeNumber(data.mean_confidence).toFixed(1)}% · ` +
      `Analysis ID: ${data.analysis_id || "N/A"} · ` +
      `Generated: ${data.timestamp || "N/A"}`;

    // Replace the preview with the server-annotated result after successful inference.
    if (data.image) {
      const preview = $("preview");
      const placeholder = $("placeholder");
      preview.src = `${data.image}${data.image.includes("?") ? "&" : "?"}t=${Date.now()}`;
      preview.style.display = "block";
      placeholder.style.display = "none";
      preview.onload = () => {
        $("imageDimensions").textContent = `${preview.naturalWidth} × ${preview.naturalHeight} px · Annotated`;
      };
    }
  }
})();
