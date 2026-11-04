"use strict";

const form = document.querySelector("#analysis-form");
const fileInput = document.querySelector("#image-file");
const uploadZone = document.querySelector("#upload-zone");
const fileName = document.querySelector("#file-name");
const feedback = document.querySelector("#form-feedback");
const thresholdBlock = document.querySelector("#threshold-block");
const segmentationMode = document.querySelector("#segmentation-mode");
const thresholdRange = document.querySelector("#threshold-dbz");

const viewer = document.querySelector("#radar-image");
const placeholder = document.querySelector("#image-placeholder");
const dimensions = document.querySelector("#image-dimensions");
const viewerStatus = document.querySelector("#viewer-status");
const viewTabs = [...document.querySelectorAll(".view-tab")];
const resultsBody = document.querySelector("#results-body");
const totalCells = document.querySelector("#total-cells");
const analyzeButton = document.querySelector("#analyze-button");
const downloadImageButton = document.querySelector("#download-image");
const downloadCsvButton = document.querySelector("#download-csv");

let selectedFile = null;
let localPreviewUrl = null;
let results = null;
let currentView = "original";

function setFeedback(message, kind = "") {
  feedback.textContent = message;
  feedback.className = `form-feedback ${kind}`;
}

function resetResults() {
  results = null;
  totalCells.innerHTML = "— <small>núcleos</small>";
  resultsBody.replaceChildren();
  const row = document.createElement("tr");
  const cell = document.createElement("td");
  cell.colSpan = 4;
  cell.className = "empty-table";
  cell.textContent = "Los núcleos identificados aparecerán aquí después del análisis.";
  row.append(cell);
  resultsBody.append(row);
  downloadCsvButton.disabled = true;
  downloadImageButton.disabled = true;
  viewTabs.forEach(tab => { tab.disabled = tab.dataset.view !== "original"; });
  document.querySelector(".viewer-status").classList.remove("ready");
  viewerStatus.textContent = "Sin procesar";
  setView("original");
}

function setView(view) {
  if (view !== "original" && !results) return;
  currentView = view;
  viewTabs.forEach(tab => {
    const active = tab.dataset.view === view;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-pressed", String(active));
  });
  const src = results ? results.images[view] : localPreviewUrl;
  viewer.hidden = !src;
  placeholder.hidden = Boolean(src);
  if (src) viewer.src = src;
  if (results) {
    const names = { original: "imagen_original.png", mask: "mascara_binaria.png", result: "nucleos_detectados.png" };
    downloadImageButton.dataset.filename = names[view];
    downloadImageButton.disabled = false;
  }
}

function setFile(file) {
  if (localPreviewUrl) URL.revokeObjectURL(localPreviewUrl);
  selectedFile = file;
  localPreviewUrl = file ? URL.createObjectURL(file) : null;
  fileName.textContent = file ? file.name : "Sin archivo seleccionado";
  fileName.classList.toggle("loaded", Boolean(file));
  dimensions.textContent = "Sin imagen";
  resetResults();
  if (file) {
    setFeedback("Imagen cargada. Configurá los parámetros y ejecutá el análisis.");
    const probe = new Image();
    probe.onload = () => { if (file === selectedFile) dimensions.textContent = `${probe.naturalWidth} × ${probe.naturalHeight} px`; };
    probe.src = localPreviewUrl;
  } else setFeedback("Seleccioná una imagen para comenzar.");
}

fileInput.addEventListener("change", () => setFile(fileInput.files[0] || null));
for (const eventType of ["dragenter", "dragover"]) {
  uploadZone.addEventListener(eventType, event => { event.preventDefault(); uploadZone.classList.add("drag-over"); });
}
for (const eventType of ["dragleave", "drop"]) {
  uploadZone.addEventListener(eventType, event => { event.preventDefault(); uploadZone.classList.remove("drag-over"); });
}
uploadZone.addEventListener("drop", event => {
  const file = event.dataTransfer?.files?.[0];
  if (file) {
    fileInput.files = event.dataTransfer.files;
    setFile(file);
  }
});

function updateSegmentOptions() {
  const dbz = segmentationMode.value === "dbz";
  thresholdBlock.hidden = !dbz;
  thresholdRange.disabled = !dbz;
}
segmentationMode.addEventListener("change", updateSegmentOptions);
thresholdRange.addEventListener("input", () => { document.querySelector("#threshold-value").textContent = `${thresholdRange.value} dBZ`; });



viewTabs.forEach(tab => tab.addEventListener("click", () => setView(tab.dataset.view)));

function appendCell(row, value) {
  const cell = document.createElement("td");
  cell.textContent = String(value);
  row.append(cell);
}

function displayResults(data) {
  results = data;
  totalCells.innerHTML = `${data.total_cells} <small>núcleos</small>`;
  dimensions.textContent = `${data.width} × ${data.height} px`;
  resultsBody.replaceChildren();
  if (!data.cells.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 4;
    cell.className = "empty-table";
    cell.textContent = "El análisis terminó, pero ningún núcleo superó los criterios seleccionados.";
    row.append(cell);
    resultsBody.append(row);
  }
  for (const item of data.cells) {
    const row = document.createElement("tr");
    appendCell(row, `#${item.id}`);
    appendCell(row, item.area.toLocaleString("es-AR"));
    appendCell(row, `(${item.centroid.x}, ${item.centroid.y})`);
    const b = item.bounding_box;
    appendCell(row, `(${b.x}, ${b.y}, ${b.width}, ${b.height})`);
    resultsBody.append(row);
  }
  viewTabs.forEach(tab => { tab.disabled = false; });
  downloadCsvButton.disabled = false;
  document.querySelector(".viewer-status").classList.add("ready");
  viewerStatus.textContent = "Análisis completo";
  setView("result");
  setFeedback(`Análisis completado: ${data.total_cells} núcleo(s) detectado(s).`, "success");
}

form.addEventListener("submit", async event => {
  event.preventDefault();
  if (!selectedFile) { setFeedback("Primero seleccioná una imagen de radar.", "error"); return; }
  if (!(["image/png", "image/jpeg"].includes(selectedFile.type))) { setFeedback("El archivo debe ser PNG o JPG.", "error"); return; }
  if (selectedFile.size > 10 * 1024 * 1024) { setFeedback("El archivo supera los 10 MB permitidos.", "error"); return; }
  const data = new FormData(form);
  data.set("image", selectedFile);

  analyzeButton.disabled = true;
  analyzeButton.textContent = "Procesando imagen…";
  setFeedback("Procesando imagen en el backend…");
  try {
    const response = await fetch("/api/analyze", { method: "POST", body: data });
    const body = await response.json();
    if (!response.ok) {
      let message = body.detail || `Error HTTP ${response.status}`;
      if (Array.isArray(message)) message = message.map(item => item.msg).join("; ");
      throw new Error(message);
    }
    displayResults(body);
  } catch (error) {
    resetResults();
    setFeedback(`No se pudo completar el análisis: ${error.message}`, "error");
  } finally {
    analyzeButton.disabled = false;
    analyzeButton.innerHTML = '<span aria-hidden="true">⌁</span> Ejecutar análisis <span class="button-arrow" aria-hidden="true">→</span>';
  }
});

function saveUrl(url, filename) {
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
}
downloadImageButton.addEventListener("click", () => {
  if (!results) return;
  saveUrl(results.images[currentView], downloadImageButton.dataset.filename || "radar.png");
});
downloadCsvButton.addEventListener("click", () => {
  if (!results) return;
  const rows = [["id", "area_px", "centroid_x", "centroid_y", "bbox_x", "bbox_y", "bbox_width", "bbox_height"]];
  for (const cell of results.cells) {
    const b = cell.bounding_box;
    rows.push([cell.id, cell.area, cell.centroid.x, cell.centroid.y, b.x, b.y, b.width, b.height]);
  }
  // UTF-8 con BOM: Excel en Windows puede reconocer los acentos.
  const content = "\ufeff" + rows.map(row => row.join(",")).join("\r\n") + "\r\n";
  const url = URL.createObjectURL(new Blob([content], { type: "text/csv;charset=utf-8" }));
  saveUrl(url, "nucleos.csv");
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

updateSegmentOptions();
