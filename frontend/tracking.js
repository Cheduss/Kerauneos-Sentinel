"use strict";

// Navegación entre los dos módulos; app.js conserva el análisis de núcleos.
const detectionModule = document.querySelector("#module-detection");
const trackingModule = document.querySelector("#module-tracking");
const detectionWorkspace = document.querySelector("#detection-workspace");
const trackingWorkspace = document.querySelector("#tracking-workspace");
const pageHeading = document.querySelector(".page-heading h1");
const pageDescription = document.querySelector(".page-heading p");
const pagePill = document.querySelector(".heading-pill");

function activateModule(module) {
  const isTracking = module === "tracking";
  detectionWorkspace.hidden = isTracking;
  trackingWorkspace.hidden = !isTracking;
  detectionModule.classList.toggle("active", !isTracking);
  trackingModule.classList.toggle("active", isTracking);
  detectionModule.setAttribute("aria-pressed", String(!isTracking));
  trackingModule.setAttribute("aria-pressed", String(isTracking));
  pageHeading.innerHTML = isTracking
    ? "Seguimiento de ecos <span>meteorológicos</span>"
    : "Explorador de núcleos <span>convectivos</span>";
  pageDescription.textContent = isTracking
    ? "Compará dos imágenes consecutivas con Lucas–Kanade y visualizá su desplazamiento aparente."
    : "Segmentá ecos, identificá regiones y explorá sus características en una sola interfaz.";
  pagePill.lastChild.textContent = isTracking ? " SEGUIMIENTO TEMPORAL" : " DETECCIÓN DE NÚCLEOS";
}

detectionModule.addEventListener("click", () => activateModule("detection"));
trackingModule.addEventListener("click", () => activateModule("tracking"));

const trackForm = document.querySelector("#tracking-form");
const trackImage1 = document.querySelector("#track-image1");
const trackImage2 = document.querySelector("#track-image2");
const trackButton = document.querySelector("#track-button");
const trackFeedback = document.querySelector("#track-feedback");
const trackEnableRoi = document.querySelector("#track-enable-roi");
const trackMode = document.querySelector("#tracking-mode");
const trackViews = [...document.querySelectorAll(".track-view-tab")];
const trackImage = document.querySelector("#tracking-image");
const trackPlaceholder = document.querySelector("#tracking-placeholder");
const trackDimensions = document.querySelector("#track-image-dimensions");
const trackStatus = document.querySelector("#track-status");
const trackStatusText = document.querySelector("#track-status-text");
const trackDx = document.querySelector("#track-dx");
const trackDy = document.querySelector("#track-dy");
const trackPoints = document.querySelector("#track-points");
const trackCandidates = document.querySelector("#track-candidates");
const trackDownloadImage = document.querySelector("#track-download-image");
const trackDownloadJson = document.querySelector("#track-download-json");

let trackResults = null;
let trackCurrentView = "first";
let trackPreviewUrls = { first: null, second: null };
let trackRequestVersion = 0;

function trackMessage(message, kind = "") {
  trackFeedback.className = `form-feedback ${kind}`;
  trackFeedback.textContent = message;
}

function trackSelectedFile(which) {
  return (which === "first" ? trackImage1 : trackImage2).files[0] || null;
}

function trackSetView(view) {
  if (view === "motion" && !trackResults) return;
  trackCurrentView = view;
  trackViews.forEach(tab => {
    const active = tab.dataset.trackView === view;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-pressed", String(active));
  });

  const src = trackResults ? trackResults.images[view] : trackPreviewUrls[view];
  trackImage.hidden = !src;
  trackPlaceholder.hidden = Boolean(src);
  if (src) trackImage.src = src;
  else trackImage.removeAttribute("src");
  trackDownloadImage.disabled = !trackResults;
  const filenames = {
    first: "radar_t1.png",
    second: "radar_t2.png",
    motion: "movimiento_radar.png"
  };
  trackDownloadImage.dataset.filename = filenames[view];
}

function trackReset() {
  trackRequestVersion += 1;
  trackResults = null;
  trackDx.textContent = "—";
  trackDy.textContent = "—";
  trackPoints.textContent = "—";
  trackCandidates.textContent = "Sin análisis";
  trackDimensions.textContent = "Sin procesar";
  trackStatus.classList.remove("ready");
  trackStatusText.textContent = "Sin procesar";
  trackDownloadJson.disabled = true;
  trackViews.find(tab => tab.dataset.trackView === "motion").disabled = true;
  trackSetView("first");
}

function trackRefreshPreview(which) {
  if (trackPreviewUrls[which]) URL.revokeObjectURL(trackPreviewUrls[which]);
  const file = trackSelectedFile(which);
  trackPreviewUrls[which] = file ? URL.createObjectURL(file) : null;
  trackReset();
  if (file) trackSetView(which);
  trackMessage(trackSelectedFile("first") && trackSelectedFile("second")
    ? "Dos imágenes listas. Seleccioná el método y ejecutá el seguimiento."
    : "Cargá ambos instantes para comenzar.");
}

trackImage1.addEventListener("change", () => trackRefreshPreview("first"));
trackImage2.addEventListener("change", () => trackRefreshPreview("second"));
trackMode.addEventListener("change", () => { trackReset(); trackMessage("Método actualizado. Ejecutá nuevamente el seguimiento."); });
trackViews.forEach(tab => tab.addEventListener("click", () => trackSetView(tab.dataset.trackView)));
trackImage.addEventListener("load", () => {
  if (!trackImage.hidden) trackDimensions.textContent = `${trackImage.naturalWidth} × ${trackImage.naturalHeight} px`;
});

trackEnableRoi.addEventListener("change", () => {
  for (const id of ["track-roi-x1", "track-roi-y1", "track-roi-x2", "track-roi-y2"]) {
    document.getElementById(id).disabled = !trackEnableRoi.checked;
  }
  trackReset();
});

function trackValidateFile(file) {
  if (!file) return "Seleccioná dos imágenes de radar.";
  const isImage = ["image/png", "image/jpeg"].includes(file.type)
    || (!file.type && /\.(png|jpe?g)$/i.test(file.name));
  if (!isImage) return `Formato no permitido: ${file.name}. Usá PNG o JPG.`;
  if (file.size > 10 * 1024 * 1024) return `El archivo ${file.name} supera los 10 MB.`;
  if (!file.size) return `El archivo ${file.name} está vacío.`;
  return null;
}

function trackGetRoi() {
  if (!trackEnableRoi.checked) return null;
  const values = ["track-roi-x1", "track-roi-y1", "track-roi-x2", "track-roi-y2"]
    .map(id => Number(document.getElementById(id).value));
  if (!values.every(Number.isInteger) || values[0] < 0 || values[1] < 0 || values[2] <= values[0] || values[3] <= values[1]) {
    throw new Error("El ROI debe cumplir 0 ≤ X1 < X2 y 0 ≤ Y1 < Y2.");
  }
  return values.join(",");
}

function trackDisplayResult(data) {
  trackResults = data;
  trackDx.textContent = `${data.dx_pixels >= 0 ? "+" : ""}${data.dx_pixels.toFixed(2)} px`;
  trackDy.textContent = `${data.dy_pixels >= 0 ? "+" : ""}${data.dy_pixels.toFixed(2)} px`;
  trackPoints.textContent = String(data.tracked_points);
  trackCandidates.textContent = data.candidate_points === null
    ? "Modo general"
    : `${data.candidate_points} candidatos iniciales`;
  trackViews.find(tab => tab.dataset.trackView === "motion").disabled = false;
  trackDownloadJson.disabled = false;
  trackStatus.classList.add("ready");
  trackStatusText.textContent = "Seguimiento completado";
  trackSetView("motion");
  trackMessage(data.tracked_points > 0
    ? `Seguimiento completo: ${data.tracked_points} puntos seguidos.`
    : "El análisis terminó, pero no encontró puntos válidos para seguir.", "success");
}

trackForm.addEventListener("submit", async event => {
  event.preventDefault();
  const first = trackSelectedFile("first");
  const second = trackSelectedFile("second");
  for (const file of [first, second]) {
    const message = trackValidateFile(file);
    if (message) { trackMessage(message, "error"); return; }
  }

  const formData = new FormData(trackForm);
  let roi;
  try { roi = trackGetRoi(); }
  catch (error) { trackMessage(error.message, "error"); return; }
  if (roi) formData.set("roi", roi);

  trackButton.disabled = true;
  trackButton.textContent = "Estimando movimiento…";
  trackMessage("Analizando dos imágenes de radar en el backend…");
  const requestVersion = ++trackRequestVersion;
  try {
    const response = await fetch("/api/track", { method: "POST", body: formData });
    const body = await response.json();
    if (requestVersion !== trackRequestVersion) return; // Un archivo cambió mientras se procesaba.
    if (!response.ok) {
      const detail = body.detail;
      const message = Array.isArray(detail)
        ? detail.map(item => item.msg).join("; ")
        : (detail || `Error HTTP ${response.status}`);
      throw new Error(message);
    }
    trackDisplayResult(body);
  } catch (error) {
    if (requestVersion !== trackRequestVersion) return;
    trackReset();
    trackMessage(`No se pudo completar el seguimiento: ${error.message}`, "error");
  } finally {
    trackButton.disabled = false;
    trackButton.innerHTML = '<span aria-hidden="true">↗</span> Estimar desplazamiento <span class="button-arrow" aria-hidden="true">→</span>';
  }
});

function trackSaveUrl(url, filename) {
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
}

trackDownloadImage.addEventListener("click", () => {
  if (trackResults) trackSaveUrl(trackResults.images[trackCurrentView], trackDownloadImage.dataset.filename);
});

trackDownloadJson.addEventListener("click", () => {
  if (!trackResults) return;
  // Se exportan datos numéricos y vectores; no se incluyen las imágenes en Base64.
  const { images, ...numericResults } = trackResults;
  const url = URL.createObjectURL(new Blob([JSON.stringify(numericResults, null, 2)], { type: "application/json" }));
  trackSaveUrl(url, "seguimiento_radar.json");
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
