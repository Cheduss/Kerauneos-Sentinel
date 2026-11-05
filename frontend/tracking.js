"use strict";

// Navegación entre Detección, Seguimiento y Glosario.
const detectionModule = document.querySelector("#module-detection");
const trackingModule = document.querySelector("#module-tracking");
const glossaryModule = document.querySelector("#module-glossary");
const detectionWorkspace = document.querySelector("#detection-workspace");
const trackingWorkspace = document.querySelector("#tracking-workspace");
const glossaryWorkspace = document.querySelector("#glossary-workspace");
const pageHeading = document.querySelector(".page-heading h1");
const pageDescription = document.querySelector(".page-heading p");
const pagePill = document.querySelector(".heading-pill");

function activateModule(module) {
  const modes = [
    { name: "detection", button: detectionModule, panel: detectionWorkspace },
    { name: "tracking", button: trackingModule, panel: trackingWorkspace },
    { name: "glossary", button: glossaryModule, panel: glossaryWorkspace }
  ];
  for (const item of modes) {
    const active = item.name === module;
    item.panel.hidden = !active;
    item.button.classList.toggle("active", active);
    item.button.setAttribute("aria-pressed", String(active));
  }

  const labels = {
    detection: {
      title: "Explorador de núcleos <span>convectivos</span>",
      description: "Segmentá ecos, identificá regiones y explorá sus características en una sola interfaz.",
      pill: " DETECCIÓN DE NÚCLEOS"
    },
    tracking: {
      title: "Seguimiento de ecos <span>meteorológicos</span>",
      description: "Compará dos imágenes con Lucas–Kanade y visualizá el desplazamiento aparente de ecos de precipitación.",
      pill: " SEGUIMIENTO TEMPORAL"
    },
    glossary: {
      title: "Glosario de <span>técnicas</span>",
      description: "Conocé los métodos de procesamiento utilizados para detectar y seguir núcleos de precipitación.",
      pill: " GUÍA DE CONCEPTOS"
    }
  };
  pageHeading.innerHTML = labels[module].title;
  pageDescription.textContent = labels[module].description;
  pagePill.lastChild.textContent = labels[module].pill;
}

detectionModule.addEventListener("click", () => activateModule("detection"));
trackingModule.addEventListener("click", () => activateModule("tracking"));
glossaryModule.addEventListener("click", () => activateModule("glossary"));

const trackForm = document.querySelector("#tracking-form");
const trackImage1 = document.querySelector("#track-image1");
const trackImage2 = document.querySelector("#track-image2");
const trackButton = document.querySelector("#track-button");
const trackFeedback = document.querySelector("#track-feedback");
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
const trackVectorsBody = document.querySelector("#track-vectors-body");
const trackVectorCount = document.querySelector("#track-vector-count");

// El resumen muestra magnitud y dirección del desplazamiento mediano,
// no la velocidad de una celda individual.
trackDx.parentElement.querySelector(".metric-label").textContent = "Magnitud del vector mediano";
trackDx.parentElement.querySelector("small").textContent = "píxeles entre T1 y T2";
trackDy.parentElement.querySelector(".metric-label").textContent = "Dirección del vector mediano";
trackDy.parentElement.querySelector("small").textContent = "0° N · 90° E · 180° S";

const trackDownloadImage = document.querySelector("#track-download-image");
const trackDownloadJson = document.querySelector("#track-download-json");

let trackResults = null;
let trackCurrentView = "first";
let trackPreviewUrls = { first: null, second: null };
let trackRequestVersion = 0;

// Ejes de imagen: X hacia la derecha, Y hacia abajo.
// Con norte arriba, atan2(dx, -dy) da el azimut horario desde el norte.
function trackVectorStatistics(vector) {
  const dx = Number(vector.x2) - Number(vector.x1);
  const dy = Number(vector.y2) - Number(vector.y1);
  const distance = Math.hypot(dx, dy);
  const direction = distance < 1e-9
    ? null
    : (Math.atan2(dx, -dy) * 180 / Math.PI + 360) % 360;
  return { dx, dy, distance, direction };
}

function trackRenderVectorTable(data) {
  trackVectorsBody.replaceChildren();
  const maxArrows = Math.max(1, Number(document.querySelector("#track-max-vectors").value) || 60);
  // Coincidir con el muestreo de draw_motion_vectors() en Python.
  const step = Math.max(1, Math.ceil(data.vectors.length / maxArrows));
  const selected = data.vectors
    .filter((_, i) => i % step === 0)
    .slice(0, maxArrows)
    .map(trackVectorStatistics)
    .filter(v => v.distance >= 1);  // min_displacement=1 en el backend

  if (!selected.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.className = "empty-table";
    cell.colSpan = 5;
    cell.textContent = "No hay flechas visibles para mostrar.";
    row.append(cell);
    trackVectorsBody.append(row);
  } else {
    for (const [i, v] of selected.entries()) {
      const row = document.createElement("tr");
      const cells = [
        String(i + 1),
        v.dx.toFixed(2),
        v.dy.toFixed(2),
        v.distance.toFixed(2),
        v.direction === null ? "—" : `${v.direction.toFixed(1)}°`
      ];
      for (const value of cells) {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      }
      trackVectorsBody.append(row);
    }
  }
  trackVectorCount.textContent = `${selected.length} flechas mostradas · ${data.tracked_points} puntos seguidos`;
}

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
  trackVectorCount.textContent = "Sin análisis";
  trackVectorsBody.innerHTML = '<tr><td class="empty-table" colspan="5">Ejecutá el seguimiento para ver el detalle de los vectores.</td></tr>';
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
trackViews.forEach(tab => tab.addEventListener("click", () => trackSetView(tab.dataset.trackView)));
trackImage.addEventListener("load", () => {
  if (!trackImage.hidden) trackDimensions.textContent = `${trackImage.naturalWidth} × ${trackImage.naturalHeight} px`;
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

function trackDisplayResult(data) {
  trackResults = data;
  const medianVector = trackVectorStatistics({
    x1: 0, y1: 0, x2: data.dx_pixels, y2: data.dy_pixels
  });
  trackDx.textContent = data.tracked_points ? `${medianVector.distance.toFixed(2)} px` : "—";
  trackDy.textContent = data.tracked_points && medianVector.direction !== null
    ? `${medianVector.direction.toFixed(1)}°` : "—";
  trackRenderVectorTable(data);
  trackPoints.textContent = String(data.tracked_points);
  trackCandidates.textContent = data.candidate_points === null
    ? "Sin dato de candidatos"
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
  // Recorte automatico gestionado exclusivamente por FastAPI.

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
  // El JSON conserva todos los puntos, con magnitud y dirección por vector.
  numericResults.vectors = trackResults.vectors.map(vector => {
    const { dx, dy, distance, direction } = trackVectorStatistics(vector);
    return {
      ...vector,
      dx_pixels: Number(dx.toFixed(2)),
      dy_pixels: Number(dy.toFixed(2)),
      displacement_pixels: Number(distance.toFixed(2)),
      direction_degrees: direction === null ? null : Number(direction.toFixed(1))
    };
  });
  const url = URL.createObjectURL(new Blob([JSON.stringify(numericResults, null, 2)], { type: "application/json" }));
  trackSaveUrl(url, "seguimiento_radar.json");
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
