"use strict";

// Ejemplos incluidos con Kerauneos-Sentinel. Se sirven como archivos estaticos
// desde /static/examples/; no se modifica el backend ni sus algoritmos.
const detectionExampleButton = document.querySelector("#load-detection-example");
const trackingExampleButton = document.querySelector("#load-tracking-examples");

async function fetchExampleImage(filename) {
  const response = await fetch(`/static/examples/${filename}`, { cache: "force-cache" });
  if (!response.ok) throw new Error(`No se pudo cargar ${filename} (HTTP ${response.status}).`);

  const blob = await response.blob();
  if (!blob.size || blob.size > 10 * 1024 * 1024) {
    throw new Error(`La imagen ${filename} está vacía o supera el limite de 10 MB.`);
  }
  return new File([blob], filename, { type: "image/png" });
}

function fillFileInput(input, file) {
  // Una asignacion real a input.files hace funcionar la validacion 'required',
  // FormData y los listeners existentes de app.js / tracking.js.
  const transfer = new DataTransfer();
  transfer.items.add(file);
  input.files = transfer.files;
  input.dispatchEvent(new Event("change", { bubbles: true }));
}

function exampleFeedback(id, message, isError = false) {
  const element = document.querySelector(id);
  element.textContent = message;
  element.className = isError ? "form-feedback error" : "form-feedback";
}

async function withLoading(button, action) {
  const originalLabel = button.textContent;
  button.disabled = true;
  button.textContent = "Cargando ejemplo…";
  try {
    await action();
  } finally {
    button.disabled = false;
    button.textContent = originalLabel;
  }
}

detectionExampleButton.addEventListener("click", async () => {
  try {
    await withLoading(detectionExampleButton, async () => {
      const file = await fetchExampleImage("ezeiza_deteccion.png");
      fillFileInput(document.querySelector("#image-file"), file);
      exampleFeedback("#form-feedback", "Ejemplo de Ezeiza cargado. Elegí los parámetros y ejecutá el análisis.");
    });
  } catch (error) {
    exampleFeedback("#form-feedback", `No se pudo cargar el ejemplo: ${error.message}`, true);
  }
});

trackingExampleButton.addEventListener("click", async () => {
  try {
    await withLoading(trackingExampleButton, async () => {
      // Ambos archivos se obtienen antes de cambiar los inputs: sin carga parcial.
      const [first, second] = await Promise.all([
        fetchExampleImage("ezeiza_t1.png"),
        fetchExampleImage("ezeiza_t2.png")
      ]);
      fillFileInput(document.querySelector("#track-image1"), first);
      fillFileInput(document.querySelector("#track-image2"), second);
      exampleFeedback("#track-feedback", "Ejemplos T1 y T2 de Ezeiza cargados. Presioná Estimar desplazamiento.");
    });
  } catch (error) {
    exampleFeedback("#track-feedback", `No se pudieron cargar los ejemplos: ${error.message}`, true);
  }
});
