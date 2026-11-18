
import base64
import tempfile
from pathlib import Path
from typing import Literal



import numpy as np

from app.crop_region import crop_radar
from app.optical_flow import estimate_motion
from app.motion_visualization import draw_motion_vectors

import cv2

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.pipeline import analyze_radar, EZEIZA_AUTO_ROI
from app.radar_profiles import EZEIZA_PROFILE, REFERENCE_COLORS
from app.storm_tracking import estimate_storm_motion


"""
GET /            Redirige a la interfaz principal.
GET /app         Muestra el frontend.
POST /api/analyze  Procesa una imagen de radar.
POST /api/track    Estima el desplazamiento entre dos capturas.
"""


app = FastAPI(
    title="Kerauneos-Sentinel API",
    description="Analisis de imagenes de radar meteorologico",
    version="0.1.0"
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend" # Indicamos dónde están los archivos del frontend para que el FASTAPI pueda mostrarlos 
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/app", include_in_schema=False)
def frontend_page():
    return FileResponse(FRONTEND_DIR / "index.html")


MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def image_to_base64(image):
    """
    Convierte una imagen de OpenCV a PNG codificado
    para enviarlo al frontend.
    """
    success, buffer = cv2.imencode(".png", image)

    if not success:
        raise RuntimeError("No se pudo codificar la imagen.")

    encoded = base64.b64encode(
        buffer.tobytes()
    ).decode("ascii")

    return f"data:image/png;base64,{encoded}"


def parse_roi(roi: str | None):
    """
    Convierte '8,92,659,739' a una tupla de enteros.
    """

    if roi is None or not roi.strip():
        return None

    try:
        coordinates = tuple(
            int(value.strip())
            for value in roi.split(",")
        )
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="ROI debe contener cuatro numeros enteros."
        )

    if len(coordinates) != 4:
        raise HTTPException(
            status_code=400,
            detail="ROI debe tener formato x1,y1,x2,y2."
        )

    return coordinates


def _has_sinarame_panel_layout(image: np.ndarray) -> bool:
    """Reconoce el esquema: corte superior + mapa + cortes laterales.

    No identifica la estacion ni certifica que la reflectividad sea correcta.
    Sirve para que una diferencia en la paleta no active el seguimiento sobre
    las secciones verticales ni sobre el corte horizontal de SINARAME.
    """
    height, width = image.shape[:2]

    # Esta plantilla se ha verificado en capturas SINARAME casi cuadradas.
    if min(height, width) < 400 or abs(width / height - 1.0) > 0.035:
        return False

    # El fondo de la pagina es claro y el sector izquierdo del mapa, oscuro.
    if np.min(image[0, 0]) < 180:
        return False

    left_area = image[
        round(height * 0.20):round(height * 0.85),
        round(width * 0.010):round(width * 0.028),
    ]
    if left_area.size == 0 or np.mean(np.max(left_area, axis=2) < 70) < 0.65:
        return False

    # Las dos secciones ajenas al mapa presentan colores de reflectividad.
    top_section = image[
        round(height * 0.045):round(height * 0.105),
        round(width * 0.10):round(width * 0.75),
    ]
    side_sections = image[
        round(height * 0.16):round(height * 0.85),
        round(width * 0.83):round(width * 0.90),
    ]

    def colored_fraction(region: np.ndarray) -> float:
        if region.size == 0:
            return 0.0
        rgb_range = (
            region.max(axis=2).astype(np.int16)
            - region.min(axis=2).astype(np.int16)
        )
        return float(np.mean(
            (rgb_range > 50) & (region.max(axis=2) > 70)
        ))

    return (
        colored_fraction(top_section) >= 0.15
        and colored_fraction(side_sections) >= 0.15
    )


def detect_sinarame_map_roi(image: np.ndarray) -> tuple[int, int, int, int] | None:
    """Determina la ROI del MAPA, dejando fuera los cortes superior y laterales.

    Usa una plantilla geometrica de SINARAME que funciona tanto para las
    capturas de Ezeiza como para las de Bernardo de Irigoyen aportadas.
    Se acepta si coincide la barra de referencia o si se reconoce la estructura
    visual. Si no se reconoce, no se inventa un recorte.
    """
    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 3
        or image.shape[2] != 3
        or image.dtype != np.uint8
        or image.size == 0
    ):
        return None

    height, width = image.shape[:2]
    scale_x = width / EZEIZA_PROFILE["width"]
    scale_y = height / EZEIZA_PROFILE["height"]

    if (
        min(height, width) < 400
        or not (0.5 <= scale_x <= 4.0)
        or abs(scale_x - scale_y) > 0.02 * max(scale_x, scale_y)
    ):
        return None

    legend_x = round(EZEIZA_PROFILE["legend_x"] * scale_x)
    dbz_max = EZEIZA_PROFILE["dbz_max"]
    dbz_min = EZEIZA_PROFILE["dbz_min"]

    matches_reference_palette = True #Comparamos algunos colores de la leyenda con los valores de referencia a ver si tiene formato conocido
    for dbz, expected_bgr in REFERENCE_COLORS.items():
        fraction = (dbz_max - dbz) / (dbz_max - dbz_min)
        legend_y = (
            EZEIZA_PROFILE["legend_y_top"]
            + fraction * (
                EZEIZA_PROFILE["legend_y_bottom"]
                - EZEIZA_PROFILE["legend_y_top"]
            )
        )
        x = legend_x
        y = round(legend_y * scale_y)

        if not (0 <= x < width and 0 <= y < height):
            matches_reference_palette = False
            break

        observed = image[y, x].astype(np.int16)
        expected = np.asarray(expected_bgr, dtype=np.int16)
        if np.max(np.abs(observed - expected)) > 45:
            matches_reference_palette = False
            break

    # No caer silenciosamente en seguimiento de la imagen completa por
    # pequeñas variaciones en el color de la leyenda o compresión JPG.
    if not matches_reference_palette and not _has_sinarame_panel_layout(image):
        return None

    x1, y1, x2, y2 = EZEIZA_AUTO_ROI
    roi = (
        round(x1 * scale_x),
        round(y1 * scale_y),
        round(x2 * scale_x),
        round(y2 * scale_y),
    )
    left, top, right, bottom = roi
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        return None

    return roi


@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse(url="/app", status_code=302)

@app.post("/api/analyze")
async def analyze_image(
    image: UploadFile = File(...),
    segmentation_mode: Literal["hsv", "dbz"] = Form("hsv"),
    threshold_dbz: float = Form(40, ge=-15, le=75),
    filter_type: Literal[
        "none", "average", "gaussian", "median"
    ] = Form("none"),
    morphology_operation: Literal[
        "none", "opening", "closing",
        "erosion", "dilation"
    ] = Form("opening"),
    min_area: int = Form(100, ge=1),
    roi: str | None = Form(None)
):
    """
    Recibe una imagen y devuelve los resultados
    del procesamiento de radar.
    """

    temporary_path = None

    try:
        filename = image.filename or ""
        extension = Path(filename).suffix.lower()

        # Validar extension
        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail="Solo se permiten PNG, JPG y JPEG."
            )

        # Limitar tamano de archivo
        content = await image.read(MAX_FILE_SIZE + 1)

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="El archivo supera los 10 MB."
            )

        if not content:
            raise HTTPException(
                status_code=400,
                detail="El archivo esta vacio."
            )

        coordinates = parse_roi(roi) #coordenadas del recorte, si fueron indicadas. Sino el pi8peline aplica recorte automartico

        # Guardar temporalmente la imagen recibida
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False
        ) as temporary:
            temporary.write(content)
            temporary_path = Path(temporary.name)

        # Ejecutar nuestros algoritmos
        try:
            analysis = analyze_radar(
                image_path=temporary_path,
                segmentation_mode=segmentation_mode,
                threshold_dbz=threshold_dbz,
                filter_type=filter_type,
                morphology_operation=morphology_operation,
                min_area=min_area,
                roi=coordinates
            )

        except (ValueError, FileNotFoundError) as error:
            raise HTTPException(
                status_code=400,
                detail=str(error)
            ) from error

        # Devolver los resultados al frontend
        return {
            "total_cells": analysis["total_cells"],
            "cells": analysis["cells"],
            "width": analysis["original"].shape[1],
            "height": analysis["original"].shape[0],
            "images": {
                "original": image_to_base64(
                    analysis["original"]
                ),
                "mask": image_to_base64(
                    analysis["mask"]
                ),
                "result": image_to_base64(
                    analysis["result"]
                )
            }
        }

    finally:
        # Eliminar archivos temporales
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

        await image.close()


async def read_uploaded_radar(file: UploadFile):
    """
    Lee y valida una imagen recibida desde el frontend.
    Devuelve la imagen como array BGR de OpenCV.
    """

    filename = file.filename or ""
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Solo se permiten PNG, JPG y JPEG."
        )

    content = await file.read(MAX_FILE_SIZE + 1)

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="La imagen supera el limite de 10 MB."
        )

    if not content:
        raise HTTPException(
            status_code=400,
            detail="La imagen esta vacia."
        )

    buffer = np.frombuffer(content, dtype=np.uint8) #Convertimos los bytes recibidos en un arreglo de NumPy y después decodificamos la imagen para trabajar con OpenCV.

    image = cv2.imdecode(
        buffer,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="No se pudo leer la imagen."
        )

    return image



@app.post("/api/track")
async def track_storms(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    max_corners: int = Form(200, ge=1, le=1000),
    max_vectors: int = Form(60, ge=1, le=200),
    roi: str | None = Form(None),
    tracking_mode: Literal["general", "precipitation"] = Form("general"),
):
    """
    Estima movimiento con Lucas-Kanade.

    Para dos capturas SINARAME con geometria compatible,
    la ROI se aplica automaticamente y SOLO internamente.
    Devuelve las capturas originales y flechas sobre la imagen completa.

    En imagenes no reconocidas no se asume ninguna ROI.
    La ROI explicita sigue funcionando por compatibilidad con la API.
    """
    try:
        full_first = await read_uploaded_radar(image1)
        full_second = await read_uploaded_radar(image2)

        if full_first.shape != full_second.shape:
            raise HTTPException(
                status_code=400,
                detail="Las imagenes deben tener las mismas dimensiones."
            )

        # ROI manual opcional (compatibilidad con versiones anteriores).
        coordinates = parse_roi(roi)
        automatic_roi = False

        #Reconoce la misma plantilla SINARAME incluso si cambio la escala.
        #Solo recortar si las DOS capturas pasan la verificacion de leyenda.
        if coordinates is None:
            roi_first = detect_sinarame_map_roi(full_first)
            roi_second = detect_sinarame_map_roi(full_second)
            if roi_first is not None and roi_first == roi_second:
                coordinates = roi_first
                automatic_roi = True
            elif (roi_first is None) != (roi_second is None):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "No se pudo reconocer la misma region del mapa "
                        "SINARAME en ambas imagenes. "
                        "No se ejecuta el seguimiento sobre los cortes."
                    ),
                )
            elif roi_first is not None and roi_second is not None:
                raise HTTPException(
                    status_code=400,
                    detail="Los mapas SINARAME no tienen una geometria compatible.",
                )

        #Trabaja con las mismas coordenadas en T1 y T2.
        if coordinates is not None:
            try:
                work_first = crop_radar(full_first, *coordinates)
                work_second = crop_radar(full_second, *coordinates)
            except ValueError as error:
                raise HTTPException(
                    status_code=400,
                    detail=str(error)
                ) from error
        else:
            work_first = full_first
            work_second = full_second

        try:
            if tracking_mode == "general":
                motion = estimate_motion(
                    work_first,
                    work_second,
                    max_corners=max_corners,
                )
            else:
                motion = estimate_storm_motion(
                    work_first,
                    work_second,
                    max_corners=max_corners,
                )

            #Dibujar sobre la ROI, de forma que el recuadro informativo
            #NO tape el encabezado con fecha/hora de la captura completa.
            work_result = draw_motion_vectors(
                work_second,
                motion,
                max_vectors=max_vectors,
            )
        except (ValueError, cv2.error) as error:
            raise HTTPException(
                status_code=400,
                detail=str(error),
            ) from error

        if coordinates is not None:
            x1, y1, x2, y2 = coordinates

            #Volvemos a colocar el resultado procesado sobre la captura completa para conservar la leyenda, la fecha y los demas paneles..
            full_result = full_second.copy()
            full_result[y1:y2, x1:x2] = work_result

            #Los vectores del JSON deben referirse a la imagen COMPLETA.
            shifted_vectors = []
            for vector in motion["vectors"]:
                shifted_vectors.append({
                    **vector,
                    "x1": round(float(vector["x1"]) + x1, 2),
                    "y1": round(float(vector["y1"]) + y1, 2),
                    "x2": round(float(vector["x2"]) + x1, 2),
                    "y2": round(float(vector["y2"]) + y1, 2),
                })
        else:
            full_result = work_result
            shifted_vectors = motion["vectors"]

        #Desplazamientos dx/dy no cambian al sumar el origen de la ROI.
        return {
            "tracking_mode": tracking_mode,
            "candidate_points": motion.get("candidate_points"),
            "tracked_points": motion["tracked_points"],
            "dx_pixels": motion["dx_pixels"],
            "dy_pixels": motion["dy_pixels"],
            "vectors": shifted_vectors,
            "roi_used": coordinates,
            "automatic_roi": automatic_roi,
            "images": {
                "first": image_to_base64(full_first),
                "second": image_to_base64(full_second),
                "motion": image_to_base64(full_result),
            },
        }
    finally:
        await image1.close()
        await image2.close()
