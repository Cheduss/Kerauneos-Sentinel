
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
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.pipeline import analyze_radar, EZEIZA_AUTO_ROI
from app.radar_profiles import validate_ezeiza_radar
from app.storm_tracking import estimate_storm_motion


"""
GET /	Comprueba que el servidor funciona
POST /api/analyze	Recibe una imagen y ejecuta mi app

"""


app = FastAPI(
    title="Kerauneos-Sentinel API",
    description="Analisis de imagenes de radar meteorologico",
    version="0.1.0"
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
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


@app.get("/")
def home():
    return {
        "project": "Kerauneos-Sentinel",
        "status": "API funcionando"
    }


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

        coordinates = parse_roi(roi)

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

    buffer = np.frombuffer(content, dtype=np.uint8)

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

    Para dos capturas completas reconocidas como Ezeiza-SINARAME,
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

        # Solo aplicar el recorte automatico si AMBOS radares
        # tienen el mismo perfil reconocido. No basta con el tamano.
        if coordinates is None:
            try:
                validate_ezeiza_radar(full_first)
                validate_ezeiza_radar(full_second)
            except ValueError:
                # Radar desconocido: procesar la captura completa.
                pass
            else:
                coordinates = EZEIZA_AUTO_ROI
                automatic_roi = True

        # Trabajar con las mismas coordenadas en T1 y T2.
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

            # Dibujar sobre la ROI, de forma que el recuadro informativo
            # NO tape el encabezado con fecha/hora de la captura completa.
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

            # Volver a insertar la region anotada en el radar completo.
            full_result = full_second.copy()
            full_result[y1:y2, x1:x2] = work_result

            # Los vectores del JSON deben referirse a la imagen COMPLETA.
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

        # Desplazamientos dx/dy no cambian al sumar el origen de la ROI.
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
