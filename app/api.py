
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

from app.pipeline import analyze_radar


"""
GET /	Comprueba que el servidor funciona
POST /api/analyze	Recibe una imagen y ejecuta mi app

"""


app = FastAPI(
    title="Kerauneos-Sentinel API",
    description="Analisis de imagenes de radar meteorologico",
    version="0.1.0"
)

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
    roi: str | None = Form(None)
):
    """
    Estima el movimiento aparente entre dos imagenes
    mediante flujo optico Lucas-Kanade.
    """

    try:
        # 1. Cargar ambas imagenes
        first = await read_uploaded_radar(image1)
        second = await read_uploaded_radar(image2)

        # 2. Validar dimensiones
        if first.shape != second.shape:
            raise HTTPException(
                status_code=400,
                detail="Las imagenes deben tener las mismas dimensiones."
            )

        # 3. Recortar la misma region en ambas imagenes
        coordinates = parse_roi(roi)

        if coordinates is not None:
            try:
                first = crop_radar(first, *coordinates)
                second = crop_radar(second, *coordinates)
            except ValueError as error:
                raise HTTPException(
                    status_code=400,
                    detail=str(error)
                ) from error

        # 4. Calcular flujo optico
        try:
            motion = estimate_motion(
                first,
                second,
                max_corners=max_corners
            )

            # 5. Dibujar vectores sobre segunda imagen
            result = draw_motion_vectors(
                second,
                motion,
                max_vectors=max_vectors
            )

        except (ValueError, cv2.error) as error:
            raise HTTPException(
                status_code=400,
                detail=str(error)
            ) from error

        # 6. Devolver resultados al frontend
        return {
            "tracked_points": motion["tracked_points"],
            "dx_pixels": motion["dx_pixels"],
            "dy_pixels": motion["dy_pixels"],
            "vectors": motion["vectors"],
            "images": {
                "first": image_to_base64(first),
                "second": image_to_base64(second),
                "motion": image_to_base64(result)
            }
        }

    finally:
        await image1.close()
        await image2.close()
