
from pathlib import Path

from app.load_an_image import load_image
from app.preprocessing import apply_filter
from app.segmentation import segment_by_color
from app.morphology import apply_morphology
from app.detection import detect_storm_cells
from app.visualization import draw_storm_cells

"""
el pipeline conecta todos los otros modulos

"""


def analyze_radar(
    image_path: str | Path,
    colors: tuple[str, ...] = ("red", "magenta"),
    filter_type: str = "none",
    morphology_operation: str = "opening",
    min_area: int = 100,
) -> dict:
    """
    Ejecuta el procesamiento completo de una imagen de radar.

    Returns:
        Diccionario con imagen original, mascara binaria,
        imagen anotada y nucleos detectados.
    """

    # 1. Cargar imagen
    image = load_image(image_path)

    # 2. Aplicar filtro espacial (opcional)
    filtered = apply_filter(
        image,
        filter_type=filter_type,
        kernel_size=3
    )

    # 3. Segmentar por colores
    mask = segment_by_color(
        filtered,
        colors=colors
    )

    # 4. Limpiar mascara
    clean_mask = apply_morphology(
        mask,
        operation=morphology_operation,
        kernel_size=3
    )

    # 5. Detectar regiones
    cells = detect_storm_cells(
        clean_mask,
        min_area=min_area
    )

    # 6. Dibujar los nucleos detectados
    result = draw_storm_cells(
        image,
        clean_mask,
        cells
    )

    return {
        "original": image,
        "mask": clean_mask,
        "result": result,
        "cells": cells,
        "total_cells": len(cells)
    }
