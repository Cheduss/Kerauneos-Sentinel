
from pathlib import Path

from app.load_an_image import load_image
from app.crop_region import crop_radar
from app.preprocessing import apply_filter
from app.segmentation import segment_by_color
from app.morphology import apply_morphology
from app.detection import detect_storm_cells
from app.visualization import draw_storm_cells


def analyze_radar(
    image_path: str | Path,
    colors: tuple[str, ...] = ("red", "magenta"),
    filter_type: str = "none",
    morphology_operation: str = "opening",
    min_area: int = 100,
    roi: tuple[int, int, int, int] | None = None,
) -> dict:
    """
    Procesa una imagen de radar y detecta regiones.

    roi: coordenadas (x1, y1, x2, y2)
         para recortar la imagen.
    """

    # 1. Cargar imagen
    image = load_image(image_path)

    # 2. Recortar region de interes si se solicita
    if roi is not None:
        if len(roi) != 4:
            raise ValueError(
                "ROI debe tener cuatro coordenadas."
            )

        image = crop_radar(image, *roi)

    # 3. Aplicar filtro espacial
    filtered = apply_filter(
        image,
        filter_type=filter_type,
        kernel_size=3
    )

    # 4. Segmentar por colores
    mask = segment_by_color(
        filtered,
        colors=colors
    )

    # 5. Aplicar operaciones morfologicas
    clean_mask = apply_morphology(
        mask,
        operation=morphology_operation,
        kernel_size=3
    )

    # 6. Detectar regiones
    cells = detect_storm_cells(
        clean_mask,
        min_area=min_area
    )

    # 7. Dibujar regiones detectadas
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
