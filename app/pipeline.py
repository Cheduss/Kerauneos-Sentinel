
from pathlib import Path

from app.load_an_image import load_image
from app.crop_region import crop_radar
from app.preprocessing import apply_filter
from app.segmentation import segment_by_color
from app.morphology import apply_morphology
from app.detection import detect_storm_cells
from app.visualization import draw_storm_cells
from app.reflectivity import segment_by_dbz
from app.radar_profiles import validate_ezeiza_radar


#se mantienen ambas funciones de segmentacion

def analyze_radar(
    image_path: str | Path,
    colors: tuple[str, ...] = ("red", "magenta"),
    filter_type: str = "none",
    morphology_operation: str = "opening",
    min_area: int = 100,
    roi: tuple[int, int, int, int] | None = None,
    segmentation_mode: str = "hsv",
    threshold_dbz: float = 40,
) -> dict:

    # 1. Cargar imagen completa
    full_image = load_image(image_path)
    image = full_image.copy()

    # 2. Recortar region principal
    if roi is not None:
        if len(roi) != 4:
            raise ValueError(
                "ROI debe tener cuatro coordenadas."
            )

        image = crop_radar(image, *roi)

    # 3. Preprocesamiento
    filtered = apply_filter(
        image,
        filter_type=filter_type,
        kernel_size=3
    )

    # 4. Seleccionar metodo de segmentacion
    if segmentation_mode == "hsv":

        mask = segment_by_color(
            filtered,
            colors=colors
        )


    elif segmentation_mode == "dbz":

        # Verificar que la imagen tenga una escala compatible
        validate_ezeiza_radar(full_image)

        # Segmentar segun reflectividad estimada
        mask = segment_by_dbz(
            filtered,
            full_image,
            threshold_dbz=threshold_dbz
        )

    else:
        raise ValueError(
            f"Modo de segmentacion desconocido: {segmentation_mode}"
        )

    # 5. Operaciones morfologicas
    clean_mask = apply_morphology(
        mask,
        operation=morphology_operation,
        kernel_size=3
    )

    # 6. Detectar nucleos
    cells = detect_storm_cells(
        clean_mask,
        min_area=min_area
    )

    # 7. Dibujar resultados
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

