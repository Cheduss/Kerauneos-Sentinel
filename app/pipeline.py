from pathlib import Path

import numpy as np

from app.load_an_image import load_image
from app.crop_region import crop_radar
from app.preprocessing import apply_filter
from app.segmentation import segment_by_color
from app.morphology import apply_morphology
from app.detection import detect_storm_cells
from app.visualization import draw_storm_cells
from app.reflectivity import segment_by_dbz
from app.radar_profiles import validate_ezeiza_radar


# Region util dentro de capturas completas del perfil Ezeiza-SINARAME.
# x2 e y2 son limites exclusivos.
EZEIZA_AUTO_ROI = (8, 92, 659, 739)


def _shift_cells(cells: list[dict], dx: int, dy: int) -> list[dict]:
    """Convierte coordenadas de la ROI a coordenadas de la imagen completa."""
    shifted = []
    for cell in cells:
        box = cell["bounding_box"]
        centroid = cell["centroid"]
        shifted.append({
            **cell,
            "centroid": {
                "x": centroid["x"] + dx,
                "y": centroid["y"] + dy,
            },
            "bounding_box": {
                **box,
                "x": box["x"] + dx,
                "y": box["y"] + dy,
            },
        })
    return shifted


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
    """
    Analiza una imagen de radar.

    Si no se indica ROI y se reconoce el perfil completo Ezeiza-SINARAME,
    recorta solo para analizar y devuelve las tres vistas en tamano completo.
    Una ROI explicita conserva el comportamiento anterior (vistas recortadas).
    """

    # 1. Leer la captura completa.
    full_image = load_image(image_path)

    # 2. Seleccionar la region de analisis.
    automatic_roi = False
    selected_roi = roi

    if roi is None:
        try:
            validate_ezeiza_radar(full_image)
        except ValueError:
            # Imagen generica: no asumir un recorte que pueda borrar datos.
            selected_roi = None
        else:
            selected_roi = EZEIZA_AUTO_ROI
            automatic_roi = True

    if selected_roi is None:
        image = full_image.copy()
    else:
        if len(selected_roi) != 4:
            raise ValueError("ROI debe tener cuatro coordenadas.")
        image = crop_radar(full_image, *selected_roi)

    # 3. Filtro espacial.
    filtered = apply_filter(
        image,
        filter_type=filter_type,
        kernel_size=3,
    )

    # 4. Segmentar.
    if segmentation_mode == "hsv":
        mask = segment_by_color(filtered, colors=colors)
    elif segmentation_mode == "dbz":
        # En modo dBZ, validar siempre la barra de colores original.
        validate_ezeiza_radar(full_image)
        mask = segment_by_dbz(
            filtered,
            full_image,
            threshold_dbz=threshold_dbz,
        )
    else:
        raise ValueError(
            f"Modo de segmentacion desconocido: {segmentation_mode}"
        )

    # 5. Limpiar la mascara y detectar nucleos dentro de la ROI.
    clean_mask = apply_morphology(
        mask,
        operation=morphology_operation,
        kernel_size=3,
    )
    cells = detect_storm_cells(clean_mask, min_area=min_area)

    # 6. En el modo automatico, volver a la geometria original.
    if automatic_roi:
        x1, y1, x2, y2 = selected_roi

        full_mask = np.zeros(full_image.shape[:2], dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = clean_mask

        full_cells = _shift_cells(cells, dx=x1, dy=y1)
        full_result = draw_storm_cells(full_image, full_mask, full_cells)

        return {
            "original": full_image,
            "mask": full_mask,
            "result": full_result,
            "cells": full_cells,
            "total_cells": len(full_cells),
            "roi_used": selected_roi,
            "automatic_roi": True,
        }

    # 7. Imagen generica o ROI seleccionada manualmente.
    result = draw_storm_cells(image, clean_mask, cells)
    return {
        "original": image,
        "mask": clean_mask,
        "result": result,
        "cells": cells,
        "total_cells": len(cells),
        "roi_used": selected_roi,
        "automatic_roi": False,
    }
