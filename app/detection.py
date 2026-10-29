
import cv2
import numpy as np


def detect_storm_cells(
    mask: np.ndarray,
    min_area: int = 100
) -> list[dict]:
    """
    Detecta regiones conectadas en una mascara binaria mediante componentes conectados.

    La cv2.connectedComponentsWithStats(mask, connectivity=8) identifica grupos de pixeles blancos conectados
    y devuelve sus propiedades. (ID, area, centroide, rectangulo que contiene la region)

    Args:
        mask: Mascara binaria con valores 0 y 255.
        min_area: Area minima en pixeles para aceptar
                  una region.

    Returns:
        Lista de diccionarios con las caracteristicas
        de cada region detectada.
    """

    # Validar la mascara
    if (
        not isinstance(mask, np.ndarray)
        or mask.ndim != 2
        or mask.size == 0
        or mask.dtype != np.uint8
        or not np.all((mask == 0) | (mask == 255))
    ):
        raise ValueError(
            "La mascara debe ser binaria de tipo uint8."
        )

    if not isinstance(min_area, int) or min_area < 1:
        raise ValueError(
            "El area minima debe ser un entero positivo."
        )

    # Identificar componentes conectados
    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    cells = []

    # La etiqueta 0 corresponde al fondo
    for label in range(1, num_labels):

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])

        # Ignorar regiones demasiado pequenas
        if area < min_area:
            continue

        center_x = float(centroids[label, 0])
        center_y = float(centroids[label, 1])

        cell = {
            "id": len(cells) + 1,
            "area": area,
            "centroid": {
                "x": round(center_x, 2),
                "y": round(center_y, 2)
            },
            "bounding_box": {
                "x": x,
                "y": y,
                "width": width,
                "height": height
            }
        }

        cells.append(cell)

    # Ordenar por superficie, de mayor a menor
    cells.sort(
        key=lambda cell: cell["area"],
        reverse=True
    )

    # Reasignar los identificadores
    for index, cell in enumerate(cells, start=1):
        cell["id"] = index

    return cells
