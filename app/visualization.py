
import cv2
import numpy as np


def draw_storm_cells(
    image: np.ndarray,
    mask: np.ndarray,
    cells: list[dict]
) -> np.ndarray:
    """
    Dibuja los nucleos detectados sobre la imagen original.
    El programa erecibe la imagen original, la mascara binaria y la lista de nucleos,
    con OPENCV se dibuja los contornos, los bounding boxes (ubicacion y dimension), centroindes,
    etiquetas de identificacion.

    cv2.connectedComponentsWithStats	Identifica las regiones de la máscara
    cv2.findContours	Obtiene los bordes de una región
    cv2.drawContours	Dibuja los contornos sobre la imagen
    cv2.rectangle	Dibuja el bounding box
    cv2.circle	Marca el centroide
    cv2.putText	Escribe el número del núcleo

    Args:
        image: Imagen original en formato BGR.
        mask: Mascara binaria de segmentacion.
        cells: Lista devuelta por detect_storm_cells().

    Returns:
        Imagen con contornos, rectangulos, centroides
        e identificadores de cada nucleo.
    """

    # Validar imagen original
    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 3
        or image.shape[2] != 3
        or image.dtype != np.uint8
        or image.size == 0
    ):
        raise ValueError("La imagen debe ser BGR de 8 bits.")

    # Validar mascara
    if (
        not isinstance(mask, np.ndarray)
        or mask.ndim != 2
        or mask.shape != image.shape[:2]
        or mask.dtype != np.uint8
        or not np.all((mask == 0) | (mask == 255))
    ):
        raise ValueError("Mascara binaria no valida.")

    # Copiar imagen para no modificar la original
    result = image.copy()

    # Obtener las etiquetas de las regiones conectadas
    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
    )

    for cell in cells:
        box = cell["bounding_box"]

        x = box["x"]
        y = box["y"]
        w = box["width"]
        h = box["height"]

        cx = round(cell["centroid"]["x"])
        cy = round(cell["centroid"]["y"])

        # Buscar la region correspondiente al nucleo
        matching_label = None

        for label in range(1, num_labels):
            if (
                stats[label, cv2.CC_STAT_LEFT] == x
                and stats[label, cv2.CC_STAT_TOP] == y
                and stats[label, cv2.CC_STAT_WIDTH] == w
                and stats[label, cv2.CC_STAT_HEIGHT] == h
                and stats[label, cv2.CC_STAT_AREA] == cell["area"]
            ):
                matching_label = label
                break

        if matching_label is None:
            raise ValueError(
                f"No se encontro la region del nucleo {cell['id']}"
            )

        # Crear mascara individual del nucleo
        component_mask = np.zeros_like(mask)
        component_mask[labels == matching_label] = 255

        # Obtener los contornos de esa region
        contours, _ = cv2.findContours(
            component_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Dibujar rectangulo verde alrededor de la region
        cv2.rectangle(
            result,
            (x - 3, y - 3),
            (x + w + 2, y + h + 2),
            (0, 255, 0),
            1
        )

        # Dibujar contorno amarillo
        cv2.drawContours(
            result,
            contours,
            -1,
            (0, 255, 255),
            2
        )

        # Marcar centroide con un punto rojo
        cv2.circle(
            result,
            (cx, cy),
            4,
            (0, 0, 255),
            -1
        )

        # Dibujar numero identificador
        cv2.putText(
            result,
            f"Nucleo {cell['id']}",
            (x, max(y - 9, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return result
