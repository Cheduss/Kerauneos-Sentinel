
import cv2
import numpy as np


"""

cv2.cvtColor	Convierte las imágenes a escala de grises
cv2.goodFeaturesToTrack	Identifica puntos visualmente destacados
cv2.calcOpticalFlowPyrLK	Calcula el desplazamiento de esos puntos mediante Lucas–Kanade
np.median	Resume los desplazamientos con un valor menos sensible a valores extremos

"""


def estimate_motion(
    image1: np.ndarray,
    image2: np.ndarray,
    max_corners: int = 200
) -> dict:
    """
    Estima desplazamientos entre dos imagenes
    mediante flujo optico de Lucas-Kanade.

    Args:
        image1: Primera imagen BGR.
        image2: Segunda imagen BGR.
        max_corners: Maximo de puntos a seguir.

    Returns:
        Diccionario con desplazamiento mediano
        y vectores de los puntos seguidos.
    """

    # Validar imagenes
    for image in (image1, image2):
        if (
            not isinstance(image, np.ndarray)
            or image.ndim != 3
            or image.shape[2] != 3
            or image.dtype != np.uint8
            or image.size == 0
        ):
            raise ValueError("Imagen BGR no valida.")

    if image1.shape != image2.shape:
        raise ValueError(
            "Las imagenes deben tener iguales dimensiones."
        )

    if not isinstance(max_corners, int) or max_corners < 1:
        raise ValueError("max_corners debe ser positivo.")

    # Convertir a escala de grises
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Identificar puntos con caracteristicas destacadas
    points1 = cv2.goodFeaturesToTrack(
        gray1,
        maxCorners=max_corners,
        qualityLevel=0.01,
        minDistance=7
    )

    # Si no hay puntos, devolver resultado vacio
    if points1 is None:
        return {
            "tracked_points": 0,
            "dx_pixels": 0.0,
            "dy_pixels": 0.0,
            "vectors": []
        }

    # Calcular flujo optico Lucas-Kanade
    points2, status, errors = cv2.calcOpticalFlowPyrLK(
        gray1,
        gray2,
        points1,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS
            | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01
        )
    )

    if points2 is None or status is None:
        raise ValueError("No se pudo calcular el flujo optico.")

    # Conservar los puntos cuyo seguimiento fue exitoso
    valid = status.reshape(-1) == 1

    old_points = points1.reshape(-1, 2)[valid]
    new_points = points2.reshape(-1, 2)[valid]

    if len(old_points) == 0:
        return {
            "tracked_points": 0,
            "dx_pixels": 0.0,
            "dy_pixels": 0.0,
            "vectors": []
        }

    # Calcular desplazamientos de cada punto
    displacement = new_points - old_points

    dx = float(np.median(displacement[:, 0]))
    dy = float(np.median(displacement[:, 1]))

    # Guardar vectores para visualizarlos despues
    vectors = []

    for old, new in zip(old_points, new_points):
        vectors.append({
            "x1": round(float(old[0]), 2),
            "y1": round(float(old[1]), 2),
            "x2": round(float(new[0]), 2),
            "y2": round(float(new[1]), 2)
        })

    return {
        "tracked_points": len(vectors),
        "dx_pixels": round(dx, 2),
        "dy_pixels": round(dy, 2),
        "vectors": vectors
    }
