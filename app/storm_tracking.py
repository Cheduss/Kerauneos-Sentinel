
import cv2
import numpy as np

from app.segmentation import segment_by_color

"""

tres mejoras fundamentales:
- Máscara de selección: solo buscamos puntos iniciales dentro de regiones rojas o magentas.
- Validación temporal: comprobamos que los puntos también terminen dentro de regiones seleccionadas en la segunda imagen.
- Verificación de ida y vuelta: seguimos un punto de T1 a T2 y después de T2 a T1. 
    Si no vuelve aproximadamente a su posición original, descartamos ese seguimiento.

"""


def estimate_storm_motion(
    image1: np.ndarray,
    image2: np.ndarray,
    colors: tuple[str, ...] = ("red", "magenta"),
    max_corners: int = 300,
    min_movement: float = 0.5,
    fb_tolerance: float = 1.5,
) -> dict:
    """
    Estima el movimiento aparente de ecos de precipitacion.

    Utiliza segmentacion por colores y flujo optico
    Lucas-Kanade con verificacion de ida y vuelta.

    Las dos imagenes deben tener igual geometria.
    """

    # 1. Validar imagenes
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

    if (
        type(max_corners) is not int
        or max_corners < 1
    ):
        raise ValueError("max_corners debe ser positivo.")

    if (
        not np.isfinite(min_movement)
        or min_movement < 0
        or not np.isfinite(fb_tolerance)
        or fb_tolerance <= 0
    ):
        raise ValueError("Parametros de seguimiento invalidos.")

    # 2. Segmentar las regiones meteorologicas
    mask1 = segment_by_color(image1, colors=colors)
    mask2 = segment_by_color(image2, colors=colors)

    # 3. Convertir a escala de grises
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    def empty_result(candidate_points=0):
        return {
            "candidate_points": candidate_points,
            "tracked_points": 0,
            "dx_pixels": 0.0,
            "dy_pixels": 0.0,
            "vectors": []
        }

    # 4. Buscar puntos solamente dentro de mask1
    points1 = cv2.goodFeaturesToTrack(
        gray1,
        maxCorners=max_corners,
        qualityLevel=0.01,
        minDistance=5,
        mask=mask1
    )

    if points1 is None:
        return empty_result()

    # 5. Calcular movimiento T1 -> T2
    points2, status2, _ = cv2.calcOpticalFlowPyrLK(
        gray1,
        gray2,
        points1,
        None,
        winSize=(21, 21),
        maxLevel=3
    )

    if points2 is None or status2 is None:
        return empty_result(len(points1))

    # 6. Verificacion inversa T2 -> T1
    points_back, status_back, _ = cv2.calcOpticalFlowPyrLK(
        gray2,
        gray1,
        points2,
        None,
        winSize=(21, 21),
        maxLevel=3
    )

    if points_back is None or status_back is None:
        return empty_result(len(points1))

    original = points1.reshape(-1, 2)
    destination = points2.reshape(-1, 2)
    recovered = points_back.reshape(-1, 2)

    # 7. Conservar seguimientos exitosos
    valid = (
        (status2.reshape(-1) == 1)
        & (status_back.reshape(-1) == 1)
    )

    # Error de ida y vuelta
    back_error = np.linalg.norm(
        recovered - original,
        axis=1
    )

    valid &= back_error <= fb_tolerance

    # 8. Verificar que el punto final este en mask2
    height, width = mask2.shape

    positions = np.rint(destination).astype(int)
    xs = positions[:, 0]
    ys = positions[:, 1]

    inside = (
        (xs >= 0)
        & (xs < width)
        & (ys >= 0)
        & (ys < height)
    )

    valid &= inside

    # Evitar indices fuera de la imagen
    safe_x = np.clip(xs, 0, width - 1)
    safe_y = np.clip(ys, 0, height - 1)

    valid &= mask2[safe_y, safe_x] > 0

    # 9. Excluir desplazamientos demasiado pequenos
    displacement = destination - original

    magnitude = np.linalg.norm(
        displacement,
        axis=1
    )

    valid &= magnitude >= min_movement

    old_points = original[valid]
    new_points = destination[valid]

    if len(old_points) == 0:
        return empty_result(len(points1))

    # 10. Calcular desplazamiento mediano
    delta = new_points - old_points

    dx = float(np.median(delta[:, 0]))
    dy = float(np.median(delta[:, 1]))

    vectors = []

    for old, new in zip(old_points, new_points):
        vectors.append({
            "x1": round(float(old[0]), 2),
            "y1": round(float(old[1]), 2),
            "x2": round(float(new[0]), 2),
            "y2": round(float(new[1]), 2)
        })

    return {
        "candidate_points": len(points1),
        "tracked_points": len(vectors),
        "dx_pixels": round(dx, 2),
        "dy_pixels": round(dy, 2),
        "vectors": vectors
    }
