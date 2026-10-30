
import cv2
import numpy as np


def draw_motion_vectors(
    image: np.ndarray,
    motion: dict,
    max_vectors: int = 60,
    min_displacement: float = 1.0
) -> np.ndarray:
    """
    DIBUJA LOS VECTORES DEL FLUJO OPTICO EN LA IMAGEN.

    Argys:
        image: Segunda imagen en formato BGR.
        motion: Resultado de estimate_motion().
        max_vectors: Maximo de flechas para dibujar.
        min_displacement: Movimiento minimo en pixeles.

        cv2.arrowedLine	Dibuja flechas de desplazamiento
        np.hypot	Calcula la magnitud del movimiento
        cv2.rectangle	Dibuja un fondo negro para las estadísticas
        cv2.putText	Muestra DX, DY y cantidad de puntos

    Returns:
        Imagen con flechas e informacion del movimiento.
    """

    # Validar imagen
    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 3
        or image.shape[2] != 3
        or image.dtype != np.uint8
        or image.size == 0
    ):
        raise ValueError("Imagen BGR no valida.")

    if (
        not isinstance(max_vectors, int)
        or isinstance(max_vectors, bool)
        or max_vectors < 1
    ):
        raise ValueError("max_vectors debe ser positivo.")

    if (
        not np.isfinite(min_displacement)
        or min_displacement < 0
    ):
        raise ValueError("Desplazamiento minimo no valido.")

    if not isinstance(motion, dict):
        raise ValueError("Resultado de movimiento no valido.")

    required = {
        "vectors", "dx_pixels",
        "dy_pixels", "tracked_points"
    }

    if not required.issubset(motion):
        raise ValueError("Faltan datos del movimiento.")

    result = image.copy()
    vectors = motion["vectors"]

    if not isinstance(vectors, list):
        raise ValueError("Los vectores deben ser una lista.")

    if not vectors:
        return result

    # Distribuir la seleccion de flechas
    step = max(
        1,
        int(np.ceil(len(vectors) / max_vectors))
    )

    selected = vectors[::step][:max_vectors]

    for vector in selected:

        try:
            x1 = float(vector["x1"])
            y1 = float(vector["y1"])
            x2 = float(vector["x2"])
            y2 = float(vector["y2"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "Vector de movimiento no valido."
            ) from error

        values = (x1, y1, x2, y2)

        if not np.all(np.isfinite(values)):
            raise ValueError("Vector con coordenadas invalidas.")

        dx = x2 - x1
        dy = y2 - y1

        displacement = np.hypot(dx, dy)

        if displacement < min_displacement:
            continue

        start = (round(x1), round(y1))
        end = (round(x2), round(y2))

        # Borde negro para mejorar el contraste
        cv2.arrowedLine(
            result,
            start,
            end,
            (0, 0, 0),
            4,
            tipLength=0.3
        )

        # Flecha cian (OpenCV usa BGR)
        cv2.arrowedLine(
            result,
            start,
            end,
            (255, 255, 0),
            2,
            tipLength=0.3
        )

    # Resumen del movimiento
    dx = motion["dx_pixels"]
    dy = motion["dy_pixels"]
    total = motion["tracked_points"]

    cv2.rectangle(
        result,
        (5, 5),
        (185, 72),
        (0, 0, 0),
        -1
    )

    lines = [
        f"DX: {dx:+.1f} px",
        f"DY: {dy:+.1f} px",
        f"Puntos: {total}"
    ]

    for index, line in enumerate(lines):
        cv2.putText(
            result,
            line,
            (12, 24 + index * 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return result
