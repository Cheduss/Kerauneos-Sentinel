
import cv2
import numpy as np


"""
LO QUE REALIZABA EL PROGRAMA ERA BUSCAR PIXELES ROJOS Y MAGENTAS, PERO CON ESTO
SE USA LA BARRA DE COLORES DEL RADAR PARA DETECTAR REGIONES QUE SUPEREN EL UMBRAL
DE REFLECTIVIDAD QUE SERÍA ELEGIDO POR EL USUARIO DEL PROGRAMA.

Esto es aparte, se consercva el HSV anterior para ver cuales cambios pueden realizarse.

NumPy	Trabajo con matrices de píxeles
cv2.cvtColor	Convierte entre espacios de color
Espacio LAB	Compara similitudes entre colores
Umbralización	Selecciona pixeles por reflectividad estimada
Máscara binaria	Separa regiones seleccionadas del fondo

"""

def segment_by_dbz(
    image: np.ndarray,
    full_image: np.ndarray,
    threshold_dbz: float = 40,
    legend_x: int = 730,
    legend_y_top: int = 102,
    legend_y_bottom: int = 714,
    legend_dbz_top: int = 75,
    legend_dbz_bottom: int = -15,
    dbz_step: int = 5,
    max_color_distance: float = 25,
    min_saturation: int = 60,
) -> np.ndarray:
    """
    Segmenta regiones de reflectividad estimada.

    image: imagen del radar que queremos analizar.
    full_image: imagen completa con barra de colores.
    threshold_dbz: umbral minimo de reflectividad.

    Returns:
        Mascara binaria de valores 0 y 255.
    """

    # 1. Validar las imagenes
    for name, img in (
        ("image", image),
        ("full_image", full_image)
    ):
        if (
            not isinstance(img, np.ndarray)
            or img.ndim != 3
            or img.shape[2] != 3
            or img.size == 0
            or img.dtype != np.uint8
        ):
            raise ValueError(
                f"{name} debe ser una imagen BGR de 8 bits."
            )

    # 2. Validar coordenadas de la leyenda
    h, w = full_image.shape[:2]

    if not (
        0 <= legend_x < w
        and 0 <= legend_y_top < legend_y_bottom < h
    ):
        raise ValueError(
            "Coordenadas de leyenda fuera de la imagen."
        )

    if (
        dbz_step <= 0
        or legend_dbz_top <= legend_dbz_bottom
        or (legend_dbz_top - legend_dbz_bottom) % dbz_step != 0
    ):
        raise ValueError("Escala de dBZ no valida.")

    if (
        not isinstance(threshold_dbz, (int, float))
        or not np.isfinite(threshold_dbz)
        or not (
            legend_dbz_bottom <= threshold_dbz <= legend_dbz_top
        )
    ):
        raise ValueError("Umbral de dBZ fuera de la escala.")

    if (
        max_color_distance <= 0
        or not (0 <= min_saturation <= 255)
    ):
        raise ValueError(
            "Parametros de comparacion de color no validos."
        )

    # 3. Valores de reflectividad de la leyenda
    dbz_values = np.arange(
        legend_dbz_top,
        legend_dbz_bottom - 1,
        -dbz_step,
        dtype=np.int16
    )

    ys = np.rint(
        np.linspace(
            legend_y_top,
            legend_y_bottom,
            len(dbz_values)
        )
    ).astype(int)

    # 4. Obtener los colores de referencia
    legend_colors = full_image[ys, legend_x]

    # Convertir colores a espacio LAB
    legend_lab = cv2.cvtColor(
        legend_colors.reshape(1, -1, 3),
        cv2.COLOR_BGR2LAB
    ).reshape(-1, 3).astype(np.float32)

    image_lab = cv2.cvtColor(
        image, cv2.COLOR_BGR2LAB
    ).astype(np.float32)

    # 5. Buscar el color de referencia mas cercano
    best_dist_sq = np.full(
        image.shape[:2],
        np.inf,
        dtype=np.float32
    )

    estimated_dbz = np.full(
        image.shape[:2],
        -999,
        dtype=np.int16
    )

    for dbz, color_lab in zip(dbz_values, legend_lab):

        diff = image_lab - color_lab
        dist_sq = np.sum(diff * diff, axis=2)

        closer = dist_sq < best_dist_sq

        best_dist_sq[closer] = dist_sq[closer]
        estimated_dbz[closer] = dbz

    # 6. Excluir parte de los colores acromaticos
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 7. Generar mascara segun el umbral
    selected = (
        (estimated_dbz >= threshold_dbz)
        & (best_dist_sq <= max_color_distance ** 2)
        & (hsv[:, :, 1] >= min_saturation)
    )

    return selected.astype(np.uint8) * 255
