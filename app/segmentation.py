
import cv2
import numpy as np


# Rangos de colores en HSV (formato de OpenCV)
COLOR_RANGES = {
    "yellow": [
        ((20, 100, 100), (35, 255, 255))
    ],
    "orange": [
        ((10, 100, 100), (20, 255, 255))
    ],
    "red": [
        ((0, 100, 100), (10, 255, 255)),
        ((170, 100, 100), (179, 255, 255))
    ],
    "magenta": [
        ((140, 100, 100), (169, 255, 255))
    ]
}


def segment_by_color(
    image: np.ndarray,
    colors: tuple[str, ...] = ("red", "magenta")
) -> np.ndarray:
    """
    Segmenta una imagen según los colores seleccionados.

    Args:
        image: Imagen BGR de OpenCV.
        colors: Colores que queremos detectar.

    Returns:
        Máscara binaria de tipo uint8:
        255 = píxel detectado
        0 = píxel no detectado
    """

    if (
        image is None
        or not isinstance(image, np.ndarray)
        or image.ndim != 3
        or image.shape[2] != 3
        or image.size == 0
        or image.dtype != np.uint8
    ):
        raise ValueError("La imagen debe ser BGR de 8 bits.")

    if not colors:
        raise ValueError("Debe seleccionarse al menos un color.")

    for color in colors:
        if color not in COLOR_RANGES:
            raise ValueError(f"Color no reconocido: {color}")

    # Convertimos de BGR a HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #convierte la imagen de BGR a HSV

    # Creamos una máscara negra
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Recorremos los colores seleccionados
    for color in colors:
        for lower, upper in COLOR_RANGES[color]:

            color_mask = cv2.inRange( #selecciona los pixeles dentro de un rango de color
                hsv,
                np.array(lower, dtype=np.uint8),
                np.array(upper, dtype=np.uint8)
            )

            # Sumamos las regiones detectadas
            mask = cv2.bitwise_or(mask, color_mask) #combina las regiones de distintos colores

    return mask

"""

nota: HSV representa los colores utilizando tono (Hue), saturacion (Saturation) y valor o brillo (value).

Esto lo que permite es seleccionar tonos especificos mas comodos que trabajando directamente en RGB.

En OPENCV, para las imagenes de 9 bits, el tono va de 0 a 179, por eso el riojo necesita dos rangos,
uno cerca de 0 y otro cerca de 179

"""