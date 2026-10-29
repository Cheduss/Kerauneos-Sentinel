
import numpy as np

"""
LO QUE HACE ESTO ES CORTAR LA REGION DE INTERES PARA QUE NO GENERE FALSOS POSITIVOS
TANTO EN LA REGION DEL CORTE VERTICAL COMO EN LAS LETRAS Y LA PARTE BLANCA.

"""

def crop_radar(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int
) -> np.ndarray:
    """
    Recorta una region rectangular de una imagen.

    Args:
        image: Imagen original como array NumPy.
        x1, y1: Esquina superior izquierda.
        x2, y2: Esquina inferior derecha (exclusiva).

    Returns:
        Imagen recortada.
    """

    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 3
        or image.shape[2] != 3
        or image.size == 0
        or image.dtype != np.uint8
    ):
        raise ValueError(
            "La imagen debe ser BGR de 8 bits."
        )

    coordinates = (x1, y1, x2, y2)

    if any(
        type(value) is not int
        for value in coordinates
    ):
        raise ValueError(
            "Las coordenadas deben ser enteros."
        )

    height, width = image.shape[:2]

    if not (
        0 <= x1 < x2 <= width
        and 0 <= y1 < y2 <= height
    ):
        raise ValueError(
            "Las coordenadas de recorte no son validas."
        )

    # Recortar utilizando NumPy
    cropped = image[y1:y2, x1:x2].copy() #selecciona filas y columnas, con cpy crea una imagen independiente

    return cropped
