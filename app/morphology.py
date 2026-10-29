
import cv2
import numpy as np

"""
LO QUE HACEMOS ACA ES LIMPIAR LAS REGIONES DFETECTADAS MEDIANTE LA SEGMENTACIÓN
PARA DESPUES PODER DETECTAR LOS NUCLEOS DE PRECIPITACIÓN.

"""


def apply_morphology(
    mask: np.ndarray,
    operation: str = "opening",
    kernel_size: int = 3,
    iterations: int = 1
) -> np.ndarray:
    """
    Aplica operaciones morfologicas a una mascara binaria.

    Args:
        mask: Imagen binaria con valores 0 y 255.
        operation: erosion, dilation, opening, closing o none.
        kernel_size: Tamano del elemento estructurante.
        iterations: Numero de iteraciones.

    Returns:
        Mascara procesada.
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
            "La mascara debe ser binaria, de tipo uint8."
        )

    # Validar el kernel
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(
            "El kernel debe ser impar y positivo."
        )

    if iterations < 1:
        raise ValueError(
            "Las iteraciones deben ser mayores a cero."
        )

    # Crear elemento estructurante
    kernel = cv2.getStructuringElement( #crea una pequeña matriz qyue determina que pixeles vecimos participan en la op. moroflogica
        #si es un kernel_size = 3, utulizamos una matriz 3x3
        cv2.MORPH_RECT,
        (kernel_size, kernel_size)
    )

    # Seleccionar operacion
    if operation == "none":
        return mask.copy()

    elif operation == "erosion":
        return cv2.erode(
            mask, kernel, iterations=iterations
        )

    elif operation == "dilation":
        return cv2.dilate(
            mask, kernel, iterations=iterations
        )

    elif operation == "opening":
        return cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=iterations
        )

    elif operation == "closing":
        return cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=iterations
        )

    else:
        raise ValueError(
            f"Operacion no reconocida: {operation}"
        )
