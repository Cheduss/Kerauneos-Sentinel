
import cv2
import numpy as np


def apply_filter(
    image: np.ndarray,
    filter_type: str = "none",
    kernel_size: int = 3
) -> np.ndarray:
    """
    Aplica un filtro espacial a una imagen.

    filter_type:
        none: sin filtro
        average: filtro promedio
        gaussian: filtro gaussiano
        median: filtro de mediana
    """

    if image is None or image.size == 0:
        raise ValueError("La imagen está vacía.")

    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("El kernel debe ser impar y positivo.")

    if filter_type == "none":
        return image.copy()

    elif filter_type == "average": #promedia los pixeles cercanos suavizando la imagen
        return cv2.blur(
            image, (kernel_size, kernel_size)
        )

    elif filter_type == "gaussian": #Suaviza dando mayor peso a los pixeles cercanos
        return cv2.GaussianBlur(
            image, (kernel_size, kernel_size), 0
        )

    elif filter_type == "median": #reemplaza cada valor por la mediana de sus vecinos (cuando hay ruiidos aislados rinde)
        return cv2.medianBlur(image, kernel_size)

    else:
        raise ValueError(
            f"Filtro no reconocido: {filter_type}"
        )
