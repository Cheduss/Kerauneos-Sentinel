
from pathlib import Path

import cv2
import numpy as np


# Formatos admitidos
SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg"}


def load_image(image_path: str | Path) -> np.ndarray: #convierto la imagen en una matriz (la represento)
    """
    Carga y valida una imagen para su procesamiento.

    Args:
        image_path: Ruta de la imagen.

    Returns:
        Imagen como array de NumPy en formato BGR.
    """

    path = Path(image_path)

    # Verificar que el archivo exista
    if not path.is_file():
        raise FileNotFoundError(
            f"No se encontró la imagen: {path}"
        )

    # Verificar el formato
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            "Formato no admitido. Usar PNG, JPG o JPEG."
        )

    # Cargar la imagen
    image = cv2.imread(str(path), cv2.IMREAD_COLOR) #leo la imagen y con el IMREAD_COLOR admito los tres canales de color

    # Verificar que se haya podido leer
    if image is None:
        raise ValueError(
            "No se pudo leer la imagen."
        )

    return image
