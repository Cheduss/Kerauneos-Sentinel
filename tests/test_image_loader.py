
import cv2
import numpy as np
import pytest

from app.load_an_image import load_image


def test_load_valid_image(tmp_path):
    # Crear imagen de prueba de 100x100 píxeles
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    image_path = tmp_path / "radar.png"
    cv2.imwrite(str(image_path), image)

    # Cargar imagen con nuestra función
    result = load_image(image_path)

    assert isinstance(result, np.ndarray)
    assert result.shape == (100, 100, 3)


def test_image_not_found(tmp_path):
    image_path = tmp_path / "no_existe.png"

    with pytest.raises(FileNotFoundError):
        load_image(image_path)


def test_invalid_format(tmp_path):
    image_path = tmp_path / "archivo.txt"
    image_path.write_text("Esto no es una imagen")

    with pytest.raises(ValueError):
        load_image(image_path)


def test_corrupted_image(tmp_path):
    image_path = tmp_path / "imagen.png"
    image_path.write_bytes(b"archivo corrupto")

    with pytest.raises(ValueError):
        load_image(image_path)
