
import cv2
import numpy as np
import pytest

from app.detection import detect_storm_cells
from app.visualization import draw_storm_cells


@pytest.fixture
def sample_data():
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    mask = np.zeros((120, 120), dtype=np.uint8)

    # Crear dos nucleos artificiales
    cv2.rectangle(mask, (20, 20), (49, 49), 255, -1)
    cv2.circle(mask, (85, 85), 12, 255, -1)

    cells = detect_storm_cells(mask, min_area=50)

    return image, mask, cells


def test_visualization_returns_image(sample_data):
    image, mask, cells = sample_data

    result = draw_storm_cells(image, mask, cells)

    assert isinstance(result, np.ndarray)
    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_visualization_draws_annotations(sample_data):
    image, mask, cells = sample_data

    result = draw_storm_cells(image, mask, cells)

    # La imagen resultante debe tener anotaciones
    assert np.any(result != image)


def test_original_image_not_modified(sample_data):
    image, mask, cells = sample_data

    original = image.copy()

    draw_storm_cells(image, mask, cells)

    assert np.array_equal(image, original)


def test_empty_detection():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)

    result = draw_storm_cells(image, mask, [])

    assert np.array_equal(result, image)
    assert result is not image


def test_invalid_mask_dimensions(sample_data):
    image, _, cells = sample_data

    wrong_mask = np.zeros((50, 50), dtype=np.uint8)

    with pytest.raises(ValueError):
        draw_storm_cells(image, wrong_mask, cells)


def test_inconsistent_detection(sample_data):
    image, mask, cells = sample_data

    # Modificamos una propiedad para que no coincida
    invalid_cells = [
        {**cells[0], "area": 9999}
    ]

    with pytest.raises(ValueError):
        draw_storm_cells(image, mask, invalid_cells)
