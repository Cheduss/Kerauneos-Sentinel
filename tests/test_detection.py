
import cv2
import numpy as np
import pytest

from app.detection import detect_storm_cells


@pytest.fixture
def binary_mask():
    """Mascara con dos regiones de distinto tamano."""

    mask = np.zeros((150, 150), dtype=np.uint8)

    # Region grande: 40 x 40 = 1600 pixeles
    cv2.rectangle(mask, (10, 10), (49, 49), 255, -1)

    # Region pequena: 20 x 20 = 400 pixeles
    cv2.rectangle(mask, (80, 80), (99, 99), 255, -1)

    return mask


def test_detect_two_cells(binary_mask):
    cells = detect_storm_cells(binary_mask)

    assert len(cells) == 2


def test_cell_areas(binary_mask):
    cells = detect_storm_cells(binary_mask)

    assert cells[0]["area"] == 1600
    assert cells[1]["area"] == 400


def test_centroid(binary_mask):
    cells = detect_storm_cells(binary_mask)

    assert cells[0]["centroid"]["x"] == 29.5
    assert cells[0]["centroid"]["y"] == 29.5


def test_bounding_box(binary_mask):
    cells = detect_storm_cells(binary_mask)

    box = cells[0]["bounding_box"]

    assert box["x"] == 10
    assert box["y"] == 10
    assert box["width"] == 40
    assert box["height"] == 40


def test_minimum_area(binary_mask):
    cells = detect_storm_cells(
        binary_mask,
        min_area=500
    )

    # Solo queda la region grande
    assert len(cells) == 1
    assert cells[0]["area"] == 1600


def test_empty_mask():
    mask = np.zeros((100, 100), dtype=np.uint8)

    cells = detect_storm_cells(mask)

    assert cells == []


def test_invalid_mask():
    with pytest.raises(ValueError):
        detect_storm_cells(None)


def test_invalid_minimum_area(binary_mask):
    with pytest.raises(ValueError):
        detect_storm_cells(binary_mask, min_area=0)


def test_diagonal_connectivity():
    mask = np.zeros((10, 10), dtype=np.uint8)

    mask[2, 2] = 255
    mask[3, 3] = 255

    cells = detect_storm_cells(mask, min_area=1)

    # Con conectividad 8 pertenecen a la misma region
    assert len(cells) == 1
    assert cells[0]["area"] == 2
