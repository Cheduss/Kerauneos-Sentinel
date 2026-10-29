
import cv2
import numpy as np
import pytest

from app.segmentation import segment_by_color


@pytest.fixture
def radar_image():
    """
    Imagen sintética con regiones de diferentes colores.
    """
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Región roja
    cv2.rectangle(image, (10, 10), (30, 30), (0, 0, 255), -1)

    # Región amarilla
    cv2.rectangle(image, (40, 10), (60, 30), (0, 255, 255), -1)

    # Región magenta
    cv2.rectangle(image, (10, 50), (30, 70), (255, 0, 255), -1)

    return image


def test_detect_red(radar_image):
    mask = segment_by_color(radar_image, ("red",))

    assert mask[20, 20] == 255
    assert mask[20, 50] == 0
    assert mask[60, 20] == 0


def test_detect_yellow(radar_image):
    mask = segment_by_color(radar_image, ("yellow",))

    assert mask[20, 50] == 255
    assert mask[20, 20] == 0


def test_detect_multiple_colors(radar_image):
    mask = segment_by_color(
        radar_image, ("red", "magenta")
    )

    assert mask[20, 20] == 255
    assert mask[60, 20] == 255
    assert mask[20, 50] == 0


def test_mask_dimensions(radar_image):
    mask = segment_by_color(radar_image)

    assert mask.shape == (100, 100)
    assert mask.dtype == np.uint8


def test_invalid_color(radar_image):
    with pytest.raises(ValueError):
        segment_by_color(radar_image, ("blue",))


def test_empty_image():
    with pytest.raises(ValueError):
        segment_by_color(None)


def test_no_colors(radar_image):
    with pytest.raises(ValueError):
        segment_by_color(radar_image, ())
