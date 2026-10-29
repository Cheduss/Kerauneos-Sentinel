
import numpy as np
import pytest

from app.crop_region import crop_radar


@pytest.fixture
def image():
    return np.zeros(
        (200, 300, 3),
        dtype=np.uint8
    )


def test_crop_dimensions(image):
    result = crop_radar(
        image, 50, 40, 250, 160
    )

    assert result.shape == (120, 200, 3)


def test_crop_preserves_pixels():
    image = np.zeros(
        (100, 100, 3),
        dtype=np.uint8
    )

    image[20:40, 30:60] = (0, 0, 255)

    result = crop_radar(
        image, 30, 20, 60, 40
    )

    assert np.all(result[:, :, 2] == 255)


def test_original_not_modified(image):
    original = image.copy()

    result = crop_radar(
        image, 10, 10, 100, 100
    )

    result[:] = 255

    assert np.array_equal(image, original)


def test_invalid_coordinates(image):
    with pytest.raises(ValueError):
        crop_radar(image, 100, 100, 50, 50)


def test_coordinates_outside_image(image):
    with pytest.raises(ValueError):
        crop_radar(image, 0, 0, 500, 500)


def test_invalid_image():
    with pytest.raises(ValueError):
        crop_radar(None, 0, 0, 50, 50)


def test_non_integer_coordinates(image):
    with pytest.raises(ValueError):
        crop_radar(image, 10.5, 20, 100, 100)


def test_full_image_crop(image):
    result = crop_radar(
        image, 0, 0, 300, 200
    )

    assert result.shape == image.shape
    assert np.array_equal(result, image)
