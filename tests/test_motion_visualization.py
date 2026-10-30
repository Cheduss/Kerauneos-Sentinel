
import numpy as np
import pytest

from app.motion_visualization import draw_motion_vectors


@pytest.fixture
def image():
    return np.zeros(
        (150, 200, 3),
        dtype=np.uint8
    )


@pytest.fixture
def motion():
    return {
        "tracked_points": 2,
        "dx_pixels": 8.0,
        "dy_pixels": -5.0,
        "vectors": [
            {
                "x1": 30, "y1": 50,
                "x2": 38, "y2": 45
            },
            {
                "x1": 80, "y1": 90,
                "x2": 88, "y2": 85
            }
        ]
    }


def test_returns_image(image, motion):
    result = draw_motion_vectors(image, motion)

    assert isinstance(result, np.ndarray)
    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_draws_arrows(image, motion):
    result = draw_motion_vectors(image, motion)

    assert np.any(result != image)


def test_original_not_modified(image, motion):
    original = image.copy()

    draw_motion_vectors(image, motion)

    assert np.array_equal(image, original)


def test_empty_vectors(image):
    motion = {
        "tracked_points": 0,
        "dx_pixels": 0.0,
        "dy_pixels": 0.0,
        "vectors": []
    }

    result = draw_motion_vectors(image, motion)

    assert np.array_equal(result, image)
    assert result is not image


def test_invalid_image(motion):
    with pytest.raises(ValueError):
        draw_motion_vectors(None, motion)


def test_invalid_max_vectors(image, motion):
    with pytest.raises(ValueError):
        draw_motion_vectors(
            image, motion, max_vectors=0
        )


def test_invalid_motion(image):
    with pytest.raises(ValueError):
        draw_motion_vectors(image, {})
