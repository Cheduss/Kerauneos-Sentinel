
import cv2
import numpy as np
import pytest

from app.optical_flow import estimate_motion


@pytest.fixture
def moving_images():
    # Imagen con varios objetos de alto contraste
    image1 = np.zeros((220, 220, 3), dtype=np.uint8)

    for x in range(20, 180, 30):
        for y in range(20, 180, 30):
            cv2.rectangle(
                image1,
                (x, y),
                (x + 12, y + 12),
                (255, 255, 255),
                -1
            )

    # Desplazar toda la imagen 5 px en X y 3 px en Y
    matrix = np.float32([
        [1, 0, 5],
        [0, 1, 3]
    ])

    image2 = cv2.warpAffine(
        image1,
        matrix,
        (220, 220)
    )

    return image1, image2


def test_motion_detected(moving_images):
    image1, image2 = moving_images

    result = estimate_motion(image1, image2)

    assert result["tracked_points"] > 0
    assert abs(result["dx_pixels"] - 5) < 1
    assert abs(result["dy_pixels"] - 3) < 1


def test_same_image(moving_images):
    image1, _ = moving_images

    result = estimate_motion(image1, image1.copy())

    assert abs(result["dx_pixels"]) < 1
    assert abs(result["dy_pixels"]) < 1


def test_invalid_dimensions(moving_images):
    image1, _ = moving_images

    image2 = np.zeros((100, 100, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        estimate_motion(image1, image2)


def test_invalid_image(moving_images):
    image1, _ = moving_images

    with pytest.raises(ValueError):
        estimate_motion(image1, None)


def test_no_features():
    image1 = np.zeros((100, 100, 3), dtype=np.uint8)
    image2 = image1.copy()

    result = estimate_motion(image1, image2)

    assert result["tracked_points"] == 0
    assert result["vectors"] == []


def test_invalid_max_corners(moving_images):
    image1, image2 = moving_images

    with pytest.raises(ValueError):
        estimate_motion(image1, image2, max_corners=0)
