
import cv2
import numpy as np
import pytest

from fastapi.testclient import TestClient
from app.api import app


client = TestClient(app)


@pytest.fixture
def moving_images():
    """Crea dos imagenes con objetos desplazados."""

    first = np.zeros((220, 220, 3), dtype=np.uint8)

    for x in range(20, 180, 30):
        for y in range(20, 180, 30):
            cv2.rectangle(
                first,
                (x, y),
                (x + 12, y + 12),
                (255, 255, 255),
                -1
            )

    matrix = np.float32([
        [1, 0, 5],
        [0, 1, 3]
    ])

    second = cv2.warpAffine(
        first,
        matrix,
        (220, 220)
    )

    return first, second


def encode_image(image):
    success, buffer = cv2.imencode(".png", image)
    assert success
    return buffer.tobytes()


def send_images(first, second, data=None):
    return client.post(
        "/api/track",
        files={
            "image1": (
                "radar_1.png",
                first,
                "image/png"
            ),
            "image2": (
                "radar_2.png",
                second,
                "image/png"
            )
        },
        data=data or {}
    )


def test_tracking_endpoint(moving_images):
    first, second = moving_images

    response = send_images(
        encode_image(first),
        encode_image(second)
    )

    assert response.status_code == 200

    result = response.json()

    assert result["tracked_points"] > 0
    assert abs(result["dx_pixels"] - 5) < 1
    assert abs(result["dy_pixels"] - 3) < 1


def test_motion_image(moving_images):
    first, second = moving_images

    response = send_images(
        encode_image(first),
        encode_image(second)
    )

    assert response.status_code == 200

    result = response.json()

    assert result["images"]["motion"].startswith(
        "data:image/png;base64,"
    )


def test_different_dimensions(moving_images):
    first, _ = moving_images

    second = np.zeros(
        (100, 100, 3),
        dtype=np.uint8
    )

    response = send_images(
        encode_image(first),
        encode_image(second)
    )

    assert response.status_code == 400


def test_invalid_image():
    response = send_images(
        b"contenido invalido",
        b"contenido invalido"
    )

    assert response.status_code == 400


def test_tracking_with_roi(moving_images):
    first, second = moving_images

    response = send_images(
        encode_image(first),
        encode_image(second),
        data={"roi": "10,10,200,200"}
    )

    assert response.status_code == 200

    result = response.json()

    assert result["tracked_points"] > 0


def test_invalid_roi(moving_images):
    first, second = moving_images

    response = send_images(
        encode_image(first),
        encode_image(second),
        data={"roi": "10,20,30"}
    )

    assert response.status_code == 400
