
import base64

import cv2
import numpy as np
import pytest

from fastapi.testclient import TestClient
from app.api import app


client = TestClient(app)


@pytest.fixture
def radar_png():
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Dos regiones artificiales
    cv2.rectangle(
        image, (10, 10), (30, 30), (0, 0, 255), -1
    )

    cv2.rectangle(
        image, (55, 55), (75, 75), (255, 0, 255), -1
    )

    success, buffer = cv2.imencode(".png", image)

    assert success
    return buffer.tobytes()


def upload_image(content, filename="radar.png", data=None):
    return client.post(
        "/api/analyze",
        files={
            "image": (filename, content, "image/png")
        },
        data=data or {}
    )


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["project"] == "Kerauneos-Sentinel"


def test_analyze_image(radar_png):
    response = upload_image(
        radar_png,
        data={
            "min_area": "20",
            "morphology_operation": "none"
        }
    )

    assert response.status_code == 200

    result = response.json()

    assert result["total_cells"] == 2
    assert len(result["cells"]) == 2
    assert result["width"] == 100
    assert result["height"] == 100


def test_result_image_is_valid(radar_png):
    response = upload_image(radar_png)

    assert response.status_code == 200

    encoded = response.json()["images"]["result"]

    assert encoded.startswith("data:image/png;base64,")

    content = base64.b64decode(encoded.split(",", 1)[1])

    image = cv2.imdecode(
        np.frombuffer(content, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )

    assert image is not None
    assert image.shape == (100, 100, 3)


def test_api_with_roi(radar_png):
    response = upload_image(
        radar_png,
        data={
            "roi": "0,0,40,40",
            "min_area": "20"
        }
    )

    assert response.status_code == 200
    assert response.json()["total_cells"] == 1
    assert response.json()["width"] == 40


def test_invalid_extension():
    response = upload_image(
        b"archivo de prueba",
        filename="radar.txt"
    )

    assert response.status_code == 415


def test_corrupted_image():
    response = upload_image(
        b"no es una imagen real"
    )

    assert response.status_code == 400


def test_invalid_roi(radar_png):
    response = upload_image(
        radar_png,
        data={"roi": "10,20,30"}
    )

    assert response.status_code == 400
