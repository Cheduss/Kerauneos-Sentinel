
import cv2
import numpy as np

from fastapi.testclient import TestClient
from app.api import app


client = TestClient(app)


def create_radar_pair():
    """
    Crea dos imagenes sinteticas con regiones
    rojas y magentas desplazadas.
    """

    first = np.zeros((240, 300, 3), dtype=np.uint8)

    for x in range(25, 210, 45):
        for y in range(35, 190, 55):

            cv2.rectangle(
                first,
                (x, y),
                (x + 19, y + 19),
                (0, 0, 255),
                -1
            )

            cv2.rectangle(
                first,
                (x + 4, y + 4),
                (x + 10, y + 10),
                (255, 0, 255),
                -1
            )

    matrix = np.float32([
        [1, 0, 8],
        [0, 1, -5]
    ])

    second = cv2.warpAffine(
        first, matrix, (300, 240)
    )

    return first, second


def encode_image(image):
    success, buffer = cv2.imencode(".png", image)
    assert success
    return buffer.tobytes()


def send_pair(first, second, mode):
    return client.post(
        "/api/track",
        files={
            "image1": (
                "radar1.png",
                encode_image(first),
                "image/png"
            ),
            "image2": (
                "radar2.png",
                encode_image(second),
                "image/png"
            )
        },
        data={"tracking_mode": mode}
    )


def test_precipitation_tracking():
    first, second = create_radar_pair()

    response = send_pair(
        first, second, "precipitation"
    )

    assert response.status_code == 200

    result = response.json()

    assert result["tracking_mode"] == "precipitation"
    assert result["tracked_points"] > 0
    assert result["candidate_points"] >= result["tracked_points"]
    assert abs(result["dx_pixels"] - 8) < 1
    assert abs(result["dy_pixels"] + 5) < 1


def test_general_tracking():
    first, second = create_radar_pair()

    response = send_pair(
        first, second, "general"
    )

    assert response.status_code == 200
    assert response.json()["tracking_mode"] == "general"
    assert "motion" in response.json()["images"]


def test_invalid_tracking_mode():
    first, second = create_radar_pair()

    response = send_pair(
        first, second, "unknown"
    )

    assert response.status_code == 422
