"""Pruebas de integracion entre interfaz, FastAPI y OpenCV."""

import base64
import re
from html.parser import HTMLParser

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.api import app


client = TestClient(app)


class HtmlIds(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if "id" in attributes:
            self.ids.add(attributes["id"])


def png_bytes(image):
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def decode_png(data_url):
    prefix = "data:image/png;base64,"
    assert data_url.startswith(prefix)
    data = base64.b64decode(data_url[len(prefix):], validate=True)
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert image is not None
    return image


def test_frontend_scripts_use_existing_html_ids():
    """Los elementos buscados desde JS deben existir en index.html."""
    response = client.get("/app")
    assert response.status_code == 200
    html = response.text

    parser = HtmlIds()
    parser.feed(html)

    for script in ("app.js", "tracking.js"):
        assert f'/static/{script}' in html
        js_response = client.get(f"/static/{script}")
        assert js_response.status_code == 200
        source = js_response.text

        ids = set(re.findall(
            r'document\.querySelector(?:All)?\(["\']#([\w-]+)', source
        ))
        ids.update(re.findall(
            r'document\.getElementById\(["\']([\w-]+)', source
        ))
        missing = ids - parser.ids
        assert not missing, f"{script} utiliza IDs inexistentes: {missing}"


def test_detection_from_frontend_returns_full_images():
    """La API entrega las tres imágenes decodificables y una tabla coherente."""
    image = np.zeros((180, 180, 3), dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (55, 55), (0, 0, 255), -1)
    cv2.rectangle(image, (100, 90), (135, 125), (255, 0, 255), -1)

    response = client.post(
        "/api/analyze",
        files={"image": ("radar.png", png_bytes(image), "image/png")},
        data={
            "segmentation_mode": "hsv",
            "filter_type": "none",
            "morphology_operation": "none",
            "min_area": "20",
        },
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["width"] == 180
    assert result["height"] == 180
    assert result["total_cells"] == len(result["cells"])
    assert result["total_cells"] >= 2
    for key in ("original", "mask", "result"):
        decoded = decode_png(result["images"][key])
        assert decoded.shape[:2] == image.shape[:2]


def test_tracking_from_frontend_returns_valid_motion_png():
    """La API de seguimiento permite mostrar la imagen con flechas."""
    first = np.zeros((240, 300, 3), dtype=np.uint8)
    for x in range(25, 210, 45):
        for y in range(35, 190, 55):
            cv2.rectangle(first, (x, y), (x + 19, y + 19), (0, 0, 255), -1)
            cv2.rectangle(first, (x + 4, y + 4), (x + 10, y + 10), (255, 0, 255), -1)
    transform = np.float32([[1, 0, 8], [0, 1, -5]])
    second = cv2.warpAffine(first, transform, (300, 240))

    response = client.post(
        "/api/track",
        files={
            "image1": ("t1.png", png_bytes(first), "image/png"),
            "image2": ("t2.png", png_bytes(second), "image/png"),
        },
        data={"tracking_mode": "precipitation"},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["tracking_mode"] == "precipitation"
    assert result["tracked_points"] > 0
    assert len(result["vectors"]) == result["tracked_points"]
    assert abs(result["dx_pixels"] - 8) < 1
    assert abs(result["dy_pixels"] + 5) < 1
    for key in ("first", "second", "motion"):
        decoded = decode_png(result["images"][key])
        assert decoded.shape[:2] == first.shape[:2]


def test_frontend_does_not_show_academic_label():
    response = client.get("/app")
    assert response.status_code == 200
    assert "Proyecto académico" not in response.text
    assert 'id="module-tracking"' in response.text
    assert 'id="module-detection"' in response.text
