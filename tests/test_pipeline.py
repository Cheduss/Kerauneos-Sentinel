
import cv2
import numpy as np

from app.pipeline import analyze_radar


def test_complete_pipeline(tmp_path):
    # Imagen artificial de radar
    image = np.zeros((200, 200, 3), dtype=np.uint8)

    # Nucleo rojo
    cv2.rectangle(
        image, (20, 20), (60, 60), (0, 0, 255), -1
    )

    # Nucleo magenta
    cv2.rectangle(
        image, (110, 110), (150, 150), (255, 0, 255), -1
    )

    path = tmp_path / "radar_test.png"
    cv2.imwrite(str(path), image)

    analysis = analyze_radar(
        image_path=path,
        min_area=100
    )

    assert analysis["total_cells"] == 2
    assert len(analysis["cells"]) == 2

    assert analysis["mask"].shape == (200, 200)
    assert analysis["result"].shape == image.shape

    # Ambas regiones tienen 41 x 41 pixeles
    assert analysis["cells"][0]["area"] == 1681
    assert analysis["cells"][1]["area"] == 1681


def test_pipeline_without_storm_cells(tmp_path):
    # Imagen sin regiones rojas ni magentas
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    path = tmp_path / "radar_empty.png"
    cv2.imwrite(str(path), image)

    analysis = analyze_radar(path)

    assert analysis["total_cells"] == 0
    assert analysis["cells"] == []


def test_pipeline_minimum_area(tmp_path):
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Region roja de 5 x 5 pixeles
    cv2.rectangle(
        image, (10, 10), (14, 14), (0, 0, 255), -1
    )

    path = tmp_path / "radar_small.png"
    cv2.imwrite(str(path), image)

    analysis = analyze_radar(
        path,
        morphology_operation="none",
        min_area=100
    )

    assert analysis["total_cells"] == 0
