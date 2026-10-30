
import cv2
import numpy as np
import pytest

from app.storm_tracking import estimate_storm_motion


@pytest.fixture
def radar_pair():
    """
    Dos imagenes sinteticas con ecos de colores
    que se desplazan 8 px a la derecha y
    5 px hacia arriba.
    """

    first = np.zeros((240, 300, 3), dtype=np.uint8)
    second = np.zeros_like(first)

    for x in range(25, 210, 45):
        for y in range(35, 190, 55):

            # Primer instante
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

            # Segundo instante: desplazamiento (8, -5)
            cv2.rectangle(
                second,
                (x + 8, y - 5),
                (x + 27, y + 14),
                (0, 0, 255),
                -1
            )

            cv2.rectangle(
                second,
                (x + 12, y - 1),
                (x + 18, y + 5),
                (255, 0, 255),
                -1
            )

    return first, second


def test_detect_motion(radar_pair):
    first, second = radar_pair

    result = estimate_storm_motion(first, second)

    assert result["tracked_points"] > 0
    assert abs(result["dx_pixels"] - 8) < 1
    assert abs(result["dy_pixels"] + 5) < 1


def test_vectors_are_valid(radar_pair):
    first, second = radar_pair

    result = estimate_storm_motion(first, second)

    assert len(result["vectors"]) == result["tracked_points"]

    for vector in result["vectors"]:
        assert "x1" in vector
        assert "y1" in vector
        assert "x2" in vector
        assert "y2" in vector


def test_without_precipitation():
    first = np.zeros((100, 100, 3), dtype=np.uint8)
    second = first.copy()

    result = estimate_storm_motion(first, second)

    assert result["tracked_points"] == 0
    assert result["vectors"] == []


def test_stationary_echoes(radar_pair):
    first, _ = radar_pair

    result = estimate_storm_motion(
        first,
        first.copy()
    )

    # Se excluyen los movimientos menores a 0.5 px
    assert result["tracked_points"] == 0


def test_different_dimensions(radar_pair):
    first, _ = radar_pair

    second = np.zeros((100, 100, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        estimate_storm_motion(first, second)


def test_invalid_max_corners(radar_pair):
    first, second = radar_pair

    with pytest.raises(ValueError):
        estimate_storm_motion(
            first, second, max_corners=0
        )


def test_invalid_image(radar_pair):
    first, _ = radar_pair

    with pytest.raises(ValueError):
        estimate_storm_motion(first, None)
