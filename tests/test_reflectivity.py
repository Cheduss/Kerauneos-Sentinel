
import numpy as np
import pytest

from app.reflectivity import segment_by_dbz


@pytest.fixture
def data():
    # Imagen artificial con una leyenda conocida
    full = np.zeros((110, 180, 3), dtype=np.uint8)

    colors = [
        (0, 0, 255),      # 50 dBZ
        (255, 0, 255),    # 45 dBZ
        (0, 255, 255),    # 40 dBZ
        (0, 255, 0),      # 35 dBZ
        (255, 0, 0)       # 30 dBZ
    ]

    for y, color in zip((10, 30, 50, 70, 90), colors):
        full[y, 140] = color

    image = np.zeros((40, 40, 3), dtype=np.uint8)

    image[2:12, 2:12] = colors[0]
    image[14:24, 14:24] = colors[1]
    image[26:36, 26:36] = colors[3]

    options = {
        "legend_x": 140,
        "legend_y_top": 10,
        "legend_y_bottom": 90,
        "legend_dbz_top": 50,
        "legend_dbz_bottom": 30
    }

    return image, full, options


def test_threshold_45(data):
    image, full, options = data

    mask = segment_by_dbz(image, full, 45, **options)

    assert mask[4, 4] == 255
    assert mask[18, 18] == 255
    assert mask[30, 30] == 0
    assert np.count_nonzero(mask) == 200


def test_threshold_50(data):
    image, full, options = data

    mask = segment_by_dbz(image, full, 50, **options)

    assert mask[4, 4] == 255
    assert mask[18, 18] == 0


def test_mask_format(data):
    image, full, options = data

    mask = segment_by_dbz(image, full, 40, **options)

    assert mask.shape == image.shape[:2]
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})


def test_invalid_legend(data):
    image, full, options = data

    with pytest.raises(ValueError):
        segment_by_dbz(
            image,
            full,
            40,
            **{**options, "legend_x": 999}
        )


def test_invalid_threshold(data):
    image, full, options = data

    with pytest.raises(ValueError):
        segment_by_dbz(image, full, 70, **options)
