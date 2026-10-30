
import numpy as np
import pytest

from app.radar_profiles import (
    EZEIZA_PROFILE,
    REFERENCE_COLORS,
    validate_ezeiza_radar,
)


@pytest.fixture
def compatible_image():
    """Crea una imagen artificial con la leyenda esperada."""

    image = np.zeros((802, 805, 3), dtype=np.uint8)

    top = EZEIZA_PROFILE["legend_y_top"]
    bottom = EZEIZA_PROFILE["legend_y_bottom"]
    x = EZEIZA_PROFILE["legend_x"]

    for dbz, color in REFERENCE_COLORS.items():

        fraction = (75 - dbz) / 90
        y = round(top + fraction * (bottom - top))

        image[y, x] = color

    return image


def test_compatible_profile(compatible_image):
    assert validate_ezeiza_radar(compatible_image) is True


def test_invalid_dimensions():
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        validate_ezeiza_radar(image)


def test_invalid_palette():
    image = np.zeros((802, 805, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        validate_ezeiza_radar(image)


def test_invalid_image():
    with pytest.raises(ValueError):
        validate_ezeiza_radar(None)


def test_modified_legend(compatible_image):
    # Alterar una muestra de la leyenda
    compatible_image[238, 730] = (0, 255, 0)

    with pytest.raises(ValueError):
        validate_ezeiza_radar(compatible_image)
