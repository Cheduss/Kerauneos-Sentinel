
import cv2
import numpy as np
import pytest

from app.morphology import apply_morphology


@pytest.fixture
def binary_mask():
    """Crea una mascara con una region y ruido aislado."""

    mask = np.zeros((100, 100), dtype=np.uint8)

    # Region principal
    cv2.rectangle(mask, (20, 20), (70, 70), 255, -1)

    # Pixel aislado
    mask[90, 90] = 255

    return mask


def test_erosion(binary_mask):
    result = apply_morphology(
        binary_mask, "erosion", kernel_size=3
    )

    assert np.count_nonzero(result) < np.count_nonzero(
        binary_mask
    )


def test_dilation(binary_mask):
    result = apply_morphology(
        binary_mask, "dilation", kernel_size=3
    )

    assert np.count_nonzero(result) > np.count_nonzero(
        binary_mask
    )


def test_opening_removes_noise(binary_mask):
    result = apply_morphology(
        binary_mask, "opening", kernel_size=3
    )

    # El pixel aislado debe desaparecer
    assert result[90, 90] == 0

    # La region principal debe mantenerse
    assert result[40, 40] == 255


def test_closing_fills_hole():
    mask = np.zeros((100, 100), dtype=np.uint8)

    cv2.rectangle(mask, (20, 20), (70, 70), 255, -1)

    # Pequeno hueco negro
    mask[40, 40] = 0

    result = apply_morphology(mask, "closing", 3)

    assert result[40, 40] == 255


def test_invalid_kernel(binary_mask):
    with pytest.raises(ValueError):
        apply_morphology(binary_mask, kernel_size=4)


def test_invalid_operation(binary_mask):
    with pytest.raises(ValueError):
        apply_morphology(binary_mask, "unknown")


def test_invalid_mask():
    with pytest.raises(ValueError):
        apply_morphology(None)


def test_none_preserves_mask(binary_mask):
    result = apply_morphology(binary_mask, "none")

    assert np.array_equal(result, binary_mask)
    assert result is not binary_mask
