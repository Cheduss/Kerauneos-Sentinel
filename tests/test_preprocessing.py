
import numpy as np
import pytest

from app.preprocessing import apply_filter


@pytest.fixture
def image():
    return np.random.default_rng(42).integers(
        0, 256, (100, 100, 3), dtype=np.uint8
    )


@pytest.mark.parametrize(
    "filter_type",
    ["none", "average", "gaussian", "median"]
)
def test_filters(image, filter_type):
    result = apply_filter(image, filter_type)

    assert isinstance(result, np.ndarray)
    assert result.shape == image.shape
    assert result.dtype == image.dtype


def test_invalid_kernel(image):
    with pytest.raises(ValueError):
        apply_filter(image, "gaussian", kernel_size=4)


def test_invalid_filter(image):
    with pytest.raises(ValueError):
        apply_filter(image, "unknown")


def test_empty_image():
    with pytest.raises(ValueError):
        apply_filter(None)


def test_none_preserves_image(image):
    result = apply_filter(image, "none")

    assert np.array_equal(result, image)
    assert result is not image
