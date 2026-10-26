import numpy as np

from ss_taker.preprocess import preprocess_image


def test_preprocess_returns_single_channel_image() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    out = preprocess_image(image)
    assert len(out.shape) == 2
    assert out.dtype == np.uint8

