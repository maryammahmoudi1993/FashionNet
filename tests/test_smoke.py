"""Quick checks that the helpers work. They do not train anything for long."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import fashion  # noqa: E402


def test_data_shapes_and_range():
    (x, y), (xv, yv), (xt, yt) = fashion.load_fashion(n_train=200, n_val=50)
    assert x.shape == (200, 28, 28, 1) and xv.shape == (50, 28, 28, 1)
    assert xt.shape == (10000, 28, 28, 1)
    assert 0.0 <= x.min() and x.max() <= 1.0
    assert set(np.unique(y)) <= set(range(10)) and len(fashion.CLASS_NAMES) == 10


def test_convolution_finds_a_vertical_edge():
    image = np.zeros((6, 6), dtype="float32")
    image[:, 3:] = 1.0                                   # dark left half, bright right half
    kernel = np.array([[-1, 0, 1]] * 3, dtype="float32")
    out = fashion.convolve2d(image, kernel)
    assert out.shape == (4, 4)
    assert out[:, 1].min() > 0 and out[:, 0].max() == 0 and out[:, 3].max() == 0


def test_models_have_expected_output_shape():
    batch = np.zeros((2, 28, 28, 1), dtype="float32")
    assert fashion.build_dense()(batch).shape == (2, 10)
    assert fashion.build_cnn()(batch).shape == (2, 10)


def test_training_is_reproducible():
    (x, y), _, _ = fashion.load_fashion(n_train=300, n_val=10)
    losses = []
    for _ in range(2):
        fashion.set_seed()
        model = fashion.compile_model(fashion.build_cnn())
        history = model.fit(x, y, epochs=1, batch_size=64, verbose=0)
        losses.append(history.history["loss"][0])
    assert losses[0] == losses[1]
