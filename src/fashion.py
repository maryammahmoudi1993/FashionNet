"""Small helpers shared by the notebooks: data loading, models and figures.

Everything that is not the lesson itself lives here, so that the notebooks stay short.
"""

import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # keep TensorFlow start-up messages quiet
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # avoid tiny run-to-run rounding differences

import keras
import matplotlib
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

SEED = 42
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
ASSETS = Path(__file__).resolve().parents[1] / "assets"


def set_seed(seed=SEED):
    """Fix every source of randomness we use, so reruns give the same numbers on a CPU."""
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def load_fashion(n_train=20000, n_val=5000):
    """Return the data as float32 arrays in [0, 1] with shape (N, 28, 28, 1).

    The 60,000 training pictures are shuffled once with a fixed seed; the first ``n_train``
    become the training set and the next ``n_val`` the validation set. The 10,000 test pictures
    are always the official ones and are never used for any decision.
    """
    (x_all, y_all), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    order = np.random.default_rng(SEED).permutation(len(x_all))
    x_all, y_all = x_all[order], y_all[order]

    def prepare(x):
        return (x.astype("float32") / 255.0)[..., None]

    train = (prepare(x_all[:n_train]), y_all[:n_train])
    val = (prepare(x_all[n_train:n_train + n_val]), y_all[n_train:n_train + n_val])
    test = (prepare(x_test), y_test)
    return train, val, test


def save_figure(fig, name):
    """Save a figure into assets/ (the README and the notebooks use these files)."""
    ASSETS.mkdir(exist_ok=True)
    fig.savefig(ASSETS / name, dpi=110, bbox_inches="tight")


def convolve2d(image, kernel):
    """Slide ``kernel`` over ``image`` and sum the products at every position ("valid" mode).

    This is exactly what a convolution layer does with one filter (without the bias).
    """
    kh, kw = kernel.shape
    out = np.zeros((image.shape[0] - kh + 1, image.shape[1] - kw + 1), dtype="float32")
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(image[i:i + kh, j:j + kw] * kernel)
    return out


def build_dense(hidden=64):
    """A network that sees only a long list of pixels (no idea which pixels are neighbours)."""
    return keras.Sequential([
        keras.layers.Input((28, 28, 1)),
        keras.layers.Flatten(),
        keras.layers.Dense(hidden, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ], name="dense_baseline")


def build_cnn(filters=(16, 32)):
    """The first CNN: two (convolution, pooling) blocks, then a decision layer."""
    return keras.Sequential([
        keras.layers.Input((28, 28, 1)),
        keras.layers.Conv2D(filters[0], 3, activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(filters[1], 3, activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation="softmax"),
    ], name="first_cnn")


def compile_model(model):
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def plot_class_grid(x, y, per_class=5):
    """One row per class, ``per_class`` examples each."""
    fig, axes = plt.subplots(10, per_class, figsize=(per_class * 1.1, 10 * 1.15))
    for c in range(10):
        idx = np.flatnonzero(y == c)[:per_class]
        for k, ax in enumerate(axes[c]):
            ax.imshow(x[idx[k], :, :, 0], cmap="gray_r")
            ax.set_xticks([]); ax.set_yticks([])
        axes[c, 0].set_ylabel(CLASS_NAMES[c], rotation=0, ha="right", va="center", fontsize=9)
    return fig


def plot_pixels(image, rows=slice(8, 20), cols=slice(8, 20)):
    """Show a picture next to a zoomed crop where every pixel value is printed."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 4.2), gridspec_kw={"width_ratios": [1, 1.5]})
    a.imshow(image, cmap="gray_r")
    r0, r1, c0, c1 = rows.start, rows.stop, cols.start, cols.stop
    a.add_patch(plt.Rectangle((c0 - .5, r0 - .5), c1 - c0, r1 - r0, fill=False, color="tab:red", lw=2))
    a.set_title("the picture (28 x 28)"); a.axis("off")
    crop = image[rows, cols]
    b.imshow(crop, cmap="gray_r", vmin=0, vmax=1)
    for i in range(crop.shape[0]):
        for j in range(crop.shape[1]):
            b.text(j, i, f"{crop[i, j]:.1f}", ha="center", va="center", fontsize=7,
                   color="white" if crop[i, j] > .5 else "black")
    b.set_title("red square, zoomed: one number per pixel"); b.axis("off")
    return fig


def plot_history(histories, labels):
    """Validation accuracy and loss per epoch for several models."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
    for h, label in zip(histories, labels):
        a.plot(range(1, len(h["val_accuracy"]) + 1), h["val_accuracy"], marker="o", label=label)
        b.plot(range(1, len(h["val_loss"]) + 1), h["val_loss"], marker="o", label=label)
    a.set_title("validation accuracy (higher is better)"); b.set_title("validation loss (lower is better)")
    for ax in (a, b):
        ax.set_xlabel("epoch"); ax.grid(alpha=.3); ax.legend()
    return fig


def plot_predictions(x, y_true, probs, n=10, wrong=False):
    """Show ``n`` test pictures with the predicted class, its confidence, and the true class."""
    pred = probs.argmax(1)
    idx = np.flatnonzero((pred != y_true) if wrong else (pred == y_true))[:n]
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.6))
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(x[i, :, :, 0], cmap="gray_r"); ax.axis("off")
        color = "tab:red" if wrong else "tab:green"
        ax.set_title(f"{CLASS_NAMES[pred[i]]} ({probs[i, pred[i]]:.0%})\ntrue: {CLASS_NAMES[y_true[i]]}",
                     fontsize=8, color=color)
    return fig
