"""Small helpers shared by the notebooks: data loading, models and figures.

Everything that is not the lesson itself lives here, so that the notebooks stay short.
"""

import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # keep TensorFlow start-up messages quiet
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


def augmentation_layers(strength="mild"):
    """Random changes applied to training pictures only (Keras switches them off for predictions).

    "mild": mirror left-right and shift by up to 6% (about 1.7 pixels).
    "strong": the same plus a small rotation and a zoom.
    """
    layers = [
        keras.layers.RandomFlip("horizontal"),            # a left sleeve may look like a right sleeve
        keras.layers.RandomTranslation(0.06, 0.06),       # shift by up to 6% of the picture size
    ]
    if strength == "strong":
        layers += [
            keras.layers.RandomRotation(0.05),            # turn by up to 5% of a full circle (18 degrees)
            keras.layers.RandomZoom(0.2),                 # zoom in or out by up to 20%
        ]
    return layers


def build_cnn(filters=(16, 32), augment=None, name=None):
    """The first CNN: two (convolution, pooling) blocks, then a decision layer.

    ``augment`` is None (no augmentation), "mild" or "strong": the random picture changes
    are placed in front of the network and are only active while training.
    """
    layers = [keras.layers.Input((28, 28, 1))]
    if augment:
        layers += augmentation_layers(augment)
    layers += [
        keras.layers.Conv2D(filters[0], 3, activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(filters[1], 3, activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation="softmax"),
    ]
    return keras.Sequential(layers, name=name or (f"cnn_{augment}" if augment else "first_cnn"))


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


def plot_gap(histories, labels):
    """Training versus validation accuracy and loss, one colour per model (dashed = training)."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for k, (h, label) in enumerate(zip(histories, labels)):
        color = f"C{k}"
        epochs = range(1, len(h["accuracy"]) + 1)
        a.plot(epochs, h["accuracy"], "--", color=color, alpha=.8, label=f"{label}: training")
        a.plot(epochs, h["val_accuracy"], "-", color=color, label=f"{label}: validation")
        b.plot(epochs, h["loss"], "--", color=color, alpha=.8, label=f"{label}: training")
        b.plot(epochs, h["val_loss"], "-", color=color, label=f"{label}: validation")
    a.set_title("accuracy (higher is better)"); b.set_title("loss (lower is better)")
    for ax in (a, b):
        ax.set_xlabel("epoch"); ax.grid(alpha=.3)
    a.set_ylim(0.7, 1.0); a.legend(fontsize=7)
    return fig


def plot_augmented(image, n=8, strength="mild"):
    """The same picture ``n`` times after random changes (the first panel is the original)."""
    layers = keras.Sequential(augmentation_layers(strength))
    fig, axes = plt.subplots(1, n + 1, figsize=(1.35 * (n + 1), 1.9))
    axes[0].imshow(image[:, :, 0], cmap="gray_r"); axes[0].set_title("original", fontsize=8)
    batch = np.repeat(image[None], n, axis=0)
    changed = layers(batch, training=True).numpy()
    for ax, im in zip(axes[1:], changed):
        ax.imshow(im[:, :, 0], cmap="gray_r", vmin=0, vmax=1)
    for ax in axes:
        ax.axis("off")
    return fig


def confusion_matrix(y_true, y_pred):
    """Rows = true class, columns = predicted class, entries = number of pictures."""
    matrix = np.zeros((10, 10), dtype=int)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def plot_confusion(matrix):
    """Confusion matrix as a heat map with the counts written in the cells."""
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(10)); ax.set_xticklabels(CLASS_NAMES, rotation=60, ha="right")
    ax.set_yticks(range(10)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("predicted class"); ax.set_ylabel("true class")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, matrix[i, j], ha="center", va="center", fontsize=7,
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    return fig


def grad_cam(model, image, class_index=None):
    """Grad-CAM heat map for one picture (Selvaraju et al., 2017).

    Steps: run the picture through the network, keep the feature maps ``A`` of the last convolution layer,
    take the gradient of the class score with respect to ``A``, average it over the positions to get one
    weight per feature map, add up the feature maps with these weights, keep only the positive part, and
    stretch the small map to the size of the picture.

    The class score is the value *before* the softmax (the "logit"). If ``class_index`` is None
    the predicted class is explained. Returns ``(heat_map, class_index, probability)``;
    the heat map has shape (28, 28) with values between 0 and 1.
    """
    conv_layer = [layer for layer in model.layers if isinstance(layer, keras.layers.Conv2D)][-1]
    x = tf.convert_to_tensor(image[None], dtype="float32")
    with tf.GradientTape() as tape:
        h, maps = x, None
        for layer in model.layers[:-1]:
            h = layer(h)
            if layer is conv_layer:
                maps = h                                   # feature maps A, shape (1, 11, 11, 32)
        head = model.layers[-1]
        logits = tf.matmul(h, head.kernel) + head.bias     # scores before the softmax
        if class_index is None:
            class_index = int(tf.argmax(logits[0]))
        score = logits[0, class_index]
    gradients = tape.gradient(score, maps)
    weights = tf.reduce_mean(gradients, axis=(1, 2))        # one weight per feature map
    cam = tf.nn.relu(tf.reduce_sum(maps * weights[:, None, None, :], axis=-1))[0]
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., None], (28, 28), method="bilinear")[..., 0].numpy()
    probability = float(tf.nn.softmax(logits)[0, class_index])
    return cam, class_index, probability


def overlay(ax, image, cam, alpha=0.55):
    """Draw the picture and the heat map on top of it (red = strong evidence)."""
    ax.imshow(image[:, :, 0], cmap="gray_r")
    ax.imshow(cam, cmap="jet", alpha=alpha, vmin=0, vmax=1)
    ax.axis("off")


def occlusion_probability_drop(model, images, centres, size=7):
    """Cover a ``size`` x ``size`` square (set to background) around each centre and measure the
    drop in the probability of the originally predicted class."""
    base = model.predict(images, verbose=0)
    predicted = base.argmax(axis=1)
    covered = images.copy()
    half = size // 2
    for k, (row, col) in enumerate(centres):
        r0, c0 = max(0, row - half), max(0, col - half)
        covered[k, r0:r0 + size, c0:c0 + size, :] = 0.0
    after = model.predict(covered, verbose=0)
    index = np.arange(len(images))
    return base[index, predicted] - after[index, predicted]


def plot_gradcam_steps(image, maps, weights, raw_map, cam):
    """The Grad-CAM recipe in six panels: picture, two feature maps, their weights, weighted sum, result."""
    best, worst = int(np.argmax(weights)), int(np.argmin(weights))
    fig, axes = plt.subplots(1, 6, figsize=(16, 2.9), gridspec_kw={"width_ratios": [1, 1, 1, 1.6, 1, 1]})
    axes[0].imshow(image[:, :, 0], cmap="gray_r"); axes[0].set_title("1. the picture", fontsize=9)
    axes[1].imshow(maps[:, :, best], cmap="viridis")
    axes[1].set_title(f"2. feature map {best}\nweight {weights[best]:+.4f} (largest)", fontsize=9)
    axes[2].imshow(maps[:, :, worst], cmap="viridis")
    axes[2].set_title(f"feature map {worst}\nweight {weights[worst]:+.4f} (smallest)", fontsize=9)
    axes[3].bar(range(len(weights)), weights, color=["tab:red" if w > 0 else "tab:blue" for w in weights])
    axes[3].set_title("3. one weight per feature map", fontsize=9); axes[3].set_xlabel("feature map")
    limit = np.abs(raw_map).max()
    axes[4].imshow(raw_map, cmap="RdBu_r", vmin=-limit, vmax=limit)
    axes[4].set_title("4. weighted sum\n(red > 0, blue < 0)", fontsize=9)
    overlay(axes[5], image, cam); axes[5].set_title("5. keep the positive part,\nstretch, overlay", fontsize=9)
    for k in (0, 1, 2, 4):
        axes[k].axis("off")
    return fig
