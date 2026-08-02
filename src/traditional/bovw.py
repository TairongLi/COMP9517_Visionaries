"""SIFT Bag-of-Visual-Words utilities for the traditional-C pipeline.

The intended workflow is::

    sampled = sample_train_descriptors(train_images)
    codebook = build_codebook(sampled, k=256)
    X_train = encode_split(train_images, codebook, k=256)
    X_valid = encode_split(valid_images, codebook, k=256)
    classifier = build_classifier("linear_svm")
    classifier.fit(X_train, y_train)

Only training images should be passed to ``sample_train_descriptors`` and
``build_codebook``.  Validation and test images are encoded with the already
fitted training codebook to prevent data leakage.
"""

from collections.abc import Iterable

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC


SIFT_DESCRIPTOR_SIZE = 128
DEFAULT_SIFT_FEATURES = 500
SUPPORTED_NORMALIZATIONS = {"l1", "l2", "hellinger"}
_EPS = np.finfo(np.float32).eps


def _positive_integer(value, name):
    """Validate and return a positive integer parameter."""

    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _validate_image(image):
    """Return an image as a non-empty, contiguous RGB uint8 array.

    Float images in the conventional [0, 1] range are scaled to [0, 255].
    Other numeric images are clipped to [0, 255].
    """

    image = np.asarray(image)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Expected an RGB image with shape (H, W, 3), got {image.shape}"
        )
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("Image height and width must be non-zero")
    if not np.issubdtype(image.dtype, np.number):
        raise TypeError(f"Image must have a numeric dtype, got {image.dtype}")
    if not np.isfinite(image).all():
        raise ValueError("Image contains NaN or infinite values")

    if image.dtype != np.uint8:
        image = image.astype(np.float32, copy=False)
        if image.size and image.min() >= 0.0 and image.max() <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)

    return np.ascontiguousarray(image)


def _validate_descriptors(descriptors, *, allow_empty=True):
    """Validate an ``(N, 128)`` SIFT descriptor matrix."""

    descriptors = np.asarray(descriptors)
    if descriptors.ndim != 2 or descriptors.shape[1] != SIFT_DESCRIPTOR_SIZE:
        raise ValueError(
            "Expected SIFT descriptors with shape (N, 128), "
            f"got {descriptors.shape}"
        )
    if not allow_empty and descriptors.shape[0] == 0:
        raise ValueError("Descriptor matrix must not be empty")
    if not np.issubdtype(descriptors.dtype, np.number):
        raise TypeError("Descriptors must have a numeric dtype")
    if not np.isfinite(descriptors).all():
        raise ValueError("Descriptors contain NaN or infinite values")
    return np.ascontiguousarray(descriptors, dtype=np.float32)


def sift_descriptors(img, max_kp=None):
    """Extract SIFT descriptors from one RGB image.

    Args:
        img: RGB image with shape ``(H, W, 3)``.
        max_kp: Maximum number of returned descriptors. Defaults to 500.

    Returns:
        A float32 array with shape ``(K, 128)``. If SIFT detects no
        keypoints, the result has shape ``(0, 128)``.
    """

    image = _validate_image(img)
    nfeatures = (
        DEFAULT_SIFT_FEATURES
        if max_kp is None
        else _positive_integer(max_kp, "max_kp")
    )

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    if descriptors is None or not keypoints:
        return np.empty((0, SIFT_DESCRIPTOR_SIZE), dtype=np.float32)

    # OpenCV normally respects nfeatures, but slicing keeps the public
    # contract true across OpenCV versions.
    descriptors = descriptors[:nfeatures]
    return np.ascontiguousarray(descriptors, dtype=np.float32)


def sample_train_descriptors(train_images, per_image=100, seed=0):
    """Randomly sample up to ``per_image`` SIFT vectors from each image.

    Sampling gives every training image a similar contribution to the visual
    vocabulary and prevents large, highly textured images from dominating it.
    Images without SIFT keypoints are safely skipped.
    """

    per_image = _positive_integer(per_image, "per_image")
    if not isinstance(train_images, Iterable):
        raise TypeError("train_images must be an iterable of RGB images")

    rng = np.random.default_rng(seed)
    sampled = []

    for image in train_images:
        descriptors = sift_descriptors(image)
        if len(descriptors) == 0:
            continue
        if len(descriptors) > per_image:
            indices = rng.choice(len(descriptors), size=per_image, replace=False)
            descriptors = descriptors[indices]
        sampled.append(descriptors)

    if not sampled:
        return np.empty((0, SIFT_DESCRIPTOR_SIZE), dtype=np.float32)
    return np.ascontiguousarray(np.vstack(sampled), dtype=np.float32)


def build_codebook(descriptors, k=512, seed=0):
    """Fit a MiniBatch K-Means visual vocabulary to training descriptors.

    Args:
        descriptors: Training-only SIFT descriptors with shape ``(N, 128)``.
        k: Number of visual words (cluster centres).
        seed: Random seed used by MiniBatch K-Means.

    Returns:
        A fitted :class:`~sklearn.cluster.MiniBatchKMeans` instance.
    """

    descriptors = _validate_descriptors(descriptors, allow_empty=False)
    k = _positive_integer(k, "k")
    if k > len(descriptors):
        raise ValueError(
            f"k ({k}) cannot exceed the number of descriptors "
            f"({len(descriptors)})"
        )

    codebook = MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        batch_size=min(4096, max(256, len(descriptors))),
        n_init=10,
        reassignment_ratio=0.01,
    )
    codebook.fit(descriptors)
    return codebook


def encode_bovw(img_descriptors, codebook, k, normalize="hellinger"):
    """Encode one image as a normalized Bag-of-Visual-Words histogram.

    ``hellinger`` applies L1 normalization followed by element-wise square
    root and L2 normalization (the RootSIFT/Root-BoVW style transform).
    """

    descriptors = _validate_descriptors(img_descriptors)
    k = _positive_integer(k, "k")
    normalize = str(normalize).lower()
    if normalize not in SUPPORTED_NORMALIZATIONS:
        raise ValueError(
            f"Unknown normalization '{normalize}'. Expected one of "
            f"{sorted(SUPPORTED_NORMALIZATIONS)}"
        )
    if not hasattr(codebook, "cluster_centers_"):
        raise ValueError("codebook must be fitted before encoding images")
    if int(codebook.n_clusters) != k:
        raise ValueError(
            f"k ({k}) does not match codebook.n_clusters "
            f"({codebook.n_clusters})"
        )

    histogram = np.zeros(k, dtype=np.float32)
    if len(descriptors) == 0:
        return histogram

    visual_words = codebook.predict(descriptors)
    histogram = np.bincount(visual_words, minlength=k).astype(np.float32)

    if normalize == "l1":
        histogram /= histogram.sum() + _EPS
    elif normalize == "l2":
        histogram /= np.linalg.norm(histogram) + _EPS
    else:  # Hellinger kernel map.
        histogram /= histogram.sum() + _EPS
        np.sqrt(histogram, out=histogram)
        histogram /= np.linalg.norm(histogram) + _EPS

    return histogram


def encode_split(images, codebook, k, **kwargs):
    """Encode an iterable of RGB images into an ``(N, k)`` feature matrix."""

    if not isinstance(images, Iterable):
        raise TypeError("images must be an iterable of RGB images")
    if not hasattr(images, "__len__"):
        images = list(images)

    k = _positive_integer(k, "k")
    features = np.zeros((len(images), k), dtype=np.float32)

    for index, image in enumerate(images):
        descriptors = sift_descriptors(image)
        features[index] = encode_bovw(
            descriptors,
            codebook,
            k,
            **kwargs,
        )

    return features


def build_classifier(name="linear_svm", seed=0):
    """Create one of the two classifiers required by traditional-C.

    ``linear_svm`` is normally the stronger and faster BoVW baseline.
    ``random_forest`` is provided as the required classical comparison.
    """

    name = str(name).lower()
    if name == "linear_svm":
        return LinearSVC(random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(
        f"Unknown classifier '{name}'. Expected 'linear_svm' or "
        "'random_forest'"
    )


def scores_of(clf, X):
    """Return an ``(N, C)`` class-score matrix for either classifier.

    LinearSVC produces decision-function values, while Random Forest produces
    probabilities. Both are suitable for ``argmax`` and top-k evaluation,
    although SVM decision values are not calibrated probabilities.
    """

    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got shape {X.shape}")

    if hasattr(clf, "decision_function"):
        scores = np.asarray(clf.decision_function(X))
        if scores.ndim == 1:
            scores = np.column_stack((-scores, scores))
    elif hasattr(clf, "predict_proba"):
        scores = np.asarray(clf.predict_proba(X))
    else:
        raise TypeError(
            "Classifier must implement decision_function or predict_proba"
        )

    if scores.ndim != 2 or scores.shape[0] != X.shape[0]:
        raise RuntimeError(f"Classifier returned invalid score shape {scores.shape}")
    return scores.astype(np.float32, copy=False)
