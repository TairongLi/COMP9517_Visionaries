import cv2
import numpy as np

from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

_EPS = 1e-12
def _validate_image(image):
    """Validate one RGB uint8 image."""

    image = np.asarray(image)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Expected RGB image with shape (H, W, 3), got {image.shape}"
        )

    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            if image.size > 0 and image.max() <= 1.0:
                image = image * 255.0

        image = np.clip(image, 0, 255).astype(np.uint8)

    return image


def sift_descriptors(img, max_kp=None):
    """Extract SIFT descriptors from one RGB image.

    Args:
        img:
            RGB uint8 image.

        max_kp:
            Optional maximum number of descriptors.

    Returns:
        Float32 array with shape (K,128).

        If no keypoints are detected,
        returns an empty array with shape (0,128).
    """

    img = _validate_image(img)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY,
    )

    sift = cv2.SIFT_create(
        nfeatures=500
    )
    keypoints, descriptors = sift.detectAndCompute(
        gray,
        None,
    )

    if descriptors is None:
        return np.empty(
            (0, 128),
            dtype=np.float32,
        )

    descriptors = descriptors.astype(
        np.float32,
        copy=False,
    )

    if (
        max_kp is not None
        and len(descriptors) > max_kp
    ):
        descriptors = descriptors[:max_kp]

    return descriptors


def sample_train_descriptors(
    train_images,
    per_image=100,
    seed=0,
):
    """Randomly sample SIFT descriptors from training images."""

    if per_image <= 0:
        raise ValueError(
            "per_image must be positive"
        )

    rng = np.random.default_rng(seed)

    descriptor_list = []

    for image in train_images:

        descriptors = sift_descriptors(image)

        if len(descriptors) == 0:
            continue

        if len(descriptors) > per_image:

            index = rng.choice(
                len(descriptors),
                per_image,
                replace=False,
            )

            descriptors = descriptors[index]

        descriptor_list.append(descriptors)

    if not descriptor_list:
        return np.empty(
            (0, 128),
            dtype=np.float32,
        )

    return np.vstack(
        descriptor_list
    ).astype(
        np.float32,
        copy=False,
    )

def build_codebook(
    descriptors,
    k=512,
    seed=0,
):
    """Build a visual vocabulary using MiniBatch KMeans.

    Args:
        descriptors:
            Float32 array with shape (N,128).

        k:
            Vocabulary size.

        seed:
            Random seed.

    Returns:
        A trained MiniBatchKMeans model.
    """

    if len(descriptors) == 0:
        raise ValueError(
            "Cannot build a codebook from empty descriptors."
        )

    codebook = MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        batch_size=4096,
        n_init=10,
    )

    codebook.fit(descriptors)

    return codebook


def encode_bovw(
    img_descriptors,
    codebook,
    k,
    normalize="hellinger",
):
    """Encode SIFT descriptors into one BoVW histogram."""

    histogram = np.zeros(
        k,
        dtype=np.float32,
    )

    if len(img_descriptors) == 0:
        return histogram

    visual_words = codebook.predict(
        img_descriptors
    )

    histogram = np.bincount(
        visual_words,
        minlength=k,
    ).astype(np.float32)

    normalize = normalize.lower()

    if normalize == "l1":

        histogram /= (
            histogram.sum() + _EPS
        )

    elif normalize == "l2":

        histogram /= (
            np.linalg.norm(histogram) + _EPS
        )

    elif normalize == "hellinger":

        # L1
        histogram /= (
            histogram.sum() + _EPS
        )

        # Power Normalization
        histogram = np.sqrt(histogram)

        # L2
        histogram /= (
            np.linalg.norm(histogram) + _EPS
        )

    else:
        raise ValueError(
            f"Unknown normalization: {normalize}"
        )

    return histogram.astype(
        np.float32,
        copy=False,
    )


def encode_split(
    images,
    codebook,
    k,
    **kwargs,
):
    """Encode one dataset split into BoVW features."""

    features = np.zeros(
        (len(images), k),
        dtype=np.float32,
    )

    for i, image in enumerate(images):

        descriptors = sift_descriptors(image)

        if len(descriptors) > 0:

            features[i] = encode_bovw(
                descriptors,
                codebook,
                k,
                **kwargs,
            )

    return features
def build_classifier(
    name="linear_svm",
    seed=0,
):
    """Build a classifier for BoVW features."""

    name = name.lower()

    if name == "linear_svm":
        return LinearSVC(
            random_state=seed,
        )

    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=seed,
            n_jobs=-1,
        )

    raise ValueError(
        f"Unknown classifier: {name}"
    )


def scores_of(
    clf,
    X,
):
    """Return class scores for evaluation.

    Returns:
        (N, C) float array.
    """

    if hasattr(clf, "decision_function"):

        score = clf.decision_function(X)

        if score.ndim == 1:
            score = np.column_stack(
                [-score, score]
            )

        return score.astype(
            np.float32,
            copy=False,
        )

    if hasattr(clf, "predict_proba"):

        return clf.predict_proba(X).astype(
            np.float32,
            copy=False,
        )

    raise ValueError(
        "Classifier does not provide scores."
    )