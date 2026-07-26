"""Handcrafted global feature descriptors for traditional image classification.

All descriptor functions accept an RGB uint8 image with shape (H, W, 3)
and return a one-dimensional float32 NumPy array.
"""

import cv2
import numpy as np
from scipy.stats import skew
from skimage.feature import hog, local_binary_pattern


_EPS = 1e-12


def _validate_image(image):
    """Validate and convert an input image to RGB uint8 format."""

    image = np.asarray(image)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Expected an RGB image with shape (H, W, 3), got {image.shape}"
        )

    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            # Support both [0, 1] and [0, 255] floating-point images.
            if image.size > 0 and image.max() <= 1.0:
                image = image * 255.0

        image = np.clip(image, 0, 255).astype(np.uint8)

    return image


def _l2_normalize(feature):
    """Return an L2-normalized one-dimensional float32 feature."""

    feature = np.asarray(feature, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(feature)

    if norm > _EPS:
        feature = feature / norm

    return feature.astype(np.float32, copy=False)


def color_histogram(image, bins=32, color_space="HSV"):
    """Extract a normalized colour histogram.

    A separate histogram is computed for each colour channel. Each channel
    histogram is L1-normalized before concatenation, and the final vector is
    L2-normalized.

    Args:
        image: RGB uint8 image with shape (H, W, 3).
        bins: Number of histogram bins per channel.
        color_space: Either "RGB" or "HSV".

    Returns:
        A float32 vector with length 3 * bins.
    """

    image = _validate_image(image)

    if not isinstance(bins, int) or bins <= 0:
        raise ValueError("bins must be a positive integer")

    color_space = color_space.upper()

    if color_space == "RGB":
        converted = image
        channel_ranges = [(0, 256), (0, 256), (0, 256)]
    elif color_space == "HSV":
        converted = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # OpenCV represents hue in [0, 179] and S/V in [0, 255].
        channel_ranges = [(0, 180), (0, 256), (0, 256)]
    else:
        raise ValueError("color_space must be either 'RGB' or 'HSV'")

    histograms = []

    for channel_index, value_range in enumerate(channel_ranges):
        histogram, _ = np.histogram(
            converted[:, :, channel_index],
            bins=bins,
            range=value_range,
        )

        histogram = histogram.astype(np.float32)
        histogram /= histogram.sum() + _EPS
        histograms.append(histogram)

    return _l2_normalize(np.concatenate(histograms))


def color_moments(image, grid_size=1, color_space="HSV"):
    """Extract colour mean, standard deviation and skewness.

    The image can be divided into a grid to retain coarse spatial information.
    Three moments are calculated for every channel in every grid region.

    Args:
        image: RGB uint8 image with shape (H, W, 3).
        grid_size: Number of rows and columns in the spatial grid.
        color_space: Either "RGB" or "HSV".

    Returns:
        A float32 vector with length grid_size * grid_size * 3 * 3.
    """

    image = _validate_image(image)

    if not isinstance(grid_size, int) or grid_size <= 0:
        raise ValueError("grid_size must be a positive integer")

    color_space = color_space.upper()

    if color_space == "RGB":
        converted = image
    elif color_space == "HSV":
        converted = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    else:
        raise ValueError("color_space must be either 'RGB' or 'HSV'")

    # Scaling values to [0, 1] makes moments from different channels comparable.
    converted = converted.astype(np.float32)

    if color_space == "HSV":
        converted[:, :, 0] /= 179.0
        converted[:, :, 1:] /= 255.0
    else:
        converted /= 255.0

    height, width = converted.shape[:2]

    if grid_size > height or grid_size > width:
        raise ValueError(
            f"grid_size={grid_size} is too large for image size "
            f"({height}, {width})"
        )

    row_boundaries = np.linspace(0, height, grid_size + 1, dtype=int)
    column_boundaries = np.linspace(0, width, grid_size + 1, dtype=int)

    moments = []

    for row in range(grid_size):
        for column in range(grid_size):
            region = converted[
                row_boundaries[row] : row_boundaries[row + 1],
                column_boundaries[column] : column_boundaries[column + 1],
            ]

            for channel_index in range(3):
                values = region[:, :, channel_index].reshape(-1)

                channel_mean = float(np.mean(values))
                channel_std = float(np.std(values))

                # Constant regions have undefined skewness. Treat them as zero.
                if channel_std <= _EPS:
                    channel_skew = 0.0
                else:
                    channel_skew = float(skew(values, bias=False))
                    if not np.isfinite(channel_skew):
                        channel_skew = 0.0

                moments.extend([channel_mean, channel_std, channel_skew])

    return np.asarray(moments, dtype=np.float32)


def lbp_feature(image, points=8, radius=1, method="uniform"):
    """Extract a normalized Local Binary Pattern histogram.

    Args:
        image: RGB uint8 image with shape (H, W, 3).
        points: Number of circularly symmetric neighbour points.
        radius: Radius of the LBP neighbourhood.
        method: LBP method accepted by scikit-image.

    Returns:
        A one-dimensional normalized float32 histogram.
    """

    image = _validate_image(image)

    if not isinstance(points, int) or points <= 0:
        raise ValueError("points must be a positive integer")

    if radius <= 0:
        raise ValueError("radius must be positive")

    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    lbp_values = local_binary_pattern(
        grayscale,
        P=points,
        R=radius,
        method=method,
    )

    if method == "uniform":
        num_bins = points + 2
        histogram_range = (0, num_bins)
    elif method == "nri_uniform":
        num_bins = points * (points - 1) + 3
        histogram_range = (0, num_bins)
    elif method == "default":
        num_bins = 2**points
        histogram_range = (0, num_bins)
    else:
        raise ValueError(
            "Supported methods are 'uniform', 'nri_uniform', and 'default'"
        )

    histogram, _ = np.histogram(
        lbp_values.ravel(),
        bins=num_bins,
        range=histogram_range,
    )

    histogram = histogram.astype(np.float32)
    histogram /= histogram.sum() + _EPS

    return _l2_normalize(histogram)


def multiscale_lbp(
    image,
    scales=((8, 1), (16, 2), (24, 3)),
    method="uniform",
):
    """Extract and concatenate LBP histograms at multiple scales.

    Args:
        image: RGB uint8 image with shape (H, W, 3).
        scales: Sequence of (points, radius) configurations.
        method: LBP encoding method.

    Returns:
        An L2-normalized concatenated float32 feature.
    """

    if not scales:
        raise ValueError("scales cannot be empty")

    features = [
        lbp_feature(
            image,
            points=int(points),
            radius=float(radius),
            method=method,
        )
        for points, radius in scales
    ]

    return _l2_normalize(np.concatenate(features))


def hog_feature(
    image,
    image_size=(128, 128),
    orientations=9,
    pixels_per_cell=(16, 16),
    cells_per_block=(2, 2),
):
    """Extract a fixed-length Histogram of Oriented Gradients feature.

    Args:
        image: RGB uint8 image with shape (H, W, 3).
        image_size: Target size as (height, width).
        orientations: Number of orientation bins.
        pixels_per_cell: Cell size as (height, width).
        cells_per_block: Number of cells in each normalization block.

    Returns:
        A one-dimensional L2-normalized float32 feature.
    """

    image = _validate_image(image)

    if len(image_size) != 2 or min(image_size) <= 0:
        raise ValueError("image_size must contain two positive integers")

    if orientations <= 0:
        raise ValueError("orientations must be positive")

    target_height, target_width = map(int, image_size)

    # cv2.resize expects size in (width, height) order.
    resized = cv2.resize(
        image,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )

    grayscale = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    grayscale = grayscale.astype(np.float32) / 255.0

    feature = hog(
        grayscale,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        feature_vector=True,
    )

    return _l2_normalize(feature)


def color_lbp_feature(
    image,
    color_bins=32,
    color_space="HSV",
    lbp_scales=((8, 1), (16, 2), (24, 3)),
):
    """Fuse complementary colour and texture features.

    Colour and LBP features are independently normalized before concatenation
    so that neither component dominates only because of its numeric scale.

    Args:
        image: RGB uint8 image with shape (H, W, 3).
        color_bins: Number of bins per colour channel.
        color_space: Colour space used for the histogram.
        lbp_scales: Multi-scale LBP configurations.

    Returns:
        An L2-normalized concatenated float32 feature.
    """

    color_feature = color_histogram(
        image,
        bins=color_bins,
        color_space=color_space,
    )

    texture_feature = multiscale_lbp(
        image,
        scales=lbp_scales,
        method="uniform",
    )

    color_feature = _l2_normalize(color_feature)
    texture_feature = _l2_normalize(texture_feature)

    return _l2_normalize(
        np.concatenate([color_feature, texture_feature])
    )


FEATURES = {
    "color_hist": color_histogram,
    "color_moments": color_moments,
    "lbp": lbp_feature,
    "multiscale_lbp": multiscale_lbp,
    "hog": hog_feature,
    "color_lbp": color_lbp_feature,
}


def extract_feature(image, feature_name, **kwargs):
    """Extract one registered feature from one image."""

    if feature_name not in FEATURES:
        available = ", ".join(sorted(FEATURES))
        raise ValueError(
            f"Unknown feature '{feature_name}'. Available features: {available}"
        )

    feature = FEATURES[feature_name](image, **kwargs)
    feature = np.asarray(feature, dtype=np.float32).reshape(-1)

    if feature.size == 0:
        raise ValueError(f"Feature '{feature_name}' returned an empty vector")

    if not np.isfinite(feature).all():
        raise ValueError(
            f"Feature '{feature_name}' contains NaN or infinite values"
        )

    return feature


def extract_features(images, feature_name, **kwargs):
    """Extract one registered descriptor from a collection of images."""

    features = [
        extract_feature(image, feature_name, **kwargs)
        for image in images
    ]

    if not features:
        return np.empty((0, 0), dtype=np.float32)

    dimensions = {feature.shape[0] for feature in features}

    if len(dimensions) != 1:
        raise ValueError(
            f"Feature '{feature_name}' produced inconsistent dimensions: "
            f"{sorted(dimensions)}"
        )

    return np.stack(features).astype(np.float32, copy=False)