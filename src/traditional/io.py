"""Traditional data helpers kept as a compatibility import module."""

import numpy as np

from .dataset import (
    build_manifest_from_folders,
    count_images_per_class,
    count_images_per_split,
    get_class_names,
    get_split,
    iter_split,
    load_manifest,
    load_rgb_image,
    verify_image_paths,
)


def load_split(dataframe, split, image_root=".", resize=256, limit=None):
    """Load one small split into memory.

    Full experiments should extract features while iterating because loading
    tens of thousands of resized images at once uses too much memory.
    """

    images = []
    labels = []
    for image, class_id, _ in iter_split(
        dataframe,
        split,
        image_root,
        resize=(resize, resize) if resize else None,
        limit=limit,
    ):
        images.append(image)
        labels.append(class_id)
    return images, np.asarray(labels, dtype=np.int64)


__all__ = [
    "build_manifest_from_folders",
    "count_images_per_class",
    "count_images_per_split",
    "get_class_names",
    "get_split",
    "iter_split",
    "load_manifest",
    "load_rgb_image",
    "load_split",
    "verify_image_paths",
]
