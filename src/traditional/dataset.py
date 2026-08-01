"""Dataset utilities for traditional computer-vision methods.

The module reads a shared manifest and loads RGB images on demand.
It does not depend on PyTorch.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


REQUIRED_COLUMNS = {
    "filepath",
    "class_id",
    "species_name",
    "split",
}

VALID_SPLITS = {
    "train",
    "validation",
    "test",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_manifest_from_folders(
    train_dir,
    test_dir,
    validation_per_class=10,
    seed=42,
):
    """Build the shared split table directly from the selected image folders.

    The selected train_mini directory contains 50 images per class. A fixed
    random subset is used for validation and the official iNat validation
    directory is kept exclusively as the final test set.
    """

    train_dir = Path(train_dir).resolve()
    test_dir = Path(test_dir).resolve()

    if not train_dir.is_dir():
        raise NotADirectoryError(f"Training directory does not exist: {train_dir}")
    if not test_dir.is_dir():
        raise NotADirectoryError(f"Test directory does not exist: {test_dir}")
    if validation_per_class <= 0:
        raise ValueError("validation_per_class must be positive")

    train_classes = {path.name: path for path in train_dir.iterdir() if path.is_dir()}
    test_classes = {path.name: path for path in test_dir.iterdir() if path.is_dir()}

    if set(train_classes) != set(test_classes):
        missing_test = sorted(set(train_classes) - set(test_classes))
        missing_train = sorted(set(test_classes) - set(train_classes))
        raise ValueError(
            "Training and test class folders do not match. "
            f"Missing from test: {missing_test[:5]}; "
            f"missing from train: {missing_train[:5]}"
        )

    rng = np.random.default_rng(seed)
    rows = []

    for class_id, folder_name in enumerate(sorted(train_classes)):
        species_name = " ".join(folder_name.split("_")[-2:])
        train_files = sorted(
            path
            for path in train_classes[folder_name].iterdir()
            if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        test_files = sorted(
            path
            for path in test_classes[folder_name].iterdir()
            if path.suffix.lower() in IMAGE_EXTENSIONS
        )

        if len(train_files) <= validation_per_class:
            raise ValueError(
                f"{folder_name} has only {len(train_files)} train_mini images"
            )

        validation_indices = set(
            rng.choice(
                len(train_files),
                size=validation_per_class,
                replace=False,
            ).tolist()
        )

        for index, path in enumerate(train_files):
            rows.append(
                {
                    "filepath": str(path),
                    "class_id": class_id,
                    "species_name": species_name,
                    "split": (
                        "validation" if index in validation_indices else "train"
                    ),
                }
            )

        for path in test_files:
            rows.append(
                {
                    "filepath": str(path),
                    "class_id": class_id,
                    "species_name": species_name,
                    "split": "test",
                }
            )

    dataframe = pd.DataFrame(rows)
    _validate_class_mapping(dataframe)
    return dataframe


def load_manifest(manifest_path):
    """Load and validate the shared dataset manifest.

    Required CSV columns:
        filepath: Relative path from image_root to the image.
        class_id: Continuous integer label from 0 to C - 1.
        species_name: Human-readable species name.
        split: train, validation, or test.

    Args:
        manifest_path: Path to manifest.csv.

    Returns:
        A validated pandas DataFrame.
    """

    manifest_path = Path(manifest_path)

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest file does not exist: {manifest_path}"
        )

    dataframe = pd.read_csv(manifest_path)

    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "Manifest is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if dataframe.empty:
        raise ValueError("Manifest is empty")

    dataframe = dataframe.copy()

    # Standardize text fields.
    dataframe["filepath"] = dataframe["filepath"].astype(str)
    dataframe["species_name"] = dataframe["species_name"].astype(str)
    dataframe["split"] = dataframe["split"].astype(str).str.lower()

    # Ensure class IDs are valid integers.
    if dataframe["class_id"].isna().any():
        raise ValueError("Manifest contains missing class_id values")

    dataframe["class_id"] = dataframe["class_id"].astype(np.int64)

    if (dataframe["class_id"] < 0).any():
        raise ValueError("class_id values cannot be negative")

    invalid_splits = set(dataframe["split"].unique()) - VALID_SPLITS

    if invalid_splits:
        raise ValueError(
            f"Invalid split values: {sorted(invalid_splits)}. "
            f"Expected: {sorted(VALID_SPLITS)}"
        )

    _validate_class_mapping(dataframe)

    return dataframe.reset_index(drop=True)


def _validate_class_mapping(dataframe):
    """Check that class IDs and species names have a one-to-one mapping."""

    id_to_species_counts = (
        dataframe.groupby("class_id")["species_name"].nunique()
    )

    if (id_to_species_counts != 1).any():
        bad_ids = id_to_species_counts[
            id_to_species_counts != 1
        ].index.tolist()

        raise ValueError(
            "Some class_id values map to multiple species names: "
            f"{bad_ids[:10]}"
        )

    species_to_id_counts = (
        dataframe.groupby("species_name")["class_id"].nunique()
    )

    if (species_to_id_counts != 1).any():
        bad_species = species_to_id_counts[
            species_to_id_counts != 1
        ].index.tolist()

        raise ValueError(
            "Some species names map to multiple class IDs: "
            f"{bad_species[:10]}"
        )

    class_ids = sorted(dataframe["class_id"].unique().tolist())
    expected_ids = list(range(len(class_ids)))

    if class_ids != expected_ids:
        raise ValueError(
            "class_id values must be continuous from 0 to C-1. "
            f"Found {class_ids[:10]}..."
        )


def get_class_names(dataframe):
    """Return class names ordered by class_id."""

    class_mapping = (
        dataframe[["class_id", "species_name"]]
        .drop_duplicates()
        .sort_values("class_id")
    )

    return class_mapping["species_name"].tolist()


def get_split(dataframe, split):
    """Return manifest rows belonging to one data split."""

    split = split.lower()

    if split not in VALID_SPLITS:
        raise ValueError(
            f"Unknown split '{split}'. "
            f"Expected one of {sorted(VALID_SPLITS)}"
        )

    split_dataframe = (
        dataframe[dataframe["split"] == split]
        .copy()
        .reset_index(drop=True)
    )

    if split_dataframe.empty:
        raise ValueError(f"Split '{split}' contains no images")

    return split_dataframe


def load_rgb_image(image_path, resize=None):
    """Load one image as an RGB uint8 NumPy array.

    Args:
        image_path: Full path to an image.
        resize: Optional (height, width) output size.

    Returns:
        RGB uint8 array with shape (H, W, 3).
    """

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")

            if resize is not None:
                if len(resize) != 2:
                    raise ValueError(
                        "resize must be a (height, width) tuple"
                    )

                height, width = map(int, resize)

                if height <= 0 or width <= 0:
                    raise ValueError(
                        "resize values must be positive"
                    )

                # PIL expects (width, height).
                image = image.resize(
                    (width, height),
                    Image.Resampling.LANCZOS,
                )

            array = np.asarray(image, dtype=np.uint8)

    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"Failed to read image '{image_path}': {error}"
        ) from error

    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(
            f"Expected RGB image, got shape {array.shape}: {image_path}"
        )

    return array


def iter_split(
    dataframe,
    split,
    image_root,
    resize=None,
    limit=None,
):
    """Yield images and labels from one split without loading all into memory.

    Yields:
        image: RGB uint8 array.
        class_id: Integer class label.
        filepath: Relative image path.
    """

    rows = get_split(dataframe, split)
    image_root = Path(image_root)

    if not image_root.is_dir():
        raise NotADirectoryError(
            f"Image root does not exist: {image_root}"
        )

    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")

        rows = rows.iloc[:limit]

    for row in rows.itertuples(index=False):
        relative_path = Path(row.filepath)
        full_path = relative_path if relative_path.is_absolute() else image_root / relative_path

        image = load_rgb_image(
            full_path,
            resize=resize,
        )

        yield image, int(row.class_id), str(row.filepath)


def count_images_per_split(dataframe):
    """Return the number of images in every split."""

    return (
        dataframe.groupby("split")
        .size()
        .reindex(sorted(VALID_SPLITS), fill_value=0)
        .rename("image_count")
    )


def count_images_per_class(dataframe):
    """Return image counts for every class and split."""

    return (
        dataframe.groupby(
            ["class_id", "species_name", "split"]
        )
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )


def verify_image_paths(
    dataframe,
    image_root,
    max_errors=20,
):
    """Check whether image paths in the manifest exist.

    Args:
        dataframe: Validated manifest DataFrame.
        image_root: Root directory used by manifest file paths.
        max_errors: Maximum number of missing paths to report.

    Returns:
        List of missing relative paths.
    """

    image_root = Path(image_root)
    missing_paths = []

    for filepath in dataframe["filepath"]:
        if not (image_root / filepath).is_file():
            missing_paths.append(filepath)

            if len(missing_paths) >= max_errors:
                break

    return missing_paths
