"""Run handcrafted descriptors or SIFT Bag-of-Visual-Words experiments.

Examples:
    python -m src.traditional.run_features --mode descriptor --descriptor lbp
    python -m src.traditional.run_features --mode bovw --k 512
"""

import argparse
import time

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm

from ..metrics import save_result
from . import bovw, features
from .dataset import (
    build_manifest_from_folders,
    get_class_names,
    get_split,
    load_manifest,
    load_rgb_image,
)


def select_rows(dataframe, split, limit_per_class=None):
    """Select one split and optionally keep a small number per class."""

    rows = get_split(dataframe, split)
    if limit_per_class is not None:
        rows = (
            rows.groupby("class_id", group_keys=False)
            .head(limit_per_class)
            .reset_index(drop=True)
        )
    return rows


def load_experiment_manifest(args):
    """Load a CSV manifest or construct one from the two image folders."""

    if args.manifest:
        return load_manifest(args.manifest)
    return build_manifest_from_folders(
        args.train_dir,
        args.test_dir,
        validation_per_class=args.validation_per_class,
        seed=args.seed,
    )


def descriptor_parameters(args):
    """Return the selected descriptor's configurable parameters."""

    if args.descriptor == "color_hist":
        return {"bins": args.color_bins, "color_space": args.color_space}
    if args.descriptor == "color_moments":
        return {"grid_size": args.grid_size, "color_space": args.color_space}
    if args.descriptor == "lbp":
        return {"points": args.lbp_points, "radius": args.lbp_radius}
    if args.descriptor == "hog":
        return {
            "image_size": (args.hog_size, args.hog_size),
            "orientations": args.hog_orientations,
            "pixels_per_cell": (args.hog_cell_size, args.hog_cell_size),
        }
    if args.descriptor == "color_lbp":
        return {"color_bins": args.color_bins, "color_space": args.color_space}
    return {}


def extract_descriptor_rows(rows, descriptor, parameters=None):
    """Load images one at a time and extract one global descriptor."""

    output = []
    for row in tqdm(
        rows.itertuples(index=False),
        total=len(rows),
        desc=f"Extracting {descriptor}",
    ):
        image = load_rgb_image(row.filepath)
        output.append(
            features.extract_feature(
                image,
                descriptor,
                **(parameters or {}),
            )
        )
    return np.stack(output).astype(np.float32)


def extract_bovw_rows(rows, codebook, k, normalization):
    """Encode image rows without keeping all source images in memory."""

    output = np.zeros((len(rows), k), dtype=np.float32)
    for index, row in enumerate(
        tqdm(rows.itertuples(index=False), total=len(rows), desc="Encoding BoVW")
    ):
        image = load_rgb_image(row.filepath)
        descriptors = bovw.sift_descriptors(image)
        output[index] = bovw.encode_bovw(
            descriptors,
            codebook,
            k,
            normalize=normalization,
        )
    return output


def train_codebook(rows, k, per_image, codebook_images, seed):
    """Sample a bounded training subset and build the visual vocabulary."""

    sample_count = min(codebook_images, len(rows))
    sampled_rows = rows.sample(n=sample_count, random_state=seed)
    sampled_images = (
        load_rgb_image(row.filepath)
        for row in tqdm(
            sampled_rows.itertuples(index=False),
            total=sample_count,
            desc="Sampling SIFT",
        )
    )
    descriptors = bovw.sample_train_descriptors(
        sampled_images,
        per_image=per_image,
        seed=seed,
    )
    return bovw.build_codebook(descriptors, k=k, seed=seed)


def align_scores(classifier, feature_matrix, num_classes):
    """Align classifier score columns with the global continuous class IDs."""

    score = bovw.scores_of(classifier, feature_matrix)
    classes = np.asarray(classifier.classes_, dtype=np.int64)
    aligned = np.full((len(feature_matrix), num_classes), -np.inf, dtype=np.float32)
    aligned[:, classes] = score
    return aligned


def run_descriptor(args, dataframe, class_names):
    """Run one global handcrafted descriptor experiment."""

    train_rows = select_rows(dataframe, "train", args.limit_per_class)
    eval_rows = select_rows(dataframe, args.evaluate_split, args.limit_per_class)
    parameters = descriptor_parameters(args)

    start = time.perf_counter()
    train_features = extract_descriptor_rows(
        train_rows,
        args.descriptor,
        parameters,
    )
    classifier = make_pipeline(
        StandardScaler(copy=False),
        SGDClassifier(
            loss="hinge",
            alpha=args.sgd_alpha,
            max_iter=args.sgd_max_iter,
            tol=1e-3,
            n_jobs=-1,
            random_state=args.seed,
            average=True,
        ),
    )
    classifier.fit(train_features, train_rows["class_id"].to_numpy())
    train_s = time.perf_counter() - start

    start = time.perf_counter()
    eval_features = extract_descriptor_rows(
        eval_rows,
        args.descriptor,
        parameters,
    )
    y_score = align_scores(classifier, eval_features, len(class_names))
    test_s = time.perf_counter() - start

    return eval_rows, y_score, train_s, test_s


def run_bovw(args, dataframe, class_names):
    """Run SIFT, BoVW and a classical classifier."""

    train_rows = select_rows(dataframe, "train", args.limit_per_class)
    eval_rows = select_rows(dataframe, args.evaluate_split, args.limit_per_class)

    start = time.perf_counter()
    codebook = train_codebook(
        train_rows,
        args.k,
        args.per_image,
        args.codebook_images,
        args.seed,
    )
    train_features = extract_bovw_rows(
        train_rows,
        codebook,
        args.k,
        args.normalization,
    )
    classifier = bovw.build_classifier(args.classifier, seed=args.seed)
    classifier.fit(train_features, train_rows["class_id"].to_numpy())
    train_s = time.perf_counter() - start

    start = time.perf_counter()
    eval_features = extract_bovw_rows(
        eval_rows,
        codebook,
        args.k,
        args.normalization,
    )
    y_score = align_scores(classifier, eval_features, len(class_names))
    test_s = time.perf_counter() - start

    return eval_rows, y_score, train_s, test_s


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["descriptor", "bovw"], required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--train-dir", default="../select_train_n_val/select_train_mini")
    parser.add_argument("--test-dir", default="../select_train_n_val/select_val_mini")
    parser.add_argument("--evaluate-split", choices=["validation", "test"], default="validation")
    parser.add_argument("--validation-per-class", type=int, default=10)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--descriptor", choices=sorted(features.FEATURES), default="color_hist")
    parser.add_argument("--color-bins", type=int, default=32)
    parser.add_argument("--color-space", choices=["RGB", "HSV"], default="HSV")
    parser.add_argument("--grid-size", type=int, default=1)
    parser.add_argument("--lbp-points", type=int, default=8)
    parser.add_argument("--lbp-radius", type=float, default=1.0)
    parser.add_argument("--hog-size", type=int, default=128)
    parser.add_argument("--hog-orientations", type=int, default=9)
    parser.add_argument("--hog-cell-size", type=int, default=16)
    parser.add_argument("--classifier", choices=["linear_svm", "random_forest"], default="linear_svm")
    parser.add_argument("--sgd-alpha", type=float, default=1e-4)
    parser.add_argument("--sgd-max-iter", type=int, default=100)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--per-image", type=int, default=100)
    parser.add_argument("--codebook-images", type=int, default=2000)
    parser.add_argument("--normalization", choices=["l1", "l2", "hellinger"], default="hellinger")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--name", default=None)
    return parser.parse_args()


def main():
    args = parse_arguments()
    dataframe = load_experiment_manifest(args)
    class_names = get_class_names(dataframe)

    if args.mode == "descriptor":
        rows, y_score, train_s, test_s = run_descriptor(
            args,
            dataframe,
            class_names,
        )
        default_name = f"{args.descriptor}_sgd_svm"
    else:
        rows, y_score, train_s, test_s = run_bovw(
            args,
            dataframe,
            class_names,
        )
        default_name = f"bovw_k{args.k}_{args.classifier}"

    y_true = rows["class_id"].to_numpy(dtype=np.int64)
    y_pred = np.argmax(y_score, axis=1)
    method_name = args.name or default_name
    path, scores = save_result(
        args.output_dir,
        method_name,
        y_true,
        y_pred,
        y_score,
        class_names,
        {"train_s": train_s, "test_s": test_s},
        filepaths=rows["filepath"].tolist(),
    )

    print(method_name)
    for name, value in scores.items():
        print(f"{name}: {value}")
    print("->", path)


if __name__ == "__main__":
    main()
