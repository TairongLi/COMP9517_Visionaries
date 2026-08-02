"""Unit tests for the traditional-C SIFT Bag-of-Visual-Words module."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "traditional" / "bovw.py"
MODULE_SPEC = importlib.util.spec_from_file_location("traditional_bovw", MODULE_PATH)
bovw = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(bovw)


class TestBovw(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        cls.images = []
        cls.labels = []
        for label in range(3):
            for _ in range(3):
                image = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
                image[label::6, :] = 255
                cls.images.append(image)
                cls.labels.append(label)
        cls.labels = np.asarray(cls.labels)

    def test_sift_output_and_blank_image(self):
        blank = np.zeros((32, 32, 3), dtype=np.uint8)
        self.assertEqual(bovw.sift_descriptors(blank).shape, (0, 128))

        descriptors = bovw.sift_descriptors(self.images[0], max_kp=10)
        self.assertEqual(descriptors.shape[1], 128)
        self.assertLessEqual(len(descriptors), 10)
        self.assertEqual(descriptors.dtype, np.float32)

    def test_descriptor_sampling_is_bounded_and_reproducible(self):
        first = bovw.sample_train_descriptors(
            self.images,
            per_image=10,
            max_total=25,
            seed=7,
        )
        second = bovw.sample_train_descriptors(
            self.images,
            per_image=10,
            max_total=25,
            seed=7,
        )
        self.assertEqual(first.shape, (25, 128))
        np.testing.assert_array_equal(first, second)

    def test_encoding_normalizations(self):
        descriptors = bovw.sample_train_descriptors(
            self.images,
            per_image=10,
            max_total=60,
            seed=7,
        )
        codebook = bovw.build_codebook(descriptors, k=6, seed=7)
        image_descriptors = bovw.sift_descriptors(self.images[0], max_kp=10)

        l1 = bovw.encode_bovw(image_descriptors, codebook, 6, normalize="l1")
        l2 = bovw.encode_bovw(image_descriptors, codebook, 6, normalize="l2")
        hellinger = bovw.encode_bovw(
            image_descriptors,
            codebook,
            6,
            normalize="hellinger",
        )
        self.assertAlmostEqual(float(l1.sum()), 1.0, places=5)
        self.assertAlmostEqual(float(np.linalg.norm(l2)), 1.0, places=5)
        self.assertAlmostEqual(float(np.linalg.norm(hellinger)), 1.0, places=5)

    def test_both_classifiers_return_multiclass_scores(self):
        descriptors = bovw.sample_train_descriptors(
            self.images,
            per_image=10,
            max_total=60,
            seed=7,
        )
        codebook = bovw.build_codebook(descriptors, k=6, seed=7)
        features = bovw.encode_split(self.images, codebook, 6)

        classifiers = (
            bovw.build_classifier("linear_svm", seed=7, C=0.5),
            bovw.build_classifier(
                "random_forest",
                seed=7,
                n_estimators=20,
            ),
        )
        for classifier in classifiers:
            classifier.fit(features, self.labels)
            scores = bovw.scores_of(classifier, features)
            self.assertEqual(scores.shape, (len(self.images), 3))
            self.assertTrue(np.isfinite(scores).all())

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            bovw.build_codebook(np.empty((0, 128), np.float32), k=2)
        with self.assertRaises(ValueError):
            bovw.sift_descriptors(self.images[0], max_kp=0)
        with self.assertRaises(ValueError):
            bovw.build_classifier("nearest_neighbor")


if __name__ == "__main__":
    unittest.main()
