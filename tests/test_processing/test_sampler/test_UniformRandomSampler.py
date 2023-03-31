# pylint: disable-all

from unittest import TestCase

import numpy as np

import hrosailing.processing.sampler as smp


class TestUniformRandomSampler(TestCase):
    def setUp(self) -> None:
        self.n_samples = 2
        self.pts = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1], [0, 0]])

    def test_UniformRandomSampler_init_Error(self):
        """
        ValueError occurs is n_samples is non-positive.
        """
        with self.assertRaises(ValueError):
            smp.UniformRandomSampler(-1)

    def test_UniformRandomSampler_sample_convex_hull(self):
        """
        Sampled points lay in the convex hull of pts.
        """

        result = smp.UniformRandomSampler(self.n_samples).sample(self.pts)
        for coordinate in result.flatten():
            self.assertGreaterEqual(coordinate, -1)
            self.assertGreaterEqual(-coordinate, -1)

    def test_sample_amount(self):
        """
        Amount of sampled points is n_samples.
        """
        result = len(smp.UniformRandomSampler(self.n_samples).sample(self.pts))
        self.assertEqual(
            result,
            self.n_samples,
            f"Expected {self.n_samples} samples but got {result} samples.",
        )

    def test_sample_edge_empty_pts(self):
        """
        EdgeCase: Sampling on empty pts.
        """
        with self.assertRaises(ValueError):
            smp.UniformRandomSampler(self.n_samples).sample(np.empty((3, 0)))
