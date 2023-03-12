from unittest import TestCase
import numpy as np

import hrosailing.processing.sampler as smp


class TestArchimedeanSampler(TestCase):
    def setUp(self) -> None:
        self.n_samples = 2
        self.pts = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1], [0, 0]])

    def test_init_Error(self):
        """
        ValueError if n_samples is non-positive.
        """
        with self.assertRaises(ValueError):
            smp.FibonacciSampler(0)

    def test_sample_convex_hull(self):
        """
        Sampled points lay in the convex hull of pts.
        """
        result = smp.FibonacciSampler(self.n_samples).sample(self.pts)
        for coordinate in result.flatten():
            self.assertGreaterEqual(coordinate, -1, "Sampled points are outside the convex hull of pts.")
            self.assertGreaterEqual(-coordinate, -1, "Sampled points are outside the convex hull of pts.")

    def test_sample_amount(self):
        """
        Amount of sampled points is n_samples.
        """
        result = len(smp.UniformRandomSampler(self.n_samples).sample(self.pts))
        self.assertEqual(result, self.n_samples, f"Expected {self.n_samples} samples but got {result} samples.")

    def test_sample_edge_empty_pts(self):
        """
        EdgeCase: Sampling on empty pts.
        """
        # TODO: this needs to be more concrete!
        result = smp.ArchimedeanSampler(self.n_samples).sample(np.array([]))
        np.testing.assert_array_equal(result, [])