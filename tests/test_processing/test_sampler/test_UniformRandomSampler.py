# pylint: disable-all

from unittest import TestCase

import numpy as np

import hrosailing.processing.sampler as smp


class TestUniformRandomSampler(TestCase):
    def setUp(self):
        self.n_samples = 2
        self.pts = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1], [0, 0]])

    def test_UniformRandomSampler_init_Error(self):
        with self.assertRaises(ValueError):
            smp.UniformRandomSampler(-1)

    def test_UniformRandomSampler_sample_convex_hull(self):
        result = smp.UniformRandomSampler(self.n_samples).sample(self.pts)
        for coordinate in result.flatten():
            self.assertGreaterEqual(
                coordinate,
                -1,
                "Sampled points are outside the convex hull of pts.",
            )
            self.assertGreaterEqual(
                -coordinate,
                -1,
                "Sampled points are outside the convex hyll of pts.",
            )

    def test_sample_amount(self):
        result = len(smp.UniformRandomSampler(self.n_samples).sample(self.pts))

        self.assertEqual(result, self.n_samples)

    def test_sample_edge_empty_pts(self):
        with self.assertRaises(ValueError):
            smp.UniformRandomSampler(self.n_samples).sample(np.empty((3, 0)))
