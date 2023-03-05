from unittest import TestCase
import numpy as np

import hrosailing.processing.sampler as smp


class TestUniformRandomSampler(TestCase):
    def setUp(self) -> None:
        self.n_samples = 2
        self.pts = [[1, 1], [-1, 1], [-1, -1], [1, -1], [0, 0]]

    def test_UniformRandomSampler_init_Error(self):
        """
        ValueError occurs is n_samples is non-positive.
        """
        with self.assertRaises(ValueError):
            smp.UniformRandomSampler(-1)

    def test_UniformRandomSampler_sample(self):
        """
        Sampled points lay in the convex hull of pts.
        """
        # TODO: this needs to be fixed, asks for (n,2) data but uses index :2 in code
        result = smp.UniformRandomSampler(self.n_samples).sample(self.pts)
        for coordinate in result.flatten():
            self.assertGreaterEqual(coordinate, -1)
            self.assertGreaterEqual(-coordinate, -1)
