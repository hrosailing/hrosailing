# pylint: disable-all

from unittest import TestCase

import numpy as np

import hrosailing.processing.sampler as smp


class TestSamplerFunctions(TestCase):
    def setUp(self):
        self.pts = np.array([[1, 6], [2, 5], [3, 4], [4, 3], [5, 2]])

    def test__create_bounds(self):
        result = smp._create_bounds(self.pts)
        expected_result = (1, 5), (2, 6)

        self.assertEqual(result, expected_result)

    def wip_test__binary_rescale(self):
        n_samples = 1
        gen_samp = 2
        st_val = 3
        result = smp._binary_rescale(n_samples, gen_samp, st_val)
        expected_result = ()

        self.assertEqual(result, expected_result)
