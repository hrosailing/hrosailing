# pylint: disable-all

from unittest import TestCase

import numpy as np

import hrosailing.processing.interpolator as int
from hrosailing.core.computing import scaled_euclidean_norm


class TestInterpolatorFunctions(TestCase):
    def setUp(self):
        self.pts = np.array(
            [
                [100, 100, 0],
                [1, 1, 1],
                [1, -1, 2],
                [-1, -1, 1],
                [-1, 1, 2],
                [0, 0, 0],
            ]
        )
        self.dist = [
            scaled_euclidean_norm(np.array([pt[:2]])) for pt in self.pts
        ]
        self.grid_pt = [0, 0]
        self.wts = np.ones(len(self.pts))

    def test__set_weights(self):
        result = int._set_weights(self.pts, self.dist)
        expected_result = [
            0.0007152431971467461,
            39.75534939,
            39.75534939,
            39.75534939,
            39.75534939,
            0.0,
        ]

        np.testing.assert_array_almost_equal(
            result,
            expected_result,
            decimal=3,
        )
