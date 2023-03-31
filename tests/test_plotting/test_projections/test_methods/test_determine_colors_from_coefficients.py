# pylint: disable-all
import unittest

import numpy as np

from hrosailing.plotting.projections import _determine_colors_from_coefficients


class TestDetermineColorsFromCoefficients(unittest.TestCase):
    def test_regular_input(self):
        # Input/Output
        result = _determine_colors_from_coefficients(
            [0.375, 0, 0.625, 1, 0.125], ((1, 0, 0), (0, 1, 0))
        )
        expected = [
            np.array((0.625, 0.375, 0)),
            np.array((1, 0, 0)),
            np.array((0.375, 0.625, 0)),
            np.array((0, 1, 0)),
            np.array((0.875, 0.125, 0)),
        ]
        for res, exp in zip(result, expected):
            np.testing.assert_array_equal(res, exp)
