import unittest

import numpy as np

from hrosailing.plotting.projections import _determine_color_gradient

class TestDetermineColorGradient(unittest.TestCase):
    def test_regular_input(self):
        #Input/Output
        result = _determine_color_gradient(
            [(1, 0, 0), (0, 1, 0)], [4, 1, 6, 9, 2]
        )
        expected = [
            np.array((0.625, 0.375, 0)),
            np.array((1, 0, 0)),
            np.array((0.375, 0.625, 0)),
            np.array((0, 1, 0)),
            np.array((0.875, 0.125, 0))
        ]
        for res, exp in zip(result, expected):
            np.testing.assert_array_equal(res, exp)