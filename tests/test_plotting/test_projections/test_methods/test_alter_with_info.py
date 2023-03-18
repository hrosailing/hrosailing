import unittest
import numpy as np

from hrosailing.plotting.projections import _alter_with_info

class TestAlterWithInfo(unittest.TestCase):
    def test_regular_input(self):
        wa = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        bsp = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        info_ = ["A", "A", "A", "B", "B", "C", "B", "B", "A"]
        result = _alter_with_info(wa, bsp, info_)
        expected = np.array([
            [0.0, 0.1, 0.2, 0.8, np.NaN, 0.3, 0.4, 0.6, 0.7, np.NaN, 0.5],
            [1, 2, 3, 9, np.NaN, 4, 5, 7, 8, np.NaN, 6]
        ])

        np.testing.assert_array_equal(result, expected)

    def test_edge_case_empty(self):
        # Input/Output with empty data
        result = _alter_with_info(np.empty((0)), np.empty((0)), [])
        expected = np.empty((2, 0))

        np.testing.assert_array_equal(result, expected)
