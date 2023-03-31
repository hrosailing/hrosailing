import unittest

import numpy as np

from hrosailing.plotting.projections import _merge


class TestMerge(unittest.TestCase):
    def test_regular_input(self):
        # Input/Output
        wa = np.array([0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1])
        intervals = [[0, 1, 4], [3], [2, 5], [6]]
        result = _merge(wa, intervals)
        expected = np.array(
            [0.1, 0.2, 0.3, np.NaN, 0.4, np.NaN, 0.3, 0.2, np.NaN, 0.1]
        )

        np.testing.assert_array_equal(result, expected)

    def test_edge_case_empty_wa(self):
        # Exception test
        with self.assertRaises(IndexError):
            _merge(np.empty((0)), [[0, 1, 4], [3], [2, 5], [6]])

    def test_edge_case_empty_intervals(self):
        # Input/Output with empty interval
        wa = np.array([0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1])
        result = _merge(wa, [])
        expected = np.empty((0))

        np.testing.assert_array_equal(result, expected)
