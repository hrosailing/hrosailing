# pylint: disable-all

from unittest import TestCase

import numpy as np

import hrosailing.processing.filter as flt


class TestBoundFilter(TestCase):
    def setUp(self):
        self.lower = 0.205
        self.upper = 5.78
        self.wts = np.array([0.2, 0.3, 0.21, 0.5, 0.8])

    def test_init_Error(self):
        with self.assertRaises(ValueError):
            flt.BoundFilter(3, 1)

    def test_repr(self):
        result = repr(flt.BoundFilter(self.lower, self.upper))
        expected_result = (
            f"BoundFilter(upper_bound={self.upper}, lower_bound={self.lower})"
        )

        self.assertEqual(result, expected_result)

    def test_filter_default(self):
        result = flt.BoundFilter().filter(self.wts)
        expected_result = [False, False, False, True, True]

        np.testing.assert_array_equal(result, expected_result)

    def test_filter_custom_upper_bound(self):
        result = flt.BoundFilter(upper_bound=self.upper).filter(self.wts)
        expected_result = [False, False, False, True, True]

        np.testing.assert_array_equal(result, expected_result)

    def test_filter_custom_lower_bound(self):
        result = flt.BoundFilter(lower_bound=self.lower).filter(self.wts)
        expected_result = [False, True, True, True, True]

        np.testing.assert_array_equal(result, expected_result)

    def test_filter_edge_empty_wts(self):
        result = flt.BoundFilter().filter(np.array([]))
        expected_result = []

        np.testing.assert_array_equal(result, expected_result)

    def test_filter_edge_bounds_in_wts(self):
        result = flt.BoundFilter(lower_bound=0.2, upper_bound=0.8).filter(
            self.wts
        )
        expected_result = np.ones(len(self.wts), dtype=bool)

        np.testing.assert_array_equal(result, expected_result)

    def test__determine_points_within_bound(self):
        result = flt.BoundFilter()._determine_points_within_bound(self.wts)
        expected_result = [False, False, False, True, True]

        np.testing.assert_array_equal(result, expected_result)
