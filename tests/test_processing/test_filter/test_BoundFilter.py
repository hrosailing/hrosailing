from unittest import TestCase

import numpy as np

import hrosailing.processing.filter as flt


class TestBoundFilter(TestCase):
    def setUp(self) -> None:
        self.lower = 0.205
        self.upper = 5.78
        self.wts = np.array([0.2, 0.3, 0.21, 0.5, 0.8])

    def test_init_Error(self):
        """
        ValueError if lower_bound > upper_bound.
        """
        with self.assertRaises(ValueError):
            flt.BoundFilter(3, 1)

    def test_repr(self):
        """
        Input/Output-Test.
        """

        result = repr(flt.BoundFilter(self.lower, self.upper))
        expected_result = (
            f"BoundFilter(upper_bound={self.upper}, lower_bound={self.lower})"
        )
        self.assertEqual(
            result,
            expected_result,
            f"Expected {expected_result} but got {result}!",
        )

    def test_filter_default(self):
        """
        Input/Output-Test.
        """

        result = flt.BoundFilter().filter(self.wts)
        expected_result = [False, False, False, True, True]
        np.testing.assert_array_equal(
            result,
            expected_result,
            f"Expected {expected_result} but got {result}!",
        )

    def test_filter_custom_upper_bound(self):
        """
        Input/Output-Test.
        """
        result = flt.BoundFilter(upper_bound=self.upper).filter(self.wts)
        expected_result = [False, False, False, True, True]
        np.testing.assert_array_equal(
            result,
            expected_result,
            f"Expected {expected_result} but got {result}!",
        )

    def test_filter_custom_lower_bound(self):
        """
        Input/Output-Test.
        """
        result = flt.BoundFilter(lower_bound=self.lower).filter(self.wts)
        expected_result = [False, True, True, True, True]
        np.testing.assert_array_equal(
            result,
            expected_result,
            f"Expected {expected_result} but got {result}!",
        )

    def test_filter_edge_empty_wts(self):
        """
        EdgeCase: Empty wts.
        """
        result = flt.BoundFilter().filter(np.array([]))
        expected_result = []
        np.testing.assert_array_equal(
            result,
            expected_result,
            f"Expected {expected_result} but got {result}!",
        )

    def test_filter_edge_bounds_in_wts(self):
        """
        EdgeCase: wts contains the bounds
        """
        result = flt.BoundFilter(lower_bound=0.2, upper_bound=0.8).filter(
            self.wts
        )
        expected_result = np.ones(len(self.wts), dtype=bool)
        np.testing.assert_array_equal(
            result,
            expected_result,
            f"Expected {expected_result} but got {result}!",
        )

    def test__determine_points_within_bound(self):
        """
        Input/Output-Test.
        """
        result = flt.BoundFilter()._determine_points_within_bound(self.wts)
        expected_result = [False, False, False, True, True]
        np.testing.assert_array_equal(
            result,
            expected_result,
            f"Expected {expected_result} but got {result}!",
        )
