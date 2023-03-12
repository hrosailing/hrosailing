from unittest import TestCase
import numpy as np

import hrosailing.processing.filter as flt


class TestQuantileFilter(TestCase):
    def setUp(self) -> None:
        self.percent = 20
        self.wts = np.array([2, 3, 2.1, 5, 8])

    def test_init_Error_upper_bound_exceeded(self):
        """
        ValueError if percent >100.
        """
        with self.assertRaises(ValueError):
            flt.QuantileFilter(101)

    def test_init_Error_lower_bound_exceeded(self):
        """
        ValueError if percent <0.
        """
        with self.assertRaises(ValueError):
            flt.QuantileFilter(-1)

    def test_repr(self):
        """
        Input/Output-Test.
        """

        result = repr(flt.QuantileFilter(self.percent))
        expected_result = f"QuantileFilter(percent={self.percent})"
        self.assertEqual(result, expected_result,
                         f"Expected {expected_result} but got {result}!")

    def test_filter_default(self):
        """
        Input/Output-Test.
        """

        result = flt.QuantileFilter().filter(self.wts)
        expected_result = [False, True, False, True, True]
        np.testing.assert_array_equal(result, expected_result,
                                      f"Expected {expected_result} but got {result}!")

    def test_filter_custom_percent(self):
        """
        Input/Output-Test.
        """

        result = flt.QuantileFilter(self.percent).filter(self.wts)
        expected_result = [False, True, True, True, True]
        np.testing.assert_array_equal(result, expected_result,
                                      f"Expected {expected_result} but got {result}!")

    def test_filter_edge_empty_wts(self):
        """
        EdgeCase: Empty wts.
        """
        result = flt.QuantileFilter().filter(np.array([]))
        expected_result = []
        np.testing.assert_array_equal(result, expected_result,
                                      f"Expected {expected_result} but got {result}!")

    def test_filter_edge_same_wts(self):
        """
        EdgeCase: Same weights, so it is hard to decide which wts to remove.
        """
        wts = np.ones(5)
        result = flt.QuantileFilter().filter(wts)
        expected_result = [False, False, False, True, True]
        np.testing.assert_array_equal(result, expected_result,
                                      f"Expected {expected_result} but got {result}!")
