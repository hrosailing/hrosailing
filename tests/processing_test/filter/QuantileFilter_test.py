from unittest import TestCase
import numpy as np

import hrosailing.processing.filter as flt


class TestQuantileFilter(TestCase):
    def setUp(self) -> None:
        self.percent = 20
        self.wts = np.array([2, 3, 2.1, 5, 8])

    def test_QuantileFilter_init_ValueError(self):
        """
        ValueError if percent not in [0, 100]
        """
        with self.subTest("upper bound exceeded"):
            with self.assertRaises(ValueError):
                flt.QuantileFilter(101)
        with self.subTest("lower bound exceeded"):
            with self.assertRaises(ValueError):
                flt.QuantileFilter(-1)

    def test_QuantileFilter_repr(self):
        """
        Input/Output-Test.
        """

        result = repr(flt.QuantileFilter(self.percent))
        expected_result = f"QuantileFilter(percent={self.percent})"
        self.assertEqual(result, expected_result,
                         f"Expected {expected_result} but got {result}!")

    def test_QuantileFilter_filter(self):
        """
        Input/Output-Test.
        """

        with self.subTest("default QuantileFilter"):
            result = flt.QuantileFilter().filter(self.wts)
            expected_result = [False, True, False, True, True]
            np.testing.assert_array_equal(result, expected_result,
                                          f"Expected {expected_result} but got {result}!")

        with self.subTest("custom percent"):
            result = flt.QuantileFilter(self.percent).filter(self.wts)
            expected_result = [False, True, True, True, True]
            np.testing.assert_array_equal(result, expected_result,
                                          f"Expected {expected_result} but got {result}!")