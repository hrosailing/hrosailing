from unittest import TestCase
import numpy as np

import hrosailing.processing.filter as flt


class TestBoundFilter(TestCase):
    def setUp(self) -> None:
        self.lower = 0.205
        self.upper = 5.78
        self.wts = np.array([.2, .3, .21, .5, .8])

    def test_BoundFilter_init_ValueError(self):
        """
        ValueError if lower_bound > upper_bound
        """
        with self.assertRaises(ValueError):
            flt.BoundFilter(3, 1)

    def test_BoundFilter_repr(self):
        """
        Input/Output-Test.
        """

        result = repr(flt.BoundFilter(self.lower, self.upper))
        expected_result = f"BoundFilter(upper_bound={self.upper}, lower_bound={self.lower})"
        self.assertEqual(result, expected_result,
                         f"Expected {expected_result} but got {result}!")

    def test_BoundFilter_filter(self):
        """
        Input/Output-Test.
        """

        with self.subTest("default BoundFilter"):
            result = flt.BoundFilter().filter(self.wts)
            expected_result = [False, False, False, True, True]
            np.testing.assert_array_equal(result, expected_result,
                                          f"Expected {expected_result} but got {result}!")

        with self.subTest("custom upper_bound"):
            result = flt.BoundFilter(upper_bound=self.upper).filter(self.wts)
            expected_result = [False, False, False, True, True]
            np.testing.assert_array_equal(result, expected_result,
                                          f"Expected {expected_result} but got {result}!")

        with self.subTest("custom lower_bound"):
            result = flt.BoundFilter(lower_bound=self.lower).filter(self.wts)
            expected_result = [False, True, True, True, True]
            np.testing.assert_array_equal(result, expected_result,
                                          f"Expected {expected_result} but got {result}!")
