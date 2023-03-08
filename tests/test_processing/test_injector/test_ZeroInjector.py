from unittest import TestCase
import numpy as np

import hrosailing.processing.injector as inj
import hrosailing.core.data as dt


class TestZeroInjector(TestCase):
    def setUp(self) -> None:
        self.n_zeros = 2
        self.wpts = dt.WeightedPoints(np.array([[12, 34, 15], [13, 40, 18]]), np.array([.3, .7]))

    def test_inject(self):
        """
        Input/Output-Test.
        """

        result = inj.ZeroInjector(self.n_zeros).inject(self.wpts)
        expected_result = dt.WeightedPoints(
            np.array([[12, 0, 0], [13, 0, 0], [12, 360, 0], [13, 360, 0]]), [1, 1, 1, 1])
        np.testing.assert_array_equal(result.data, expected_result.data,
                                      f"Expected {expected_result} but got {result}!")
        np.testing.assert_array_equal(result.weights, expected_result.weights,
                                      f"Expected {expected_result} but got {result}!")
