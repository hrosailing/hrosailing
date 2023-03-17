from unittest import TestCase
import numpy as np

import hrosailing.processing.interpolator as itp
import hrosailing.core.data as dt


class TestIDWInterpolator(TestCase):
    def setUp(self) -> None:
        self.norm = lambda x: .3 * np.linalg.norm(x, ord=2, axis=1)

        self.wpts = dt.WeightedPoints(
            np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, .5], [1, -1, 1], [10, 10, .5]]),
            np.ones(5))

        self.grid_pt = np.array([0, 0])

    def test_interpolate_default(self):
        """
        Input/Output-Test.
        """
        result = itp.ImprovedIDWInterpolator().interpolate(self.wpts, self.grid_pt)
        expected_result = 0.87445589
        self.assertAlmostEqual(result, expected_result, places=1,
                               msg=f"Expected {expected_result} but got {result}!")

    def test_interpolate_custom_norm(self):
        """
        Input/Output-Test.
        """
        result = itp.ImprovedIDWInterpolator(norm=self.norm).interpolate(self.wpts, self.grid_pt)
        expected_result = 0.8340573528145949
        self.assertAlmostEqual(result, expected_result, places=1,
                               msg=f"Expected {expected_result} but got {result}!")
