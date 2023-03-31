"""
Tests
"""

from unittest import TestCase

import numpy as np

import hrosailing.core.data as dt
import hrosailing.processing.interpolator as itp


class TestIDWInterpolator(TestCase):
    def setUp(self) -> None:
        self.s = 0.5
        self.norm = lambda x: 0.3 * np.linalg.norm(x, ord=2, axis=1)
        self.distr = lambda distances, old_weights, *params: np.ones(
            len(distances)
        )
        self.params = (1,)

        self.wpts = dt.WeightedPoints(
            np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 0.5], [1, -1, 1]]),
            np.ones(4),
        )

        self.grid_pt = np.array([0, 0])

    def test_init_Error(self):
        """
        ValueError if s non-positive.
        """
        with self.assertRaises(ValueError):
            itp.ArithmeticMeanInterpolator(s=0)

    def test_repr(self):
        """
        Input/Output-Test.
        """

        result = repr(
            itp.ArithmeticMeanInterpolator(
                self.s, self.norm, self.distr, self.params
            )
        )
        expected_result = (
            f"ArithmeticMeanInterpolator(s={self.s},"
            f" norm={self.norm.__name__}, distribution={self.distr},"
            f" params={self.params})"
        )
        self.assertEqual(
            result,
            expected_result,
            f"Expected {expected_result} but got {result}!",
        )

    def test_interpolate_default(self):
        """
        Input/Output-Test.
        """

        result = itp.ArithmeticMeanInterpolator().interpolate(
            self.wpts, self.grid_pt
        )
        expected_result = 0.875
        self.assertAlmostEqual(
            result,
            expected_result,
            msg=f"Expected {expected_result} but got {result}!",
        )

    def test_interpolate_custom_s(self):
        """
        Input/Output-Test.
        """

        result = itp.ArithmeticMeanInterpolator(self.s).interpolate(
            self.wpts, self.grid_pt
        )
        expected_result = 0.4375
        self.assertAlmostEqual(
            result,
            expected_result,
            msg=f"Expected {expected_result} but got {result}!",
        )

    def test_interpolate_custom_norm(self):
        """
        Input/Output-Test.
        """

        result = itp.ArithmeticMeanInterpolator(norm=self.norm).interpolate(
            self.wpts, self.grid_pt
        )
        expected_result = 0.875
        self.assertEqual(
            result,
            expected_result,
            f"Expected {expected_result} but got {result}!",
        )

    def test_interpolate_custom_distribution(self):
        """
        Input/Output-Test.
        """

        result = itp.ArithmeticMeanInterpolator(
            distribution=self.distr
        ).interpolate(self.wpts, self.grid_pt)
        expected_result = 0.875
        self.assertEqual(
            result,
            expected_result,
            f"Expected {expected_result} but got {result}!",
        )

    def test_interpolate_custom_params(self):
        """
        Input/Output-Test.
        """

        result = itp.ArithmeticMeanInterpolator(
            params=self.params
        ).interpolate(self.wpts, self.grid_pt)
        expected_result = 0.875
        self.assertEqual(
            result,
            expected_result,
            f"Expected {expected_result} but got {result}!",
        )

    def test_interpolate_edge_grid_pt_in_wpts(self):
        """
        EdgeCase: grid_pt is in wpts.
        """
        result = itp.ArithmeticMeanInterpolator().interpolate(
            dt.WeightedPoints(
                np.array([[-1, 1, 1], [-1, -1, 0.5], [0, 0, 3]]), np.ones(3)
            ),
            self.grid_pt,
        )
        expected_result = 3
        self.assertEqual(
            result,
            expected_result,
            f"Expected {expected_result} but got {result}!",
        )
