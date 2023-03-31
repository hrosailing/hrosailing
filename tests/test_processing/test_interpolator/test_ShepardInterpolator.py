"""
Tests
"""

from unittest import TestCase

import numpy as np

import hrosailing.processing.interpolator as int
import hrosailing.processing.neighbourhood as nbh


class TestShepardInterpolator(TestCase):
    def setUp(self) -> None:
        self.tol = 0.001
        self.slope = 0.2
        self.norm = lambda x: np.linalg.norm(x, ord=2, axis=0)
        self.nbh = nbh.Ball()

    def test_init_Error_tol(self):
        """
        ValueError if tol is non-positive.
        """

        with self.assertRaises(ValueError):
            int.ShepardInterpolator(self.nbh, tol=0)

    def test_init_Error_slope(self):
        """
        ValueError if slope is non-positive.
        """

        with self.assertRaises(ValueError):
            int.ShepardInterpolator(self.nbh, slope=0)

    def test_repr(self):
        """
        Input/Output-Test.
        """

        result = repr(
            int.ShepardInterpolator(self.nbh, self.tol, self.slope, self.norm)
        )
        expected_result = (
            f"ShepardInterpolator( tol={self.tol}, slope_scal={self.slope},"
            f" norm={self.norm.__name__})"
        )
        self.assertEqual(
            result,
            expected_result,
            f"Expected {expected_result} but got {result}!",
        )
