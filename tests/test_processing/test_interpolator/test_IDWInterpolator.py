from unittest import TestCase

import numpy as np

import hrosailing.core.data as dt
import hrosailing.processing.interpolator as int


class TestIDWInterpolator(TestCase):
    def setUp(self) -> None:
        self.p = 1
        self.norm = lambda x: 0.3 * np.linalg.norm(x, ord=2, axis=1)
        self.wpts = dt.WeightedPoints(
            np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 0.5], [1, -1, 0.5]]),
            np.ones(4),
        )

        self.grid_pt = np.array([0, 0])

    def test_init_Error(self):
        """
        p < 0 causes ValueError.
        """
        with self.assertRaises(ValueError):
            int.IDWInterpolator(-1)

    def test_repr(self):
        """
        Input/Output-Test.
        """

        result = repr(int.IDWInterpolator(self.p, self.norm))
        expected_result = (
            f"IDWInterpolator(p={self.p}, norm={self.norm.__name__})"
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

        result = int.IDWInterpolator().interpolate(self.wpts, self.grid_pt)
        expected_result = 0.75
        self.assertEqual(
            result,
            expected_result,
            f"Expected {expected_result} but got {result}!",
        )

    def test_interpolate_custom_p(self):
        """
        Input/Output-Test.
        """
        result = int.IDWInterpolator(self.p).interpolate(
            self.wpts, self.grid_pt
        )
        expected_result = 0.75
        self.assertEqual(
            result,
            expected_result,
            f"Expected {expected_result} but got {result}!",
        )

    def test_interpolate_custom_norm(self):
        """
        Input/Output-Test.
        """
        result = int.IDWInterpolator(norm=self.norm).interpolate(
            self.wpts, self.grid_pt
        )
        expected_result = 0.75
        self.assertEqual(
            result,
            expected_result,
            f"Expected {expected_result} but got {result}!",
        )

    def test_interpolate_edge_empty_wpts(self):
        """
        EdgeCase: Empty wpts.
        """
        with self.assertRaises(ValueError):
            int.IDWInterpolator().interpolate(
                dt.WeightedPoints(np.array([]), np.array([])), self.grid_pt
            )

    def test_interpolate_edge_float_p(self):
        """
        EdgeCase: p is a float value.
        """
        with self.assertRaises(TypeError):
            int.IDWInterpolator(p=0.5)
