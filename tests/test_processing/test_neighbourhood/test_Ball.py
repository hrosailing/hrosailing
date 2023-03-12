from unittest import TestCase
import numpy as np

import hrosailing.processing.neighbourhood as nbh


class TestBall(TestCase):
    def setUp(self) -> None:
        self.radius = 0.02
        self.norm = lambda x: 0.03 * np.linalg.norm(x, ord=2, axis=1)
        self.pts = [[1, 2], [3, 1], [0.5, 0.5]]

    def test_init_Error(self):
        """
        ValueError occurs if radius is negative.
        """
        with self.assertRaises(ValueError):
            nbh.Ball(-1)

    def test_repr(self):
        """
        Input/Output-Test.
        """

        result = repr(nbh.Ball(self.radius, self.norm))
        expected_result = f"Ball(norm={self.norm.__name__}, radius={self.radius})"
        self.assertEqual(result, expected_result,
                         f"Expected {expected_result} but got {result}!")

    def test_is_contained_in_default(self):
        """
        Input/Output-Test.
        """

        result = nbh.Ball().is_contained_in(self.pts)
        expected_result = [True, False, True]
        np.testing.assert_array_equal(result, expected_result,
                         f"Expected {expected_result} but got {result}!")

    def test_is_contained_in_custom_radius(self):
        """
        Input/Output-Test.
        """

        result = nbh.Ball(self.radius).is_contained_in(self.pts)
        expected_result = [False, False, True]
        np.testing.assert_array_equal(result, expected_result,
                                      f"Expected {expected_result} but got {result}!")

    def test_is_contained_in_custom_norm(self):
        """
        Input/Output-Test.
        """

        result = nbh.Ball(norm=self.norm).is_contained_in(self.pts)
        expected_result = [False, False, True]
        np.testing.assert_array_equal(result, expected_result,
                                      f"Expected {expected_result} but got {result}!")

    def test_is_contained_in_edge_empty_pts(self):
        """
        EdgeCase: Empty pts.
        """
        result = nbh.Ball().is_contained_in([])
        expected_result = []
        np.testing.assert_array_equal(result, expected_result,
                                      f"Expected {expected_result} but got {result}!")
