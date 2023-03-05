from unittest import TestCase
import numpy as np

import hrosailing.processing.neighbourhood as nbh


class TestBall(TestCase):
    def setUp(self) -> None:
        self.radius = 0.02
        self.norm = lambda x: 0.03 * np.linalg.norm(x, ord=2, axis=1)
        self.pts = [[1, 2], [3, 1], [0.5, 0.5]]

    def test_Ball_init_Error(self):
        """
        ValueError occurs if radius is negative.
        """
        with self.assertRaises(ValueError):
            nbh.Ball(-1)

    def test_Ball_repr(self):
        """
        Input/Output-Test.
        """

        result = repr(nbh.Ball(self.radius, self.norm))
        expected_result = f"Ball(norm={self.norm.__name__}, radius={self.radius})"
        self.assertEqual(result, expected_result,
                         f"Expected {expected_result} but got {result}!")

    def test_Ball_is_contained_in(self):
        """
        Input/Output-Test.
        """

        with self.subTest("default Ball"):
            result = nbh.Ball().is_contained_in(self.pts)
            expected_result = [True, False, True]
            np.testing.assert_array_equal(result, expected_result,
                             f"Expected {expected_result} but got {result}!")

        with self.subTest("custom radius"):
            result = nbh.Ball(self.radius).is_contained_in(self.pts)
            expected_result = [False, False, True]
            np.testing.assert_array_equal(result, expected_result,
                                          f"Expected {expected_result} but got {result}!")

        with self.subTest("custom norm"):
            result = nbh.Ball(norm=self.norm).is_contained_in(self.pts)
            expected_result = [False, False, True]
            np.testing.assert_array_equal(result, expected_result,
                                          f"Expected {expected_result} but got {result}!")
