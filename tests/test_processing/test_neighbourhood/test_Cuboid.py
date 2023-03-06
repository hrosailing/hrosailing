from unittest import TestCase
import numpy as np

import hrosailing.processing.neighbourhood as nbh


class TestCuboid(TestCase):
    def setUp(self) -> None:
        self.norm = lambda x: 0.03 * np.linalg.norm(x, ord=2, axis=0)
        self.dimensions = (0.5, 0.5)
        self.pts = [[.01, .02], [.3, 1], [0.5, 0.5]]

    def test_Cuboid_repr(self):
        """
        Input/Output-Test.
        """

        result = repr(nbh.Cuboid(self.norm, self.dimensions))
        expected_result = f"Cuboid(norm={self.norm.__name__}, dimensions={self.dimensions})"
        self.assertEqual(result, expected_result,
                         f"Expected {expected_result} but got {result}!")

    def test_Cuboid_is_contained_in(self):
        """
        Input/Output-Test.
        """
        with self.subTest("default Cuboid"):
            result = nbh.Cuboid().is_contained_in(self.pts)
            expected_result = [True, False, False]
            np.testing.assert_array_equal(result, expected_result,
                                          f"Expected {expected_result} but got {result}!")

        with self.subTest("custom norm"):
            result = nbh.Cuboid(self.norm).is_contained_in(self.pts)
            expected_result = [True, True, True]
            np.testing.assert_array_equal(result, expected_result,
                                          f"Expected {expected_result} but got {result}!")

        with self.subTest("custom dimensions"):
            result = nbh.Cuboid(dimensions=self.dimensions).is_contained_in(self.pts)
            expected_result = [True, False, True]
            np.testing.assert_array_equal(result, expected_result,
                                          f"Expected {expected_result} but got {result}!")