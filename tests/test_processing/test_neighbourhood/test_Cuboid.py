from unittest import TestCase

import numpy as np

import hrosailing.processing.neighbourhood as nbh


class TestCuboid(TestCase):
    def setUp(self) -> None:
        self.norm = lambda x: 0.03 * np.linalg.norm(x, ord=2, axis=0)
        self.dimensions = (0.5, 0.5)
        self.pts = [[0.01, 0.02], [0.3, 1], [0.5, 0.5]]

    def test_repr(self):
        """
        Input/Output-Test.
        """

        result = repr(nbh.Cuboid(self.norm, self.dimensions))
        expected_result = (
            f"Cuboid(norm={self.norm.__name__}, attributes={self.dimensions})"
        )
        self.assertEqual(
            result,
            expected_result,
            f"Expected {expected_result} but got {result}!",
        )

    def test_is_contained_in_default(self):
        """
        Input/Output-Test.
        """

        with self.subTest("default Cuboid"):
            result = nbh.Cuboid().is_contained_in(self.pts)
            expected_result = [True, False, False]
            np.testing.assert_array_equal(
                result,
                expected_result,
                f"Expected {expected_result} but got {result}!",
            )

    def test_is_contained_in_custom_norm(self):
        """
        Input/Output-Test.
        """

        result = nbh.Cuboid(self.norm).is_contained_in(self.pts)
        expected_result = [True, True, True]
        np.testing.assert_array_equal(
            result,
            expected_result,
            f"Expected {expected_result} but got {result}!",
        )

    def test_is_contained_in_custom_dimensions(self):
        """
        Input/Output-Test.
        """

        result = nbh.Cuboid(dimensions=self.dimensions).is_contained_in(
            self.pts
        )
        expected_result = [True, False, True]
        np.testing.assert_array_equal(
            result,
            expected_result,
            f"Expected {expected_result} but got {result}!",
        )

    def test_is_contained_in_edge_empty_pts(self):
        """
        EdgeCase: Empty pts.
        """
        result = nbh.Cuboid().is_contained_in([])
        expected_result = []
        np.testing.assert_array_equal(
            result,
            expected_result,
            f"Expected {expected_result} but got {result}!",
        )

    def test_is_contained_in_edge_dimensions_0(self):
        """
        EdgeCase: attributes = (0, 0).
        """
        result = nbh.Cuboid(dimensions=(0, 0)).is_contained_in(self.pts)
        expected_result = np.zeros(3, dtype=bool)
        np.testing.assert_array_equal(
            result,
            expected_result,
            f"Expected {expected_result} but got {result}!",
        )
