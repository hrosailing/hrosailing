from unittest import TestCase
import numpy as np

import hrosailing.processing.neighbourhood as nbh


class TestEllipsoid(TestCase):
    def setUp(self) -> None:
        self.lin_trans = np.array([[2, 0], [0, 2]])
        self.radius = 0.02
        self.norm = lambda x: 0.03 * np.linalg.norm(x, ord=2, axis=1)
        self.pts = [[1, 2], [3, 1], [0.5, 0.5]]

    def test_Ellipsoid_init_Errors(self):
        """
        ValueError if lin_trans has wrong shape, is singular or if radius is non-positive.
        """
        with self.subTest("lin_trans wrong shape"):
            with self.assertRaises(ValueError):
                nbh.Ellipsoid(lin_trans=[[1, 2, 3], [1, 2, 3]])

        with self.subTest("lin_trans singular"):
            with self.assertRaises(ValueError):
                nbh.Ellipsoid([[0, 0], [0, 0]])

        with self.subTest("radius non-positive"):
            with self.assertRaises(ValueError):
                nbh.Ellipsoid(radius=0)

    def test_Ellipsoid_repr(self):
        """
        Input/Output-Test.
        """

        result = repr(nbh.Ellipsoid(self.lin_trans, self.norm, self.radius))
        expected_result = f"Ellipsoid(lin_trans={np.linalg.inv(self.lin_trans)}, " \
                          f"norm={self.norm.__name__}, radius={self.radius})"
        self.assertEqual(result, expected_result,
                         f"Expected {expected_result} but got {result}!")

    def test_Ellipsoid_is_contained_in(self):
        """
        Input/Output-Test.
        """
        with self.subTest("default Ellipsoid"):
            result = nbh.Ellipsoid().is_contained_in(self.pts)
            expected_result = [True, False, True]
            np.testing.assert_array_equal(result, expected_result,
                             f"Expected {expected_result} but got {result}!")

        with self.subTest("custom lin_trans"):
            result = nbh.Ellipsoid(lin_trans=self.lin_trans).is_contained_in(self.pts)
            expected_result = [True, True, True]
            np.testing.assert_array_equal(result, expected_result,
                                          f"Expected {expected_result} but got {result}!")

        with self.subTest("custom norm"):
            result = nbh.Ellipsoid(norm=self.norm).is_contained_in(self.pts)
            expected_result = [False, False, True]
            np.testing.assert_array_equal(result, expected_result,
                                          f"Expected {expected_result} but got {result}!")

        with self.subTest("custom radius"):
            result = nbh.Ellipsoid(radius=self.radius).is_contained_in(self.pts)
            expected_result = [False, False, True]
            np.testing.assert_array_equal(result, expected_result,
                                          f"Expected {expected_result} but got {result}!")
