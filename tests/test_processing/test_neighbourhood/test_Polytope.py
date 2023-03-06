from unittest import TestCase
import numpy as np

import hrosailing.processing.neighbourhood as nbh


class TestPolytope(TestCase):
    def setUp(self) -> None:
        self.mat = np.array([[-1, 0], [0, -1], [1, 2], [2, 1]])
        self.b = np.asarray([.2, .5, 1, 2])
        self.pts = [[.01, .02], [.3, 1], [0.5, 0.5]]

    def test_Polytope_init_Errors(self):
        """
        ValueErrors if mat or b has incorrect shape
        """
        with self.subTest("mat has wrong shape"):
            with self.assertRaises(ValueError):
                nbh.Polytope(mat=[[1, 2, 3], [1, 2, 3], [1, 2, 3]])

        with self.subTest("b has wrong shape"):
            with self.assertRaises(ValueError):
                nbh.Polytope(b=[1, 2, 3])

    def test_Polytope_repr(self):
        """
        Input/Output-Test.
        """

        result = repr(nbh.Polytope(self.mat, self.b))
        expected_result = f"Polytope(mat={self.mat}, b={self.b})"
        self.assertEqual(result, expected_result,
                         f"Expected {expected_result} but got {result}!")

    def test_Polytope_is_contained_in(self):
        """
        Input/Output-Test.
        """
        with self.subTest("default Polytope"):
            result = nbh.Polytope().is_contained_in(self.pts)
            expected_result = [True, False, False]
            np.testing.assert_array_equal(result, expected_result,
                         f"Expected {expected_result} but got {result}!")

        with self.subTest("custom mat"):
            result = nbh.Polytope(self.mat).is_contained_in(self.pts)
            expected_result = [True, False, False]
            np.testing.assert_array_equal(result, expected_result,
                                          f"Expected {expected_result} but got {result}!")

        with self.subTest("custom b"):
            result = nbh.Polytope(b=self.b).is_contained_in(self.pts)
            expected_result = [True, False, False]
            np.testing.assert_array_equal(result, expected_result,
                                          f"Expected {expected_result} but got {result}!")
