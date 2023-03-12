from unittest import TestCase
import numpy as np

import hrosailing.processing.neighbourhood as nbh


class TestPolytope(TestCase):
    def setUp(self) -> None:
        self.mat = np.array([[-1, 0], [0, -1], [1, 2], [2, 1]])
        self.b = np.asarray([.2, .5, 1, 2])
        self.pts = [[.01, .02], [.3, 1], [0.5, 0.5]]

    def test_init_Error_mat_wrong_shape(self):
        """
        ValueError if mat has incorrect shape.
        """
        with self.assertRaises(ValueError):
            nbh.Polytope(mat=[[1, 2, 3], [1, 2, 3], [1, 2, 3]])

    def test_init_Error_b_wrong_shape(self):
        """
        ValueError if b has incorrect shape.
        """
        with self.assertRaises(ValueError):
            nbh.Polytope(b=[1, 2, 3])

    def test_repr(self):
        """
        Input/Output-Test.
        """

        result = repr(nbh.Polytope(self.mat, self.b))
        expected_result = f"Polytope(mat={self.mat}, b={self.b})"
        self.assertEqual(result, expected_result,
                         f"Expected {expected_result} but got {result}!")

    def test_is_contained_in_default(self):
        """
        Input/Output-Test.
        """
        result = nbh.Polytope().is_contained_in(self.pts)
        expected_result = [True, False, False]
        np.testing.assert_array_equal(result, expected_result,
                     f"Expected {expected_result} but got {result}!")

    def test_is_contained_in_custom_mat(self):
        """
        Input/Output-Test.
        """

        result = nbh.Polytope(self.mat).is_contained_in(self.pts)
        expected_result = [True, False, False]
        np.testing.assert_array_equal(result, expected_result,
                                      f"Expected {expected_result} but got {result}!")

    def test_is_contained_in_custom_b(self):
        """
        Input/Output-Test.
        """
        result = nbh.Polytope(b=self.b).is_contained_in(self.pts)
        expected_result = [True, False, False]
        np.testing.assert_array_equal(result, expected_result,
                                      f"Expected {expected_result} but got {result}!")

    def test_is_contained_in_edge_empty_pts(self):
        """
        EdgeCase: Empty pts.
        """
        result = nbh.Polytope().is_contained_in([])
        expected_result = []
        np.testing.assert_array_equal(result, expected_result,
                                      f"Expected {expected_result} but got {result}!")

