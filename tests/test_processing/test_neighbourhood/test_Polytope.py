# pylint: disable-all

from unittest import TestCase

import numpy as np

import hrosailing.processing.neighbourhood as nbh


class TestPolytope(TestCase):
    def setUp(self):
        self.mat = np.array([[-1, 0], [0, -1], [1, 2], [2, 1]])
        self.b = np.asarray([0.2, 0.5, 1, 2])
        self.pts = [[0.01, 0.02], [0.3, 1], [0.5, 0.5]]

    def test_init_Error_mat_wrong_shape(self):
        with self.assertRaises(ValueError):
            nbh.Polytope(mat=[[1, 2, 3], [1, 2, 3], [1, 2, 3]])

    def test_init_Error_b_wrong_shape(self):
        with self.assertRaises(ValueError):
            nbh.Polytope(b=[1, 2, 3])

    def test_repr(self):
        result = repr(nbh.Polytope(self.mat, self.b))
        expected_result = f"Polytope(mat={self.mat}, b={self.b})"

        self.assertEqual(result, expected_result)

    def test_is_contained_in_default(self):
        result = nbh.Polytope().is_contained_in(self.pts)
        expected_result = [True, False, False]

        np.testing.assert_array_equal(result, expected_result)

    def test_is_contained_in_custom_mat(self):
        result = nbh.Polytope(self.mat).is_contained_in(self.pts)
        expected_result = [True, False, False]

        np.testing.assert_array_equal(result, expected_result)

    def test_is_contained_in_custom_b(self):
        result = nbh.Polytope(b=self.b).is_contained_in(self.pts)
        expected_result = [True, False, False]

        np.testing.assert_array_equal(result, expected_result)

    def test_is_contained_in_edge_empty_pts(self):
        result = nbh.Polytope().is_contained_in([])
        expected_result = []

        np.testing.assert_array_equal(result, expected_result)
