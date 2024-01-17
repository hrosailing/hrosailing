# pylint: disable-all

from unittest import TestCase

import numpy as np

import hrosailing.processing.neighbourhood as nbh


class TestEllipsoid(TestCase):
    def setUp(self):
        self.lin_trans = np.array([[2, 0], [0, 2]])
        self.radius = 0.02
        self.norm = lambda x: 0.03 * np.linalg.norm(x, ord=2, axis=1)
        self.pts = np.array([[1, 2], [3, 1], [0.5, 0.5]])

    def test_init_Error_lin_trans_wrong_shape(self):
        with self.assertRaises(ValueError):
            nbh.Ellipsoid(lin_trans=[[1, 2, 3], [1, 2, 3]])

    def test_init_Error_lin_trans_singular(self):
        with self.assertRaises(ValueError):
            nbh.Ellipsoid([[0, 0], [0, 0]])

    def test_init_Error_radius_non_positive(self):
        with self.assertRaises(ValueError):
            nbh.Ellipsoid(radius=0)

    def test_repr(self):
        result = repr(nbh.Ellipsoid(self.lin_trans, self.norm, self.radius))
        expected_result = (
            f"Ellipsoid(lin_trans={np.linalg.inv(self.lin_trans)}, "
            f"norm={self.norm.__name__}, radius={self.radius})"
        )

        self.assertEqual(result, expected_result)

    def test_is_contained_in_default(self):
        result = nbh.Ellipsoid().is_contained_in(self.pts)
        expected_result = [True, False, True]

        np.testing.assert_array_equal(result, expected_result)

    def test_is_contained_in_custom_lin_trans(self):
        result = nbh.Ellipsoid(lin_trans=self.lin_trans).is_contained_in(
            self.pts
        )
        expected_result = [True, True, True]

        np.testing.assert_array_equal(result, expected_result)

    def test_is_contained_in_custom_norm(self):
        result = nbh.Ellipsoid(norm=self.norm).is_contained_in(self.pts)
        expected_result = [False, False, True]

        np.testing.assert_array_equal(result, expected_result)

    def test_is_contained_in_custom_radius(self):
        result = nbh.Ellipsoid(radius=self.radius).is_contained_in(self.pts)
        expected_result = [False, False, True]

        np.testing.assert_array_equal(result, expected_result)

    def test_is_contained_in_edge_empty_pts(self):
        result = nbh.Ellipsoid().is_contained_in([])
        expected_result = []

        np.testing.assert_array_equal(result, expected_result)

    def test__transform_ellipsoid_to_ball(self):
        result = nbh.Ellipsoid()._transform_ellipsoid_to_ball(self.pts)
        expected_result = self.pts

        np.testing.assert_array_equal(result, expected_result)
