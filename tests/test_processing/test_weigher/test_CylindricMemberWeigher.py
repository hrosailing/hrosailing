# pylint: disable-all

from unittest import TestCase

import numpy as np

import hrosailing.core.data as dt
import hrosailing.processing.weigher as wgh
from hrosailing.processing.weigher import (
    hrosailing_standard_scaled_euclidean_norm as st_norm,
)


class TestCylindricMemberWeigher(TestCase):
    def setUp(self):
        self.radius = 0.002
        self.length = 0.0
        self.norm = lambda x: 25 * st_norm()(x)
        self.dimensions = ["TWS", "TWA"]
        self.data = dt.Data().from_dict(
            {
                "TWS": [1.0, 0.5, 1.0],
                "TWA": [2.0, 0.5, 0.0],
                "BSP": [0.0, 1.0, 2.0],
            }
        )
        self.np_arr = np.array(
            [[1, 2, 1], [0.96, 1.5, 0.96], [0.5, 0.5, 0.5], [1.0, 0, 2]]
        )

    def test_init_Error_radius(self):
        with self.assertRaises(ValueError):
            wgh.CylindricMemberWeigher(radius=0)

    def test_init_Error_length(self):
        with self.assertRaises(ValueError):
            wgh.CylindricMemberWeigher(length=-1)

    def test_repr(self):
        result = repr(
            wgh.CylindricMemberWeigher(
                self.radius, self.length, self.norm, self.dimensions
            )
        )
        expected_result = (
            f"CylindricMemberWeigher(radius={self.radius},"
            f"length={self.length}, norm={self.norm.__name__})"
        )

        self.assertEqual(result, expected_result)

    def test_weigh_default(self):
        result = wgh.CylindricMemberWeigher().weigh(self.np_arr)
        expected_result = [1, 1, 0, 0]

        np.testing.assert_array_equal(result, expected_result)

    def test_weigh_custom_radius(self):
        result = wgh.CylindricMemberWeigher(self.radius).weigh(self.np_arr)
        expected_result = [0, 0, 0, 0]

        np.testing.assert_array_equal(result, expected_result)

    def test_weigh_custom_norm(self):
        result = wgh.CylindricMemberWeigher(norm=self.norm).weigh(self.np_arr)
        expected_result = [0, 0, 0, 0]

        np.testing.assert_array_equal(result, expected_result)

    def test_weigh_custom_length(self):
        result = wgh.CylindricMemberWeigher(length=self.length).weigh(
            self.np_arr
        )
        expected_result = [0, 0, 0, 0]

        np.testing.assert_array_equal(result, expected_result)

    def test_weigh_edge_empty_array(self):
        result = wgh.CylindricMemberWeigher().weigh(np.array([]))
        expected_result = []

        np.testing.assert_array_equal(result, expected_result)

    def test__calculate_weight(self):
        result = wgh.CylindricMemberWeigher()._calculate_weight(
            np.array([1, 2]), np.array([[1, 1], [1, 2], [2, 1], [2, 2]])
        )
        expected_result = 0

        self.assertEqual(result, expected_result)

    def test__count_points_in_cylinder(self):
        result = wgh.CylindricMemberWeigher()._count_points_in_cylinder(
            np.array([1, 2]), np.array([[1, 1], [1, 2], [2, 1], [2, 2]])
        )
        expected_result = 1

        np.testing.assert_array_equal(result, expected_result)
