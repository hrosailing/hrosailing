# pylint: disable-all

from unittest import TestCase

import numpy as np

import hrosailing.core.data as dt
import hrosailing.processing.weigher as wgh
from hrosailing.processing.weigher import (
    hrosailing_standard_scaled_euclidean_norm as st_norm,
)


class TestCylindricMeanWeigher(TestCase):
    def setUp(self):
        self.radius = 1 / 3 * 0.05
        self.norm = lambda x: 3 * st_norm(["TWS", "TWA"])(x)
        self.dimensions = ["TWS", "TWA"]
        self.data = dt.Data().from_dict(
            {
                "TWS": [1.0, 0.5, 0.25],
                "TWA": [2.0, 0.5, 0.0],
                "BSP": [0.0, 1.0, 2.0],
            }
        )
        self.np_arr = np.array([[1, 2, 0], [0.5, 0.5, 1], [0.25, 0, 2]])

    def test_init_Error(self):
        with self.assertRaises(ValueError):
            wgh.CylindricMeanWeigher(radius=0)

    def test_repr(self):
        result = repr(
            wgh.CylindricMeanWeigher(self.radius, self.norm, self.dimensions)
        )
        expected_result = (
            f"CylindricMeanWeigher(radius={self.radius},"
            f" norm={self.norm.__name__})"
        )

        self.assertEqual(result, expected_result)

    def test_weigh_default_nparr(self):
        result = wgh.CylindricMeanWeigher().weigh(self.np_arr)
        expected_result = [0, 1, 0]

        np.testing.assert_array_equal(result, expected_result)

    def test_weigh_custom_radius_nparr(self):
        result = wgh.CylindricMeanWeigher(radius=self.radius).weigh(
            self.np_arr
        )
        expected_result = [1, 0, 0]

        np.testing.assert_array_equal(result, expected_result)

    def test_weigh_custom_norm_nparr(self):
        result = wgh.CylindricMeanWeigher(norm=self.norm).weigh(self.np_arr)
        expected_result = [1, 0, 0]

        np.testing.assert_array_equal(result, expected_result)

    def test_weigh_default_data(self):
        result = wgh.CylindricMeanWeigher().weigh(self.data)
        expected_result = [0, 1, 0]

        np.testing.assert_array_equal(result, expected_result)

    def test_weigh_custom_radius_data(self):
        result = wgh.CylindricMeanWeigher(radius=self.radius).weigh(self.data)
        expected_result = [1, 0, 0]

        np.testing.assert_array_equal(result, expected_result)

    def test_weigh_custom_norm_data(self):
        result = wgh.CylindricMeanWeigher(norm=self.norm).weigh(self.data)
        expected_result = [1, 0, 0]

        np.testing.assert_array_equal(result, expected_result)

    def test_weigh_edge_empty_array(self):
        result = wgh.CylindricMeanWeigher().weigh(np.array([]))
        expected_result = []

        np.testing.assert_array_equal(result, expected_result)

    def test__calculate_weight(self):
        result = wgh.CylindricMeanWeigher()._calculate_weight(
            self.np_arr[0, :2],
            self.np_arr[1:, :2],
            self.np_arr[1:, 2],
            self.np_arr[0, 2],
            self.dimensions,
        )
        expected_result = 3

        self.assertEqual(result, expected_result)

    def test__determine_points_in_cylinder(self):
        result = wgh.CylindricMeanWeigher()._determine_points_in_cylinder(
            self.np_arr[0, :2],
            self.np_arr[1:, :2],
            self.np_arr[1:, 2],
            self.dimensions,
        )
        expected_result = [1.0, 2.0]

        np.testing.assert_array_equal(result, expected_result)
