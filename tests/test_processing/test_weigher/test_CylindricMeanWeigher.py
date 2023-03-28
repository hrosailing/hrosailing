from unittest import TestCase
import numpy as np

import hrosailing.processing.weigher as wgh
import hrosailing.core.data as dt
from hrosailing.processing.weigher import hrosailing_standard_scaled_euclidean_norm as st_norm


class TestCylindricMeanWeigher(TestCase):
    def setUp(self) -> None:
        self.radius = 1 / 3 * 0.05
        self.norm = lambda x: 3 * st_norm(["TWS", "TWA"])(x)
        self.dimensions = ["TWS", "TWA"]
        self.data = dt.Data().from_dict({"TWS": [1.0, .5, .25], "TWA": [2.0, .5, 0.0], "BSP": [0.0, 1.0, 2.0]})
        self.np_arr = np.array([[1, 2, 0], [0.5, 0.5, 1], [0.25, 0, 2]])

    def test_init_Error(self):
        """
        ValueError if radius < 0.
        """
        with self.assertRaises(ValueError):
            wgh.CylindricMeanWeigher(radius=0)

    def test_repr(self):
        """
        Input/Output-Test.
        """

        result = repr(wgh.CylindricMeanWeigher(self.radius, self.norm, self.dimensions))
        expected_result = f"CylindricMeanWeigher(radius={self.radius}, norm={self.norm.__name__})"
        self.assertEqual(result, expected_result,
                         f"Expected {expected_result} but got {result}!")

    def test_weigh_default_nparr(self):
        """
        Input/Output-Test.
        """
        result = wgh.CylindricMeanWeigher().weigh(self.np_arr)
        expected_result = [0, 1, 0]
        np.testing.assert_array_equal(result, expected_result,
                                      f"Expected {expected_result} but got {result}!")

    def test_weigh_custom_radius_nparr(self):
        """
        Input/Output-Test.
        """
        result = wgh.CylindricMeanWeigher(radius=self.radius).weigh(self.np_arr)
        expected_result = [1, 0, 0]
        np.testing.assert_array_equal(result, expected_result,
                                      f"Expected {expected_result} but got {result}!")

    def test_weigh_custom_norm_nparr(self):
        """
        Input/Output-Test.
        """
        result = wgh.CylindricMeanWeigher(norm=self.norm).weigh(self.np_arr)
        expected_result = [1, 0, 0]
        np.testing.assert_array_equal(result, expected_result,
                                      f"Expected {expected_result} but got {result}!")

    def test_weigh_default_data(self):
        """
        Input/Output-Test.
        """
        result = wgh.CylindricMeanWeigher().weigh(self.data)
        expected_result = [0, 1, 0]
        np.testing.assert_array_equal(result, expected_result,
                                      f"Expected {expected_result} but got {result}!")

    def test_weigh_custom_radius_data(self):
        """
        Input/Output-Test.
        """
        result = wgh.CylindricMeanWeigher(radius=self.radius).weigh(self.data)
        expected_result = [1, 0, 0]
        np.testing.assert_array_equal(result, expected_result,
                                      f"Expected {expected_result} but got {result}!")

    def test_weigh_custom_norm_data(self):
        """
        Input/Output-Test.
        """
        result = wgh.CylindricMeanWeigher(norm=self.norm).weigh(self.data)
        expected_result = [1, 0, 0]
        np.testing.assert_array_equal(result, expected_result,
                                      f"Expected {expected_result} but got {result}!")

    def test_weigh_edge_empty_array(self):
        """
        EdgeCase: Empty array.
        """
        result = wgh.CylindricMeanWeigher().weigh(np.array([]))
        expected_result = []
        np.testing.assert_array_equal(result, expected_result,
                                      f"Expected {expected_result} but got {result}!")

    def test__calculate_weight(self):
        """
        Input/Output-Test.
        """

        result = wgh.CylindricMeanWeigher()._calculate_weight(self.np_arr[0, :2], self.np_arr[1:, :2],
                                                              self.np_arr[1:, 2], self.np_arr[0, 2], self.dimensions)
        expected_result = 3
        self.assertEqual(result, expected_result, f"Expected {expected_result} but got {result}!")

    def test__determine_points_in_cylinder(self):
        """
        Input/Output-Test.
        """

        result = wgh.CylindricMeanWeigher()._determine_points_in_cylinder(self.np_arr[0, :2], self.np_arr[1:, :2],
                                                                          self.np_arr[1:, 2], self.dimensions)
        expected_result = [1., 2.]
        np.testing.assert_array_equal(result, expected_result,
                                      f"Expected {expected_result} but got {result}!")
