from unittest import TestCase
import numpy as np

import hrosailing.processing.weigher as wgh
import hrosailing.core.data as dt
import hrosailing.core.computing as comp


class TestCylindricMeanWeigher(TestCase):
    def setUp(self) -> None:
        self.radius = 1/3 * 0.05
        self.norm = lambda x: 3 * comp.scaled_euclidean_norm(x)
        self.dimensions = 3
        self.data = dt.Data().from_dict({})
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

    def test_weigh_default(self):
        """
        Input/Output-Test.
        """
        result = wgh.CylindricMeanWeigher().weigh(self.np_arr)
        expected_result = [0, 1, 0]
        np.testing.assert_array_equal(result, expected_result,
                                      f"Expected {expected_result} but got {result}!")

    def test_weigh_custom_radius(self):
        """
        Input/Output-Test.
        """
        # TODO: this gets the wrong result
        result = wgh.CylindricMeanWeigher(radius=self.radius).weigh(self.np_arr)
        expected_result = [0, 1, 0]
        np.testing.assert_array_equal(result, expected_result,
                                      f"Expected {expected_result} but got {result}!")

    def test_weigh_custom_norm(self):
        """
        Input/Output-Test.
        """
        result = wgh.CylindricMeanWeigher(norm=self.norm).weigh(self.np_arr)
        expected_result = [0, 1, 0]
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
