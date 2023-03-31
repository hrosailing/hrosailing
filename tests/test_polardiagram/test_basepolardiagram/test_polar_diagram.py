"""
Tests
"""

import unittest

import numpy as np

from tests.test_polardiagram.dummy_classes import DummyPolarDiagram


class TestPolarDiagram(unittest.TestCase):
    def setUp(self) -> None:
        self.pd = DummyPolarDiagram()
        self.wind = np.array([[1, 2], [2, 3], [3, 4]])

    def test_get_slices_without_full_info(self):
        # Input/Output
        ws, slices = self.pd.get_slices(
            ws=[1, 2, 3], n_steps=1, full_info=False, test_key="test"
        )
        np.testing.assert_array_equal(ws, np.array([1, 1.5, 2, 2.5, 3]))
        np.testing.assert_array_equal(slices, [[[1], [1], [1]]])

    def test_get_slices_with_full_info(self):
        # Input/Output
        ws, slices, info = self.pd.get_slices(
            ws=[1, 2, 3], n_steps=1, full_info=True, test_key="test"
        )
        np.testing.assert_array_equal(ws, np.array([1, 1.5, 2, 2.5, 3]))
        np.testing.assert_array_equal(slices, [[[1], [1], [1]]])
        self.assertIsNone(info)

    def test_get_windspeeds_ws_None(self):
        # Input/Output
        result = self.pd._get_windspeeds(None, None)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_get_windspeeds_ws_float(self):
        # Input/Output
        result = self.pd._get_windspeeds(3.5, None)
        np.testing.assert_array_equal(result, [3.5])

    def test_get_windspeeds_ws_not_number_or_iterable(self):
        # Exception test
        with self.assertRaises(TypeError):
            self.pd._get_windspeeds(self.pd, None)

    def test_get_windspeeds_not_all_numbers(self):
        # Exception test
        with self.assertRaises(TypeError):
            self.pd._get_windspeeds(["hallo", 1, 2, self.pd], None)

    def test_get_windspeeds_nsteps_negative(self):
        # Exception test
        with self.assertRaises(ValueError):
            self.pd._get_windspeeds(None, -1)

    def test_get_windspeeds_regular_input(self):
        # Input/Output test
        result = self.pd._get_windspeeds([1, 2, 3], 1)
        np.testing.assert_array_equal(result, [1, 1.5, 2, 2.5, 3])

    def test_get_wind_array(self):
        # Input/Output
        result = self.pd._get_wind(self.wind)
        np.testing.assert_array_equal(result, [[1, 2], [2, 3], [3, 4]])

    def test_get_wind_array_transposed(self):
        # Input/Output
        result = self.pd._get_wind(np.array([[1, 2, 3], [2, 3, 4]]))
        np.testing.assert_array_equal(result, [[1, 2], [2, 3], [3, 4]])

    def test_get_wind_array_wrong_shape(self):
        # Exception Test
        with self.assertRaises(ValueError):
            self.pd._get_wind(np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]]))

    def test_get_wind_wrong_tuple(self):
        # Exception Test
        with self.assertRaises(ValueError):
            self.pd._get_wind((np.array([1]), np.array([2]), np.array([3])))

    def test_get_wind_tuple(self):
        # Input/Output
        result = self.pd._get_wind((np.array([1, 2, 3]), np.array([4, 5, 6])))
        np.testing.assert_array_equal(
            result,
            [
                [1, 4],
                [2, 4],
                [3, 4],
                [1, 5],
                [2, 5],
                [3, 5],
                [1, 6],
                [2, 6],
                [3, 6],
            ],
        )

    def test_get_wind_wrong_type(self):
        # Exception test
        with self.assertRaises(TypeError):
            self.pd._get_wind("test")

    def test_get_points_wind_is_None(self):
        # Input/Output
        result = self.pd.get_points(None)
        np.testing.assert_array_equal(result, [[1, 1, 1]])

    def test_get_points_regular_input(self):
        # Input/Output
        result = self.pd.get_points(self.wind)
        np.testing.assert_array_equal(
            result, [[1, 2, 1], [2, 3, 1], [3, 4, 1]]
        )
