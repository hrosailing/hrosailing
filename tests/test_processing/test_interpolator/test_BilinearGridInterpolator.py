# pylint: disable-all

from unittest import TestCase

import numpy as np

import hrosailing.core.data as dt
import hrosailing.processing.interpolator as itp

from hrosailing.core.exceptions import BilinearInterpolatorOutsideGridException
from hrosailing.core.exceptions import BilinearInterpolatorNoGridException

class TestBilinearGridInterpolator(TestCase):
    def setUp(self):

        self.wpts = dt.WeightedPoints(
            np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 0.5], [1, -1, 1]]),
            np.ones(4),
        )

        self.grid_pt = np.array([0, 0])

    def test_repr(self):
        result = repr(
            itp.BilinearGridInterpolator()
        )
        expected_result = (
            f"BilinearGridInterpolator()"
        )

        self.assertEqual(result, expected_result)

    def test_interpolate_default(self):
        result = itp.BilinearGridInterpolator().interpolate(
            self.wpts, self.grid_pt
        )
        expected_result = 0.875

        self.assertAlmostEqual(result, expected_result)

    def test_interpolate_edge_grid_pt_in_wpts(self):
        result = itp.BilinearGridInterpolator().interpolate(
            dt.WeightedPoints(
                np.array([[-1, 1, 1], [-1, -1, 0.5], [0, 0, 3]]), np.ones(3)
            ),
            self.grid_pt,
        )
        expected_result = 3

        self.assertEqual(result, expected_result)

    def test_simple(self):

        wpts = dt.WeightedPoints(
            np.array([[0, 0, 0], 
                      [0, 1, 1], 
                      [1, 0, 2],
                      [1, 1, 4],
                      ]),
            np.ones(3),
        )

        grid_pt = np.array([0, 0])
        expected_result = 0.0
        result = itp.BilinearGridInterpolator().interpolate(
            wpts, grid_pt
        )
        self.assertAlmostEqual(result, expected_result)

        grid_pt = np.array([1, 1])
        expected_result = 4.0
        result = itp.BilinearGridInterpolator().interpolate(
            wpts, grid_pt
        )
        self.assertAlmostEqual(result, expected_result)

        grid_pt = np.array([0, 1])
        expected_result = 1.0
        result = itp.BilinearGridInterpolator().interpolate(
            wpts, grid_pt
        )
        self.assertAlmostEqual(result, expected_result)

        grid_pt = np.array([1, 0])
        expected_result = 2.0
        result = itp.BilinearGridInterpolator().interpolate(
            wpts, grid_pt
        )
        self.assertAlmostEqual(result, expected_result)

        grid_pt = np.array([1, 1])
        expected_result = 4.0
        result = itp.BilinearGridInterpolator().interpolate(
            wpts, grid_pt
        )
        self.assertAlmostEqual(result, expected_result)

        grid_pt = np.array([0.0, 0.5])
        expected_result = 0.5
        result = itp.BilinearGridInterpolator().interpolate(
            wpts, grid_pt
        )
        self.assertAlmostEqual(result, expected_result)

        grid_pt = np.array([0.5, 0.0])
        expected_result = 1.0
        result = itp.BilinearGridInterpolator().interpolate(
            wpts, grid_pt
        )
        self.assertAlmostEqual(result, expected_result)

        grid_pt = np.array([1.0, 0.5])
        expected_result = 3.0
        result = itp.BilinearGridInterpolator().interpolate(
            wpts, grid_pt
        )
        self.assertAlmostEqual(result, expected_result)

        grid_pt = np.array([0.5, 1.0])
        expected_result = 2.5
        result = itp.BilinearGridInterpolator().interpolate(
            wpts, grid_pt
        )
        self.assertAlmostEqual(result, expected_result)

        grid_pt = np.array([0.75, 1.0])
        expected_result = 3.25
        result = itp.BilinearGridInterpolator().interpolate(
            wpts, grid_pt
        )
        self.assertAlmostEqual(result, expected_result)

    def test_grid_Error(self):

        wpts = dt.WeightedPoints(
            np.array([[0, 0, 0], 
                      [0, 1, 1], 
                      [1, 0, 2],
                      [1, 1, 4],
                      ]),
            np.ones(3),
        )

        grid_pt = np.array([-0.1, 0])
        with self.assertRaises(BilinearInterpolatorOutsideGridException):
            itp.BilinearGridInterpolator().interpolate(wpts, grid_pt)
        
        grid_pt = np.array([0.0, -0.1])
        with self.assertRaises(BilinearInterpolatorOutsideGridException):
            itp.BilinearGridInterpolator().interpolate(wpts, grid_pt)

        grid_pt = np.array([1.1, 0.5])
        with self.assertRaises(BilinearInterpolatorOutsideGridException):
            itp.BilinearGridInterpolator().interpolate(wpts, grid_pt)

        grid_pt = np.array([0.1, 2.33])
        with self.assertRaises(BilinearInterpolatorOutsideGridException):
            itp.BilinearGridInterpolator().interpolate(wpts, grid_pt)

