# pylint: disable=missing-docstring
# pylint: disable=too-many-public-methods
# pylint: disable=import-outside-toplevel

import unittest

import numpy as np

import matplotlib.pyplot as plt

import hrosailing.polardiagram as pol
from hrosailing.polardiagram._basepolardiagram import (
    PolarDiagramException,
    PolarDiagramInitializationException,
)


class PolarDiagramPointCloudTest(unittest.TestCase):
    def setUp(self):
        self.points = np.array(
            [
                [2, 10, 1.1],
                [2, 15, 1.5],
                [2, 20, 1.7],
                [2, 25, 2.1],
                [4, 10, 2],
                [4, 15, 2.4],
                [4, 20, 2.6],
                [4, 25, 3],
                [6, 10, 3],
                [6, 15, 3.1],
                [6, 20, 3.5],
                [6, 25, 3.8],
                [8, 10, 4],
                [8, 15, 4.1],
                [8, 20, 4.4],
                [8, 25, 4.6],
            ]
        )
        self.pc = pol.PolarDiagramPointcloud(self.points)

    def test_init(self):
        np.testing.assert_array_equal(self.pc.points, self.points)

    def test_wind_speeds(self):
        np.testing.assert_array_equal(self.pc.wind_speeds, [2, 4, 6, 8])

    def test_wind_angles(self):
        np.testing.assert_array_equal(self.pc.wind_angles, [10, 15, 20, 25])

    def test_points(self):
        np.testing.assert_array_equal(self.pc.points, self.points)

    def test_add_points(self):
        self.pc.add_points([[2.3, 15.5, 1.65], [3.7, 20.1, 2.43]])
        self.points = np.row_stack(
            (self.points, np.array([[2.3, 15.5, 1.65], [3.7, 20.1, 2.43]]))
        )
        np.testing.assert_array_equal(self.pc.points, self.points)

    def test_symmetric_polar_diagram(self):
        sym_pc = self.pc.symmetrize()
        sym_pts = self.pc.points
        sym_pts[:, 1] = 360 - sym_pts[:, 1]
        pts = np.row_stack((self.pc.points, sym_pts))
        np.testing.assert_array_equal(sym_pc.points, pts)

    def test_get_slice(self):
        ws, wa, bsp = self.pc.get_slices(4)
        self.assertEqual(ws, [4])
        np.testing.assert_array_equal(wa[0], np.deg2rad([10, 15, 20, 25]))
        np.testing.assert_array_equal(bsp[0], np.array([2, 2.4, 2.6, 3]))

    def test_get_slice_exception(self):
        with self.assertRaises(PolarDiagramException):
            self.pc.get_slices(0)

    def test_get_slices_list(self):
        ws, wa, bsp = self.pc.get_slices([4, 8])
        self.assertEqual(ws, [4, 8])
        self.assertEqual(type(wa), list)
        self.assertEqual(type(bsp), list)
        self.assertEqual(len(wa), 2)
        self.assertEqual(len(bsp), 2)
        np.testing.assert_array_equal(wa[0], np.deg2rad([10, 15, 20, 25]))
        np.testing.assert_array_equal(wa[1], np.deg2rad([10, 15, 20, 25]))
        np.testing.assert_array_equal(bsp[0], np.array([2, 2.4, 2.6, 3]))
        np.testing.assert_array_equal(bsp[1], np.array([4, 4.1, 4.4, 4.6]))

    def test_get_slices_exception_empty(self):
        with self.assertRaises(PolarDiagramException):
            self.pc.get_slices([])

    def test_get_slices_exception_no_slices(self):
        with self.assertRaises(PolarDiagramException):
            self.pc.get_slices([0, 2])

    def test_get_slices_range(self):
        ws, wa, bsp = self.pc.get_slices((3, 9))
        self.assertEqual(ws, [3, 4.2, 5.4, 6.6, 7.8, 9])
        self.assertEqual(type(wa), list)
        self.assertEqual(type(bsp), list)
        self.assertEqual(len(wa), len(ws))
        self.assertEqual(len(bsp), len(ws))
        bsps = [
            np.array([1.1, 2, 1.5, 2.4, 1.7, 2.6, 2.1, 3]),
            np.array([2, 2.4, 2.6, 3]),
            np.array([3, 3.1, 3.5, 3.8]),
            np.array([3, 3.1, 3.5, 3.8]),
            np.array([4, 4.1, 4.4, 4.6]),
            np.array([4, 4.1, 4.4, 4.6]),
        ]
        answers = [
            np.deg2rad([10, 10, 15, 15, 20, 20, 25, 25]),
            np.deg2rad([10, 15, 20, 25]),
            np.deg2rad([10, 15, 20, 25]),
            np.deg2rad([10, 15, 20, 25]),
            np.deg2rad([10, 15, 20, 25]),
            np.deg2rad([10, 15, 20, 25]),
        ]
        for i in range(6):
            np.testing.assert_array_equal(wa[i], answers[i])
            np.testing.assert_array_equal(bsp[i], bsps[i])

    def test_get_slices_range_no_slices(self):
        with self.assertRaises(PolarDiagramException):
            self.pc.get_slices((0, 1))

    def test_get_slices_range_empty_interval(self):
        with self.assertRaises(PolarDiagramException):
            self.pc.get_slices((1, 0))