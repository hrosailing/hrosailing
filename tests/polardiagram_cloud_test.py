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

    def test_boat_speeds(self):
        np.testing.assert_array_equal(self.pc.boat_speeds, [1.1, 1.5, 1.7, 2.1, 2, 2.4, 2.6, 3,
                                                            3, 3.1, 3.5, 3.8, 4, 4.1, 4.4, 4.6])

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

    def test_plot_polar(self):
        f, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        self.pc.plot_polar(ax=ax)
        k = 0
        for i in range(0, 4):
            with self.subTest(i=i):
                x_plot = ax.lines[i].get_xdata()
                y_plot = ax.lines[i].get_ydata()
                np.testing.assert_array_equal(x_plot, np.deg2rad(self.pc.wind_angles))
                np.testing.assert_array_equal(y_plot, self.pc.boat_speeds[k:k+4])
                k = k+4

    def test_plot_polar_single_element_ws(self):
        f, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        self.pc.plot_polar(ws=4, ax=ax)
        x_plot = ax.lines[0].get_xdata()
        y_plot = ax.lines[0].get_ydata()
        np.testing.assert_array_equal(x_plot, np.deg2rad(self.pc.wind_angles))
        np.testing.assert_array_equal(y_plot, self.pc.boat_speeds[4:8])

    def test_plot_polar_interval_ws(self):
        f, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        self.pc.plot_polar(ws=(4, 8), ax=ax)
        k = 4
        for i in range(0, 4):
            with self.subTest(i=i):
                x_plot = ax.lines[i].get_xdata()
                y_plot = ax.lines[i].get_ydata()
                np.testing.assert_array_equal(x_plot, np.deg2rad(self.pc.wind_angles))
                np.testing.assert_array_equal(y_plot, self.pc.boat_speeds[k:k + 4])
                if i == 1:
                    k = 8
                else:
                    k = k + 4

    def test_plot_polar_mixed_list_ws(self):
        f, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        self.pc.plot_polar(ws=[(4, 8), 2], ax=ax)
        sorted_wind_angles = sorted(np.deg2rad(np.concatenate((self.pc.wind_angles,
                                                               self.pc.wind_angles,
                                                               self.pc.wind_angles))))
        for i in range(0, 2):
            with self.subTest(i=i):
                x_plot = ax.lines[i].get_xdata()
                y_plot = ax.lines[i].get_ydata()
                if i == 1:
                    np.testing.assert_array_equal(x_plot, np.deg2rad(self.pc.wind_angles))
                    np.testing.assert_array_equal(y_plot, self.pc.boat_speeds[0:4])
                else:
                    np.testing.assert_array_equal(x_plot, sorted_wind_angles)
                    np.testing.assert_array_equal(y_plot, [2, 3, 4, 2.4, 3.1, 4.1, 2.6, 3.5, 4.4, 3, 3.8, 4.6])

    def test_plot_polar_mixed_tuple_ws(self):
        f, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        self.pc.plot_polar(ws=((4, 8), 2), ax=ax)
        sorted_wind_angles = sorted(np.deg2rad(np.concatenate((self.pc.wind_angles,
                                                               self.pc.wind_angles,
                                                               self.pc.wind_angles))))
        for i in range(0, 2):
            with self.subTest(i=i):
                x_plot = ax.lines[i].get_xdata()
                y_plot = ax.lines[i].get_ydata()
                if i == 1:
                    np.testing.assert_array_equal(x_plot, np.deg2rad(self.pc.wind_angles))
                    np.testing.assert_array_equal(y_plot, self.pc.boat_speeds[0:4])
                else:
                    np.testing.assert_array_equal(x_plot, sorted_wind_angles)
                    np.testing.assert_array_equal(y_plot, [2, 3, 4, 2.4, 3.1, 4.1, 2.6, 3.5, 4.4, 3, 3.8, 4.6])

    def test_plot_polar_mixed_set_ws(self):
        f, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        self.pc.plot_polar(ws={(4, 8), 2}, ax=ax)
        sorted_wind_angles = sorted(np.deg2rad(np.concatenate((self.pc.wind_angles,
                                                               self.pc.wind_angles,
                                                               self.pc.wind_angles))))
        for i in range(0, 2):
            with self.subTest(i=i):
                x_plot = ax.lines[i].get_xdata()
                y_plot = ax.lines[i].get_ydata()
                if i == 1:
                    np.testing.assert_array_equal(x_plot, np.deg2rad(self.pc.wind_angles))
                    np.testing.assert_array_equal(y_plot, self.pc.boat_speeds[0:4])
                else:
                    np.testing.assert_array_equal(x_plot, sorted_wind_angles)
                    np.testing.assert_array_equal(y_plot, [2, 3, 4, 2.4, 3.1, 4.1, 2.6, 3.5, 4.4, 3, 3.8, 4.6])

    def test_plot_polar_n_steps(self):
        f, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        self.pc.plot_polar(ws=(4, 8), ax=ax, n_steps=3)
        # test for ws still missing
        k = 4
        for i in range(0, 3):
            with self.subTest(i=i):
                x_plot = ax.lines[i].get_xdata()
                y_plot = ax.lines[i].get_ydata()
                np.testing.assert_array_equal(x_plot, np.deg2rad(self.pc.wind_angles))
                np.testing.assert_array_equal(y_plot, self.pc.boat_speeds[k:k+4])
                k = k+4

    def test_plot_polar_range_single_ws(self):
        f, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        self.pc.plot_polar(ws=4, ax=ax, range_=2)
        sorted_wind_angles = sorted(np.deg2rad(np.concatenate((self.pc.wind_angles,
                                                               self.pc.wind_angles,
                                                               self.pc.wind_angles))))
        x_plot = ax.lines[0].get_xdata()
        y_plot = ax.lines[0].get_ydata()
        np.testing.assert_array_equal(x_plot, sorted_wind_angles)
        np.testing.assert_array_equal(y_plot, [1.1, 2, 3, 1.5, 2.4, 3.1, 1.7, 2.6, 3.5, 2.1, 3, 3.8])

    def test_plot_polar_range_mixed_list(self):
        pd = pol.from_csv("../examples/csv-format-examples/cloud_hro_format_example.csv")
        f, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        pd.plot_polar(ws=[(14, 20), 8], ax=ax, range_=2)
        sorted_wind_angles = sorted(np.deg2rad(np.concatenate((pd.wind_angles,
                                                               pd.wind_angles,
                                                               pd.wind_angles))))
        for i in range(0, 2):
            with self.subTest(i=i):
                x_plot = ax.lines[i].get_xdata()
                y_plot = ax.lines[i].get_ydata()
                np.testing.assert_array_equal(x_plot, sorted_wind_angles)
                if i == 0:
                    np.testing.assert_array_equal(y_plot, [5.47, 5.81, 5.66, 5.67, 6.17, 5.94, 5.95, 6.86, 6.27, 7.35,
                                                           6.7, 6.49, 7.48, 6.79, 8.76, 7.32, 6.62, 9.74, 8.34, 6.77,
                                                           6.22, 6.22, 5.78, 7.32])
                else:
                    np.testing.assert_array_equal(y_plot, [3.74, 4.96, 4.48, 3.98, 5.18, 4.73, 4.16, 5.35, 4.93, 5.64,
                                                           5.19, 4.35, 5.22, 4.39, 5.68, 5.11, 4.23, 5.58, 5.33, 4.64,
                                                           3.72, 4.1, 3.21, 4.87])

    def test_plot_polar_range_mixed_tuple(self):
        pd = pol.from_csv("../examples/csv-format-examples/cloud_hro_format_example.csv")
        f, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        pd.plot_polar(ws=((14, 20), 8), ax=ax, range_=2)
        sorted_wind_angles = sorted(np.deg2rad(np.concatenate((pd.wind_angles,
                                                               pd.wind_angles,
                                                               pd.wind_angles))))
        for i in range(0, 2):
            with self.subTest(i=i):
                x_plot = ax.lines[i].get_xdata()
                y_plot = ax.lines[i].get_ydata()
                np.testing.assert_array_equal(x_plot, sorted_wind_angles)
                if i == 0:
                    np.testing.assert_array_equal(y_plot, [5.47, 5.81, 5.66, 5.67, 6.17, 5.94, 5.95, 6.86, 6.27, 7.35,
                                                           6.7, 6.49, 7.48, 6.79, 8.76, 7.32, 6.62, 9.74, 8.34, 6.77,
                                                           6.22, 6.22, 5.78, 7.32])
                else:
                    np.testing.assert_array_equal(y_plot, [3.74, 4.96, 4.48, 3.98, 5.18, 4.73, 4.16, 5.35, 4.93, 5.64,
                                                           5.19, 4.35, 5.22, 4.39, 5.68, 5.11, 4.23, 5.58, 5.33, 4.64,
                                                           3.72, 4.1, 3.21, 4.87])

    def test_plot_polar_range_mixed_set(self):
        pd = pol.from_csv("../examples/csv-format-examples/cloud_hro_format_example.csv")
        f, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        pd.plot_polar(ws={(14, 20), 8}, ax=ax, range_=2)
        sorted_wind_angles = sorted(np.deg2rad(np.concatenate((pd.wind_angles,
                                                               pd.wind_angles,
                                                               pd.wind_angles))))
        for i in range(0, 2):
            with self.subTest(i=i):
                x_plot = ax.lines[i].get_xdata()
                y_plot = ax.lines[i].get_ydata()
                np.testing.assert_array_equal(x_plot, sorted_wind_angles)
                if i == 0:
                    np.testing.assert_array_equal(y_plot, [5.47, 5.81, 5.66, 5.67, 6.17, 5.94, 5.95, 6.86, 6.27, 7.35,
                                                           6.7, 6.49, 7.48, 6.79, 8.76, 7.32, 6.62, 9.74, 8.34, 6.77,
                                                           6.22, 6.22, 5.78, 7.32])
                else:
                    np.testing.assert_array_equal(y_plot, [3.74, 4.96, 4.48, 3.98, 5.18, 4.73, 4.16, 5.35, 4.93, 5.64,
                                                           5.19, 4.35, 5.22, 4.39, 5.68, 5.11, 4.23, 5.58, 5.33, 4.64,
                                                           3.72, 4.1, 3.21, 4.87])

    def test_plot_polar_exception_single_element_ws(self):
        with self.assertRaises(PolarDiagramException):
            self.pc.plot_polar(ws=10)

    def test_plot_polar_exception_interval_ws(self):
        with self.assertRaises(PolarDiagramException):
            self.pc.plot_polar(ws=(4, 10))

    def test_plot_polar_exception_mixed_iterable_ws(self):
        with self.subTest(i=0):
            with self.assertRaises(PolarDiagramException):
                self.pc.plot_polar(ws=[(10, 20), 4])
        with self.subTest(i=1):
            with self.assertRaises(PolarDiagramException):
                self.pc.plot_polar(ws=((10, 20), 4))
        with self.subTest(i=2):
            with self.assertRaises(PolarDiagramException):
                self.pc.plot_polar(ws={(10, 20), 4})

    def test_plot_flat(self):
        f, ax = plt.subplots()
        self.pc.plot_flat(ax=ax)
        k = 0
        for i in range(0, len(self.pc.wind_speeds)):
            with self.subTest(i=i):
                x_plot = ax.lines[i].get_xdata()
                y_plot = ax.lines[i].get_ydata()
                np.testing.assert_array_almost_equal(x_plot, self.pc.wind_angles, 10)
                np.testing.assert_array_equal(y_plot, self.pc.boat_speeds[k:k + 4])
                k = k + 4

    def test_plot_flat_single_element_ws(self):
        f, ax = plt.subplots()
        self.pc.plot_flat(ws=4, ax=ax)
        x_plot = ax.lines[0].get_xdata()
        y_plot = ax.lines[0].get_ydata()
        np.testing.assert_array_almost_equal(x_plot, self.pc.wind_angles, 10)
        np.testing.assert_array_equal(y_plot, self.pc.boat_speeds[4:8])

    def test_plot_flat_interval_ws(self):
        f, ax = plt.subplots()
        self.pc.plot_flat(ws=(4, 8), ax=ax)
        k = 4
        for i in range(0, 4):
            with self.subTest(i=i):
                x_plot = ax.lines[i].get_xdata()
                y_plot = ax.lines[i].get_ydata()
                np.testing.assert_array_almost_equal(x_plot, self.pc.wind_angles, 10)
                np.testing.assert_array_equal(y_plot, self.pc.boat_speeds[k:k + 4])
                if i == 1:
                    k = 8
                else:
                    k = k + 4

    def test_plot_flat_mixed_list_ws(self):
        f, ax = plt.subplots()
        self.pc.plot_flat(ws=[(4, 8), 2], ax=ax)
        sorted_wind_angles = sorted(np.concatenate((self.pc.wind_angles,
                                                    self.pc.wind_angles,
                                                    self.pc.wind_angles)))
        for i in range(0, 2):
            with self.subTest(i=i):
                x_plot = ax.lines[i].get_xdata()
                y_plot = ax.lines[i].get_ydata()
                if i == 1:
                    np.testing.assert_array_almost_equal(x_plot, self.pc.wind_angles, 10)
                    np.testing.assert_array_equal(y_plot, self.pc.boat_speeds[0:4])
                else:
                    np.testing.assert_array_almost_equal(x_plot, sorted_wind_angles, 10)
                    np.testing.assert_array_equal(y_plot, [2, 3, 4, 2.4, 3.1, 4.1, 2.6, 3.5, 4.4, 3, 3.8, 4.6])

    def test_plot_flat_mixed_tuple_ws(self):
        f, ax = plt.subplots()
        self.pc.plot_flat(ws=((4, 8), 2), ax=ax)
        sorted_wind_angles = sorted(np.concatenate((self.pc.wind_angles,
                                                    self.pc.wind_angles,
                                                    self.pc.wind_angles)))
        for i in range(0, 2):
            with self.subTest(i=i):
                x_plot = ax.lines[i].get_xdata()
                y_plot = ax.lines[i].get_ydata()
                if i == 1:
                    np.testing.assert_array_almost_equal(x_plot, self.pc.wind_angles, 10)
                    np.testing.assert_array_equal(y_plot, self.pc.boat_speeds[0:4])
                else:
                    np.testing.assert_array_almost_equal(x_plot, sorted_wind_angles, 10)
                    np.testing.assert_array_equal(y_plot, [2, 3, 4, 2.4, 3.1, 4.1, 2.6, 3.5, 4.4, 3, 3.8, 4.6])

    def test_plot_flat_mixed_set_ws(self):
        f, ax = plt.subplots()
        self.pc.plot_flat(ws={(4, 8), 2}, ax=ax)
        sorted_wind_angles = sorted(np.concatenate((self.pc.wind_angles,
                                                    self.pc.wind_angles,
                                                    self.pc.wind_angles)))
        for i in range(0, 2):
            with self.subTest(i=i):
                x_plot = ax.lines[i].get_xdata()
                y_plot = ax.lines[i].get_ydata()
                if i == 1:
                    np.testing.assert_array_almost_equal(x_plot, self.pc.wind_angles, 10)
                    np.testing.assert_array_equal(y_plot, self.pc.boat_speeds[0:4])
                else:
                    np.testing.assert_array_almost_equal(x_plot, sorted_wind_angles, 10)
                    np.testing.assert_array_equal(y_plot, [2, 3, 4, 2.4, 3.1, 4.1, 2.6, 3.5, 4.4, 3, 3.8, 4.6])

    def test_plot_flat_n_steps(self):
        f, ax = plt.subplots()
        self.pc.plot_flat(ws=(4, 8), ax=ax, n_steps=3)
        # test for ws still missing
        k = 4
        for i in range(0, 3):
            with self.subTest(i=i):
                x_plot = ax.lines[i].get_xdata()
                y_plot = ax.lines[i].get_ydata()
                np.testing.assert_array_almost_equal(x_plot, self.pc.wind_angles, 10)
                np.testing.assert_array_equal(y_plot, self.pc.boat_speeds[k:k+4])
                k = k+4

    def test_plot_flat_range_single_ws(self):
        f, ax = plt.subplots()
        self.pc.plot_flat(ws=4, ax=ax, range_=2)
        sorted_wind_angles = sorted(np.concatenate((self.pc.wind_angles,
                                                    self.pc.wind_angles,
                                                    self.pc.wind_angles)))
        x_plot = ax.lines[0].get_xdata()
        y_plot = ax.lines[0].get_ydata()
        np.testing.assert_array_almost_equal(x_plot, sorted_wind_angles, 10)
        np.testing.assert_array_equal(y_plot, [1.1, 2, 3, 1.5, 2.4, 3.1, 1.7, 2.6, 3.5, 2.1, 3, 3.8])

    def test_plot_flat_range_mixed_list(self):
        pd = pol.from_csv("../examples/csv-format-examples/cloud_hro_format_example.csv")
        f, ax = plt.subplots()
        pd.plot_flat(ws=[(14, 20), 8], ax=ax, range_=2)
        sorted_wind_angles = sorted(np.concatenate((pd.wind_angles,
                                                    pd.wind_angles,
                                                    pd.wind_angles)))
        for i in range(0, 2):
            with self.subTest(i=i):
                x_plot = ax.lines[i].get_xdata()
                y_plot = ax.lines[i].get_ydata()
                np.testing.assert_array_almost_equal(x_plot, sorted_wind_angles, 10)
                if i == 0:
                    np.testing.assert_array_equal(y_plot, [5.47, 5.81, 5.66, 5.67, 6.17, 5.94, 5.95, 6.86, 6.27, 7.35,
                                                           6.7, 6.49, 7.48, 6.79, 8.76, 7.32, 6.62, 9.74, 8.34, 6.77,
                                                           6.22, 6.22, 5.78, 7.32])
                else:
                    np.testing.assert_array_equal(y_plot, [3.74, 4.96, 4.48, 3.98, 5.18, 4.73, 4.16, 5.35, 4.93, 5.64,
                                                           5.19, 4.35, 5.22, 4.39, 5.68, 5.11, 4.23, 5.58, 5.33, 4.64,
                                                           3.72, 4.1, 3.21, 4.87])

    def test_plot_flat_range_mixed_tuple(self):
        pd = pol.from_csv("../examples/csv-format-examples/cloud_hro_format_example.csv")
        f, ax = plt.subplots()
        pd.plot_flat(ws=((14, 20), 8), ax=ax, range_=2)
        sorted_wind_angles = sorted(np.concatenate((pd.wind_angles,
                                                    pd.wind_angles,
                                                    pd.wind_angles)))
        for i in range(0, 2):
            with self.subTest(i=i):
                x_plot = ax.lines[i].get_xdata()
                y_plot = ax.lines[i].get_ydata()
                np.testing.assert_array_almost_equal(x_plot, sorted_wind_angles, 10)
                if i == 0:
                    np.testing.assert_array_equal(y_plot, [5.47, 5.81, 5.66, 5.67, 6.17, 5.94, 5.95, 6.86, 6.27, 7.35,
                                                           6.7, 6.49, 7.48, 6.79, 8.76, 7.32, 6.62, 9.74, 8.34, 6.77,
                                                           6.22, 6.22, 5.78, 7.32])
                else:
                    np.testing.assert_array_equal(y_plot, [3.74, 4.96, 4.48, 3.98, 5.18, 4.73, 4.16, 5.35, 4.93, 5.64,
                                                           5.19, 4.35, 5.22, 4.39, 5.68, 5.11, 4.23, 5.58, 5.33, 4.64,
                                                           3.72, 4.1, 3.21, 4.87])

    def test_plot_flat_range_mixed_set(self):
        pd = pol.from_csv("../examples/csv-format-examples/cloud_hro_format_example.csv")
        f, ax = plt.subplots()
        pd.plot_flat(ws={(14, 20), 8}, ax=ax, range_=2)
        sorted_wind_angles = sorted(np.concatenate((pd.wind_angles,
                                                    pd.wind_angles,
                                                    pd.wind_angles)))
        for i in range(0, 2):
            with self.subTest(i=i):
                x_plot = ax.lines[i].get_xdata()
                y_plot = ax.lines[i].get_ydata()
                np.testing.assert_array_almost_equal(x_plot, sorted_wind_angles, 10)
                if i == 0:
                    np.testing.assert_array_equal(y_plot, [5.47, 5.81, 5.66, 5.67, 6.17, 5.94, 5.95, 6.86, 6.27, 7.35,
                                                           6.7, 6.49, 7.48, 6.79, 8.76, 7.32, 6.62, 9.74, 8.34, 6.77,
                                                           6.22, 6.22, 5.78, 7.32])
                else:
                    np.testing.assert_array_equal(y_plot, [3.74, 4.96, 4.48, 3.98, 5.18, 4.73, 4.16, 5.35, 4.93, 5.64,
                                                           5.19, 4.35, 5.22, 4.39, 5.68, 5.11, 4.23, 5.58, 5.33, 4.64,
                                                           3.72, 4.1, 3.21, 4.87])

    def test_plot_flat_exception_single_element_ws(self):
        with self.assertRaises(PolarDiagramException):
            self.pc.plot_flat(ws=10)

    def test_plot_flat_exception_interval_ws(self):
        with self.assertRaises(PolarDiagramException):
            self.pc.plot_flat(ws=(4, 10))

    def test_plot_flat_exception_mixed_iterable_ws(self):
        with self.subTest(i=0):
            with self.assertRaises(PolarDiagramException):
                self.pc.plot_flat(ws=[(10, 20), 4])
        with self.subTest(i=1):
            with self.assertRaises(PolarDiagramException):
                self.pc.plot_flat(ws=((10, 20), 4))
        with self.subTest(i=2):
            with self.assertRaises(PolarDiagramException):
                self.pc.plot_flat(ws={(10, 20), 4})

    def test_plot_3d(self):
        f, ax = plt.subplots(subplot_kw={'projection': '3d'})
        self.pc.plot_3d(ax=ax)
        # not finished yet

    def test_plot_3d_colors(self):
        f, ax = plt.subplots(subplot_kw={'projection': '3d'})
        self.pc.plot_3d(ax=ax, colors=('blue', 'red'))
        # not finished yet

    def test_plot_3d_exception_empty_cloud(self):
        pd_empty = pol.PolarDiagramPointcloud(np.empty((0, 3)))
        with self.assertRaises(PolarDiagramException):
            pd_empty.plot_3d()

    def test_plot_color_gradient(self):
        # test not implemented yet
        pass

    def test_plot_color_gradient_exception_empty_cloud(self):
        pd_empty = pol.PolarDiagramPointcloud(np.empty((0, 3)))
        with self.assertRaises(PolarDiagramException):
            pd_empty.plot_color_gradient()

    def test_plot_convex_hull(self):
        # test not implemented yet
        pass

    def test_plot_convex_hull_exception_single_element_ws(self):
        with self.assertRaises(PolarDiagramException):
            self.pc.plot_convex_hull(ws=10)

    def test_plot_convex_hull_exception_interval_ws(self):
        with self.assertRaises(PolarDiagramException):
            self.pc.plot_convex_hull(ws=(4, 10))

    def test_plot_convex_hull_exception_mixed_iterable_ws(self):
        with self.subTest(i=0):
            with self.assertRaises(PolarDiagramException):
                self.pc.plot_convex_hull(ws=[(10, 20), 4])
        with self.subTest(i=1):
            with self.assertRaises(PolarDiagramException):
                self.pc.plot_convex_hull(ws=((10, 20), 4))
        with self.subTest(i=2):
            with self.assertRaises(PolarDiagramException):
                self.pc.plot_convex_hull(ws={(10, 20), 4})