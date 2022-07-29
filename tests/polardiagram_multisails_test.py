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


class PolarDiagramMultiSailsTest(unittest.TestCase):
    def setUp(self):
        self.wind_speeds = np.array([42, 44, 46])
        self.wind_angles = np.array([21, 42, 84])
        self.boat_speeds1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.boat_speeds2 = np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]])
        self.tbl_1 = pol.PolarDiagramTable(self.wind_speeds, self.wind_angles, self.boat_speeds1)
        self.tbl_2 = pol.PolarDiagramTable(self.wind_speeds, self.wind_angles, self.boat_speeds2)
        self.mts = pol.PolarDiagramMultiSails((self.tbl_1, self.tbl_2))

        # example for a bigger multisails-pd:
        self.wind_speeds_big = np.array([6, 8, 10, 12, 14, 16, 20])
        self.wind_angles_big = np.array([52, 60, 75, 90, 110, 120, 135, 150])
        self.boat_speeds_big_1 = [
            [3.74, 4.48, 4.96, 5.27, 5.47, 5.66, 5.81],
            [3.98, 4.73, 5.18, 5.44, 5.67, 5.94, 6.17],
            [4.16, 4.93, 5.35, 5.66, 5.95, 6.27, 6.86],
            [4.35, 5.19, 5.64, 6.09, 6.49, 6.7, 7.35],
            [4.39, 5.22, 5.68, 6.19, 6.79, 7.48, 8.76],
            [4.23, 5.11, 5.58, 6.06, 6.62, 7.32, 9.74],
            [3.72, 4.64, 5.33, 5.74, 6.22, 6.77, 8.34],
            [3.21, 4.1, 4.87, 5.4, 5.78, 6.22, 7.32]
        ]
        self.boat_speeds_big_2 = [
            [3.74, 4.48, 4.96, 5.27, 5.47, 5.66, 5.81],
            [3.98, 4.73, 5.18, 5.44, 5.67, 5.94, 6.17],
            [4.16, 4.93, 5.35, 5.66, 5.95, 6.27, 6.86],
            [4.35, 5.19, 5.64, 6.09, 6.49, 6.7, 7.35],
            [4.39, 5.22, 5.68, 6.19, 6.79, 7.48, 8.76],
            [4.23, 5.11, 5.58, 6.06, 6.62, 7.32, 9.74],
            [3.72, 4.64, 5.33, 5.74, 6.22, 6.77, 8.34],
            [3.21, 4.1, 4.87, 5.4, 5.78, 6.22, 7.32]
        ]
        self.tbl_big_1 = pol.PolarDiagramTable(self.wind_speeds_big, self.wind_angles_big, self.boat_speeds_big_1)
        self.tbl_big_2 = pol.PolarDiagramTable(self.wind_speeds_big, self.wind_angles_big, self.boat_speeds_big_2)
        self.mts_big = pol.PolarDiagramMultiSails((self.tbl_big_1, self.tbl_big_2))

    def test_init(self):
        np.testing.assert_array_equal(self.mts.sails, ['Sail 0', 'Sail 1'])
        np.testing.assert_array_equal(self.mts.wind_speeds, self.wind_speeds)

        def tbl_enumerate(n):
            if n == 0:
                return self.tbl_1
            elif n == 1:
                return self.tbl_2
            else:
                return None

        for i, table in enumerate(self.mts.tables):
            with self.subTest(i=i):
                np.testing.assert_array_equal(table.wind_speeds, self.wind_speeds)
                np.testing.assert_array_equal(table.wind_angles, self.wind_angles)
                np.testing.assert_array_equal(table.boat_speeds, tbl_enumerate(i).boat_speeds)

    def test_init_exception_different_ws(self):
        tbl_exc1 = pol.PolarDiagramTable(self.wind_speeds, self.wind_angles, self.boat_speeds2)
        tbl_exc2 = pol.PolarDiagramTable([43, 45, 47], self.wind_angles, self.boat_speeds2)
        with self.assertRaises(PolarDiagramInitializationException):
            pol.PolarDiagramMultiSails((tbl_exc1, tbl_exc2))

    def test_sails(self):
        np.testing.assert_array_equal(self.mts.sails, ['Sail 0', 'Sail 1'])

    def test_ws(self):
        np.testing.assert_array_equal(self.mts.wind_speeds, self.wind_speeds)

    def test_tbls(self):
        def tbl_enumerate(n):
            if n == 0:
                return self.tbl_1
            elif n == 1:
                return self.tbl_2
            else:
                return None

        for i, table in enumerate(self.mts.tables):
            with self.subTest(i=i):
                np.testing.assert_array_equal(table.wind_speeds, self.wind_speeds)
                np.testing.assert_array_equal(table.wind_angles, self.wind_angles)
                np.testing.assert_array_equal(table.boat_speeds, tbl_enumerate(i).boat_speeds)

    def test_symmetrize(self):
        def sym_boat_speeds(n):
            if n == 0:
                return np.row_stack((self.boat_speeds1, np.flip(self.boat_speeds1, axis=0)))
            elif n == 1:
                return np.row_stack((self.boat_speeds2, np.flip(self.boat_speeds2, axis=0)))
            else:
                return None

        sym_pd = self.mts.symmetrize()

        sym_wind_angles = np.concatenate(
            [self.wind_angles, 360 - np.flip(self.wind_angles)]
        )

        np.testing.assert_array_equal(sym_pd.wind_speeds, self.wind_speeds)
        np.testing.assert_array_equal(sym_pd.sails, ['Sail 0', 'Sail 1'])
        for i, table in enumerate(sym_pd.tables):
            with self.subTest(i=i):
                np.testing.assert_array_equal(table.wind_speeds, self.wind_speeds)
                np.testing.assert_array_equal(table.wind_angles, sym_wind_angles)
                np.testing.assert_array_equal(table.boat_speeds, sym_boat_speeds(i))

    def test_get_one_slice(self):
        ws, wa, bsp, members = self.mts.get_slices(42)
        wind_angles = np.concatenate(((np.deg2rad(self.tbl_1.wind_angles)),
                                                         (np.deg2rad(self.tbl_2.wind_angles))))
        self.assertEqual(ws, [42])
        np.testing.assert_array_equal(np.asarray(wa).flat, wind_angles)
        np.testing.assert_array_equal(np.asarray(bsp).flat, [1, 4, 7, 10, 13, 16])
        np.testing.assert_array_equal(members, 3*['Sail 0'] + 3*['Sail 1'])


    def test_get_multiple_slices_interval(self):
        ws, wa, bsp, members = self.mts.get_slices((42, 46))
        wind_angles = np.concatenate(((np.deg2rad(self.tbl_1.wind_angles)),
                                                         (np.deg2rad(self.tbl_2.wind_angles))))
        corr_wind_angles = np.concatenate((wind_angles, wind_angles, wind_angles))
        self.assertEqual(ws, [42, 44, 46])
        np.testing.assert_array_equal(np.asarray(wa).flat, corr_wind_angles)
        np.testing.assert_array_equal(bsp,
                                      [[1, 4, 7, 10, 13, 16],
                                       [2, 5, 8, 11, 14, 17],
                                       [3, 6, 9, 12, 15, 18]])
        np.testing.assert_array_equal(members, 3 * ['Sail 0'] + 3 * ['Sail 1'])

    def test_get_multiple_slices_tuple(self):
        ws, wa, bsp, members = self.mts.get_slices((42, 44, 46))
        wind_angles = np.concatenate(((np.deg2rad(self.tbl_1.wind_angles)),
                                      (np.deg2rad(self.tbl_2.wind_angles))))
        corr_wind_angles = np.concatenate((wind_angles, wind_angles, wind_angles))
        self.assertEqual(ws, [42, 44, 46])
        np.testing.assert_array_equal(np.asarray(wa).flat, corr_wind_angles)
        np.testing.assert_array_equal(bsp,
                                      [[1, 4, 7, 10, 13, 16],
                                       [2, 5, 8, 11, 14, 17],
                                       [3, 6, 9, 12, 15, 18]])
        np.testing.assert_array_equal(members, 3 * ['Sail 0'] + 3 * ['Sail 1'])

    def test_get_multiple_slices_list(self):
        ws, wa, bsp, members = self.mts.get_slices([42, 44, 46])
        wind_angles = np.concatenate(((np.deg2rad(self.tbl_1.wind_angles)),
                                      (np.deg2rad(self.tbl_2.wind_angles))))
        corr_wind_angles = np.concatenate((wind_angles, wind_angles, wind_angles))
        self.assertEqual(ws, [42, 44, 46])
        np.testing.assert_array_equal(np.asarray(wa).flat, corr_wind_angles)
        np.testing.assert_array_equal(bsp,
                                      [[1, 4, 7, 10, 13, 16],
                                       [2, 5, 8, 11, 14, 17],
                                       [3, 6, 9, 12, 15, 18]])
        np.testing.assert_array_equal(members, 3 * ['Sail 0'] + 3 * ['Sail 1'])

    def test_get_multiple_slices_set(self):
        ws, wa, bsp, members = self.mts.get_slices({42, 44, 46})
        wind_angles = np.concatenate(((np.deg2rad(self.tbl_1.wind_angles)),
                                      (np.deg2rad(self.tbl_2.wind_angles))))
        corr_wind_angles = np.concatenate((wind_angles, wind_angles, wind_angles))
        self.assertEqual(ws, [42, 44, 46])
        np.testing.assert_array_equal(np.asarray(wa).flat, corr_wind_angles)
        np.testing.assert_array_equal(bsp,
                                      [[1, 4, 7, 10, 13, 16],
                                       [2, 5, 8, 11, 14, 17],
                                       [3, 6, 9, 12, 15, 18]])
        np.testing.assert_array_equal(members, 3 * ['Sail 0'] + 3 * ['Sail 1'])

    def test_get_all_slices(self):
        ws, wa, bsp, members = self.mts.get_slices(None)
        wind_angles = np.concatenate(((np.deg2rad(self.tbl_1.wind_angles)),
                                      (np.deg2rad(self.tbl_2.wind_angles))))
        corr_wind_angles = np.concatenate((wind_angles, wind_angles, wind_angles))
        self.assertEqual(ws, [42, 44, 46])
        np.testing.assert_array_equal(np.asarray(wa).flat, corr_wind_angles)
        np.testing.assert_array_equal(bsp,
                                      [[1, 4, 7, 10, 13, 16],
                                       [2, 5, 8, 11, 14, 17],
                                       [3, 6, 9, 12, 15, 18]])
        np.testing.assert_array_equal(members, 3 * ['Sail 0'] + 3 * ['Sail 1'])

    def test_plot_polar(self):
        self.mts.plot_polar()
        bsps = [[1, 4, 7],
                [2, 5, 8],
                [3, 6, 9],
                [10, 13, 16],
                [11, 14, 17],
                [12, 15, 18]]
        for i in range(6):
            with self.subTest(i=i):
                x_plot = plt.gca().lines[i].get_xdata()
                y_plot = plt.gca().lines[i].get_ydata()
                np.testing.assert_array_equal(x_plot, np.deg2rad(self.wind_angles))
                np.testing.assert_array_equal(y_plot, bsps[i])

    def test_plot_polar_single_element_ws(self):
        self.mts.plot_polar(ws=42)
        bsps = [[1, 4, 7],
                [10, 13, 16]]
        for i in range(2):
            with self.subTest(i=i):
                x_plot = plt.gca().lines[i].get_xdata()
                y_plot = plt.gca().lines[i].get_ydata()
                np.testing.assert_array_equal(x_plot, np.deg2rad(self.wind_angles))
                np.testing.assert_array_equal(y_plot, bsps[i])

    def test_plot_polar_interval_ws(self):
        self.mts.plot_polar(ws=(40, 45))
        bsps = [[1, 4, 7],
                [2, 5, 8],
                [10, 13, 16],
                [11, 14, 17]]
        for i in range(4):
            with self.subTest(i=i):
                x_plot = plt.gca().lines[i].get_xdata()
                y_plot = plt.gca().lines[i].get_ydata()
                np.testing.assert_array_equal(x_plot, np.deg2rad(self.wind_angles))
                np.testing.assert_array_equal(y_plot, bsps[i])

    def test_plot_polar_iterable_list_ws(self):
        self.mts.plot_polar(ws=[42, 44, 46])
        bsps = [[1, 4, 7],
                [2, 5, 8],
                [3, 6, 9],
                [10, 13, 16],
                [11, 14, 17],
                [12, 15, 18]]
        for i in range(6):
            with self.subTest(i=i):
                x_plot = plt.gca().lines[i].get_xdata()
                y_plot = plt.gca().lines[i].get_ydata()
                np.testing.assert_array_equal(x_plot, np.deg2rad(self.wind_angles))
                np.testing.assert_array_equal(y_plot, bsps[i])

    def test_plot_polar_iterable_tuple_ws(self):
        self.mts.plot_polar(ws=(42, 44, 46))
        bsps = [[1, 4, 7],
                [2, 5, 8],
                [3, 6, 9],
                [10, 13, 16],
                [11, 14, 17],
                [12, 15, 18]]
        for i in range(6):
            with self.subTest(i=i):
                x_plot = plt.gca().lines[i].get_xdata()
                y_plot = plt.gca().lines[i].get_ydata()
                np.testing.assert_array_equal(x_plot, np.deg2rad(self.wind_angles))
                np.testing.assert_array_equal(y_plot, bsps[i])

    def test_plot_polar_iterable_set_ws(self):
        self.mts.plot_polar(ws={42, 44, 46})
        bsps = [[1, 4, 7],
                [2, 5, 8],
                [3, 6, 9],
                [10, 13, 16],
                [11, 14, 17],
                [12, 15, 18]]
        for i in range(6):
            with self.subTest(i=i):
                x_plot = plt.gca().lines[i].get_xdata()
                y_plot = plt.gca().lines[i].get_ydata()
                np.testing.assert_array_equal(x_plot, np.deg2rad(self.wind_angles))
                np.testing.assert_array_equal(y_plot, bsps[i])

    def test_plot_polar_axes_instance(self):
        f, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        self.mts.plot_polar(ax=ax)
        gca = plt.gca()
        np.testing.assert_array_equal(ax.__dict__, gca.__dict__)

    def test_plot_polar_single_color_passed(self):
        self.mts.plot_polar(colors="purple")
        for i in range(6):
            with self.subTest(i=i):
                self.assertEqual(plt.gca().lines[i].get_color(), "purple")

    def test_plot_polar_two_colors_passed(self):
        self.mts.plot_polar(colors=["red", "blue"])
        np.testing.assert_array_equal(plt.gca().lines[0].get_color(), [1, 0, 0])
        np.testing.assert_array_equal(plt.gca().lines[1].get_color(), [0.5, 0, 0.5])
        np.testing.assert_array_equal(plt.gca().lines[2].get_color(), [0, 0, 1])
        np.testing.assert_array_equal(plt.gca().lines[3].get_color(), [1, 0, 0])
        np.testing.assert_array_equal(plt.gca().lines[4].get_color(), [0.5, 0, 0.5])
        np.testing.assert_array_equal(plt.gca().lines[5].get_color(), [0, 0, 1])

    def test_plot_polar_more_than_two_colors_passed(self):
        self.mts_big.plot_polar(ws=[6, 8, 10, 12], colors=["red", "yellow", "orange"])
        np.testing.assert_array_equal(plt.gca().lines[0].get_color(), "red")
        np.testing.assert_array_equal(plt.gca().lines[1].get_color(), "yellow")
        np.testing.assert_array_equal(plt.gca().lines[2].get_color(), "orange")
        np.testing.assert_array_equal(plt.gca().lines[3].get_color(), "blue")
        np.testing.assert_array_equal(plt.gca().lines[4].get_color(), "red")
        np.testing.assert_array_equal(plt.gca().lines[5].get_color(), "yellow")
        np.testing.assert_array_equal(plt.gca().lines[6].get_color(), "orange")
        np.testing.assert_array_equal(plt.gca().lines[7].get_color(), "blue")

    def test_plot_polar_ws_color_pairs_passed(self):
        self.mts.plot_polar(colors=((42, "purple"), (44, "blue"), (46, "red")))
        np.testing.assert_array_equal(plt.gca().lines[0].get_color(), "purple")
        np.testing.assert_array_equal(plt.gca().lines[1].get_color(), "blue")
        np.testing.assert_array_equal(plt.gca().lines[2].get_color(), "red")
        np.testing.assert_array_equal(plt.gca().lines[0].get_color(), "purple")
        np.testing.assert_array_equal(plt.gca().lines[1].get_color(), "blue")
        np.testing.assert_array_equal(plt.gca().lines[2].get_color(), "red")

    def test_plot_polar_ws_color_pairs_unsorted_passed(self):
        self.mts.plot_polar(colors=((42, "purple"), (46, "red"), (44, "blue")))
        np.testing.assert_array_equal(plt.gca().lines[0].get_color(), "purple")
        np.testing.assert_array_equal(plt.gca().lines[1].get_color(), "blue")
        np.testing.assert_array_equal(plt.gca().lines[2].get_color(), "red")
        np.testing.assert_array_equal(plt.gca().lines[3].get_color(), "purple")
        np.testing.assert_array_equal(plt.gca().lines[4].get_color(), "blue")
        np.testing.assert_array_equal(plt.gca().lines[5].get_color(), "red")

    def test_plot_polar_show_legend(self):
        self.mts.plot_polar(colors=["red", "purple", "blue"], show_legend=True)
        self.assertNotEqual(None, plt.gca().get_legend())
        legend = plt.gca().get_legend()
        texts = legend.__dict__["texts"]
        texts = str(texts)
        self.assertEqual(texts, "[Text(0, 0, 'TWS 42.0'), Text(0, 0, 'TWS 44.0'), Text(0, 0, 'TWS 46.0')]")
        # not finished: colors in legend not tested yet

    def test_plot_polar_plot_kw(self):
        self.mts.plot_polar(ls=":", lw=1.5, marker="o")
        for i in range(6):
            with self.subTest(i=i):
                line = plt.gca().lines[i]
                self.assertEqual(line.get_linestyle(), ':')
                self.assertEqual(line.get_linewidth(), 1.5)
                self.assertEqual(line.get_marker(), 'o')

    def test_plot_polar_exception_ws_not_in_self_wind_speeds(self):
        with self.subTest(i=0):
            with self.assertRaises(PolarDiagramException):
                self.mts.plot_polar(ws=40)
        with self.subTest(i=1):
            with self.assertRaises(PolarDiagramException):
                self.mts.plot_polar(ws=[42, 43, 44])

    def test_plot_polar_exception_no_slice_in_interval(self):
        with self.subTest(i=0):
            with self.assertRaises(PolarDiagramException):
                self.mts.plot_polar(ws=(60, 70))
        with self.subTest(i=1):
            with self.assertRaises(PolarDiagramException):
                self.mts.plot_polar(ws=(50, 0))

    def test_plot_flat(self):
        self.mts.plot_flat()
        bsps = [[1, 4, 7],
                [2, 5, 8],
                [3, 6, 9],
                [10, 13, 16],
                [11, 14, 17],
                [12, 15, 18]]
        for i in range(6):
            with self.subTest(i=i):
                x_plot = plt.gca().lines[i].get_xdata()
                y_plot = plt.gca().lines[i].get_ydata()
                np.testing.assert_array_equal(x_plot, self.wind_angles)
                np.testing.assert_array_equal(y_plot, bsps[i])

    def test_plot_flat_single_element_ws(self):
        self.mts.plot_flat(ws=42)
        bsps = [[1, 4, 7],
                [10, 13, 16]]
        for i in range(2):
            with self.subTest(i=i):
                x_plot = plt.gca().lines[i].get_xdata()
                y_plot = plt.gca().lines[i].get_ydata()
                np.testing.assert_array_equal(x_plot, self.wind_angles)
                np.testing.assert_array_equal(y_plot, bsps[i])

    def test_plot_flat_interval_ws(self):
        self.mts.plot_flat(ws=(40, 45))
        bsps = [[1, 4, 7],
                [2, 5, 8],
                [10, 13, 16],
                [11, 14, 17]]
        for i in range(4):
            with self.subTest(i=i):
                x_plot = plt.gca().lines[i].get_xdata()
                y_plot = plt.gca().lines[i].get_ydata()
                np.testing.assert_array_equal(x_plot, self.wind_angles)
                np.testing.assert_array_equal(y_plot, bsps[i])

    def test_plot_flat_iterable_list_ws(self):
        self.mts.plot_flat(ws=[42, 44, 46])
        bsps = [[1, 4, 7],
                [2, 5, 8],
                [3, 6, 9],
                [10, 13, 16],
                [11, 14, 17],
                [12, 15, 18]]
        for i in range(6):
            with self.subTest(i=i):
                x_plot = plt.gca().lines[i].get_xdata()
                y_plot = plt.gca().lines[i].get_ydata()
                np.testing.assert_array_equal(x_plot, self.wind_angles)
                np.testing.assert_array_equal(y_plot, bsps[i])

    def test_plot_flat_iterable_tuple_ws(self):
        self.mts.plot_flat(ws=(42, 44, 46))
        bsps = [[1, 4, 7],
                [2, 5, 8],
                [3, 6, 9],
                [10, 13, 16],
                [11, 14, 17],
                [12, 15, 18]]
        for i in range(6):
            with self.subTest(i=i):
                x_plot = plt.gca().lines[i].get_xdata()
                y_plot = plt.gca().lines[i].get_ydata()
                np.testing.assert_array_equal(x_plot, self.wind_angles)
                np.testing.assert_array_equal(y_plot, bsps[i])

    def test_plot_flat_iterable_set_ws(self):
        self.mts.plot_flat(ws={42, 44, 46})
        bsps = [[1, 4, 7],
                [2, 5, 8],
                [3, 6, 9],
                [10, 13, 16],
                [11, 14, 17],
                [12, 15, 18]]
        for i in range(6):
            with self.subTest(i=i):
                x_plot = plt.gca().lines[i].get_xdata()
                y_plot = plt.gca().lines[i].get_ydata()
                np.testing.assert_array_equal(x_plot, self.wind_angles)
                np.testing.assert_array_equal(y_plot, bsps[i])

    def test_plot_flat_axes_instances(self):
        f, ax = plt.subplots()
        self.mts.plot_flat(ax=ax)
        gca = plt.gca()
        np.testing.assert_array_equal(ax.__dict__, gca.__dict__)

    def test_plot_flat_single_color_passed(self):
        self.mts.plot_flat(colors="purple")
        for i in range(6):
            with self.subTest(i=i):
                self.assertEqual(plt.gca().lines[i].get_color(), "purple")

    def test_plot_flat_two_colors_passed(self):
        self.mts.plot_flat(colors=["red", "blue"])
        np.testing.assert_array_equal(plt.gca().lines[0].get_color(), [1, 0, 0])
        np.testing.assert_array_equal(plt.gca().lines[1].get_color(), [0.5, 0, 0.5])
        np.testing.assert_array_equal(plt.gca().lines[2].get_color(), [0, 0, 1])
        np.testing.assert_array_equal(plt.gca().lines[3].get_color(), [1, 0, 0])
        np.testing.assert_array_equal(plt.gca().lines[4].get_color(), [0.5, 0, 0.5])
        np.testing.assert_array_equal(plt.gca().lines[5].get_color(), [0, 0, 1])

    def test_plot_flat_more_than_two_colors_passed(self):
        self.mts_big.plot_flat(ws=[6, 8, 10, 12], colors=["red", "yellow", "orange"])
        np.testing.assert_array_equal(plt.gca().lines[0].get_color(), "red")
        np.testing.assert_array_equal(plt.gca().lines[1].get_color(), "yellow")
        np.testing.assert_array_equal(plt.gca().lines[2].get_color(), "orange")
        np.testing.assert_array_equal(plt.gca().lines[3].get_color(), "blue")
        np.testing.assert_array_equal(plt.gca().lines[4].get_color(), "red")
        np.testing.assert_array_equal(plt.gca().lines[5].get_color(), "yellow")
        np.testing.assert_array_equal(plt.gca().lines[6].get_color(), "orange")
        np.testing.assert_array_equal(plt.gca().lines[7].get_color(), "blue")

    def test_plot_flat_ws_color_pairs_passed(self):
        self.mts.plot_flat(colors=((42, "purple"), (44, "blue"), (46, "red")))
        np.testing.assert_array_equal(plt.gca().lines[0].get_color(), "purple")
        np.testing.assert_array_equal(plt.gca().lines[1].get_color(), "blue")
        np.testing.assert_array_equal(plt.gca().lines[2].get_color(), "red")
        np.testing.assert_array_equal(plt.gca().lines[0].get_color(), "purple")
        np.testing.assert_array_equal(plt.gca().lines[1].get_color(), "blue")
        np.testing.assert_array_equal(plt.gca().lines[2].get_color(), "red")

    def test_plot_flat_ws_color_pairs_unsorted_passed(self):
        self.mts.plot_flat(colors=((42, "purple"), (46, "red"), (44, "blue")))
        np.testing.assert_array_equal(plt.gca().lines[0].get_color(), "purple")
        np.testing.assert_array_equal(plt.gca().lines[1].get_color(), "blue")
        np.testing.assert_array_equal(plt.gca().lines[2].get_color(), "red")
        np.testing.assert_array_equal(plt.gca().lines[3].get_color(), "purple")
        np.testing.assert_array_equal(plt.gca().lines[4].get_color(), "blue")
        np.testing.assert_array_equal(plt.gca().lines[5].get_color(), "red")

    def test_plot_flat_show_legend(self):
        self.mts.plot_flat(colors=["red", "purple", "blue"], show_legend=True)
        self.assertNotEqual(None, plt.gca().get_legend())
        legend = plt.gca().get_legend()
        texts = legend.__dict__["texts"]
        texts = str(texts)
        self.assertEqual(texts, "[Text(0, 0, 'TWS 42.0'), Text(0, 0, 'TWS 44.0'), Text(0, 0, 'TWS 46.0')]")
        # not finished: colors in legend not tested yet

    def test_plot_flat_plot_kw(self):
        self.mts.plot_flat(ls=":", lw=1.5, marker="o")
        for i in range(6):
            with self.subTest(i=i):
                line = plt.gca().lines[i]
                self.assertEqual(line.get_linestyle(), ':')
                self.assertEqual(line.get_linewidth(), 1.5)
                self.assertEqual(line.get_marker(), 'o')

    def test_plot_flat_exception_ws_not_in_self_wind_speeds(self):
        with self.subTest(i=0):
            with self.assertRaises(PolarDiagramException):
                self.mts.plot_flat(ws=40)
        with self.subTest(i=1):
            with self.assertRaises(PolarDiagramException):
                self.mts.plot_flat(ws=[42, 43, 44])

    def test_plot_flat_exception_no_slice_in_interval(self):
        with self.subTest(i=0):
            with self.assertRaises(PolarDiagramException):
                self.mts.plot_flat(ws=(60, 70))
        with self.subTest(i=1):
            with self.assertRaises(PolarDiagramException):
                self.mts.plot_flat(ws=(50, 0))

    def test_plot_3d(self):
        # test not implemented yet
        pass

    def test_plot_color_gradient(self):
        pass

    def test_plot_convex_hull(self):
        # not finished yet: wa and bsp not tested
        self.mts.plot_convex_hull()
        for i in range(6):
            with self.subTest(i=i):
                x_plot = plt.gca().lines[i].get_xdata()
                y_plot = plt.gca().lines[i].get_ydata()

    def test_plot_convex_hull_axes_instance(self):
        f, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        self.mts.plot_convex_hull(ax=ax)
        gca = plt.gca()
        np.testing.assert_array_equal(ax.__dict__, gca.__dict__)

    def test_plot_convex_hull_single_color_passed(self):
        self.mts.plot_convex_hull(colors="purple")
        for i in range(6):
            with self.subTest(i=i):
                self.assertEqual(plt.gca().lines[i].get_color(), "purple")

    def test_plot_convex_hull_two_colors_passed(self):
        self.mts.plot_convex_hull(colors=["red", "blue"])
        np.testing.assert_array_equal(plt.gca().lines[0].get_color(), [1, 0, 0])
        np.testing.assert_array_equal(plt.gca().lines[1].get_color(), [0.5, 0, 0.5])
        np.testing.assert_array_equal(plt.gca().lines[2].get_color(), [0, 0, 1])
        np.testing.assert_array_equal(plt.gca().lines[3].get_color(), [1, 0, 0])
        np.testing.assert_array_equal(plt.gca().lines[4].get_color(), [0.5, 0, 0.5])
        np.testing.assert_array_equal(plt.gca().lines[5].get_color(), [0, 0, 1])

    def test_plot_convex_hull_more_than_two_colors_passed(self):
        self.mts_big.plot_convex_hull(ws=[6, 8, 10, 12], colors=["red", "yellow", "orange"])
        np.testing.assert_array_equal(plt.gca().lines[0].get_color(), "red")
        np.testing.assert_array_equal(plt.gca().lines[1].get_color(), "yellow")
        np.testing.assert_array_equal(plt.gca().lines[2].get_color(), "orange")
        np.testing.assert_array_equal(plt.gca().lines[3].get_color(), "blue")
        np.testing.assert_array_equal(plt.gca().lines[4].get_color(), "red")
        np.testing.assert_array_equal(plt.gca().lines[5].get_color(), "yellow")
        np.testing.assert_array_equal(plt.gca().lines[6].get_color(), "orange")
        np.testing.assert_array_equal(plt.gca().lines[7].get_color(), "blue")

    def test_plot_convex_hull_ws_color_pairs_passed(self):
        self.mts.plot_convex_hull(colors=((42, "purple"), (44, "blue"), (46, "red")))
        np.testing.assert_array_equal(plt.gca().lines[0].get_color(), "purple")
        np.testing.assert_array_equal(plt.gca().lines[1].get_color(), "blue")
        np.testing.assert_array_equal(plt.gca().lines[2].get_color(), "red")
        np.testing.assert_array_equal(plt.gca().lines[0].get_color(), "purple")
        np.testing.assert_array_equal(plt.gca().lines[1].get_color(), "blue")
        np.testing.assert_array_equal(plt.gca().lines[2].get_color(), "red")

    def test_plot_convex_hull_ws_color_pairs_unsorted_passed(self):
        self.mts.plot_convex_hull(colors=((42, "purple"), (46, "red"), (44, "blue")))
        np.testing.assert_array_equal(plt.gca().lines[0].get_color(), "purple")
        np.testing.assert_array_equal(plt.gca().lines[1].get_color(), "blue")
        np.testing.assert_array_equal(plt.gca().lines[2].get_color(), "red")
        np.testing.assert_array_equal(plt.gca().lines[3].get_color(), "purple")
        np.testing.assert_array_equal(plt.gca().lines[4].get_color(), "blue")
        np.testing.assert_array_equal(plt.gca().lines[5].get_color(), "red")

    def test_plot_convex_hull_show_legend(self):
        self.mts.plot_convex_hull(colors=["red", "purple", "blue"], show_legend=True)
        self.assertNotEqual(None, plt.gca().get_legend())
        legend = plt.gca().get_legend()
        texts = legend.__dict__["texts"]
        texts = str(texts)
        self.assertEqual(texts, "[Text(0, 0, 'TWS 42.0'), Text(0, 0, 'TWS 44.0'), Text(0, 0, 'TWS 46.0')]")
        # not finished: colors in legend not tested yet

    def test_plot_convex_hull_plot_kw(self):
        self.mts.plot_convex_hull(ls=":", lw=1.5, marker="o")
        for i in range(6):
            with self.subTest(i=i):
                line = plt.gca().lines[i]
                self.assertEqual(line.get_linestyle(), ':')
                self.assertEqual(line.get_linewidth(), 1.5)
                self.assertEqual(line.get_marker(), 'o')

    def test_plot_convex_hull_exception_ws_not_in_self_wind_speeds(self):
        with self.subTest(i=0):
            with self.assertRaises(PolarDiagramException):
                self.mts.plot_convex_hull(ws=40)
        with self.subTest(i=1):
            with self.assertRaises(PolarDiagramException):
                self.mts.plot_convex_hull(ws=[42, 43, 44])

    def test_plot_convex_hull_exception_no_slice_in_interval(self):
        with self.subTest(i=0):
            with self.assertRaises(PolarDiagramException):
                self.mts.plot_convex_hull(ws=(60, 70))
        with self.subTest(i=1):
            with self.assertRaises(PolarDiagramException):
                self.mts.plot_convex_hull(ws=(50, 0))