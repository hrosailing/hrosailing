# pylint: disable=missing-docstring
# pylint: disable=too-many-public-methods
# pylint: disable=import-outside-toplevel
import itertools
import unittest

import numpy as np

import matplotlib.pyplot as plt

import math

import hrosailing.polardiagram as pol
from hrosailing.polardiagram._basepolardiagram import (
    PolarDiagramException,
    PolarDiagramInitializationException,
)
import _test_plot_functions as helper_functions


class PolarDiagramTableTest(unittest.TestCase):
    def setUp(self):
        self.ws_resolution = np.array([2, 4, 6, 8])

        self.wa_resolution = np.array([10, 15, 20, 25])
        self.bsp = np.array(
            [
                [1, 2, 3, 4],
                [1.5, 2.4, 3.1, 4.1],
                [1.7, 2.6, 3.5, 4.4],
                [2, 3, 3.8, 4.6],
            ]
        )
        self.pd = pol.PolarDiagramTable(
            self.ws_resolution, self.wa_resolution, self.bsp
        )

    @staticmethod
    def test_default_init():
        pd = pol.PolarDiagramTable()
        np.testing.assert_array_equal(pd.wind_speeds, np.arange(2, 42, 2))
        np.testing.assert_array_equal(pd.wind_angles, np.arange(0, 360, 5))
        np.testing.assert_array_equal(pd.boat_speeds, np.zeros((72, 20)))

    def test_init(self):
        np.testing.assert_array_equal(self.pd.wind_speeds, self.ws_resolution)
        np.testing.assert_array_equal(self.pd.wind_angles, self.wa_resolution)
        np.testing.assert_array_equal(self.pd.boat_speeds, self.bsp)

    def test_init_only_ws_resolution(self):
        pd = pol.PolarDiagramTable(ws_resolution=self.ws_resolution)
        np.testing.assert_array_equal(pd.wind_speeds, self.ws_resolution)
        np.testing.assert_array_equal(pd.wind_angles, np.arange(0, 360, 5))
        np.testing.assert_array_equal(pd.boat_speeds, np.zeros((72, 4)))

    def test_init_only_wa_resolution(self):
        pd = pol.PolarDiagramTable(wa_resolution=self.wa_resolution)
        np.testing.assert_array_equal(pd.wind_speeds, np.arange(2, 42, 2))
        np.testing.assert_array_equal(pd.wind_angles, self.wa_resolution)
        np.testing.assert_array_equal(pd.boat_speeds, np.zeros((4, 20)))

    def test_init_ws_wa_resolution(self):
        pd = pol.PolarDiagramTable(
            ws_resolution=self.ws_resolution, wa_resolution=self.wa_resolution
        )
        np.testing.assert_array_equal(pd.wind_speeds, self.ws_resolution)
        np.testing.assert_array_equal(pd.wind_angles, self.wa_resolution)
        np.testing.assert_array_equal(pd.boat_speeds, np.zeros((4, 4)))

    @staticmethod
    def test_init_only_bsps():
        bsps = np.random.rand(72, 20)
        pd = pol.PolarDiagramTable(bsps=bsps)
        np.testing.assert_array_equal(pd.wind_speeds, np.arange(2, 42, 2))
        np.testing.assert_array_equal(pd.wind_angles, np.arange(0, 360, 5))
        np.testing.assert_array_equal(pd.boat_speeds, bsps)

    def test_init_ws_resolution_bsps(self):
        bsps = np.random.rand(72, 4)
        pd = pol.PolarDiagramTable(ws_resolution=self.ws_resolution, bsps=bsps)
        np.testing.assert_array_equal(pd.wind_speeds, self.ws_resolution)
        np.testing.assert_array_equal(pd.wind_angles, np.arange(0, 360, 5))
        np.testing.assert_array_equal(pd.boat_speeds, bsps)

    def test_init_wa_resolution_bsps(self):
        bsps = np.random.rand(4, 20)
        pd = pol.PolarDiagramTable(wa_resolution=self.wa_resolution, bsps=bsps)
        np.testing.assert_array_equal(pd.wind_speeds, np.arange(2, 42, 2))
        np.testing.assert_array_equal(pd.wind_angles, self.wa_resolution)
        np.testing.assert_array_equal(pd.boat_speeds, bsps)

    def test_init_ws_resolution_not_array_like(self):
        ws_resolution = [{2, 4, 6, 8}, {2: 0, 4: 0, 6: 0, 8: 0}]
        for i, ws in enumerate(ws_resolution):
            with self.subTest(i=i):
                with self.assertRaises(ValueError):
                    pol.PolarDiagramTable(ws_resolution=ws)

    def test_init_wa_resolution_not_array_like(self):
        wa_resolution = [{10, 15, 20, 25}, {10: 0, 15: 0, 20: 0, 25: 0}]
        for i, wa in enumerate(wa_resolution):
            with self.subTest(i=i):
                with self.assertRaises(ValueError):
                    pol.PolarDiagramTable(wa_resolution=wa)

    def test_init_exception_empty_bsps(self):
        with self.assertRaises(PolarDiagramInitializationException):
            pol.PolarDiagramTable(bsps=[])

    def test_init_exception_not_ndim2(self):
        with self.assertRaises(PolarDiagramInitializationException):
            pol.PolarDiagramTable(bsps=[[[0]]])

    def test_init_exception_wrong_shape(self):
        with self.assertRaises(PolarDiagramInitializationException):
            pol.PolarDiagramTable(bsps=[[0]])

    def test_init_unsorted_ws_resolution(self):
        ws_resolution = [8, 2, 6, 4]
        bsps = [
            [4, 1, 3, 2],
            [4.1, 1.5, 3.1, 2.4],
            [4.4, 1.7, 3.5, 2.6],
            [4.6, 2, 3.8, 3],
        ]
        pd = pol.PolarDiagramTable(ws_resolution, self.wa_resolution, bsps)
        np.testing.assert_array_equal(pd.wind_speeds, self.ws_resolution)
        np.testing.assert_array_equal(pd.wind_angles, self.wa_resolution)
        np.testing.assert_array_equal(pd.boat_speeds, self.bsp)

    def test_init_unsorted_wa_resolution(self):
        wa_resolution = [20, 15, 25, 10]
        bsps = [
            [1.7, 2.6, 3.5, 4.4],
            [1.5, 2.4, 3.1, 4.1],
            [2, 3, 3.8, 4.6],
            [1, 2, 3, 4],
        ]
        pd = pol.PolarDiagramTable(self.ws_resolution, wa_resolution, bsps)
        np.testing.assert_array_equal(pd.wind_speeds, self.ws_resolution)
        np.testing.assert_array_equal(pd.wind_angles, self.wa_resolution)
        np.testing.assert_array_equal(pd.boat_speeds, self.bsp)

    def test_init_unsorted_ws_wa_resolution(self):
        ws_resolution = [8, 2, 6, 4]
        wa_resolution = [20, 15, 25, 10]
        bsps = [
            [4.4, 1.7, 3.5, 2.6],
            [4.1, 1.5, 3.1, 2.4],
            [4.6, 2, 3.8, 3],
            [4, 1, 3, 2],
        ]
        pd = pol.PolarDiagramTable(ws_resolution, wa_resolution, bsps)
        np.testing.assert_array_equal(pd.wind_speeds, self.ws_resolution)
        np.testing.assert_array_equal(pd.wind_angles, self.wa_resolution)
        np.testing.assert_array_equal(pd.boat_speeds, self.bsp)

    def test_wind_speeds(self):
        np.testing.assert_array_equal(self.pd.wind_speeds, self.ws_resolution)

    def test_wind_angles(self):
        np.testing.assert_array_equal(self.pd.wind_angles, self.wa_resolution)

    def test_boat_speeds(self):
        np.testing.assert_array_equal(self.pd.boat_speeds, self.bsp)

    def test_symmetric_polar_diagram_no_180(self):
        sym_pd = self.pd.symmetrize()

        sym_wa_resolution = np.concatenate(
            [self.wa_resolution, 360 - np.flip(self.wa_resolution)]
        )
        sym_bsp = np.row_stack((self.bsp, np.flip(self.bsp, axis=0)))

        np.testing.assert_array_equal(sym_pd.wind_speeds, self.pd.wind_speeds)
        np.testing.assert_array_equal(sym_pd.wind_angles, sym_wa_resolution)
        np.testing.assert_array_equal(sym_pd.boat_speeds, sym_bsp)

    def test_symmetric_polar_diagram_w_180_and_0(self):
        wa_resolution = [0, 90, 180]
        pd = pol.PolarDiagramTable(
            ws_resolution=self.ws_resolution,
            wa_resolution=wa_resolution,
            bsps=self.bsp[:3, :],
        )
        sym_pd = pd.symmetrize()
        sym_bsps = np.row_stack((self.bsp[:3, :], self.bsp[1, :]))
        sym_wa_resolution = np.array([0, 90, 180, 270])
        np.testing.assert_array_equal(sym_pd.wind_speeds, self.ws_resolution)
        np.testing.assert_array_equal(sym_pd.wind_angles, sym_wa_resolution)
        np.testing.assert_array_equal(sym_pd.boat_speeds, sym_bsps)

    def test_change_one_entry(self):
        self.pd.change_entries(new_bsps=2.1, ws=2, wa=25)
        self.bsp[3, 0] = 2.1
        np.testing.assert_array_equal(self.pd.boat_speeds, self.bsp)

    def test_change_one_column(self):
        self.pd.change_entries(new_bsps=[3.5, 3.7, 3.9, 4.1], ws=8)
        self.bsp[:, 3] = [3.5, 3.7, 3.9, 4.1]
        np.testing.assert_array_equal(self.pd.boat_speeds, self.bsp)

    def test_change_one_row(self):
        self.pd.change_entries(new_bsps=[1.9, 2.7, 3.6, 4.4], wa=20)
        self.bsp[2, :] = [1.9, 2.7, 3.6, 4.4]
        np.testing.assert_array_equal(self.pd.boat_speeds, self.bsp)

    def test_change_subarray(self):
        self.pd.change_entries(
            new_bsps=[[2.3, 3.0], [2.5, 3.4]], ws=[4, 6], wa=[15, 20]
        )
        mask = np.zeros((4, 4), dtype=bool)
        mask[1, 1] = True
        mask[1, 2] = True
        mask[2, 1] = True
        mask[2, 2] = True
        self.bsp[mask] = np.array([[2.3, 3.0], [2.5, 3.4]]).flat
        np.testing.assert_array_equal(self.bsp, self.pd.boat_speeds)

    def test_change_entries_exceptions_empty(self):
        with self.assertRaises(PolarDiagramException):
            self.pd.change_entries(new_bsps=[])

    def test_change_entries_exception_wrong_shape(self):
        new_bsps = [
            [1],
            [[1, 2], [3, 4]],
            [[1, 2], [3, 4], [5, 6], [7, 8]],
            [[1, 2, 3, 4], [5, 6, 7, 8]],
        ]
        for i, new_bsp in enumerate(new_bsps):
            with self.subTest(i=i):
                with self.assertRaises(PolarDiagramException):
                    self.pd.change_entries(new_bsps=new_bsp)

    def test_change_entries_exception_not_array_like(self):
        new_bsps = [{}, set(), {1: 1, 2: 2, 3: 3, 4: 4}, {1, 2, 3, 4}]
        for i, new_bsp in enumerate(new_bsps):
            with self.subTest(i=i):
                with self.assertRaises(PolarDiagramException):
                    self.pd.change_entries(new_bsps=new_bsp)

    def test_get_one_slice(self):
        ws, wa, bsp = self.pd.get_slices(2)
        self.assertEqual(ws, [2])
        np.testing.assert_array_equal(wa, np.deg2rad(self.pd.wind_angles))
        np.testing.assert_array_equal(bsp.ravel(), self.pd.boat_speeds[:, 0])

    def test_get_multiple_slices_interval(self):
        ws, wa, bsp = self.pd.get_slices((2, 6))
        self.assertEqual(ws, [2, 4, 6])
        np.testing.assert_array_equal(wa, np.deg2rad(self.pd.wind_angles))
        np.testing.assert_array_equal(bsp, self.pd.boat_speeds[:, :3])

    def test_get_multiple_slices_list(self):
        ws, wa, bsp = self.pd.get_slices([2, 4, 8])
        self.assertEqual(ws, [2, 4, 8])
        np.testing.assert_array_equal(wa, np.deg2rad(self.pd.wind_angles))
        np.testing.assert_array_equal(bsp, self.pd.boat_speeds[:, [0, 1, 3]])

    def test_get_multiple_slices_tuple(self):
        ws, wa, bsp = self.pd.get_slices((2, 4, 8))
        self.assertEqual(ws, [2, 4, 8])
        np.testing.assert_array_equal(wa, np.deg2rad(self.pd.wind_angles))
        np.testing.assert_array_equal(bsp, self.pd.boat_speeds[:, [0, 1, 3]])

    def test_get_multiple_slices_set(self):
        ws, wa, bsp = self.pd.get_slices({2, 4, 8})
        self.assertEqual(ws, [2, 4, 8])
        np.testing.assert_array_equal(wa, np.deg2rad(self.pd.wind_angles))
        np.testing.assert_array_equal(bsp, self.pd.boat_speeds[:, [0, 1, 3]])

    def test_get_all_slices(self):
        ws, wa, bsp = self.pd.get_slices()
        self.assertEqual(ws, [2, 4, 6, 8])
        np.testing.assert_array_equal(wa, np.deg2rad(self.pd.wind_angles))
        np.testing.assert_array_equal(bsp, self.bsp)

    def test_get_slice_exceptions(self):
        slices = [[], 0, [0, 2, 4]]
        for i, slice_ in enumerate(slices):
            with self.subTest(i=i):
                with self.assertRaises(PolarDiagramException):
                    self.pd.get_slices(slice_)

    def test_plot_polar(self):
        plt.close()
        self.pd.plot_polar()
        wa = np.deg2rad(self.wa_resolution)
        bsp = self.bsp.T
        for i in range(len(self.ws_resolution)):
            with self.subTest(i=i):
                helper_functions.curve_table_plot_polar_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp)

    def test_plot_polar_single_element_ws(self):
        plt.close()
        self.pd.plot_polar(ws=2)
        ws, wa, bsp = self.pd.get_slices(ws=2)
        x_plot = plt.gca().lines[0].get_xdata()
        y_plot = plt.gca().lines[0].get_ydata()
        np.testing.assert_array_equal(x_plot, wa)
        np.testing.assert_array_equal(y_plot, np.asarray(bsp).flat)

    def test_plot_polar_interval_ws(self):
        plt.close()
        self.pd.plot_polar(ws=(4, 8))
        ws, wa, bsp = self.pd.get_slices(ws=(4, 8))
        bsp = bsp.T
        for i in range(3):
            with self.subTest(i=i):
                helper_functions.curve_table_plot_polar_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp)

    def test_plot_polar_iterable_list_ws(self):
        plt.close()
        self.pd.plot_polar(ws=[2, 4, 6])
        ws, wa, bsp = self.pd.get_slices(ws=[2, 4, 6])
        bsp = bsp.T
        for i in range(3):
            with self.subTest(i=i):
                helper_functions.curve_table_plot_polar_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp)

    def test_plot_polar_iterable_tuple_ws(self):
        plt.close()
        self.pd.plot_polar(ws=(2, 4, 6))
        ws, wa, bsp = self.pd.get_slices(ws=(2, 4, 6))
        bsp = bsp.T
        for i in range(3):
            with self.subTest(i=i):
                helper_functions.curve_table_plot_polar_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp)

    def test_plot_polar_iterable_set_ws(self):
        plt.close()
        self.pd.plot_polar(ws={2, 4, 6})
        ws, wa, bsp = self.pd.get_slices(ws={2, 4, 6})
        bsp = bsp.T
        for i in range(3):
            with self.subTest(i=i):
                helper_functions.curve_table_plot_polar_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp)

    def test_plot_polar_axes_keywords(self):
        # test not implemented yet
        pass

    def test_plot_polar_single_color(self):
        plt.close()
        self.pd.plot_polar(colors="purple")
        for i in range(4):
            with self.subTest(i=i):
                self.assertEqual(plt.gca().lines[i].get_color(), "purple")

    def test_plot_polar_two_colors_passed(self):
        plt.close()
        self.pd.plot_polar(ws=[4, 6, 8], colors=["red", "blue"])
        helper_functions.comparing_colors_two_colors_passed()

    def test_plot_polar_more_than_two_colors_passed(self):
        plt.close()
        self.pd.plot_polar(ws=[2, 4, 6, 8], colors=["red", "yellow", "orange"])
        helper_functions.comparing_colors_more_than_two_colors_passed()

    def test_plot_polar_ws_color_pairs_passed(self):
        plt.close()
        self.pd.plot_polar(ws=[4, 6, 8], colors=((4, "purple"), (6, "blue"), (8, "red")))
        helper_functions.comparing_colors_ws_color_pairs_passed()

    def test_plot_polar_ws_color_pairs_unsorted_passed(self):
        plt.close()
        self.pd.plot_polar(ws=[4, 6, 8], colors=((4, "purple"), (8, "red"), (6, "blue")))
        helper_functions.comparing_colors_ws_color_pairs_passed()

    def test_plot_polar_show_legend(self):
        plt.close()
        self.pd.plot_polar(ws=[2, 4, 6], colors=["red", "purple", "blue"], show_legend=True)
        helper_functions.test_cloud_table_comparing_show_legend(self, plt.gca().get_legend())

    def test_plot_polar_legend_kw(self):
        plt.close()
        self.pd.plot_polar(ws=[2, 4, 6], colors=["red", "purple", "blue"], show_legend=True,
                           legend_kw={'labels': ["ws 2", "ws 4", "ws 6"], 'loc': 'upper left'})
        helper_functions.test_cloud_table_comparing_legend_keywords(self, plt.gca().get_legend())

    def test_plot_polar_show_colorbar(self):
        plt.close()
        self.pd.plot_polar(ws=[2, 4, 6], colors=("red", "blue"), show_legend=True)
        helper_functions.test_comparing_show_colorbar(self, "True Wind Speed")

    def test_plot_polar_plot_kw(self):
        plt.close()
        self.pd.plot_polar(ls=":", lw=1.5, marker="o")
        for i in range(4):
            with self.subTest(i=i):
                helper_functions.test_comparing_plot_kw(self, i)

    def test_plot_polar_exception_ws_not_in_self_wind_speeds(self):
        with self.subTest(i=0):
            with self.assertRaises(PolarDiagramException):
                self.pd.plot_polar(ws=3)
        with self.subTest(i=1):
            with self.assertRaises(PolarDiagramException):
                self.pd.plot_polar(ws=[2, 4, 5])

    def test_plot_polar_exception_no_slice_in_interval(self):
        with self.subTest(i=0):
            with self.assertRaises(PolarDiagramException):
                self.pd.plot_polar(ws=(9, 10))
        with self.subTest(i=1):
            with self.assertRaises(PolarDiagramException):
                self.pd.plot_polar(ws=(2, 0))

    def test_plot_flat(self):
        plt.close()
        self.pd.plot_flat()
        ws, wa, bsp = self.pd.get_slices(None)
        bsp = bsp.T
        for i in range(len(self.ws_resolution)):
            with self.subTest(i=i):
                helper_functions.curve_table_plot_flat_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp)

    def test_plot_flat_single_element_ws(self):
        plt.close()
        self.pd.plot_flat(ws=2)
        ws, wa, bsp = self.pd.get_slices(ws=2)
        x_plot = plt.gca().lines[0].get_xdata()
        y_plot = plt.gca().lines[0].get_ydata()
        np.testing.assert_array_equal(x_plot, np.rad2deg(wa))
        np.testing.assert_array_equal(y_plot, np.asarray(bsp).flat)

    def test_plot_flat_interval_ws(self):
        plt.close()
        self.pd.plot_flat(ws=(4, 8))
        ws, wa, bsp = self.pd.get_slices(ws=(4, 8))
        bsp = bsp.T
        for i in range(3):
            with self.subTest(i=i):
                helper_functions.curve_table_plot_flat_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp)

    def test_plot_flat_iterable_list_ws(self):
        plt.close()
        self.pd.plot_flat(ws=[2, 4, 6])
        ws, wa, bsp = self.pd.get_slices(ws=[2, 4, 6])
        bsp = bsp.T
        for i in range(3):
            with self.subTest(i=i):
                helper_functions.curve_table_plot_flat_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp)

    def test_plot_flat_iterable_tuple_ws(self):
        plt.close()
        self.pd.plot_flat(ws=(2, 4, 6))
        ws, wa, bsp = self.pd.get_slices(ws=(2, 4, 6))
        bsp = bsp.T
        for i in range(3):
            with self.subTest(i=i):
                helper_functions.curve_table_plot_flat_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp)

    def test_plot_flat_iterable_set_ws(self):
        plt.close()
        self.pd.plot_flat(ws={2, 4, 6})
        ws, wa, bsp = self.pd.get_slices(ws={2, 4, 6})
        bsp = bsp.T
        for i in range(3):
            with self.subTest(i=i):
                helper_functions.curve_table_plot_flat_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp)

    def test_plot_flat_axes_keywords(self):
        # test not implemented yet
        pass

    def test_plot_flat_single_color(self):
        plt.close()
        self.pd.plot_flat(colors="purple")
        for i in range(4):
            with self.subTest(i=i):
                self.assertEqual(plt.gca().lines[i].get_color(), "purple")

    def test_plot_flat_two_colors_passed(self):
        plt.close()
        self.pd.plot_flat(ws=[4, 6, 8], colors=["red", "blue"])
        helper_functions.comparing_colors_two_colors_passed()

    def test_plot_flat_more_than_two_colors_passed(self):
        plt.close()
        self.pd.plot_flat(ws=[2, 4, 6, 8], colors=["red", "yellow", "orange"])
        helper_functions.comparing_colors_more_than_two_colors_passed()

    def test_plot_flat_ws_color_pairs_passed(self):
        plt.close()
        self.pd.plot_flat(ws=[4, 6, 8], colors=((4, "purple"), (6, "blue"), (8, "red")))
        helper_functions.comparing_colors_ws_color_pairs_passed()

    def test_plot_flat_ws_color_pairs_unsorted_passed(self):
        plt.close()
        self.pd.plot_flat(ws=[4, 6, 8], colors=((4, "purple"), (8, "red"), (6, "blue")))
        helper_functions.comparing_colors_ws_color_pairs_passed()

    def test_plot_flat_show_legend(self):
        plt.close()
        self.pd.plot_flat(ws=[2, 4, 6], colors=["red", "purple", "blue"], show_legend=True)
        helper_functions.test_cloud_table_comparing_show_legend(self, plt.gca().get_legend())

    def test_plot_flat_legend_kw(self):
        plt.close()
        self.pd.plot_flat(ws=[2, 4, 6], colors=["red", "purple", "blue"], show_legend=True,
                          legend_kw={'labels': ["ws 2", "ws 4", "ws 6"], 'loc': 'upper left'})
        helper_functions.test_cloud_table_comparing_legend_keywords(self, plt.gca().get_legend())

    def test_plot_flat_show_colorbar(self):
        plt.close()
        self.pd.plot_flat(ws=[2, 4, 6], colors=("red", "blue"), show_legend=True)
        helper_functions.test_comparing_show_colorbar(self, "True Wind Speed")

    def test_plot_flat_plot_kw(self):
        plt.close()
        self.pd.plot_flat(ls=":", lw=1.5, marker="o")
        for i in range(4):
            with self.subTest(i=i):
                helper_functions.test_comparing_plot_kw(self, i)

    def test_plot_flat_exception_ws_not_in_self_wind_speeds(self):
        with self.subTest(i=0):
            with self.assertRaises(PolarDiagramException):
                self.pd.plot_flat(ws=3)
        with self.subTest(i=1):
            with self.assertRaises(PolarDiagramException):
                self.pd.plot_flat(ws=[2, 4, 5])

    def test_plot_flat_exception_no_slice_in_interval(self):
        with self.subTest(i=0):
            with self.assertRaises(PolarDiagramException):
                self.pd.plot_flat(ws=(9, 10))
        with self.subTest(i=1):
            with self.assertRaises(PolarDiagramException):
                self.pd.plot_flat(ws=(2, 0))

    def test_plot_3d(self):
        plt.close()
        self.pd.plot_3d()
        wind_speeds = plt.gca().collections[0]._vec[0]
        bsp_sinus_wa = plt.gca().collections[0]._vec[1]
        bsp_cosinus_wa = plt.gca().collections[0]._vec[2]
        wss, was, bsps = self.pd.get_slices()
        ws_results = list(np.asarray([4 * [2], 4 * [4], 4 * [6], 4 * [8]]).flat)
        bsp_sin_wa_results = []
        bsp_cos_wa_results = []
        for i in range(4):
            for bsp, wa in zip(bsps.T[i], was):
                bsp_sin_wa_results.append(bsp * math.sin(wa))
                bsp_cos_wa_results.append(bsp * math.cos(wa))

        for triple in zip(wind_speeds, bsp_sinus_wa, bsp_cosinus_wa):
            self.assertIn(triple, zip(ws_results, bsp_sin_wa_results, bsp_cos_wa_results))
        for result in zip(ws_results, bsp_sin_wa_results, bsp_cos_wa_results):
            self.assertIn(result, zip(wind_speeds, bsp_sinus_wa, bsp_cosinus_wa))

    def test_plot_3d_ax_instance(self):
        # test not finished yet
        plt.close()
        ax = plt.axes(projection='3d', label='axes label')
        self.pd.plot_3d(ax=ax)
        self.assertEqual(plt.gca().get_label(), 'axes label')

    def test_plot_3d_color_pair(self):
        # test not finished yet
        plt.close()
        self.pd.plot_3d(colors=('blue', 'red'))
        #print(plt.gca().collections[0].__dict__)

    def test_plot_color_gradient(self):
        plt.close()
        self.pd.plot_color_gradient()
        ws_wa_list = [list(item) for item in plt.gca().collections[0]._offsets]
        all_combinations_ws_wa = [list(item) for item in itertools.product(self.ws_resolution, self.wa_resolution)]
        self.assertEqual(len(ws_wa_list), len(all_combinations_ws_wa))
        self.assertCountEqual(ws_wa_list, all_combinations_ws_wa)
        colors = [item[:-1] for item in plt.gca().collections[0]._facecolors]
        flat_bsp = self.bsp.flat
        helper_functions.cloud_table_plot_color_gradient_calculations(self, ws_wa_list,
                                                                      all_combinations_ws_wa,
                                                                      flat_bsp,
                                                                      colors)

    # test for plot_color_gradient when the colors are given as a color pair
    # def test_plot_color_gradient_color_pair(self):

    # test for plot_color_gradient when the marker and the marker size are given
    # def test_plot_color_gradient_marker_ms(self):

    def test_plot_color_gradient_show_colorbar(self):
        plt.close()
        self.pd.plot_color_gradient(show_legend=True)
        helper_functions.test_comparing_show_colorbar(self, "Boat Speed")

    def test_plot_color_gradient_legend_kw(self):
        # test not finished yet
        plt.close()
        self.pd.plot_color_gradient(show_legend=True, orientation="horizontal", ticklocation="top")
        colorbar_axes = None
        for axes in plt.gcf().axes:
            if axes.get_label() == "<colorbar>":
                colorbar_axes = axes
        print(colorbar_axes.__dict__)
        # plt.show()

    def test_plot_convex_hull(self):
        # test not finished yet
        plt.close()
        self.pd.plot_convex_hull()

    def test_plot_convex_hull_single_element_ws(self):
        # test not finished yet
        plt.close()
        self.pd.plot_convex_hull(ws=2)
        x_data = plt.gca().lines[0].get_xdata()
        y_data = plt.gca().lines[0].get_ydata()
        np.testing.assert_array_equal(x_data, np.deg2rad([10, 15, 25, 10]))
        np.testing.assert_array_equal(y_data, [1, 1.5, 2, 1])

    def test_plot_convex_hull_interval_ws(self):
        # test not finished yet
        plt.close()
        self.pd.plot_convex_hull(ws=(2, 6))

    def test_plot_convex_hull_iterable_list_ws(self):
        # test not finished yet
        plt.close()
        self.pd.plot_convex_hull(ws=[2, 4, 6])

    def test_plot_convex_hull_iterable_tuple_ws(self):
        # test not finished yet
        plt.close()
        self.pd.plot_convex_hull(ws=(2, 4, 6))

    def test_plot_convex_hull_iterable_set_ws(self):
        # test not finished yet
        plt.close()
        self.pd.plot_convex_hull(ws={2, 4, 6})

    def test_plot_convex_hull_axes_keywords(self):
        # test not implemented yet
        pass

    def test_plot_convex_hull_single_color(self):
        plt.close()
        self.pd.plot_convex_hull(colors="purple")
        for i in range(4):
            with self.subTest(i=i):
                self.assertEqual(plt.gca().lines[i].get_color(), "purple")

    def test_plot_convex_hull_two_colors_passed(self):
        plt.close()
        self.pd.plot_convex_hull(ws=[4, 6, 8], colors=["red", "blue"])
        helper_functions.comparing_colors_two_colors_passed()

    def test_plot_convex_hull_more_than_two_colors_passed(self):
        plt.close()
        self.pd.plot_convex_hull(ws=[2, 4, 6, 8], colors=["red", "yellow", "orange"])
        helper_functions.comparing_colors_more_than_two_colors_passed()

    def test_plot_convex_hull_ws_color_pairs_passed(self):
        plt.close()
        self.pd.plot_convex_hull(ws=[4, 6, 8], colors=((4, "purple"), (6, "blue"), (8, "red")))
        helper_functions.comparing_colors_ws_color_pairs_passed()

    def test_plot_convex_hull_ws_color_pairs_unsorted_passed(self):
        plt.close()
        self.pd.plot_convex_hull(ws=[4, 6, 8], colors=((4, "purple"), (8, "red"), (6, "blue")))
        helper_functions.comparing_colors_ws_color_pairs_passed()

    def test_plot_convex_hull_show_legend(self):
        plt.close()
        self.pd.plot_convex_hull(ws=[2, 4, 6], colors=["red", "purple", "blue"], show_legend=True)
        helper_functions.test_cloud_table_comparing_show_legend(self, plt.gca().get_legend())

    def test_plot_convex_hull_legend_kw(self):
        plt.close()
        self.pd.plot_convex_hull(ws=[2, 4, 6], colors=["red", "purple", "blue"], show_legend=True,
                                 legend_kw={'labels': ["ws 2", "ws 4", "ws 6"], 'loc': 'upper left'})
        helper_functions.test_cloud_table_comparing_legend_keywords(self, plt.gca().get_legend())

    def test_plot_convex_hull_show_colorbar(self):
        plt.close()
        self.pd.plot_convex_hull(ws=[2, 4, 6], colors=("red", "blue"), show_legend=True)
        helper_functions.test_comparing_show_colorbar(self, "True Wind Speed")

    def test_plot_convex_hull_plot_kw(self):
        plt.close()
        self.pd.plot_convex_hull(ls=":", lw=1.5, marker="o")
        for i in range(4):
            with self.subTest(i=i):
                helper_functions.test_comparing_plot_kw(self, i)

    def test_plot_convex_hull_exception_ws_not_in_self_wind_speeds(self):
        with self.subTest(i=0):
            with self.assertRaises(PolarDiagramException):
                self.pd.plot_convex_hull(ws=3)
        with self.subTest(i=1):
            with self.assertRaises(PolarDiagramException):
                self.pd.plot_convex_hull(ws=[2, 4, 5])

    def test_plot_convex_hull_exception_no_slice_in_interval(self):
        with self.subTest(i=0):
            with self.assertRaises(PolarDiagramException):
                self.pd.plot_convex_hull(ws=(9, 10))
        with self.subTest(i=1):
            with self.assertRaises(PolarDiagramException):
                self.pd.plot_convex_hull(ws=(2, 0))
