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
        f, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        self.pd.plot_polar(ax=ax)
        for i in range(len(self.ws_resolution)):
            with self.subTest(i=i):
                x_plot = ax.lines[i].get_xdata()
                y_plot = ax.lines[i].get_ydata()
                np.testing.assert_array_equal(x_plot, np.deg2rad(self.wa_resolution))
                np.testing.assert_array_equal(y_plot, self.bsp[:, i])

    def test_plot_polar_single_element_ws(self):
        f, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        self.pd.plot_polar(ws=2, ax=ax)
        x_plot = ax.lines[0].get_xdata()
        y_plot = ax.lines[0].get_ydata()
        np.testing.assert_array_equal(x_plot, np.deg2rad(self.wa_resolution))
        np.testing.assert_array_equal(y_plot, self.bsp[:, 0])

    def test_plot_polar_interval_ws(self):
        f, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        self.pd.plot_polar(ws=(4, 8), ax=ax)
        for i in range(0, 3):
            with self.subTest(i=i):
                x_plot = ax.lines[i].get_xdata()
                y_plot = ax.lines[i].get_ydata()
                np.testing.assert_array_equal(x_plot, np.deg2rad(self.wa_resolution))
                print(y_plot)
                print(self.bsp[:, i+1])
                np.testing.assert_array_equal(y_plot, self.bsp[:, i+1])

    def test_plot_polar_iterable_list_ws(self):
        f, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        self.pd.plot_polar(ws=[2, 4, 6], ax=ax)
        for i in range(0, 3):
            with self.subTest(i=i):
                x_plot = ax.lines[i].get_xdata()
                y_plot = ax.lines[i].get_ydata()
                np.testing.assert_array_equal(x_plot, np.deg2rad(self.wa_resolution))
                np.testing.assert_array_equal(y_plot, self.bsp[:, i])

    def test_plot_polar_iterable_tuple_ws(self):
        f, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        self.pd.plot_polar(ws=(2, 4, 6), ax=ax)
        for i in range(0, 3):
            with self.subTest(i=i):
                x_plot = ax.lines[i].get_xdata()
                y_plot = ax.lines[i].get_ydata()
                np.testing.assert_array_equal(x_plot, np.deg2rad(self.wa_resolution))
                np.testing.assert_array_equal(y_plot, self.bsp[:, i])

    def test_plot_polar_iterable_set_ws(self):
        f, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        self.pd.plot_polar(ws={2, 4, 6}, ax=ax)
        for i in range(0, 3):
            with self.subTest(i=i):
                x_plot = ax.lines[i].get_xdata()
                y_plot = ax.lines[i].get_ydata()
                np.testing.assert_array_equal(x_plot, np.deg2rad(self.wa_resolution))
                np.testing.assert_array_equal(y_plot, self.bsp[:, i])

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
        f, ax = plt.subplots()
        self.pd.plot_flat(ax=ax)
        for i in range(len(self.ws_resolution)):
            with self.subTest(i=i):
                x_plot = ax.lines[i].get_xdata()
                y_plot = ax.lines[i].get_ydata()
                np.testing.assert_array_equal(np.deg2rad(x_plot), np.deg2rad(self.wa_resolution))
                np.testing.assert_array_equal(y_plot, self.bsp[:, i])

    def test_plot_flat_single_element_ws(self):
        f, ax = plt.subplots()
        self.pd.plot_flat(ws=2, ax=ax)
        x_plot = ax.lines[0].get_xdata()
        y_plot = ax.lines[0].get_ydata()
        np.testing.assert_array_equal(np.deg2rad(x_plot), np.deg2rad(self.wa_resolution))
        np.testing.assert_array_equal(y_plot, self.bsp[:, 0])

    def test_plot_flat_interval_ws(self):
        f, ax = plt.subplots()
        self.pd.plot_flat(ws=(4, 8), ax=ax)
        for i in range(0, 3):
            with self.subTest(i=i):
                x_plot = ax.lines[i].get_xdata()
                y_plot = ax.lines[i].get_ydata()
                np.testing.assert_array_equal(np.deg2rad(x_plot), np.deg2rad(self.wa_resolution))
                np.testing.assert_array_equal(y_plot, self.bsp[:, i+1])

    def test_plot_flat_iterable_list_ws(self):
        f, ax = plt.subplots()
        self.pd.plot_flat(ws=[2, 4, 6], ax=ax)
        for i in range(0, 3):
            with self.subTest(i=i):
                x_plot = ax.lines[i].get_xdata()
                y_plot = ax.lines[i].get_ydata()
                np.testing.assert_array_equal(np.deg2rad(x_plot), np.deg2rad(self.wa_resolution))
                np.testing.assert_array_equal(y_plot, self.bsp[:, i])

    def test_plot_flat_iterable_tuple_ws(self):
        f, ax = plt.subplots()
        self.pd.plot_flat(ws=(2, 4, 6), ax=ax)
        for i in range(0, 3):
            with self.subTest(i=i):
                x_plot = ax.lines[i].get_xdata()
                y_plot = ax.lines[i].get_ydata()
                np.testing.assert_array_equal(np.deg2rad(x_plot), np.deg2rad(self.wa_resolution))
                np.testing.assert_array_equal(y_plot, self.bsp[:, i])

    def test_plot_flat_iterable_set_ws(self):
        f, ax = plt.subplots()
        self.pd.plot_flat(ws={2, 4, 6}, ax=ax)
        for i in range(0, 3):
            with self.subTest(i=i):
                x_plot = ax.lines[i].get_xdata()
                y_plot = ax.lines[i].get_ydata()
                np.testing.assert_array_equal(np.deg2rad(x_plot), np.deg2rad(self.wa_resolution))
                np.testing.assert_array_equal(y_plot, self.bsp[:, i])

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
        f, ax = plt.subplots(subplot_kw={'projection': '3d'})
        self.pd.plot_3d(ax=ax)

    def test_plot_3d_colors(self):
        f, ax = plt.subplots(subplot_kw={'projection': '3d'})
        self.pd.plot_3d(ax=ax, colors=('blue', 'red'))