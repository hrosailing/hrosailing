"""

"""

# Author: Valentin F. Dannenberg


import unittest

import numpy as np

import hrosailing.polardiagram as pol
from hrosailing.polardiagram import (
    PolarDiagramException,
    PolarDiagramInitializationException
)
from hrosailing.wind import WindConversionException


class PolarDiagramTableTest(unittest.TestCase):
    def setUp(self):
        self.ws_res = np.array([2, 4, 6, 8])
        self.wa_res = np.array([10, 15, 20, 25])
        self.bsp = np.array(
            [
                [1, 2, 3, 4],
                [1.5, 2.4, 3.1, 4.1],
                [1.7, 2.6, 3.5, 4.4],
                [2, 3, 3.8, 4.6],
            ]
        )
        self.pd = pol.PolarDiagramTable(self.ws_res, self.wa_res, self.bsp)

    @staticmethod
    def test_default_init():
        pd = pol.PolarDiagramTable()
        np.testing.assert_array_equal(pd.wind_speeds, np.arange(2, 42, 2))
        np.testing.assert_array_equal(pd.wind_angles, np.arange(0, 360, 5))
        np.testing.assert_array_equal(pd.boat_speeds, np.zeros((72, 20)))

    def test_init(self):
        np.testing.assert_array_equal(self.pd.wind_speeds, self.ws_res)
        np.testing.assert_array_equal(self.pd.wind_angles, self.wa_res)
        np.testing.assert_array_equal(self.pd.boat_speeds, self.bsp)

    def test_init_only_ws_res(self):
        pd = pol.PolarDiagramTable(ws_res=self.ws_res)
        np.testing.assert_array_equal(pd.wind_speeds, self.ws_res)
        np.testing.assert_array_equal(pd.wind_angles, np.arange(0, 360, 5))
        np.testing.assert_array_equal(pd.boat_speeds, np.zeros((72, 4)))

    def test_init_only_wa_res(self):
        pd = pol.PolarDiagramTable(wa_res=self.wa_res)
        np.testing.assert_array_equal(pd.wind_speeds, np.arange(2, 42, 2))
        np.testing.assert_array_equal(pd.wind_angles, self.wa_res)
        np.testing.assert_array_equal(pd.boat_speeds, np.zeros((4, 20)))

    def test_init_ws_wa_res(self):
        pd = pol.PolarDiagramTable(ws_res=self.ws_res, wa_res=self.wa_res)
        np.testing.assert_array_equal(pd.wind_speeds, self.ws_res)
        np.testing.assert_array_equal(pd.wind_angles, self.wa_res)
        np.testing.assert_array_equal(pd.boat_speeds, np.zeros((4, 4)))

    @staticmethod
    def test_init_only_bsps():
        bsps = np.random.rand(72, 20)
        pd = pol.PolarDiagramTable(bsps=bsps)
        np.testing.assert_array_equal(pd.wind_speeds, np.arange(2, 42, 2))
        np.testing.assert_array_equal(pd.wind_angles, np.arange(0, 360, 5))
        np.testing.assert_array_equal(pd.boat_speeds, bsps)

    def test_init_ws_res_bsps(self):
        bsps = np.random.rand(72, 4)
        pd = pol.PolarDiagramTable(ws_res=self.ws_res, bsps=bsps)
        np.testing.assert_array_equal(pd.wind_speeds, self.ws_res)
        np.testing.assert_array_equal(pd.wind_angles, np.arange(0, 360, 5))
        np.testing.assert_array_equal(pd.boat_speeds, bsps)

    def test_init_wa_res_bsps(self):
        bsps = np.random.rand(4, 20)
        pd = pol.PolarDiagramTable(wa_res=self.wa_res, bsps=bsps)
        np.testing.assert_array_equal(pd.wind_speeds, np.arange(2, 42, 2))
        np.testing.assert_array_equal(pd.wind_angles, self.wa_res)
        np.testing.assert_array_equal(pd.boat_speeds, bsps)

    def test_init_ws_res_not_array_like(self):
        ws_res = [{2, 4, 6, 8}, {2: 0, 4: 0, 6: 0, 8: 0}]
        for i, ws in enumerate(ws_res):
            with self.subTest(i=i):
                with self.assertRaises(ValueError):
                    pol.PolarDiagramTable(ws_res=ws)

    def test_init_wa_res_not_array_like(self):
        wa_res = [{10, 15, 20, 25}, {10: 0, 15: 0, 20: 0, 25: 0}]
        for i, wa in enumerate(wa_res):
            with self.subTest(i=i):
                with self.assertRaises(ValueError):
                    pol.PolarDiagramTable(wa_res=wa)

    def test_init_exception_empty_bsps(self):
        with self.assertRaises(PolarDiagramInitializationException):
            pol.PolarDiagramTable(bsps=[])

    def test_init_exception_not_ndim2(self):
        with self.assertRaises(PolarDiagramInitializationException):
            pol.PolarDiagramTable(bsps=[[[0]]])

    def test_init_exception_wrong_shape(self):
        with self.assertRaises(PolarDiagramInitializationException):
            pol.PolarDiagramTable(bsps=[[0]])

    def test_init_unsorted_ws_res(self):
        ws_res = [8, 2, 6, 4]
        bsps = [
            [4, 1, 3, 2],
            [4.1, 1.5, 3.1, 2.4],
            [4.4, 1.7, 3.5, 2.6],
            [4.6, 2, 3.8, 3],
        ]
        pd = pol.PolarDiagramTable(ws_res, self.wa_res, bsps)
        np.testing.assert_array_equal(pd.wind_speeds, self.ws_res)
        np.testing.assert_array_equal(pd.wind_angles, self.wa_res)
        np.testing.assert_array_equal(pd.boat_speeds, self.bsp)

    def test_init_unsorted_wa_res(self):
        wa_res = [20, 15, 25, 10]
        bsps = [
            [1.7, 2.6, 3.5, 4.4],
            [1.5, 2.4, 3.1, 4.1],
            [2, 3, 3.8, 4.6],
            [1, 2, 3, 4],
        ]
        pd = pol.PolarDiagramTable(self.ws_res, wa_res, bsps)
        np.testing.assert_array_equal(pd.wind_speeds, self.ws_res)
        np.testing.assert_array_equal(pd.wind_angles, self.wa_res)
        np.testing.assert_array_equal(pd.boat_speeds, self.bsp)

    def test_init_unsorted_ws_wa_res(self):
        ws_res = [8, 2, 6, 4]
        wa_res = [20, 15, 25, 10]
        bsps = [
            [4.4, 1.7, 3.5, 2.6],
            [4.1, 1.5, 3.1, 2.4],
            [4.6, 2, 3.8, 3],
            [4, 1, 3, 2],
        ]
        pd = pol.PolarDiagramTable(ws_res, wa_res, bsps)
        np.testing.assert_array_equal(pd.wind_speeds, self.ws_res)
        np.testing.assert_array_equal(pd.wind_angles, self.wa_res)
        np.testing.assert_array_equal(pd.boat_speeds, self.bsp)

    def test_wind_speeds(self):
        np.testing.assert_array_equal(self.pd.wind_speeds, self.ws_res)

    def test_wind_angles(self):
        np.testing.assert_array_equal(self.pd.wind_angles, self.wa_res)

    def test_boat_speeds(self):
        np.testing.assert_array_equal(self.pd.boat_speeds, self.bsp)

    def test_symmetric_polar_diagram_no_180(self):
        sym_pd = self.pd.symmetrize()

        sym_wa_res = np.concatenate([self.wa_res, 360 - np.flip(self.wa_res)])
        sym_bsp = np.row_stack((self.bsp, np.flip(self.bsp, axis=0)))

        np.testing.assert_array_equal(sym_pd.wind_speeds, self.pd.wind_speeds)
        np.testing.assert_array_equal(sym_pd.wind_angles, sym_wa_res)
        np.testing.assert_array_equal(sym_pd.boat_speeds, sym_bsp)

    def test_symmetric_polar_diagram_w_180_and_0(self):
        wa_res = [0, 90, 180]
        pd = pol.PolarDiagramTable(
            ws_res=self.ws_res, wa_res=wa_res, bsps=self.bsp[:3, :]
        )
        sym_pd = pd.symmetrize()
        sym_bsps = np.row_stack((self.bsp[:3, :], self.bsp[1, :]))
        sym_wa_res = np.array([0, 90, 180, 270])
        np.testing.assert_array_equal(sym_pd.wind_speeds, self.ws_res)
        np.testing.assert_array_equal(sym_pd.wind_angles, sym_wa_res)
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

    def test_default_init(self):
        pc = pol.PolarDiagramPointcloud()
        self.assertEqual(pc.points.size, False)

    def test_init(self):
        np.testing.assert_array_equal(self.pc.points, self.points)

    def test_init_exception_wrong_size(self):
        with self.assertRaises(WindConversionException):
            pol.PolarDiagramPointcloud(pts=[0])

    def test_wind_speeds(self):
        np.testing.assert_array_equal(self.pc.wind_speeds, [2, 4, 6, 8])

    def test_wind_angles(self):
        np.testing.assert_array_equal(self.pc.wind_angles, [10, 15, 20, 25])

    def test_points(self):
        np.testing.assert_array_equal(self.pc.points, self.points)

    def test_add_points_with_no_points(self):
        pc = pol.PolarDiagramPointcloud()
        pc.add_points(self.points)
        np.testing.assert_array_equal(pc.points, self.points)

    def test_add_points(self):
        self.pc.add_points([[2.3, 15.5, 1.65], [3.7, 20.1, 2.43]])
        self.points = np.row_stack(
            (self.points, np.array([[2.3, 15.5, 1.65], [3.7, 20.1, 2.43]]))
        )
        np.testing.assert_array_equal(self.pc.points, self.points)

    def test_add_points_exception_empty_new_pts(self):
        with self.assertRaises(WindConversionException):
            self.pc.add_points(new_pts=[])

    def test_add_points_exception_wrong_shape(self):
        new_pts = [[0], [1, 2], [1, 2, 3, 4]]
        for i, new_pt in enumerate(new_pts):
            with self.subTest(i=i):
                with self.assertRaises(WindConversionException):
                    self.pc.add_points(new_pts=new_pt)

    def test_add_points_exception_not_array_like(self):
        new_pts = [{}, set(), {1: 1, 2: 2, 3: 3}, {1, 2, 3}]
        for i, new_pt in enumerate(new_pts):
            with self.subTest(i=i):
                with self.assertRaises(WindConversionException):
                    self.pc.add_points(new_pts=new_pt)

    def test_symmetric_polar_diagram_no_points(self):
        pc = pol.PolarDiagramPointcloud()
        sym_pc = pc.symmetrize()
        self.assertEqual(sym_pc.points.size, False)

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
            np.array([1.1, 1.5, 1.7, 2.1, 2, 2.4, 2.6, 3]),
            np.array([2, 2.4, 2.6, 3]),
            np.array([3, 3.1, 3.5, 3.8]),
            np.array([3, 3.1, 3.5, 3.8]),
            np.array([4, 4.1, 4.4, 4.6]),
            np.array([4, 4.1, 4.4, 4.6]),
        ]
        answers = [np.deg2rad([10, 15, 20, 25, 10, 15, 20, 25]), np.deg2rad([10, 15, 20, 25]), np.deg2rad([10, 15, 20, 25]),
                   np.deg2rad([10, 15, 20, 25]), np.deg2rad([10, 15, 20, 25]), np.deg2rad([10, 15, 20, 25])]
        for i in range(6):
            np.testing.assert_array_equal(wa[i], answers[i])
            np.testing.assert_array_equal(bsp[i], bsps[i])

    def test_get_slices_range_no_slices(self):
        with self.assertRaises(PolarDiagramException):
            self.pc.get_slices((0, 1))

    def test_get_slices_range_empty_interval(self):
        with self.assertRaises(PolarDiagramException):
            self.pc.get_slices((1, 0))


class PolarDiagramCurveTest(unittest.TestCase):
    def setUp(self):
        def func(ws, wa , *params):
            return params[0] * np.asarray(ws) * np.asarray(wa) + params[1]

        self.f = func
        self.params = 1, 2
        self.radians = False
        self.c = pol.PolarDiagramCurve(
            self.f, *self.params, radians=self.radians
        )

    def test_init(self):
        self.assertEqual(self.c.curve.__name__, "func")
        self.assertEqual(self.c.parameters, (1, 2))
        self.assertEqual(self.c.radians, False)

    def test_init_exception_not_callable(self):
        with self.assertRaises(PolarDiagramInitializationException):
            f = 5
            params = 1, 2
            pol.PolarDiagramCurve(f, params)

    def test_not_enough_params(self):
        with self.assertRaises(PolarDiagramInitializationException):
            pol.PolarDiagramCurve(self.f, radians=False)

        with self.assertRaises(PolarDiagramInitializationException):
            pol.PolarDiagramCurve(self.f, 1, radians=False)

    def test_more_params_then_needed(self):
        pol.PolarDiagramCurve(self.f, 1, 2, 3, radians=False)
        self.assertTrue(True)

    def test_curve(self):
        self.assertEqual(self.c.curve.__name__, "func")

    def test_parameters(self):
        self.assertEqual(self.c.parameters, (1, 2))

    def test_radians(self):
        self.assertEqual(self.c.radians, False)

    def test_call_scalar(self):
        import random

        for _ in range(500):
            ws = random.randrange(2, 40)
            wa = random.randrange(0, 360)
            self.assertEqual(self.c(ws, wa), ws * wa + 2)

    def test_call_array(self):
        for _ in range(500):
            ws = np.random.rand(100)
            wa = np.random.rand(100)
            np.testing.assert_array_equal(self.c(ws, wa), ws * wa + 2)

    def test_symmetrize(self):
        import random

        sym_c = self.c.symmetrize()
        for _ in range(500):
            ws = random.randrange(2, 40)
            wa = random.randrange(0, 360)
            np.testing.assert_array_equal(
                sym_c(ws, wa), 1 / 2 * (self.c(ws, wa) + self.c(ws, 360 - wa))
            )

    def test_get_slice(self):
        ws, wa, bsp = self.c.get_slices(10)
        self.assertEqual(ws, [10])
        np.testing.assert_array_equal(
            wa, np.deg2rad(np.linspace(0, 360, 1000))
        )
        np.testing.assert_array_equal(
            bsp[0], self.c(np.array(ws * 1000), np.linspace(0, 360, 1000))
        )

    def test_get_slices_list(self):
        ws, wa, bsp = self.c.get_slices([10, 12, 14])
        self.assertEqual(ws, [10, 12, 14])
        np.testing.assert_array_equal(
            wa, np.deg2rad(np.linspace(0, 360, 1000))
        )
        for i, w in enumerate(ws):
            np.testing.assert_array_equal(
                bsp[i], self.c(np.array([w] * 1000), np.linspace(0, 360, 1000))
            )

    def test_get_slices_tuple(self):
        ws, wa, bsp = self.c.get_slices((10, 15), stepsize=100)
        self.assertEqual(ws, list(np.linspace(10, 15, 100)))
        np.testing.assert_array_equal(
            wa, np.deg2rad(np.linspace(0, 360, 1000))
        )
        for i, w in enumerate(ws):
            np.testing.assert_array_equal(
                bsp[i], self.c(np.array([w] * 1000), np.linspace(0, 360, 1000))
            )
