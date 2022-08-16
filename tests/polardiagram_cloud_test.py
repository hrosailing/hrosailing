# pylint: disable=missing-docstring
# pylint: disable=too-many-public-methods
# pylint: disable=import-outside-toplevel
import itertools
import unittest

import numpy as np

import matplotlib.pyplot as plt

import hrosailing.polardiagram as pol
from hrosailing.polardiagram._basepolardiagram import (
    PolarDiagramException,
    PolarDiagramInitializationException,
)
import _test_plot_functions as helper_functions


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

        #example for a bigger pointcloud:
        self.points_big_pc = np.array(
            [
                [6,52,3.74],
                [6,60,3.98],
                [6,75,4.16],
                [6,90,4.35],
                [6,110,4.39],
                [6,120,4.23],
                [6,135,3.72],
                [6,150,3.21],
                [8,52,4.48],
                [8,60,4.73],
                [8,75,4.93],
                [8,90,5.19],
                [8,110,5.22],
                [8,120,5.11],
                [8,135,4.64],
                [8,150,4.1],
                [10,52,4.96],
                [10,60,5.18],
                [10,75,5.35],
                [10,90,5.64],
                [10,110,5.68],
                [10,120,5.58],
                [10,135,5.33],
                [10,150,4.87],
                [12,52,5.27],
                [12,60,5.44],
                [12,75,5.66],
                [12,90,6.09],
                [12,110,6.19],
                [12,120,6.06],
                [12,135,5.74],
                [12,150,5.4],
                [14,52,5.47],
                [14,60,5.67],
                [14,75,5.95],
                [14,90,6.49],
                [14,110,6.79],
                [14,120,6.62],
                [14,135,6.22],
                [14,150,5.78],
                [16,52,5.66],
                [16,60,5.94],
                [16,75,6.27],
                [16,90,6.7],
                [16,110,7.48],
                [16,120,7.32],
                [16,135,6.77],
                [16,150,6.22],
                [20,52,5.81],
                [20,60,6.17],
                [20,75,6.86],
                [20,90,7.35],
                [20,110,8.76],
                [20,120,9.74],
                [20,135,8.34],
                [20,150,7.32]
            ]
        )

        self.big_pc = pol.PolarDiagramPointcloud(self.points_big_pc)

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

    def test_get_all_slices(self):
        ws, wa, bsp = self.pc.get_slices(None)
        wind_angles = 4 * [np.deg2rad([10, 15, 20, 25])]
        boat_speeds = [
            [1.1, 1.5, 1.7, 2.1],
            [2, 2.4, 2.6, 3],
            [3, 3.1, 3.5, 3.8],
            [4, 4.1, 4.4, 4.6]
        ]
        self.assertEqual(ws, [2, 4, 6, 8])
        for i in range(4):
            with self.subTest(i=i):
                np.testing.assert_array_equal(wa[i], wind_angles[i])
                np.testing.assert_array_equal(bsp[i], boat_speeds[i])

    def test_get_slices_interval(self):
        ws, wa, bsp = self.pc.get_slices(ws=(4, 8))
        np.testing.assert_array_equal(ws, np.linspace(4, 8, 4))
        wind_angles = 4 * [np.deg2rad([10, 15, 20, 25])]
        boat_speeds = [
            [2, 2.4, 2.6, 3],
            [3, 3.1, 3.5, 3.8],
            [3, 3.1, 3.5, 3.8],
            [4, 4.1, 4.4, 4.6]
        ]
        np.testing.assert_array_equal(ws, np.linspace(4, 8, 4))
        for i in range(4):
            with self.subTest(i=i):
                np.testing.assert_array_equal(wa[i], wind_angles[i])
                np.testing.assert_array_equal(bsp[i], boat_speeds[i])

    def test_get_slices_list(self):
        ws, wa, bsp = self.pc.get_slices(ws=[4, 8])
        self.assertEqual(ws, [4, 8])
        self.assertEqual(type(wa), list)
        self.assertEqual(type(bsp), list)
        self.assertEqual(len(wa), 2)
        self.assertEqual(len(bsp), 2)
        np.testing.assert_array_equal(wa[0], np.deg2rad([10, 15, 20, 25]))
        np.testing.assert_array_equal(wa[1], np.deg2rad([10, 15, 20, 25]))
        np.testing.assert_array_equal(bsp[0], np.array([2, 2.4, 2.6, 3]))
        np.testing.assert_array_equal(bsp[1], np.array([4, 4.1, 4.4, 4.6]))

    def test_get_slices_mixed_iterable_list_slices_in_table(self):
        ws, wa, bsp = self.pc.get_slices(ws=[(4, 8), 2])
        self.assertEqual(ws, [6, 2])
        zipped_tuples = [[list(item) for item in zip(wa[0], bsp[0])],
                         [list(item) for item in zip(wa[1], bsp[1])]]
        wind_speeds = [[4, 6, 8], [2]]
        for i in range(2):
            for wa_bsp_tuple in zipped_tuples[i]:
                boolean = False
                for point in self.pc.points:
                    if point[0] in wind_speeds[i] and np.deg2rad(point[1]) == wa_bsp_tuple[0]\
                            and point[2] == wa_bsp_tuple[1]:
                        boolean = True
                if not boolean:
                    self.fail()

    def test_get_slices_mixed_iterable_list_table_values_in_slices(self):
        ws, wa, bsp = self.pc.get_slices(ws=[(4, 8), 2])
        self.assertEqual(ws, [6, 2])
        zipped_tuples = [[list(item) for item in zip(wa[0], bsp[0])],
                         [list(item) for item in zip(wa[1], bsp[1])]]
        wind_speeds = [[4, 6, 8], [2]]
        for i in range(2):
            for point in self.pc.points:
                boolean = False
                if point[0] in wind_speeds[i]:
                    for wa_bsp_tuple in zipped_tuples[i]:
                        if np.deg2rad(point[1]) == wa_bsp_tuple[0] and point[2] == wa_bsp_tuple[1]:
                            boolean = True
                    if not boolean:
                        self.fail()

    def test_get_slices_mixed_iterable_tuple_slices_in_table(self):
        ws, wa, bsp = self.pc.get_slices(ws=((4, 8), 2))
        self.assertEqual(ws, [6, 2])
        zipped_tuples = [[list(item) for item in zip(wa[0], bsp[0])],
                         [list(item) for item in zip(wa[1], bsp[1])]]
        wind_speeds = [[4, 6, 8], [2]]
        for i in range(2):
            for wa_bsp_tuple in zipped_tuples[i]:
                boolean = False
                for point in self.pc.points:
                    if point[0] in wind_speeds[i] and np.deg2rad(point[1]) == wa_bsp_tuple[0] \
                            and point[2] == wa_bsp_tuple[1]:
                        boolean = True
                if not boolean:
                    self.fail()

    def test_get_slices_mixed_iterable_tuple_table_values_in_slices(self):
        ws, wa, bsp = self.pc.get_slices(ws=((4, 8), 2))
        self.assertEqual(ws, [6, 2])
        zipped_tuples = [[list(item) for item in zip(wa[0], bsp[0])],
                         [list(item) for item in zip(wa[1], bsp[1])]]
        wind_speeds = [[4, 6, 8], [2]]
        for i in range(2):
            for point in self.pc.points:
                boolean = False
                if point[0] in wind_speeds[i]:
                    for wa_bsp_tuple in zipped_tuples[i]:
                        if np.deg2rad(point[1]) == wa_bsp_tuple[0] and point[2] == wa_bsp_tuple[1]:
                            boolean = True
                    if not boolean:
                        self.fail()

    def test_get_slices_mixed_iterable_set_slices_in_table(self):
        ws, wa, bsp = self.pc.get_slices(ws={(4, 8), 2})
        self.assertIn(ws, [[6, 2], [2, 6]])
        zipped_tuples = [[list(item) for item in zip(wa[0], bsp[0])],
                         [list(item) for item in zip(wa[1], bsp[1])]]
        wind_speeds = [[4, 6, 8], [2]]
        for i in range(2):
            for wa_bsp_tuple in zipped_tuples[i]:
                boolean = False
                for point in self.pc.points:
                    if ws[0] == 6:
                        if point[0] in wind_speeds[i] and np.deg2rad(point[1]) == wa_bsp_tuple[0] \
                                and point[2] == wa_bsp_tuple[1]:
                            boolean = True
                    else:
                        if point[0] in wind_speeds[i-1] and np.deg2rad(point[1]) == wa_bsp_tuple[0] \
                                and point[2] == wa_bsp_tuple[1]:
                            boolean = True
                if not boolean:
                    self.fail()

    def test_get_slices_mixed_iterable_set_table_values_in_slices(self):
        ws, wa, bsp = self.pc.get_slices(ws={(4, 8), 2})
        self.assertIn(ws, [[6, 2], [2, 6]])
        zipped_tuples = [[list(item) for item in zip(wa[0], bsp[0])],
                         [list(item) for item in zip(wa[1], bsp[1])]]
        wind_speeds = [[4, 6, 8], [2]]
        for i in range(2):
            for point in self.pc.points:
                boolean = False
                if point[0] in wind_speeds[i]:
                    if ws[0] == 6:
                        for wa_bsp_tuple in zipped_tuples[i]:
                            if np.deg2rad(point[1]) == wa_bsp_tuple[0] and point[2] == wa_bsp_tuple[1]:
                                boolean = True
                    else:
                        for wa_bsp_tuple in zipped_tuples[i-1]:
                            if np.deg2rad(point[1]) == wa_bsp_tuple[0] and point[2] == wa_bsp_tuple[1]:
                                boolean = True
                    if not boolean:
                        self.fail()

    def test_get_slices_exception_empty(self):
        with self.assertRaises(PolarDiagramException):
            self.pc.get_slices([])

    def test_get_slices_exception_no_slices(self):
        with self.assertRaises(PolarDiagramException):
            self.pc.get_slices([0, 2])

    def test_get_slices_range(self):
        ws, wa, bsp = self.pc.get_slices(ws=(3, 9))
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

    def test_get_slices_n_steps(self):
        ws, wa, bsp = self.pc.get_slices(ws=(4, 8), n_steps=3)
        self.assertEqual(ws, [4, 6, 8])
        boat_speeds = [
            [2, 2.4, 2.6, 3],
            [3, 3.1, 3.5, 3.8],
            [4, 4.1, 4.4, 4.6]
        ]
        for i in range(3):
            with self.subTest(i=i):
                np.testing.assert_array_equal(wa[i], np.deg2rad([10, 15, 20, 25]))
                np.testing.assert_array_equal(bsp[i], boat_speeds[i])

    def test_get_slices_range_single_ws_slice_in_table(self):
        ws, wa, bsp = self.pc.get_slices(ws=4, range_=2)
        self.assertEqual(ws, [4])
        zipped_tuples = [list(item) for item in zip(wa[0], bsp[0])]
        wind_speeds = [2, 4, 6]
        for wa_bsp_tuple in zipped_tuples:
            boolean = False
            for point in self.pc.points:
                if point[0] in wind_speeds and np.deg2rad(point[1]) == wa_bsp_tuple[0] \
                        and point[2] == wa_bsp_tuple[1]:
                    boolean = True
            if not boolean:
                self.fail()

    def test_get_slices_range_single_ws_table_values_in_slice(self):
        ws, wa, bsp = self.pc.get_slices(ws=4, range_=2)
        self.assertEqual(ws, [4])
        zipped_tuples = [list(item) for item in zip(wa[0], bsp[0])]
        wind_speeds = [2, 4, 6]
        for i in range(2):
            for point in self.pc.points:
                boolean = False
                if point[0] in wind_speeds:
                    for wa_bsp_tuple in zipped_tuples:
                        if np.deg2rad(point[1]) == wa_bsp_tuple[0] and point[2] == wa_bsp_tuple[1]:
                            boolean = True
                    if not boolean:
                        self.fail()

    def test_get_slices_range_mixed_list_slices_in_table(self):
        pd = self.big_pc
        ws, wa, bsp = pd.get_slices(ws=[(14, 20), 8], range_=2)
        np.testing.assert_array_equal(ws, [17, 8])
        zipped_tuples = [[list(item) for item in zip(wa[0], bsp[0])],
                         [list(item) for item in zip(wa[1], bsp[1])]]
        wind_speeds = [[14, 16, 20], [6, 8, 10]]
        for i in range(2):
            for wa_bsp_tuple in zipped_tuples[i]:
                boolean = False
                for point in pd.points:
                    if point[0] in wind_speeds[i] and np.deg2rad(point[1]) == wa_bsp_tuple[0] \
                            and point[2] == wa_bsp_tuple[1]:
                        boolean = True
                if not boolean:
                    self.fail()

    def test_get_slices_range_mixed_list_table_values_in_slices(self):
        pd = self.big_pc
        ws, wa, bsp = pd.get_slices(ws=[(14, 20), 8], range_=2)
        np.testing.assert_array_equal(ws, [17, 8])
        zipped_tuples = [[list(item) for item in zip(wa[0], bsp[0])],
                         [list(item) for item in zip(wa[1], bsp[1])]]
        wind_speeds = [[14, 16, 20], [6, 8, 10]]
        for i in range(2):
            for point in pd.points:
                boolean = False
                if point[0] in wind_speeds[i]:
                    for wa_bsp_tuple in zipped_tuples[i]:
                        if np.deg2rad(point[1]) == wa_bsp_tuple[0] and point[2] == wa_bsp_tuple[1]:
                            boolean = True
                    if not boolean:
                        self.fail()

    def test_get_slices_range_mixed_tuple_slices_in_table(self):
        pd = self.big_pc
        ws, wa, bsp = pd.get_slices(ws=((14, 20), 8), range_=2)
        np.testing.assert_array_equal(ws, [17, 8])
        zipped_tuples = [[list(item) for item in zip(wa[0], bsp[0])],
                         [list(item) for item in zip(wa[1], bsp[1])]]
        wind_speeds = [[14, 16, 20], [6, 8, 10]]
        for i in range(2):
            for wa_bsp_tuple in zipped_tuples[i]:
                boolean = False
                for point in pd.points:
                    if point[0] in wind_speeds[i] and np.deg2rad(point[1]) == wa_bsp_tuple[0] \
                            and point[2] == wa_bsp_tuple[1]:
                        boolean = True
                if not boolean:
                    self.fail()

    def test_get_slices_range_mixed_tuple_table_values_in_slices(self):
        pd = self.big_pc
        ws, wa, bsp = pd.get_slices(ws=((14, 20), 8), range_=2)
        np.testing.assert_array_equal(ws, [17, 8])
        zipped_tuples = [[list(item) for item in zip(wa[0], bsp[0])],
                         [list(item) for item in zip(wa[1], bsp[1])]]
        wind_speeds = [[14, 16, 20], [6, 8, 10]]
        for i in range(2):
            for point in pd.points:
                boolean = False
                if point[0] in wind_speeds[i]:
                    for wa_bsp_tuple in zipped_tuples[i]:
                        if np.deg2rad(point[1]) == wa_bsp_tuple[0] and point[2] == wa_bsp_tuple[1]:
                            boolean = True
                    if not boolean:
                        self.fail()

    def test_get_slices_range_mixed_set_slices_in_table(self):
        pd = self.big_pc
        ws, wa, bsp = pd.get_slices(ws={(14, 20), 8}, range_=2)
        self.assertIn(ws, [[17, 8], [8, 17]])
        zipped_tuples = [[list(item) for item in zip(wa[0], bsp[0])],
                         [list(item) for item in zip(wa[1], bsp[1])]]
        wind_speeds = [[14, 16, 20], [6, 8, 10]]
        for i in range(2):
            for wa_bsp_tuple in zipped_tuples[i]:
                boolean = False
                for point in pd.points:
                    if ws[0] == 17:
                        if point[0] in wind_speeds[i] and np.deg2rad(point[1]) == wa_bsp_tuple[0] \
                                and point[2] == wa_bsp_tuple[1]:
                            boolean = True
                    else:
                        if point[0] in wind_speeds[i - 1] and np.deg2rad(point[1]) == wa_bsp_tuple[0] \
                                and point[2] == wa_bsp_tuple[1]:
                            boolean = True
                if not boolean:
                    self.fail()

    def test_get_slices_range_mixed_set_table_values_in_slices(self):
        pd = self.big_pc
        ws, wa, bsp = pd.get_slices(ws={(14, 20), 8}, range_=2)
        self.assertIn(ws, [[17, 8], [8, 17]])
        zipped_tuples = [[list(item) for item in zip(wa[0], bsp[0])],
                         [list(item) for item in zip(wa[1], bsp[1])]]
        wind_speeds = [[14, 16, 20], [6, 8, 10]]
        for i in range(2):
            for point in pd.points:
                boolean = False
                if point[0] in wind_speeds[i]:
                    if ws[0] == 17:
                        for wa_bsp_tuple in zipped_tuples[i]:
                            if np.deg2rad(point[1]) == wa_bsp_tuple[0] and point[2] == wa_bsp_tuple[1]:
                                boolean = True
                    else:
                        for wa_bsp_tuple in zipped_tuples[i-1]:
                            if np.deg2rad(point[1]) == wa_bsp_tuple[0] and point[2] == wa_bsp_tuple[1]:
                                boolean = True
                    if not boolean:
                        self.fail()

    def test_get_slices_exception_nonpositive_range_(self):
        with self.subTest(i=0):
            with self.assertRaises(PolarDiagramException):
                self.pc.get_slices(ws=4, range_=0)
        with self.subTest(i=1):
            with self.assertRaises(PolarDiagramException):
                self.pc.get_slices(ws=4, range_=-20)

    def test_plot_polar(self):
        plt.close()
        self.pc.plot_polar()
        ws, wa, bsp = self.pc.get_slices(None)
        for i in range(4):
            with self.subTest(i=i):
                helper_functions.cloud_plot_polar_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp)

    def test_plot_polar_single_element_ws(self):
        plt.close()
        self.pc.plot_polar(ws=4)
        ws, wa, bsp = self.pc.get_slices(ws=4)
        helper_functions.cloud_plot_polar_comparing_x_plot_wa_y_plot_bsp_single_ws(wa, bsp)

    def test_plot_polar_interval_ws(self):
        plt.close()
        self.pc.plot_polar(ws=(4, 8))
        ws, wa, bsp = self.pc.get_slices(ws=(4, 8))
        for i in range(4):
            with self.subTest(i=i):
                helper_functions.cloud_plot_polar_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp)

    def test_plot_polar_mixed_list_ws(self):
        plt.close()
        self.pc.plot_polar(ws=[(4, 8), 2])
        ws, wa, bsp = self.pc.get_slices(ws=[(4, 8), 2])
        for i in range(2):
            with self.subTest(i=i):
                helper_functions.cloud_plot_polar_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp)

    def test_plot_polar_mixed_tuple_ws(self):
        plt.close()
        self.pc.plot_polar(ws=((4, 8), 2))
        ws, wa, bsp = self.pc.get_slices(ws=((4, 8), 2))
        for i in range(2):
            with self.subTest(i=i):
                helper_functions.cloud_plot_polar_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp)

    def test_plot_polar_mixed_set_ws(self):
        plt.close()
        self.pc.plot_polar(ws={(4, 8), 2})
        ws, wa, bsp = self.pc.get_slices(ws={(4, 8), 2})
        for i in range(2):
            with self.subTest(i=i):
                helper_functions.cloud_plot_polar_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp)

    def test_plot_polar_n_steps(self):
        plt.close()
        self.pc.plot_polar(ws=(4, 8), n_steps=3)
        # test for ws still missing
        ws, wa, bsp = self.pc.get_slices(ws=(4, 8), n_steps=3)
        for i in range(3):
            with self.subTest(i=i):
                helper_functions.cloud_plot_polar_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp)

    def test_plot_polar_range_single_ws(self):
        plt.close()
        self.pc.plot_polar(ws=4, range_=2)
        ws, wa, bsp = self.pc.get_slices(ws=4, range_=2)
        helper_functions.cloud_plot_polar_comparing_x_plot_wa_y_plot_bsp_single_ws(wa, bsp)

    def test_plot_polar_range_mixed_list(self):
        plt.close()
        pd = self.big_pc
        pd.plot_polar(ws=[(14, 20), 8], range_=2)
        ws, wa, bsp = pd.get_slices(ws=[(14, 20), 8], range_=2)
        for i in range(2):
            with self.subTest(i=i):
                helper_functions.cloud_plot_polar_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp)

    def test_plot_polar_range_mixed_tuple(self):
        plt.close()
        pd = self.big_pc
        pd.plot_polar(ws=((14, 20), 8), range_=2)
        ws, wa, bsp = pd.get_slices(ws=((14, 20), 8), range_=2)
        for i in range(2):
            with self.subTest(i=i):
                helper_functions.cloud_plot_polar_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp)

    def test_plot_polar_range_mixed_set(self):
        plt.close()
        pd = self.big_pc
        pd.plot_polar(ws={(14, 20), 8}, range_=2)
        ws, wa, bsp = pd.get_slices(ws={(14, 20), 8}, range_=2)
        for i in range(2):
            with self.subTest(i=i):
                helper_functions.cloud_plot_polar_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp)

    def test_plot_polar_axes_keywords(self):
        # test not implemented yet
        pass

    def test_plot_polar_single_color(self):
        plt.close()
        self.pc.plot_polar(colors="purple")
        for i in range(4):
            with self.subTest(i=i):
                self.assertEqual(plt.gca().lines[i].get_color(), "purple")

    def test_plot_polar_two_colors_passed(self):
        plt.close()
        self.pc.plot_polar(ws=[4, 6, 8], colors=["red", "blue"])
        helper_functions.comparing_colors_two_colors_passed()

    def test_plot_polar_more_than_two_colors_passed(self):
        plt.close()
        self.pc.plot_polar(ws=[2, 4, 6, 8], colors=["red", "yellow", "orange"])
        helper_functions.comparing_colors_more_than_two_colors_passed()

    def test_plot_polar_ws_color_pairs_passed(self):
        self.pc.plot_polar(ws=[4, 6, 8], colors=((4, "purple"), (6, "blue"), (8, "red")))
        helper_functions.comparing_colors_ws_color_pairs_passed()

    def test_plot_polar_ws_color_pairs_unsorted_passed(self):
        plt.close()
        self.pc.plot_polar(ws=[4, 6, 8], colors=((4, "purple"), (8, "red"), (6, "blue")))
        helper_functions.comparing_colors_ws_color_pairs_passed()

    def test_plot_polar_show_legend(self):
        plt.close()
        self.pc.plot_polar(ws=[2, 4, 6], colors=["red", "purple", "blue"], show_legend=True)
        helper_functions.test_cloud_table_comparing_show_legend(self, plt.gca().get_legend())

    def test_plot_polar_legend_kw(self):
        plt.close()
        self.pc.plot_polar(ws=[2, 4, 6], colors=["red", "purple", "blue"], show_legend=True,
                           legend_kw={'labels': ["ws 2", "ws 4", "ws 6"], 'loc': 'upper left'})
        helper_functions.test_cloud_table_comparing_legend_keywords(self, plt.gca().get_legend())

    def test_plot_polar_show_colorbar(self):
        plt.close()
        self.pc.plot_polar(ws=[2, 4, 6], colors=("red", "blue"), show_legend=True)
        helper_functions.test_comparing_show_colorbar(self)

    def test_plot_polar_plot_kw(self):
        plt.close()
        self.pc.plot_polar(ls=":", lw=1.5, marker="o")
        for i in range(4):
            with self.subTest(i=i):
                helper_functions.test_comparing_plot_kw(self, i)

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
        plt.close()
        self.pc.plot_flat()
        ws, wa, bsp = self.pc.get_slices(None)
        for i in range(len(self.pc.wind_speeds)):
            with self.subTest(i=i):
                helper_functions.cloud_plot_flat_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp)

    def test_plot_flat_single_element_ws(self):
        plt.close()
        self.pc.plot_flat(ws=4)
        ws, wa, bsp = self.pc.get_slices(ws=4)
        helper_functions.cloud_plot_flat_comparing_x_plot_wa_y_plot_bsp_single_ws(wa, bsp)

    def test_plot_flat_interval_ws(self):
        plt.close()
        self.pc.plot_flat(ws=(4, 8))
        ws, wa, bsp = self.pc.get_slices(ws=(4, 8))
        for i in range(4):
            with self.subTest(i=i):
                helper_functions.cloud_plot_flat_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp)

    def test_plot_flat_mixed_list_ws(self):
        plt.close()
        self.pc.plot_flat(ws=[(4, 8), 2])
        ws, wa, bsp = self.pc.get_slices(ws=[(4, 8), 2])
        for i in range(2):
            with self.subTest(i=i):
                helper_functions.cloud_plot_flat_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp)

    def test_plot_flat_mixed_tuple_ws(self):
        plt.close()
        self.pc.plot_flat(ws=((4, 8), 2))
        ws, wa, bsp = self.pc.get_slices(ws=((4, 8), 2))
        for i in range(2):
            with self.subTest(i=i):
                helper_functions.cloud_plot_flat_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp)

    def test_plot_flat_mixed_set_ws(self):
        plt.close()
        self.pc.plot_flat(ws={(4, 8), 2})
        ws, wa, bsp = self.pc.get_slices(ws={(4, 8), 2})
        for i in range(2):
            with self.subTest(i=i):
                helper_functions.cloud_plot_flat_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp)

    def test_plot_flat_n_steps(self):
        plt.close()
        self.pc.plot_flat(ws=(4, 8), n_steps=3)
        # test for ws still missing
        ws, wa, bsp = self.pc.get_slices(ws=(4, 8), n_steps=3)
        for i in range(3):
            with self.subTest(i=i):
                helper_functions.cloud_plot_flat_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp)

    def test_plot_flat_range_single_ws(self):
        plt.close()
        self.pc.plot_flat(ws=4, range_=2)
        ws, wa, bsp = self.pc.get_slices(ws=4, range_=2)
        helper_functions.cloud_plot_flat_comparing_x_plot_wa_y_plot_bsp_single_ws(wa, bsp)

    def test_plot_flat_range_mixed_list(self):
        plt.close()
        pd = self.big_pc
        pd.plot_flat(ws=[(14, 20), 8], range_=2)
        ws, wa, bsp = pd.get_slices(ws=[(14, 20), 8], range_=2)
        for i in range(2):
            with self.subTest(i=i):
                helper_functions.cloud_plot_flat_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp)

    def test_plot_flat_range_mixed_tuple(self):
        plt.close()
        pd = self.big_pc
        pd.plot_flat(ws=((14, 20), 8), range_=2)
        ws, wa, bsp = pd.get_slices(ws=((14, 20), 8), range_=2)
        for i in range(2):
            with self.subTest(i=i):
                helper_functions.cloud_plot_flat_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp)

    def test_plot_flat_range_mixed_set(self):
        plt.close()
        pd = self.big_pc
        pd.plot_flat(ws={(14, 20), 8}, range_=2)
        ws, wa, bsp = pd.get_slices(ws={(14, 20), 8}, range_=2)
        for i in range(2):
            with self.subTest(i=i):
                helper_functions.cloud_plot_flat_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp)

    def test_plot_flat_axes_keywords(self):
        # test not implemented yet
        pass

    def test_plot_flat_single_color(self):
        plt.close()
        self.pc.plot_flat(colors="purple")
        for i in range(4):
            with self.subTest(i=i):
                self.assertEqual(plt.gca().lines[i].get_color(), "purple")

    def test_plot_flat_two_colors_passed(self):
        plt.close()
        self.pc.plot_flat(ws=[4, 6, 8], colors=["red", "blue"])
        helper_functions.comparing_colors_two_colors_passed()

    def test_plot_flat_more_than_two_colors_passed(self):
        plt.close()
        self.pc.plot_flat(ws=[2, 4, 6, 8], colors=["red", "yellow", "orange"])
        helper_functions.comparing_colors_more_than_two_colors_passed()

    def test_plot_flat_ws_color_pairs_passed(self):
        plt.close()
        self.pc.plot_flat(ws=[4, 6, 8], colors=((4, "purple"), (6, "blue"), (8, "red")))
        helper_functions.comparing_colors_ws_color_pairs_passed()

    def test_plot_flat_ws_color_pairs_unsorted_passed(self):
        plt.close()
        self.pc.plot_flat(ws=[4, 6, 8], colors=((4, "purple"), (8, "red"), (6, "blue")))
        helper_functions.comparing_colors_ws_color_pairs_passed()

    def test_plot_flat_show_legend(self):
        plt.close()
        self.pc.plot_flat(ws=[2, 4, 6], colors=["red", "purple", "blue"], show_legend=True)
        helper_functions.test_cloud_table_comparing_show_legend(self, plt.gca().get_legend())

    def test_plot_flat_legend_kw(self):
        plt.close()
        self.pc.plot_flat(ws=[2, 4, 6], colors=["red", "purple", "blue"], show_legend=True,
                          legend_kw={'labels': ["ws 2", "ws 4", "ws 6"], 'loc': 'upper left'})
        helper_functions.test_cloud_table_comparing_legend_keywords(self, plt.gca().get_legend())

    def test_plot_flat_show_colorbar(self):
        plt.close()
        self.pc.plot_flat(ws=[2, 4, 6], colors=("red", "blue"), show_legend=True)
        helper_functions.test_comparing_show_colorbar(self)

    def test_plot_flat_plot_kw(self):
        plt.close()
        self.pc.plot_flat(ls=":", lw=1.5, marker="o")
        for i in range(4):
            with self.subTest(i=i):
                helper_functions.test_comparing_plot_kw(self, i)

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
        # not finished yet
        plt.close()
        self.pc.plot_3d()
        print(plt.gca().collections[0]._offsets3d)

    def test_plot_3d_colors(self):
        # not finished yet
        plt.close()
        self.pc.plot_3d(colors=('blue', 'red'))

    def test_plot_3d_exception_empty_cloud(self):
        pd_empty = pol.PolarDiagramPointcloud(np.empty((0, 3)))
        with self.assertRaises(PolarDiagramException):
            pd_empty.plot_3d()

    def test_plot_color_gradient(self):
        # test not finished yet
        plt.close()
        ax = plt.axes()
        self.pc.plot_color_gradient(ax=ax, show_legend=True)
        ws_wa_list = [list(item) for item in np.array(ax.collections[0]._offsets)]
        all_combinations_ws_wa = [list(item) for item in itertools.product(self.pc.wind_speeds, self.pc.wind_angles)]
        self.assertEqual(len(ws_wa_list), len(all_combinations_ws_wa))
        self.assertCountEqual(ws_wa_list, all_combinations_ws_wa)

    def test_plot_color_gradient_exception_empty_cloud(self):
        pd_empty = pol.PolarDiagramPointcloud(np.empty((0, 3)))
        with self.assertRaises(PolarDiagramException):
            pd_empty.plot_color_gradient()

    def test_plot_convex_hull(self):
        # test not finished yet
        plt.close()
        self.pc.plot_convex_hull()

    def test_plot_convex_hull_single_element_ws(self):
        plt.close()
        self.pc.plot_convex_hull(ws=2)
        x_data = plt.gca().lines[0].get_xdata()
        y_data = plt.gca().lines[0].get_ydata()
        np.testing.assert_array_equal(x_data, np.deg2rad([10, 15, 25, 10]))
        np.testing.assert_array_equal(y_data, [1.1, 1.5, 2.1, 1.1])

    def test_plot_convex_hull_interval_ws(self):
        # test not finished yet
        plt.close()
        self.pc.plot_convex_hull(ws=(2, 6))

    def test_plot_convex_hull_iterable_list_ws(self):
        # test not finished yet
        plt.close()
        self.pc.plot_convex_hull(ws=[2, 4, 6])

    def test_plot_convex_hull_iterable_tuple_ws(self):
        # test not finished yet
        plt.close()
        self.pc.plot_convex_hull(ws=(2, 4, 6))

    def test_plot_convex_hull_iterable_set_ws(self):
        # test not finished yet
        plt.close()
        self.pc.plot_convex_hull(ws={2, 4, 6})

    def test_plot_convex_hull_axes_keywords(self):
        # test not implemented yet
        pass

    def test_plot_convex_hull_single_color(self):
        plt.close()
        self.pc.plot_convex_hull(colors="purple")
        for i in range(4):
            with self.subTest(i=i):
                self.assertEqual(plt.gca().lines[i].get_color(), "purple")

    def test_plot_convex_hull_two_colors_passed(self):
        plt.close()
        self.pc.plot_convex_hull(ws=[4, 6, 8], colors=["red", "blue"])
        helper_functions.comparing_colors_two_colors_passed()

    def test_plot_convex_hull_more_than_two_colors_passed(self):
        plt.close()
        self.pc.plot_convex_hull(ws=[2, 4, 6, 8], colors=["red", "yellow", "orange"])
        helper_functions.comparing_colors_more_than_two_colors_passed()

    def test_plot_convex_hull_ws_color_pairs_passed(self):
        plt.close()
        self.pc.plot_convex_hull(ws=[4, 6, 8], colors=((4, "purple"), (6, "blue"), (8, "red")))
        helper_functions.comparing_colors_ws_color_pairs_passed()

    def test_plot_convex_hull_ws_color_pairs_unsorted_passed(self):
        plt.close()
        self.pc.plot_convex_hull(ws=[4, 6, 8], colors=((4, "purple"), (8, "red"), (6, "blue")))
        helper_functions.comparing_colors_ws_color_pairs_passed()

    def test_plot_convex_hull_show_legend(self):
        plt.close()
        self.pc.plot_convex_hull(ws=[2, 4, 6], colors=["red", "purple", "blue"], show_legend=True)
        helper_functions.test_cloud_table_comparing_show_legend(self, plt.gca().get_legend())

    def test_plot_convex_hull_legend_kw(self):
        plt.close()
        self.pc.plot_convex_hull(ws=[2, 4, 6], colors=["red", "purple", "blue"], show_legend=True,
                                 legend_kw={'labels': ["ws 2", "ws 4", "ws 6"], 'loc': 'upper left'})
        helper_functions.test_cloud_table_comparing_legend_keywords(self, plt.gca().get_legend())

    def test_plot_convex_hull_show_colorbar(self):
        plt.close()
        self.pc.plot_convex_hull(ws=[2, 4, 6], colors=("red", "blue"), show_legend=True)
        helper_functions.test_comparing_show_colorbar(self)

    def test_plot_convex_hull_plot_kw(self):
        plt.close()
        self.pc.plot_convex_hull(ls=":", lw=1.5, marker="o")
        for i in range(4):
            with self.subTest(i=i):
                helper_functions.test_comparing_plot_kw(self, i)

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
