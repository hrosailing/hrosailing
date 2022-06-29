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
        self.boat_speeds1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        self.boat_speeds2 = [[10, 11, 12], [13, 14, 15], [16, 17, 18]]
        self.tbl_1 = pol.PolarDiagramTable(self.wind_speeds, self.wind_angles, self.boat_speeds1)
        self.tbl_2 = pol.PolarDiagramTable(self.wind_speeds, self.wind_angles, self.boat_speeds2)
        self.mts = pol.PolarDiagramMultiSails((self.tbl_1, self.tbl_2))

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