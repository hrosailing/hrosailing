# pylint: disable=missing-docstring

import unittest

import numpy as np

from os.path import exists

import hrosailing.polardiagram as pol
from hrosailing.polardiagram import FileReadingException


class FileReadingTest(unittest.TestCase):
    def test_read_nonexistent_file(self):
        with self.assertRaises(OSError):
            pol.from_csv("nonexistent.csv")

    def test_from_csv_read_correct_existent_file(self):
        files = [
            ("../tests/testfiles/pd-hro.csv", "hro"),
            ("../tests/testfiles/pc-hro.csv", "hro"),
            ("../tests/testfiles/pd-orc.csv", "orc"),
            ("../tests/testfiles/pd-opencpn.csv", "opencpn"),
            ("../tests/testfiles/pd-array.csv", "array"),
        ]
        for i, (file, fmt) in enumerate(files):
            with self.subTest(i=i):
                pol.from_csv(file, fmt=fmt)

    def test_from_csv_format_hro_works_correctly(self):
        files = ["../tests/testfiles/pd-hro.csv", "../tests/testfiles/pc-hro.csv"]
        for i, file in enumerate(files):
            with self.subTest(i=i):
                pd = pol.from_csv(file)
                np.testing.assert_array_equal(
                    pd.wind_speeds, np.array([2, 4, 6, 8, 10])
                )
                np.testing.assert_array_equal(
                    pd.wind_angles, np.array([10, 15, 20, 25, 30])
                )
                np.testing.assert_array_equal(
                    pd.boat_speeds.ravel(), np.zeros((25,))
                )

    @staticmethod
    def test_from_csv_format_orc_works_correctly():
        pd = pol.from_csv("../tests/testfiles/pd-orc.csv", fmt="orc")
        np.testing.assert_array_equal(
            pd.wind_speeds, np.array([6, 8, 10, 12, 14, 16, 20])
        )
        np.testing.assert_array_equal(
            pd.wind_angles, np.array([52, 60, 75, 90, 110, 120, 135, 150])
        )
        np.testing.assert_array_equal(
            pd.boat_speeds,
            np.array(
                [
                    [5.06, 5.99, 6.62, 6.89, 6.98, 7.02, 7.02],
                    [5.37, 6.29, 6.84, 7.09, 7.19, 7.23, 7.25],
                    [5.64, 6.54, 7, 7.27, 7.45, 7.55, 7.63],
                    [5.57, 6.53, 7.07, 7.34, 7.56, 7.79, 8.1],
                    [5.44, 6.57, 7.15, 7.51, 7.86, 8.15, 8.51],
                    [5.26, 6.39, 7.06, 7.45, 7.83, 8.22, 8.91],
                    [4.76, 5.86, 6.73, 7.21, 7.56, 7.96, 8.81],
                    [4.04, 5.1, 6.02, 6.77, 7.2, 7.54, 8.29],
                ]
            ),
        )

    @staticmethod
    def test_from_csv_format_opencpn_works_correctly():
        pd = pol.from_csv("../tests/testfiles/pd-opencpn.csv", fmt="opencpn")
        np.testing.assert_array_equal(pd.wind_speeds, np.arange(2, 42, 2))
        np.testing.assert_array_equal(pd.wind_angles, np.arange(5, 185, 5))
        np.testing.assert_array_equal(pd.boat_speeds, np.zeros((36, 20)))

    @staticmethod
    def test_from_csv_format_array_works_correctly():
        pd = pol.from_csv("../tests/testfiles/pd-array.csv", fmt="array")
        np.testing.assert_array_equal(
            pd.wind_speeds, np.array([0, 4, 6, 8, 10, 12, 14, 16, 20, 25, 30])
        )
        np.testing.assert_array_equal(
            pd.wind_angles,
            np.array(
                [0, 5, 10, 15, 20, 25, 32, 36, 40, 45, 52, 60, 70, 80, 90]
            ),
        )
        np.testing.assert_array_equal(
            pd.boat_speeds,
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0.3, 0.5, 0.5, 0.6, 0.8, 0.8, 0.9, 0.9, 0.9, 0.8],
                    [0, 0.6, 0.9, 1, 1.2, 1.6, 1.7, 1.7, 1.8, 1.7, 1.6],
                    [0, 0.9, 1.4, 1.5, 1.8, 2.4, 2.6, 2.6, 2.6, 2.6, 2.5],
                    [0, 1.1, 1.6, 1.7, 2.1, 2.8, 2.9, 3.0, 3.0, 3.0, 2.8],
                    [0, 1.3, 1.9, 2, 2.5, 3.3, 3.5, 3.5, 3.6, 3.5, 3.4],
                    [0, 2.1, 3.2, 3.3, 4.1, 5.5, 5.8, 5.9, 6, 5.9, 5.6],
                    [0, 2.5, 3.7, 3.7, 4.6, 6, 6.3, 6.4, 6.5, 6.5, 6.4],
                    [0, 2.8, 3.9, 3.9, 5, 6.3, 6.4, 6.6, 6.7, 6.8, 6.8],
                    [0, 3.1, 4.2, 4.5, 5.5, 6.7, 6.9, 7, 7, 7.1, 7.1],
                    [0, 3.5, 4.4, 4.9, 5.9, 7.0, 7.2, 7.3, 7.3, 7.4, 7.5],
                    [0, 3.8, 5.2, 5.7, 6.6, 7.2, 7.4, 7.5, 7.6, 7.7, 7.7],
                    [0, 4, 5.4, 6.4, 6.8, 7.3, 7.5, 7.7, 7.8, 8, 8.1],
                    [0, 4.1, 5.5, 6.5, 6.9, 7.4, 7.6, 7.8, 8.1, 8.3, 8.4],
                    [0, 4.1, 5.5, 6.6, 7.1, 7.4, 7.6, 7.8, 8.3, 8.5, 8.7],
                ]
            ),
        )

    def test_from_csv_exception_unknown_format(self):
        with self.assertRaises(FileReadingException):
            pol.from_csv("unknown_format_example.csv", fmt="unknown")

    def test_from_csv_exception_unknown_hro_subclass(self):
        with self.assertRaises(FileReadingException):
            pol.from_csv("../tests/testfiles/unknown_subclass_hro_format_example.csv")


def reading_suite():
    suite = unittest.TestSuite()
    suite.addTests(
        [
            FileReadingTest("test_read_nonexistent_file"),
            FileReadingTest("test_from_csv_read_correct_existent_file"),
            FileReadingTest("test_from_csv_format_hro_works_correctly"),
            FileReadingTest("test_from_csv_format_orc_works_correctly"),
            FileReadingTest("test_from_csv_format_opencpn_works_correctly"),
            FileReadingTest("test_from_csv_format_array_works_correctly"),
            FileReadingTest("test_from_csv_exception_unknown_format"),
            FileReadingTest("test_from_csv_exception_unknown_hro_subclass"),
        ]
    )

    return suite


class FileWritingTest(unittest.TestCase):
    def test_to_csv_pd_curve(self):
        pd_curve = pol.from_csv("../examples/csv-format-examples/curve_hro_format_example.csv")
        pd_curve.to_csv("../tests/testfiles/to_csv_pd_curve.csv")
        self.assertEqual(True, exists("../tests/testfiles/to_csv_pd_curve.csv"))
        pd_curve_2 = pol.from_csv("../tests/testfiles/to_csv_pd_curve.csv")
        self.assertEqual(pd_curve.__dict__, pd_curve_2.__dict__)


    def test_to_csv_pd_multisails(self):
        pd_ms = pol.from_csv("../examples/csv-format-examples/multisails_hro_format_example.csv")
        pd_ms.to_csv("../tests/testfiles/to_csv_pd_multisails.csv")
        self.assertEqual(True, exists("../tests/testfiles/to_csv_pd_multisails.csv"))
        pd_ms_2 = pol.from_csv("../tests/testfiles/to_csv_pd_multisails.csv")
        np.testing.assert_array_equal(pd_ms.sails, pd_ms_2.sails)
        np.testing.assert_array_equal(pd_ms.wind_speeds, pd_ms_2.wind_speeds)
        tables_1 = pd_ms.tables
        tables_2 = pd_ms_2.tables
        for i, table_list in enumerate([tables_1, tables_2]):
            with self.subTest(i=i):
                np.testing.assert_array_equal(table_list[0].wind_speeds, table_list[1].wind_speeds)
                np.testing.assert_array_equal(table_list[0].wind_angles, table_list[1].wind_angles)
                np.testing.assert_array_equal(table_list[0].boat_speeds, table_list[1].boat_speeds)

    def test_to_csv_pd_pointcloud(self):
        pd_cloud = pol.from_csv("../examples/csv-format-examples/cloud_hro_format_example.csv")
        pd_cloud.to_csv("../tests/testfiles/to_csv_pd_cloud.csv")
        self.assertEqual(True, exists("../tests/testfiles/to_csv_pd_cloud.csv"))
        pd_cloud_2 = pol.from_csv("../tests/testfiles/to_csv_pd_cloud.csv")
        self.assertEqual(pd_cloud.__dict__.keys(), pd_cloud_2.__dict__.keys())
        for i, key in enumerate(pd_cloud.__dict__.keys()):
            with self.subTest(i=i):
                np.testing.assert_array_equal(pd_cloud.__dict__[key], pd_cloud_2.__dict__[key])

    def test_to_csv_pd_table(self):
        pd_table = pol.from_csv("../examples/csv-format-examples/table_hro_format_example.csv")
        pd_table.to_csv("../tests/testfiles/to_csv_pd_table.csv")
        self.assertEqual(True, exists("../tests/testfiles/to_csv_pd_table.csv"))
        pd_table_2 = pol.from_csv("../tests/testfiles/to_csv_pd_table.csv")
        self.assertEqual(pd_table.__dict__.keys(), pd_table_2.__dict__.keys())
        for i, key in enumerate(pd_table.__dict__.keys()):
            with self.subTest(i=i):
                np.testing.assert_array_equal(pd_table.__dict__[key], pd_table_2.__dict__[key])

    def test_to_csv_array(self):
        pd_1 = pol.from_csv("../examples/csv-format-examples/array_format_example.csv", fmt="array")
        pd_1.to_csv("../tests/testfiles/to_csv_array.csv", fmt="array")
        self.assertEqual(True, exists("../tests/testfiles/to_csv_array.csv"))
        pd_2 = pol.from_csv("../tests/testfiles/to_csv_array.csv", fmt="array")
        self.assertEqual(pd_1.__dict__.keys(), pd_2.__dict__.keys())
        for i, key in enumerate(pd_1.__dict__.keys()):
            with self.subTest(i=i):
                np.testing.assert_array_equal(pd_1.__dict__[key], pd_2.__dict__[key])

    def test_to_csv_opencpn(self):
        pd_1 = pol.from_csv("../examples/csv-format-examples/opencpn_format_example.csv", fmt="opencpn")
        pd_1.to_csv("../tests/testfiles/to_csv_opencpn.csv", fmt="opencpn")
        self.assertEqual(True, exists("../tests/testfiles/to_csv_opencpn.csv"))
        pd_2 = pol.from_csv("../tests/testfiles/to_csv_opencpn.csv", fmt="opencpn")
        self.assertEqual(pd_1.__dict__.keys(), pd_2.__dict__.keys())
        for i, key in enumerate(pd_1.__dict__.keys()):
            with self.subTest(i=i):
                np.testing.assert_array_equal(pd_1.__dict__[key], pd_2.__dict__[key])

    def test_to_csv_orc(self):
        pd_1 = pol.from_csv("../examples/csv-format-examples/orc_format_example.csv", fmt="orc")
        pd_1.to_csv("../tests/testfiles/to_csv_orc.csv", fmt="orc")
        self.assertEqual(True, exists("../tests/testfiles/to_csv_orc.csv"))
        pd_2 = pol.from_csv("../tests/testfiles/to_csv_orc.csv", fmt="orc")
        self.assertEqual(pd_1.__dict__.keys(), pd_2.__dict__.keys())
        for i, key in enumerate(pd_1.__dict__.keys()):
            with self.subTest(i=i):
                np.testing.assert_array_equal(pd_1.__dict__[key], pd_2.__dict__[key])


def writing_suite():
    suite = unittest.TestSuite()
    suite.addTests([
        FileWritingTest("test_to_csv_pd_curve"),
        FileWritingTest("test_to_csv_pd_multisails"),
        FileWritingTest("test_to_csv_pd_pointcloud"),
        FileWritingTest("test_to_csv_pd_table"),
        FileWritingTest("test_to_csv_array"),
        FileWritingTest("test_to_csv_opencpn"),
        FileWritingTest("test_to_csv_orc")
    ])

    return suite
