# pylint: disable=missing-docstring

import unittest

import numpy as np

import hrosailing.polardiagram as pol
from hrosailing.polardiagram import FileReadingException


class FileReadingTest(unittest.TestCase):
    def test_read_nonexistent_file(self):
        with self.assertRaises(OSError):
            pol.from_csv("nonexistent.csv")

    def test_from_csv_read_correct_existent_file(self):
        files = [
            ("tests/testfiles/pd-hro.csv", "hro"),
            ("tests/testfiles/pc-hro.csv", "hro"),
            ("tests/testfiles/pd-orc.csv", "orc"),
            ("tests/testfiles/pd-opencpn.csv", "opencpn"),
            ("tests/testfiles/pd-array.csv", "array"),
        ]
        for i, (file, fmt) in enumerate(files):
            with self.subTest(i=i):
                pol.from_csv(file, fmt=fmt)

    def test_from_csv_format_hro_works_correctly(self):
        files = ["tests/testfiles/pd-hro.csv", "tests/testfiles/pc-hro.csv"]
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
        pd = pol.from_csv("tests/testfiles/pd-orc.csv", fmt="orc")
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
        pd = pol.from_csv("tests/testfiles/pd-opencpn.csv", fmt="opencpn")
        np.testing.assert_array_equal(pd.wind_speeds, np.arange(2, 42, 2))
        np.testing.assert_array_equal(pd.wind_angles, np.arange(5, 185, 5))
        np.testing.assert_array_equal(pd.boat_speeds, np.zeros((36, 20)))

    @staticmethod
    def test_from_csv_format_array_works_correctly():
        pd = pol.from_csv("tests/testfiles/pd-array.csv", fmt="array")
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
        ]
    )

    return suite


class FileWritingTest(unittest.TestCase):
    pass


def writing_suite():
    suite = unittest.TestSuite()
    suite.addTests([])

    return suite
