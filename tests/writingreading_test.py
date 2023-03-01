# pylint: disable=missing-docstring
import os
import unittest

import numpy as np

from os.path import exists

import hrosailing.polardiagram as pol


class FileReadingTest(unittest.TestCase):
    def setUp(self):
        # create needed files
        self.pc_hro_path = "pc-hro.csv"
        self.pd_hro_path = "pd-hro.csv"
        self.array_path = "pd-array.csv"
        self.opencpn_path = "pd-opencpn.csv"
        self.orc_path = "pd-orc.csv"
        self.unknown_subclass_hro_path = "unknown_subclass_hro_format_example.csv"

        l_pc_hro = [
            "PolarDiagramPointcloud\n",
            "TWS,TWA,BSP\n",
            "2,10,0\n",
            "2,15,0\n",
            "2,20,0\n",
            "2,25,0\n",
            "2,30,0\n",
            "4,10,0\n",
            "4,15,0\n",
            "4,20,0\n",
            "4,25,0\n",
            "4,30,0\n",
            "6,10,0\n",
            "6,15,0\n",
            "6,20,0\n",
            "6,25,0\n",
            "6,30,0\n",
            "8,10,0\n",
            "8,15,0\n",
            "8,20,0\n",
            "8,25,0\n",
            "8,30,0\n",
            "10,10,0\n",
            "10,15,0\n",
            "10,20,0\n",
            "10,25,0\n",
            "10,30,0"
        ]
        with open(self.pc_hro_path, "w", encoding="utf-8") as file:
            file.writelines(l_pc_hro)

        l_pd_hro = [
            "PolarDiagramTable\n",
            "TWS\n",
            "2,4,6,8,10\n",
            "TWA\n",
            "10,15,20,25,30\n",
            "BSP\n",
            "0,0,0,0,0\n",
            "0,0,0,0,0\n",
            "0,0,0,0,0\n",
            "0,0,0,0,0\n",
            "0,0,0,0,0"
        ]
        with open(self.pd_hro_path, "w", encoding="utf-8") as file:
            file.writelines(l_pd_hro)

        l_array = [
            "TWA\TWS	0	4	6	8	10	12	14	16	20	25	30\n",
            "0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0\n",
            "5	0.0	0.3	0.5	0.5	0.6	0.8	0.8	0.9	0.9	0.9	0.8\n",
            "10	0.0	0.6	0.9	1.0	1.2	1.6	1.7	1.7	1.8	1.7	1.6\n",
            "15	0.0	0.9	1.4	1.5	1.8	2.4	2.6	2.6	2.6	2.6	2.5\n",
            "20	0.0	1.1	1.6	1.7	2.1	2.8	2.9	3.0	3.0	3.0	2.8\n",
            "25	0.0	1.3	1.9	2.0	2.5	3.3	3.5	3.5	3.6	3.5	3.4\n",
            "32	0.0	2.1	3.2	3.3	4.1	5.5	5.8	5.9	6.0	5.9	5.6\n",
            "36	0.0	2.5	3.7	3.7	4.6	6.0	6.3	6.4	6.5	6.5	6.4\n",
            "40	0.0	2.8	3.9	3.9	5.0	6.3	6.4	6.6	6.7	6.8	6.8\n",
            "45	0.0	3.1	4.2	4.5	5.5	6.7	6.9	7.0	7.0	7.1	7.1\n",
            "52	0.0	3.5	4.4	4.9	5.9	7.0	7.2	7.3	7.3	7.4	7.5\n",
            "60	0.0	3.8	5.2	5.7	6.6	7.2	7.4	7.5	7.6	7.7	7.7\n",
            "70	0.0	4.0	5.4	6.4	6.8	7.3	7.5	7.7	7.8	8.0	8.1\n",
            "80	0.0	4.1	5.5	6.5	6.9	7.4	7.6	7.8	8.1	8.3	8.4\n",
            "90	0.0	4.1	5.5	6.6	7.1	7.4	7.6	7.8	8.3	8.5	8.7"
        ]
        with open(self.array_path, "w", encoding="utf-8") as file:
            file.writelines(l_array)

        l_opencpn = [
            "TWA\TWS,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40\n",
            "5°,,,,,,,,,,,,,,,,,,,,\n",
            "10°,,,,,,,,,,,,,,,,,,,,\n",
            "15°,,,,,,,,,,,,,,,,,,,,\n",
            "20°,,,,,,,,,,,,,,,,,,,,\n",
            "25°,,,,,,,,,,,,,,,,,,,,\n",
            "30°,,,,,,,,,,,,,,,,,,,,\n",
            "35°,,,,,,,,,,,,,,,,,,,,\n",
            "40°,,,,,,,,,,,,,,,,,,,,\n",
            "45°,,,,,,,,,,,,,,,,,,,,\n",
            "50°,,,,,,,,,,,,,,,,,,,,\n",
            "55°,,,,,,,,,,,,,,,,,,,,\n",
            "60°,,,,,,,,,,,,,,,,,,,,\n",
            "65°,,,,,,,,,,,,,,,,,,,,\n",
            "70°,,,,,,,,,,,,,,,,,,,,\n",
            "75°,,,,,,,,,,,,,,,,,,,,\n",
            "80°,,,,,,,,,,,,,,,,,,,,\n",
            "85°,,,,,,,,,,,,,,,,,,,,\n",
            "90°,,,,,,,,,,,,,,,,,,,,\n",
            "95°,,,,,,,,,,,,,,,,,,,,\n",
            "100°,,,,,,,,,,,,,,,,,,,,\n",
            "105°,,,,,,,,,,,,,,,,,,,,\n",
            "110°,,,,,,,,,,,,,,,,,,,,\n",
            "115°,,,,,,,,,,,,,,,,,,,,\n",
            "120°,,,,,,,,,,,,,,,,,,,,\n",
            "125°,,,,,,,,,,,,,,,,,,,,\n",
            "130°,,,,,,,,,,,,,,,,,,,,\n",
            "135°,,,,,,,,,,,,,,,,,,,,\n",
            "140°,,,,,,,,,,,,,,,,,,,,\n",
            "145°,,,,,,,,,,,,,,,,,,,,\n",
            "150°,,,,,,,,,,,,,,,,,,,,\n",
            "155°,,,,,,,,,,,,,,,,,,,,\n",
            "160°,,,,,,,,,,,,,,,,,,,,\n",
            "165°,,,,,,,,,,,,,,,,,,,,\n",
            "170°,,,,,,,,,,,,,,,,,,,,\n",
            "175°,,,,,,,,,,,,,,,,,,,,\n",
            "180°,,,,,,,,,,,,,,,,,,,,"
        ]
        with open(self.opencpn_path, "w", encoding="utf-8") as file:
            file.writelines(l_opencpn)

        l_orc = [
            "twa/tws;6;8;10;12;14;16;20\n",
            "0;0;0;0;0;0;0;0\n",
            "52;5.06;5.99;6.62;6.89;6.98;7.02;7.02\n",
            "60;5.37;6.29;6.84;7.09;7.19;7.23;7.25\n",
            "75;5.64;6.54;7;7.27;7.45;7.55;7.63\n",
            "90;5.57;6.53;7.07;7.34;7.56;7.79;8.1\n",
            "110;5.44;6.57;7.15;7.51;7.86;8.15;8.51\n",
            "120;5.26;6.39;7.06;7.45;7.83;8.22;8.91\n",
            "135;4.76;5.86;6.73;7.21;7.56;7.96;8.81\n",
            "150;4.04;5.1;6.02;6.77;7.2;7.54;8.29"
        ]
        with open(self.orc_path, "w", encoding="utf-8") as file:
            file.writelines(l_orc)

        l_unknown_subclass_hro = [
            "PolarDiagramUnknown\n",
            "Wind speed resolution\n",
            "6,8,10,12,14,16,20\n",
            "Wind angle resolution\n",
            "52,60,75,90,110,120,135,150\n",
            "Boat speeds\n",
            "3.74,4.48,4.96,5.27,5.47,5.66,5.81\n",
            "3.98,4.73,5.18,5.44,5.67,5.94,6.17\n",
            "4.16,4.93,5.35,5.66,5.95,6.27,6.86\n",
            "4.35,5.19,5.64,6.09,6.49,6.7,7.35\n",
            "4.39,5.22,5.68,6.19,6.79,7.48,8.76\n",
            "4.23,5.11,5.58,6.06,6.62,7.32,9.74\n",
            "3.72,4.64,5.33,5.74,6.22,6.77,8.34\n",
            "3.21,4.1,4.87,5.4,5.78,6.22,7.32"
        ]
        with open(self.unknown_subclass_hro_path, "w", encoding="utf-8") as file:
            file.writelines(l_unknown_subclass_hro)

    def tearDown(self):
        os.remove(self.pd_hro_path)
        os.remove(self.pc_hro_path)
        os.remove(self.array_path)
        os.remove(self.opencpn_path)
        os.remove(self.orc_path)
        os.remove(self.unknown_subclass_hro_path)

    def test_read_nonexistent_file(self):
        with self.assertRaises(OSError):
            pol.from_csv("nonexistent.csv")

    def test_from_csv_read_correct_existent_file(self):
        files = [
            (self.pd_hro_path, "hro"),
            (self.pc_hro_path, "hro"),
            (self.orc_path, "orc"),
            (self.opencpn_path, "opencpn"),
            (self.array_path, "array"),
        ]
        for i, (file, fmt) in enumerate(files):
            with self.subTest(i=i):
                pol.from_csv(file, fmt=fmt)

    def test_from_csv_format_hro_works_correctly(self):
        files = [self.pd_hro_path, self.pc_hro_path]
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

    def test_from_csv_format_orc_works_correctly(self):
        pd = pol.from_csv(self.orc_path, fmt="orc")
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

    def test_from_csv_format_opencpn_works_correctly(self):
        pd = pol.from_csv(self.opencpn_path, fmt="opencpn")
        np.testing.assert_array_equal(pd.wind_speeds, np.arange(2, 42, 2))
        np.testing.assert_array_equal(pd.wind_angles, np.arange(5, 185, 5))
        np.testing.assert_array_equal(pd.boat_speeds, np.zeros((36, 20)))

    def test_from_csv_format_array_works_correctly(self):
        pd = pol.from_csv(self.array_path, fmt="array")
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
            pol.from_csv(self.unknown_subclass_hro_path)


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
    curve_path = "curve_hro_format_example.csv"
    multisails_path = "multisails_hro_format_example.csv"
    cloud_path = "cloud_hro_format_example.csv"
    table_path = "table_hro_format_example.csv"
    array_path = "array_format_example.csv"
    opencpn_path = "opencpn_format_example.csv"
    orc_path = "orc_format_example.csv"

    def setUp(self):
        # create needed files
        l_curve = [
            "PolarDiagramCurve\n",
            "Function:ws_s_wa_gauss_and_square\n",
            "Radians:False\n",
            "Parameters:1.7:0.2:0.0005:0:0.0:142:477:0:204:1260\n"
        ]
        with open(self.curve_path, "w", encoding="utf-8") as file:
            file.writelines(l_curve)

        l_multisails = [
            "PolarDiagramMultiSails\n",
            "TWS\n",
            "6,8,10,12,14,16,20\n",
            "Sail 1\n",
            "TWA\n",
            "52,60,75,90,110,120,135,150\n",
            "BSP\n",
            "3.74,4.48,4.96,5.27,5.47,5.66,5.81\n",
            "3.98,4.73,5.18,5.44,5.67,5.94,6.17\n",
            "4.16,4.93,5.35,5.66,5.95,6.27,6.86\n",
            "4.35,5.19,5.64,6.09,6.49,6.7,7.35\n",
            "4.39,5.22,5.68,6.19,6.79,7.48,8.76\n",
            "4.23,5.11,5.58,6.06,6.62,7.32,9.74\n",
            "3.72,4.64,5.33,5.74,6.22,6.77,8.34\n",
            "3.21,4.1,4.87,5.4,5.78,6.22,7.32\n",
            "Sail 2\n",
            "TWA\n",
            "52,60,75,90,110,120,135,150\n",
            "BSP\n",
            "3.74,4.48,4.96,5.27,5.47,5.66,5.81\n",
            "3.98,4.73,5.18,5.44,5.67,5.94,6.17\n",
            "4.16,4.93,5.35,5.66,5.95,6.27,6.86\n",
            "4.35,5.19,5.64,6.09,6.49,6.7,7.35\n",
            "4.39,5.22,5.68,6.19,6.79,7.48,8.76\n",
            "4.23,5.11,5.58,6.06,6.62,7.32,9.74\n",
            "3.72,4.64,5.33,5.74,6.22,6.77,8.34\n",
            "3.21,4.1,4.87,5.4,5.78,6.22,7.32"
        ]
        with open(self.multisails_path, "w", encoding="utf-8") as file:
            file.writelines(l_multisails)

        l_cloud = [
            "PolarDiagramPointcloud\n",
            "TWS,TWA,BSP\n",
            "6,52,3.74\n",
            "6,60,3.98\n",
            "6,75,4.16\n",
            "6,90,4.35\n",
            "6,110,4.39\n",
            "6,120,4.23\n",
            "6,135,3.72\n",
            "6,150,3.21\n",
            "8,52,4.48\n",
            "8,60,4.73\n",
            "8,75,4.93\n",
            "8,90,5.19\n",
            "8,110,5.22\n",
            "8,120,5.11\n",
            "8,135,4.64\n",
            "8,150,4.1\n",
            "10,52,4.96\n",
            "10,60,5.18\n",
            "10,75,5.35\n",
            "10,90,5.64\n",
            "10,110,5.68\n",
            "10,120,5.58\n",
            "10,135,5.33\n",
            "10,150,4.87\n",
            "12,52,5.27\n",
            "12,60,5.44\n",
            "12,75,5.66\n",
            "12,90,6.09\n",
            "12,110,6.19\n"
            "12,120,6.06\n",
            "12,135,5.74\n",
            "12,150,5.4\n",
            "14,52,5.47\n",
            "14,60,5.67\n",
            "14,75,5.95\n",
            "14,90,6.49\n",
            "14,110,6.79\n",
            "14,120,6.62\n",
            "14,135,6.22\n",
            "14,150,5.78\n",
            "16,52,5.66\n",
            "16,60,5.94\n",
            "16,75,6.27\n",
            "16,90,6.7\n",
            "16,110,7.48\n",
            "16,120,7.32\n",
            "16,135,6.77\n",
            "16,150,6.22\n",
            "20,52,5.81\n",
            "20,60,6.17\n",
            "20,75,6.86\n",
            "20,90,7.35\n",
            "20,110,8.76\n",
            "20,120,9.74\n",
            "20,135,8.34\n",
            "20,150,7.32"
        ]
        with open(self.cloud_path, "w", encoding="utf-8") as file:
            file.writelines(l_cloud)

        l_table = [
            "PolarDiagramTable\n",
            "TWS\n",
            "6,8,10,12,14,16,20\n",
            "TWA\n",
            "52,60,75,90,110,120,135,150\n",
            "BSP\n",
            "3.74,4.48,4.96,5.27,5.47,5.66,5.81\n",
            "3.98,4.73,5.18,5.44,5.67,5.94,6.17\n",
            "4.16,4.93,5.35,5.66,5.95,6.27,6.86\n",
            "4.35,5.19,5.64,6.09,6.49,6.7,7.35\n",
            "4.39,5.22,5.68,6.19,6.79,7.48,8.76\n",
            "4.23,5.11,5.58,6.06,6.62,7.32,9.74\n",
            "3.72,4.64,5.33,5.74,6.22,6.77,8.34\n",
            "3.21,4.1,4.87,5.4,5.78,6.22,7.32"
        ]
        with open(self.table_path, "w", encoding="utf-8") as file:
            file.writelines(l_table)

        l_array = [
            "TWA\TWS 6   8   10  12  14  16  20\n",
            "52  3.74    4.48    4.96    5.27    5.47    5.66    5.81\n",
            "60  3.98    4.73    5.18    5.44    5.67    5.94    6.17\n",
            "75  4.16    4.93    5.35    5.66    5.95    6.27    6.86\n",
            "90  4.35    5.19    5.64    6.09    6.49    6.7 7.35\n",
            "110 4.39    5.22    5.68    6.19    6.79    7.48    8.76\n",
            "120 4.23    5.11    5.58    6.06    6.62    7.32    9.74\n",
            "135 3.72    4.64    5.33    5.74    6.22    6.77    8.34\n",
            "150 3.21    4.1 4.87    5.4 5.78    6.22    7.32"
        ]
        with open(self.array_path, "w", encoding="utf-8") as file:
            file.writelines(l_array)

        l_opencpn = [
            "TWA\TWS,6,8,10,12,14,16,20\n",
            "52°,3.74,4.48,4.96,5.27,5.47,5.66,5.81\n",
            "60°,3.98,4.73,5.18,5.44,5.67,5.94,6.17\n",
            "75°,4.16,4.93,5.35,5.66,5.95,6.27,6.86\n",
            "90°,4.35,5.19,5.64,6.09,6.49,6.7,7.35\n",
            "110°,4.39,5.22,5.68,6.19,6.79,7.48,8.76\n",
            "120°,4.23,5.11,5.58,6.06,6.62,7.32,9.74\n",
            "135°,3.72,4.64,5.33,5.74,6.22,6.77,8.34\n",
            "150°,3.21,4.1,4.87,5.4,5.78,6.22,7.32"
        ]
        with open(self.opencpn_path, "w", encoding="utf-8") as file:
            file.writelines(l_opencpn)

        l_orc = [
            "twa/tws;6;8;10;12;14;16;20\n",
            "0;0;0;0;0;0;0;0\n",
            "52;3.74;4.48;4.96;5.27;5.47;5.66;5.81\n",
            "60;3.98;4.73;5.18;5.44;5.67;5.94;6.17\n",
            "75;4.16;4.93;5.35;5.66;5.95;6.27;6.86\n",
            "90;4.35;5.19;5.64;6.09;6.49;6.7;7.35\n",
            "110;4.39;5.22;5.68;6.19;6.79;7.48;8.76\n",
            "120;4.23;5.11;5.58;6.06;6.62;7.32;9.74\n",
            "135;3.72;4.64;5.33;5.74;6.22;6.77;8.34\n",
            "150;3.21;4.1;4.87;5.4;5.78;6.22;7.32"
        ]
        with open(self.orc_path, "w", encoding="utf-8") as file:
            file.writelines(l_orc)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.curve_path)
        os.remove(cls.multisails_path)
        os.remove(cls.cloud_path)
        os.remove(cls.table_path)
        os.remove(cls.array_path)
        os.remove(cls.opencpn_path)
        os.remove(cls.orc_path)
        os.remove('to_csv_pd_curve.csv')
        os.remove('to_csv_pd_multisails.csv')
        os.remove('to_csv_pd_cloud.csv')
        os.remove('to_csv_pd_table.csv')
        os.remove('to_csv_array.csv')
        os.remove('to_csv_opencpn.csv')
        os.remove('to_csv_orc.csv')

    def test_to_csv_pd_curve(self):
        pd_curve = pol.from_csv(self.curve_path)
        pd_curve.to_csv("to_csv_pd_curve.csv")
        self.assertEqual(True, exists("to_csv_pd_curve.csv"))
        pd_curve_2 = pol.from_csv("to_csv_pd_curve.csv")
        self.assertEqual(pd_curve.__dict__, pd_curve_2.__dict__)

    def test_to_csv_pd_multisails(self):
        pd_ms = pol.from_csv(self.multisails_path)
        pd_ms.to_csv("to_csv_pd_multisails.csv")
        self.assertEqual(True, exists("to_csv_pd_multisails.csv"))
        pd_ms_2 = pol.from_csv("to_csv_pd_multisails.csv")
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
        pd_cloud = pol.from_csv(self.cloud_path)
        pd_cloud.to_csv("to_csv_pd_cloud.csv")
        self.assertEqual(True, exists("to_csv_pd_cloud.csv"))
        pd_cloud_2 = pol.from_csv("to_csv_pd_cloud.csv")
        self.assertEqual(pd_cloud.__dict__.keys(), pd_cloud_2.__dict__.keys())
        for i, key in enumerate(pd_cloud.__dict__.keys()):
            with self.subTest(i=i):
                np.testing.assert_array_equal(pd_cloud.__dict__[key], pd_cloud_2.__dict__[key])

    def test_to_csv_pd_table(self):
        pd_table = pol.from_csv(self.table_path)
        pd_table.to_csv("to_csv_pd_table.csv")
        self.assertEqual(True, exists("to_csv_pd_table.csv"))
        pd_table_2 = pol.from_csv("to_csv_pd_table.csv")
        self.assertEqual(pd_table.__dict__.keys(), pd_table_2.__dict__.keys())
        for i, key in enumerate(pd_table.__dict__.keys()):
            with self.subTest(i=i):
                np.testing.assert_array_equal(pd_table.__dict__[key], pd_table_2.__dict__[key])

    def test_to_csv_array(self):
        pd_1 = pol.from_csv(self.array_path, fmt="array")
        pd_1.to_csv("to_csv_array.csv", fmt="array")
        self.assertEqual(True, exists("to_csv_array.csv"))
        pd_2 = pol.from_csv("to_csv_array.csv", fmt="array")
        self.assertEqual(pd_1.__dict__.keys(), pd_2.__dict__.keys())
        for i, key in enumerate(pd_1.__dict__.keys()):
            with self.subTest(i=i):
                np.testing.assert_array_equal(pd_1.__dict__[key], pd_2.__dict__[key])

    def test_to_csv_opencpn(self):
        pd_1 = pol.from_csv(self.opencpn_path, fmt="opencpn")
        pd_1.to_csv("to_csv_opencpn.csv", fmt="opencpn")
        self.assertEqual(True, exists("to_csv_opencpn.csv"))
        pd_2 = pol.from_csv("to_csv_opencpn.csv", fmt="opencpn")
        self.assertEqual(pd_1.__dict__.keys(), pd_2.__dict__.keys())
        for i, key in enumerate(pd_1.__dict__.keys()):
            with self.subTest(i=i):
                np.testing.assert_array_equal(pd_1.__dict__[key], pd_2.__dict__[key])

    def test_to_csv_orc(self):
        pd_1 = pol.from_csv(self.orc_path, fmt="orc")
        pd_1.to_csv("to_csv_orc.csv", fmt="orc")
        self.assertEqual(True, exists("to_csv_orc.csv"))
        pd_2 = pol.from_csv("to_csv_orc.csv", fmt="orc")
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
