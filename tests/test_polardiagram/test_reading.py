import unittest
import os
import csv

import numpy as np

from hrosailing.polardiagram._reading import from_csv, _read_intern_format, \
    _read_extern_format, _read_from_array, _read_orc_format, _read_wind_speeds, \
    _read_wind_angles_and_boat_speeds, _read_opencpn_format
from hrosailing.polardiagram import PolarDiagram
from hrosailing.core.exceptions import FileReadingException

# Method `_read_extern_format` don't has a designated test case
# and is tested in multiple test cases


class TestReadOpenCPNFormat(unittest.TestCase):
    def setUp(self) -> None:
        self.path = "example.csv"
        lines = [
            "TWA\\TWS,6,8,10,12,14,16,20\n",
            "52°,3.74,4.48,4.96,5.27,5.47,5.66,5.81\n",
            "60°,3.98,4.73,5.18,5.44,5.67,5.94,6.17\n",
            "75°,4.16,4.93,5.35,5.66,5.95,6.27,6.86\n",
            "90°,4.35,5.19,5.64,6.09,6.49,6.7,7.35\n",
            "110°,4.39,5.22,5.68,6.19,6.79,7.48,8.76\n",
            "120°,4.23,5.11,5.58,6.06,6.62,7.32,9.74\n",
            "135°,3.72,4.64,5.33,5.74,6.22,6.77,8.34\n",
            "150°,3.21,4.1,4.87,5.4,5.78,6.22,7.32\n",
        ]
        with open(self.path, "w", encoding="utf-8") as file:
            file.writelines(lines)

    def test_call(self):
        # Input/Output
        with open(self.path, "r", encoding="utf-8") as file:
            ws_res, wa_res, bsps = _read_opencpn_format(file)
        self.assertEqual(ws_res, [6, 8, 10, 12, 14, 16, 20])
        self.assertEqual(wa_res, [52, 60, 75, 90, 110, 120, 135, 150])
        self.assertEqual(
            bsps,
            [
                [3.74,4.48,4.96,5.27,5.47,5.66,5.81],
                [3.98,4.73,5.18,5.44,5.67,5.94,6.17],
                [4.16,4.93,5.35,5.66,5.95,6.27,6.86],
                [4.35,5.19,5.64,6.09,6.49,6.7,7.35],
                [4.39,5.22,5.68,6.19,6.79,7.48,8.76],
                [4.23,5.11,5.58,6.06,6.62,7.32,9.74],
                [3.72,4.64,5.33,5.74,6.22,6.77,8.34],
                [3.21,4.1,4.87,5.4,5.78,6.22,7.32],
            ]
        )

    def test_read_extern_format(self):
        # Execution Test
        with open(self.path, "r", encoding="utf-8") as file:
            _read_extern_format(file, "opencpn")

    def tearDown(self) -> None:
        if os.path.isfile(self.path):
            os.remove(self.path)


class TestReadWindAnglesAndBoatSpeeds(unittest.TestCase):
    def setUp(self) -> None:
        self.path = "example.csv"
        lines = [
            "52°,3.74,4.48,4.96,5.27,5.47,5.66,5.81\n",
            "60°,3.98,4.73,5.18,5.44,5.67,5.94,6.17\n",
            "75°,4.16,4.93,5.35,5.66,5.95,6.27,6.86\n",
            "90°,4.35,5.19,5.64,6.09,6.49,6.7,7.35\n",
            "110°,4.39,5.22,5.68,6.19,6.79,7.48,8.76\n",
            "120°,4.23,5.11,5.58,6.06,6.62,7.32,9.74\n",
            "135°,3.72,4.64,5.33,5.74,6.22,6.77,8.34\n",
            "150°,3.21,4.1,4.87,5.4,5.78,6.22,7.32\n",
        ]
        with open(self.path, "w", encoding="utf-8") as file:
            file.writelines(lines)

    def test_call(self):
        # Input/Output
        with open(self.path, "r", encoding="utf-8") as file:
            csv_reader = csv.reader(file, delimiter=",")
            wa_res, bsps = _read_wind_angles_and_boat_speeds(csv_reader)
        self.assertEqual(wa_res, [52, 60, 75, 90, 110, 120, 135, 150])
        self.assertEqual(
            bsps,
            [
                [3.74,4.48,4.96,5.27,5.47,5.66,5.81],
                [3.98,4.73,5.18,5.44,5.67,5.94,6.17],
                [4.16,4.93,5.35,5.66,5.95,6.27,6.86],
                [4.35,5.19,5.64,6.09,6.49,6.7,7.35],
                [4.39,5.22,5.68,6.19,6.79,7.48,8.76],
                [4.23,5.11,5.58,6.06,6.62,7.32,9.74],
                [3.72,4.64,5.33,5.74,6.22,6.77,8.34],
                [3.21,4.1,4.87,5.4,5.78,6.22,7.32],
            ]
        )

    def tearDown(self) -> None:
        if os.path.isfile(self.path):
            os.remove(self.path)


class TestReadWindSpeeds(unittest.TestCase):
    def setUp(self) -> None:
        self.path = "example.csv"
        lines = [
            "TWA\\TWS,6,8,10,12,14,16,20\n",
            "52°,3.74,4.48,4.96,5.27,5.47,5.66,5.81\n",
            "60°,3.98,4.73,5.18,5.44,5.67,5.94,6.17\n",
            "75°,4.16,4.93,5.35,5.66,5.95,6.27,6.86\n",
            "90°,4.35,5.19,5.64,6.09,6.49,6.7,7.35\n",
            "110°,4.39,5.22,5.68,6.19,6.79,7.48,8.76\n",
            "120°,4.23,5.11,5.58,6.06,6.62,7.32,9.74\n",
            "135°,3.72,4.64,5.33,5.74,6.22,6.77,8.34\n",
            "150°,3.21,4.1,4.87,5.4,5.78,6.22,7.32\n",
        ]
        with open(self.path, "w", encoding="utf-8") as file:
            file.writelines(lines)

    def test_call(self):
        # Input/Output
        with open(self.path, "r", encoding="utf-8") as file:
            csv_reader = csv.reader(file, delimiter=",")
            ws_res = _read_wind_speeds(csv_reader)
            self.assertEqual(ws_res, [6, 8, 10, 12, 14, 16, 20])

    def tearDown(self) -> None:
        if os.path.isfile(self.path):
            os.remove(self.path)


class TestReadORCFormat(unittest.TestCase):
    def setUp(self) -> None:
        self.path = "example.csv"
        lines = [
            "twa/tws;6;8;10;12;14;16;20\n",
            "0;0;0;0;0;0;0;0\n",
            "52;3.74;4.48;4.96;5.27;5.47;5.66;5.81\n",
            "60;3.98;4.73;5.18;5.44;5.67;5.94;6.17\n",
            "75;4.16;4.93;5.35;5.66;5.95;6.27;6.86\n",
            "90;4.35;5.19;5.64;6.09;6.49;6.7;7.35\n",
            "110;4.39;5.22;5.68;6.19;6.79;7.48;8.76\n",
            "120;4.23;5.11;5.58;6.06;6.62;7.32;9.74\n",
            "135;3.72;4.64;5.33;5.74;6.22;6.77;8.34\n",
            "150;3.21;4.1;4.87;5.4;5.78;6.22;7.32\n",
        ]
        with open(self.path, "w", encoding="utf-8") as file:
            file.writelines(lines)

    def test_call(self):
        # Input/Output
        with open(self.path, "r", encoding="utf-8") as file:
            ws_res, wa_res, bsps = _read_orc_format(file)
        self.assertEqual(ws_res, [6, 8, 10, 12, 14, 16, 20])
        self.assertEqual(wa_res, [52, 60, 75, 90, 110, 120, 135, 150])
        self.assertEqual(
            bsps,
            [
                [3.74,4.48,4.96,5.27,5.47,5.66,5.81],
                [3.98,4.73,5.18,5.44,5.67,5.94,6.17],
                [4.16,4.93,5.35,5.66,5.95,6.27,6.86],
                [4.35,5.19,5.64,6.09,6.49,6.7,7.35],
                [4.39,5.22,5.68,6.19,6.79,7.48,8.76],
                [4.23,5.11,5.58,6.06,6.62,7.32,9.74],
                [3.72,4.64,5.33,5.74,6.22,6.77,8.34],
                [3.21,4.1,4.87,5.4,5.78,6.22,7.32],
            ]
        )

    def test_read_extern_format(self):
        # Execution Test
        with open(self.path, "r", encoding="utf-8") as file:
            _read_extern_format(file, "orc")

    def tearDown(self) -> None:
        if os.path.isfile(self.path):
            os.remove(self.path)


class TestReadFromArray(unittest.TestCase):
    def setUp(self) -> None:
        self.path = "example.csv"
        lines = [
            "TWA\\TWS 6   8   10  12  14  16  20\n",
            "52  3.74    4.48    4.96    5.27    5.47    5.66    5.81\n",
            "60  3.98    4.73    5.18    5.44    5.67    5.94    6.17\n",
            "75  4.16    4.93    5.35    5.66    5.95    6.27    6.86\n",
            "90  4.35    5.19    5.64    6.09    6.49    6.7 7.35\n",
            "110 4.39    5.22    5.68    6.19    6.79    7.48    8.76\n",
            "120 4.23    5.11    5.58    6.06    6.62    7.32    9.74\n",
            "135 3.72    4.64    5.33    5.74    6.22    6.77    8.34\n",
            "150 3.21    4.1 4.87    5.4 5.78    6.22    7.32\n",
        ]
        with open(self.path, "w", encoding="utf-8") as file:
            file.writelines(lines)

    def test_call(self):
        # Input/Output
        with open(self.path, "r", encoding="utf-8") as file:
            ws_res, wa_res, bsps = _read_from_array(file)
        np.testing.assert_array_equal(ws_res, [6, 8, 10, 12, 14, 16, 20])
        np.testing.assert_array_equal(wa_res, [52, 60, 75, 90, 110, 120, 135, 150])
        np.testing.assert_array_equal(
            bsps,
            [
                [3.74,4.48,4.96,5.27,5.47,5.66,5.81],
                [3.98,4.73,5.18,5.44,5.67,5.94,6.17],
                [4.16,4.93,5.35,5.66,5.95,6.27,6.86],
                [4.35,5.19,5.64,6.09,6.49,6.7,7.35],
                [4.39,5.22,5.68,6.19,6.79,7.48,8.76],
                [4.23,5.11,5.58,6.06,6.62,7.32,9.74],
                [3.72,4.64,5.33,5.74,6.22,6.77,8.34],
                [3.21,4.1,4.87,5.4,5.78,6.22,7.32],
            ]
        )

    def test_read_extern_format(self):
        # Execution Test
        with open(self.path, "r", encoding="utf-8") as file:
            _read_extern_format(file, "array")

    def tearDown(self) -> None:
        if os.path.isfile(self.path):
            os.remove(self.path)

class DummyPolarDiagram(PolarDiagram):
    """
    Needed for the following tests
    """
    def __init__(self, teststr):
        super().__init__()
        self.teststr = teststr

    def to_csv(self, csv_path):
        pass

    @classmethod
    def __from_csv__(cls, file):
        return cls(file.readline())

    def __call__(self, ws, wa):
        pass

    def symmetrize(self):
        pass

    @property
    def default_points(self):
        return None

    @property
    def default_slices(self):
        return None

    def ws_to_slices(self, ws, **kwargs):
        pass


class TestReadInternFormat(unittest.TestCase):
    def setUp(self) -> None:
        self.path = "example.csv"
        lines = [
            "DummyPolarDiagram\n"
            "Test"
        ]
        with open(self.path, "w", encoding="utf-8") as file:
            file.writelines(lines)

        self.path_fail = "example_fail.csv"
        lines = [
            "NonExistentPolarDiagram\n",
            "Test"
        ]
        with open(self.path_fail, "w", encoding="utf-8") as file:
            file.writelines(lines)

    def test_existing_polar_diagram_class(self):
        # Input/Output
        with open(self.path, "r", encoding="utf-8") as file:
            pd = _read_intern_format(file)

        self.assertEqual(pd.teststr, "Test")

    def test_non_existent_polar_diagram_class(self):
        # Exception test
        with self.assertRaises(FileReadingException):
            with open(self.path_fail, "r", encoding="utf-8") as file:
                _read_intern_format(file)

    def tearDown(self) -> None:
        if os.path.isfile(self.path):
            os.remove(self.path)
        if os.path.isfile(self.path_fail):
            os.remove(self.path_fail)


class TestFromCSV(unittest.TestCase):
    def setUp(self) -> None:
        self.path_extern = "example_extern.csv"
        lines = [
            "TWA\\TWS,6,8,10,12,14,16,20\n",
            "52°,3.74,4.48,4.96,5.27,5.47,5.66,5.81\n",
            "60°,3.98,4.73,5.18,5.44,5.67,5.94,6.17\n",
            "75°,4.16,4.93,5.35,5.66,5.95,6.27,6.86\n",
            "90°,4.35,5.19,5.64,6.09,6.49,6.7,7.35\n",
            "110°,4.39,5.22,5.68,6.19,6.79,7.48,8.76\n",
            "120°,4.23,5.11,5.58,6.06,6.62,7.32,9.74\n",
            "135°,3.72,4.64,5.33,5.74,6.22,6.77,8.34\n",
            "150°,3.21,4.1,4.87,5.4,5.78,6.22,7.32\n",
        ]
        with open(self.path_extern, "w", encoding="utf-8") as file:
            file.writelines(lines)

        self.path_intern = "example_intern.csv"
        lines = [
            "DummyPolarDiagram\n"
            "Test"
        ]
        with open(self.path_intern, "w", encoding="utf-8") as file:
            file.writelines(lines)

    def test_wrong_format(self):
        # Exception test
        with self.assertRaises(ValueError):
            from_csv(self.path_extern, fmt="wrong_format")

    def test_intern_format(self):
        from_csv(self.path_intern)

    def test_extern_format(self):
        from_csv(self.path_extern, fmt="opencpn")

    def tearDown(self) -> None:
        if os.path.isfile(self.path_extern):
            os.remove(self.path_extern)
        if os.path.isfile(self.path_intern):
            os.remove(self.path_intern)