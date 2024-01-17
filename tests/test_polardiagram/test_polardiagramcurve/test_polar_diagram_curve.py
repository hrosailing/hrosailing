# pylint: disable-all

import os

import numpy as np

from hrosailing.core.modelfunctions import s_shaped
from hrosailing.polardiagram import PolarDiagramCurve
from tests.utils_for_testing import hroTestCase


class TestPolarDiagramCurve(hroTestCase):
    def setUp(self):
        self.fun = lambda ws, wa, p: ws + wa + p
        self.params = (5,)
        self.pd = PolarDiagramCurve(self.fun, *self.params)
        self.named_pd = PolarDiagramCurve(s_shaped, 1, 2, 3, 4)

        with open("read_example.csv", "w", encoding="utf-8") as file:
            file.write(
                "Function:s_shaped\nRadians:False\nParameters:1:2:3:4\n"
            )

        with open("read_example_invalid.csv", "w", encoding="utf-8") as file:
            file.write(
                "Function:non_existent_function\nRadians:False\nParameters:1:2:3:4\n"
            )

    def tearDown(self):
        if os.path.isfile("write_example.csv"):
            os.remove("write_example.csv")

        if os.path.isfile("read_example.csv"):
            os.remove("read_example.csv")

        if os.path.isfile("read_example_invalid.csv"):
            os.remove("read_example_invalid.csv")

    def test_init_not_callable(self):
        with self.assertRaises(TypeError):
            PolarDiagramCurve("test")

    def test_init_not_enough_params(self):
        with self.assertRaises(ValueError):
            PolarDiagramCurve(lambda ws, wa, p: p, 5, 6, 7, 8, 9, 10)

    def test_init_regular_input(self):
        PolarDiagramCurve(lambda ws, wa, p: p, 4, radians=False)

    def test_default_points(self):
        result = self.pd.default_points

        self.assertEqual(result.ndim, 2)
        self.assertEqual(result.shape[1], 3)

        for point in result:
            ws, wa, bsp = point
            self.assertEqual(ws + wa + 5, bsp)

    def test_get_slices(self):
        self.pd.get_slices([1, 2, 3], 2, full_info=True, wa_resolution=2)

    def test_ws_to_slices(self):
        result = self.pd.ws_to_slices([1, 2, 3], wa_resolution=3)
        expected = [
            np.array([[1, 1, 1], [0, 180, 360], [6, 186, 6]]),
            np.array([[2, 2, 2], [0, 180, 360], [7, 187, 7]]),
            np.array([[3, 3, 3], [0, 180, 360], [8, 188, 8]]),
        ]

        self.assertEqual(len(result), len(expected))

        for result_, expected_ in zip(result, expected):
            np.testing.assert_array_equal(result_, expected_)

    def test_check_enough_params_enough_params(self):
        result = PolarDiagramCurve._check_enough_params(
            lambda x, y, z: 5, (5,)
        )

        self.assertTrue(result)

    def test_check_enough_params_not_enough_params(self):
        result = PolarDiagramCurve._check_enough_params(
            lambda x, y, z, a: 5, (5,)
        )

        self.assertFalse(result)

    def test_repr(self):
        result = repr(self.named_pd)

        self.assertEqual(
            result, "PolarDiagramCurve(f=s_shaped, 1, 2, 3, 4, radians=False)"
        )

    def test_call_wa_is_array(self):
        result = self.pd(np.array([1, 2, 3]), np.array([1, 2, 360]))

        np.testing.assert_array_equal(result, [7, 9, 8])

    def test_call_with_radians(self):
        pd = PolarDiagramCurve(self.fun, *self.params, radians=True)
        result = pd(np.array([1, 2, 3]), np.array([1, 2, 2 * np.pi]))

        np.testing.assert_array_almost_equal(result, [7, 9, 8])

    def test_curve(self):
        self.assertEqual(self.pd.curve(1, 1, 1), 3)

    def test_default_slices(self):
        result = self.pd.default_slices

        self.assertEqual(result.ndim, 1)

    def test_parameters(self):
        self.assertEqual(self.pd.parameters, (5,))

    def test_radians(self):
        self.assertEqual(self.pd.radians, False)

    def test_to_csv(self):
        self.named_pd.to_csv("example.csv")
        with open("example.csv", "r", encoding="utf-8") as file:
            content = file.read()

        self.assertEqual(
            content,
            "PolarDiagramCurve\nFunction:s_shaped\nRadians:False\nParameters:1:2:3:4\n",
        )

    def test_from_csv_valid(self):
        with open("read_example.csv", "r", encoding="utf-8") as file:
            pd = PolarDiagramCurve.__from_csv__(file)

        self.assertEqual(pd.curve.__name__, "s_shaped")
        self.assertEqual(pd.parameters, (1, 2, 3, 4))
        self.assertFalse(pd.radians)

    def test_from_csv_invalid(self):
        with self.assertRaises(RuntimeError):
            with open(
                "read_example_invalid.csv", "r", encoding="utf-8"
            ) as file:
                PolarDiagramCurve.__from_csv__(file)

    def test_symmetrize(self):
        symm_pd = self.pd.symmetrize()

        self.assertEqual(symm_pd(3, 4), symm_pd(3, 360 - 4))
        self.assertEqual(symm_pd(5, 0), symm_pd(5, 360))
        self.assertEqual(symm_pd(1, 40), symm_pd(1, 360 - 40))
        self.assertEqual(symm_pd(12, 370), symm_pd(12, 360 - 30))
