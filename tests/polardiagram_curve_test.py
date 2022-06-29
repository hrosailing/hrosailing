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


class PolarDiagramCurveTest(unittest.TestCase):
    def setUp(self):
        def func(ws, wa, *params):
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
        ws, wa, bsp = self.c.get_slices((10, 15), n_steps=100)
        self.assertEqual(ws, list(np.linspace(10, 15, 100)))
        np.testing.assert_array_equal(
            wa, np.deg2rad(np.linspace(0, 360, 1000))
        )
        for i, w in enumerate(ws):
            np.testing.assert_array_equal(
                bsp[i], self.c(np.array([w] * 1000), np.linspace(0, 360, 1000))
            )