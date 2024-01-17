# pylint: disable-all

from unittest import TestCase

import numpy as np

from hrosailing.core.data import Data
from hrosailing.processing.weigher import FuzzyBool


class TestFuzzyBool(TestCase):
    def setUp(self):
        self.ev_fun = lambda x: 1 / (1 + np.exp(-10 * x))
        self.ev_fun2 = lambda x: 1 / (1 + np.exp(-5 * x))
        self.data = Data().from_dict({"TWS": [0.25, -0.5, 2.0]})

    def test_call(self):
        result = [
            FuzzyBool(self.ev_fun)(0.25),
            FuzzyBool(self.ev_fun)(-0.5),
            FuzzyBool(self.ev_fun)(2),
        ]
        expected_result = [0.924142, 0.00669285, 0.999999997]

        np.testing.assert_array_almost_equal(
            result,
            expected_result,
            decimal=4,
        )

    def test_str(self):
        result = str(FuzzyBool(self.ev_fun))
        expected_result = "Fuzzy-Bool"

        self.assertEqual(result, expected_result)

    def test_fuzzy_and(self):
        result = [
            FuzzyBool.fuzzy_and(
                FuzzyBool(self.ev_fun), FuzzyBool(self.ev_fun2)
            )(0.25),
            FuzzyBool.fuzzy_and(
                FuzzyBool(self.ev_fun), FuzzyBool(self.ev_fun2)
            )(-0.5),
            FuzzyBool.fuzzy_and(
                FuzzyBool(self.ev_fun), FuzzyBool(self.ev_fun2)
            )(2),
        ]
        expected_result = [0.7773, 0.00669285, 0.9999546021313]

        np.testing.assert_array_almost_equal(
            result,
            expected_result,
            decimal=4,
        )

    def test_fuzzy_or(self):
        result = [
            FuzzyBool.fuzzy_or(
                FuzzyBool(self.ev_fun), FuzzyBool(self.ev_fun2)
            )(0.25),
            FuzzyBool.fuzzy_or(
                FuzzyBool(self.ev_fun), FuzzyBool(self.ev_fun2)
            )(-0.5),
            FuzzyBool.fuzzy_or(
                FuzzyBool(self.ev_fun), FuzzyBool(self.ev_fun2)
            )(2),
        ]
        expected_result = [0.924142, 0.07585818002124355, 0.9999546021313]

        np.testing.assert_array_almost_equal(
            result,
            expected_result,
            decimal=4,
        )

    def test_fuzzy_not(self):
        result = [
            FuzzyBool.fuzzy_not(FuzzyBool(self.ev_fun2))(0.25),
            FuzzyBool.fuzzy_not(FuzzyBool(self.ev_fun2))(-0.5),
            FuzzyBool.fuzzy_not(FuzzyBool(self.ev_fun2))(2),
        ]
        expected_result = [
            1 - 0.7773,
            1 - 0.07585818002124355,
            1 - 0.9999546021313,
        ]

        np.testing.assert_array_almost_equal(
            result,
            expected_result,
            decimal=4,
        )

    def test_sigmoid(self):
        result = [
            FuzzyBool.sigmoid(0, 10, -1)(0.25),
            FuzzyBool.sigmoid(0, 10, -1)(-0.5),
            FuzzyBool.sigmoid(0, 10, -1)(2),
        ]
        expected_result = [
            FuzzyBool(self.ev_fun)(0.25),
            FuzzyBool(self.ev_fun)(-0.5),
            FuzzyBool(self.ev_fun)(2),
        ]

        self.assertEqual(result, expected_result)

    def test__concat_repr(self):
        fuzz1 = FuzzyBool(self.ev_fun)
        fuzz1.repr = "Fuzzy-1"
        fuzz2 = FuzzyBool(self.ev_fun)
        fuzz2.repr = "Fuzzy-2"
        fuzz = FuzzyBool(self.ev_fun)
        fuzz._concat_repr(fuzz1, fuzz2, "and")

        result = str(fuzz)
        expected_result = "Fuzzy-1 and Fuzzy-2"

        self.assertEqual(result, expected_result)

    def test__concat_repr_with_brackets(self):
        fuzz1 = FuzzyBool(self.ev_fun)
        fuzz1.repr = "Fuzzy 1"
        fuzz2 = FuzzyBool(self.ev_fun)
        fuzz2.repr = "Fuzzy 2"
        fuzz = FuzzyBool(self.ev_fun)
        fuzz._concat_repr(fuzz1, fuzz2, "and")

        result = str(fuzz)
        expected_result = "(Fuzzy 1) and (Fuzzy 2)"

        self.assertEqual(result, expected_result)

    def test_and(self):
        fuzz1 = FuzzyBool(self.ev_fun)
        fuzz2 = FuzzyBool(self.ev_fun2)
        fuzz1 & fuzz2

    def test_or(self):
        fuzz1 = FuzzyBool(self.ev_fun)
        fuzz2 = FuzzyBool(self.ev_fun2)
        fuzz1 | fuzz2

    def test_invert(self):
        fuzz1 = FuzzyBool(self.ev_fun)
        ~fuzz1

    def test_getitem(self):
        fuzz = FuzzyBool(self.ev_fun)["TWS"]

        self.assertAlmostEqual(fuzz({"TWS": 0.25}), 0.92414182)
        self.assertAlmostEqual(fuzz({"TWS": -0.5}), 0.00669285)
        self.assertAlmostEqual(fuzz({"TWS": 2.0}), 1.0)
