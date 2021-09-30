# pylint: disable=missing-docstring

from unittest import TestCase

import numpy as np

from hrosailing.wind import WindConversionException
from hrosailing.wind import _convert_wind as convert_wind
from hrosailing.wind import _set_resolution as set_resolution


class ResolutionTest(TestCase):
    @staticmethod
    def test_resolution_None():
        np.testing.assert_array_equal(
            set_resolution(None, "s"), np.arange(2, 42, 2)
        )
        np.testing.assert_array_equal(
            set_resolution(None, "a"), np.arange(0, 360, 5)
        )

    def test_resolution_iters(self):
        iters = [[1, 2, 3, 4], (1, 2, 3, 4), np.array([1, 2, 3, 4])]
        for i, iter_ in enumerate(iters):
            with self.subTest(i=i):
                np.testing.assert_array_equal(
                    set_resolution(iter_, "s"), np.asarray(iter_)
                )

    def test_resolution_nums(self):
        nums = [1, 1.5]
        for i, num in enumerate(nums):
            with self.subTest(i=i):
                np.testing.assert_array_equal(
                    set_resolution(num, "s"), np.arange(num, 40, num)
                )
                np.testing.assert_array_equal(
                    set_resolution(num, "a"), np.arange(num, 360, num)
                )

    def test_resolution_exception_set_dict(self):
        iters = [{}, set()]
        for i, iter_ in enumerate(iters):
            with self.subTest(i=i):
                with self.assertRaises(ValueError):
                    set_resolution(iter_, "s")

    def test_resolution_exception_empty_iter(self):
        iters = [[], (), np.array([])]
        for i, iter_ in enumerate(iters):
            with self.subTest(i=i):
                with self.assertRaises(ValueError):
                    set_resolution(iter_, "s")

    def test_resolution_exception_None_in_iter(self):
        iters = [[None], (None,), np.array([None]), np.atleast_1d(None)]
        for i, iter_ in enumerate(iters):
            with self.subTest(i=i):
                with self.assertRaises(ValueError):
                    set_resolution(iter_, "s")

    def test_resolution_exception_infinite_or_nan_vals(self):
        iters = [
            [1, 2, 3, np.inf],
            [1, 2, 3, np.NINF],
            [1, 2, 3, np.nan],
            [1, 2, 3, np.PINF],
        ]
        for i, iter_ in enumerate(iters):
            with self.subTest(i=i):
                with self.assertRaises(ValueError):
                    set_resolution(iter_, "s")

    def test_resolution_exception_zero_nums(self):
        nums = [0, 0.0]
        for i, num in enumerate(nums):
            with self.subTest(i=i):
                with self.assertRaises(ValueError):
                    set_resolution(num, "s")

    def test_resolution_exception_negative_nums(self):
        nums = [-1, -1.5]
        for i, num in enumerate(nums):
            with self.subTest(i=i):
                with self.assertRaises(ValueError):
                    set_resolution(num, "s")


class ConvertWind(TestCase):
    def test_convert_wind_iters(self):
        iters = [[[0, 0, 3]], ((0, 0, 3),), np.array([[0, 0, 3]])]
        for i, iter_ in enumerate(iters):
            with self.subTest(i=i):
                tw = np.array([[3, 0, 3]])
                aw = np.array([[3, 180, 3]])
                np.testing.assert_array_equal(
                    convert_wind(iter_, 1, False, False), tw
                )
                np.testing.assert_array_equal(
                    convert_wind(iter_, -1, False, False), aw
                )

    def test_non_array_like(self):
        iters = [{}, set()]
        for i, iter_ in enumerate(iters):
            with self.subTest(i=i):
                with self.assertRaises(WindConversionException):
                    convert_wind(iter_, 1, False, False)

    def test_empty_iter(self):
        iters = [[], (), np.array([])]
        for i, iter_ in enumerate(iters):
            with self.subTest(i=i):
                with self.assertRaises(WindConversionException):
                    convert_wind(iter_, 1, False, False)

    def test_wrong_shape(self):
        iters = [
            [[0, 0]],
            [0, 0],
            [0, 0, 0],
            ((0, 0),),
            (0, 0),
            (0, 0, 0),
            np.array([0, 0]),
            np.array([0, 0, 0]),
        ]
        for i, iter_ in enumerate(iters):
            with self.subTest(i=i):
                with self.assertRaises(WindConversionException):
                    convert_wind(iter_, 1, False, False)

    def test_dont_convert_when_tw_is_True(self):
        iters = [[[0, 0, 3]], ((0, 0, 3),), np.array([[0, 0, 3]])]
        for i, iter_ in enumerate(iters):
            with self.subTest(i=i):
                wind = np.array([[0, 0, 3]])
                np.testing.assert_array_equal(
                    convert_wind(iter_, 1, True, False), wind
                )
                np.testing.assert_array_equal(
                    convert_wind(iter_, -1, True, False), wind
                )

    def test_check_finite_on(self):
        iters = [
            np.array([np.NaN]),
            np.array([np.inf]),
            np.array([np.NINF]),
            np.array([np.PINF]),
        ]
        for i, iter_ in enumerate(iters):
            with self.subTest(i=i):
                with self.assertRaises(ValueError):
                    convert_wind(iter_, 1, False, True)

    def test_check_finite_off(self):
        iters = [
            np.array([[1, 2, np.NaN]]),
            np.array([[1, 2, np.inf]]),
            np.array([[1, 2, np.NINF]]),
            np.array([[1, 2, np.PINF]]),
        ]
        answers = [
            np.array([[np.NaN, np.NaN, np.NaN]]),
            np.array([[np.inf, np.NaN, np.inf]]),
            np.array([[np.NaN, np.NaN, np.NINF]]),
            np.array([[np.inf, np.NaN, np.PINF]]),
        ]
        for i, iter_ in enumerate(iters):
            with self.subTest(i=i):
                np.testing.assert_array_equal(
                    convert_wind(iter_, 1, False, False), answers[i]
                )
