from unittest import TestCase

import numpy as np

from hrosailing.wind import (
    _set_resolution as set_resolution,
    _convert_wind as convert_wind,
    apparent_wind_to_true,
    true_wind_to_apparent,
)


class ResolutionTest(TestCase):
    @staticmethod
    def test_resolution_None():
        np.testing.assert_array_equal(
            set_resolution(None, "speed"), np.arange(2, 42, 2)
        )
        np.testing.assert_array_equal(
            set_resolution(None, "angle"), np.arange(0, 360, 5)
        )

    def test_resolution_iters(self):
        iters = [[1, 2, 3, 4], (1, 2, 3, 4), np.array([1, 2, 3, 4])]
        for i, iter_ in enumerate(iters):
            with self.subTest(i=i):
                np.testing.assert_array_equal(
                    set_resolution(iter_, "speed"), np.asarray(iter_)
                )

    def test_resolution_nums(self):
        nums = [1, 1.5]
        for i, num in enumerate(nums):
            with self.subTest(i=i):
                np.testing.assert_array_equal(
                    set_resolution(num, "speed"), np.arange(num, 40, num)
                )
                np.testing.assert_array_equal(
                    set_resolution(num, "angle"), np.arange(num, 360, num)
                )

    def test_resolution_exception_set_dict(self):
        iters = [{}, set()]
        for i, iter_ in enumerate(iters):
            with self.subTest(i=i):
                with self.assertRaises(ValueError):
                    set_resolution(iter_, "speed")

    def test_resolution_exception_empty_iter(self):
        iters = [[], (), np.array([])]
        for i, iter_ in enumerate(iters):
            with self.subTest(i=i):
                with self.assertRaises(ValueError):
                    set_resolution(iter_, "speed")

    def test_resolution_exception_None_in_iter(self):
        iters = [[None], (None,), np.array([None]), np.atleast_1d(None)]
        for i, iter_ in enumerate(iters):
            with self.subTest(i=i):
                with self.assertRaises(ValueError):
                    set_resolution(iter_, "speed")

    def test_resolution_exception_infinite_or_nan_vals(self):
        iters = [[1, 2, 3, np.inf], [1, 2, 3, np.NINF], [1, 2, 3, np.nan],
                 [1, 2, 3, np.PINF]]
        for i, iter_ in enumerate(iters):
            with self.subTest(i=i):
                with self.assertRaises(ValueError):
                    set_resolution(iter_, "speed")

    def test_resolution_exception_zero_nums(self):
        nums = [0, 0.0]
        for i, num in enumerate(nums):
            with self.subTest(i=i):
                with self.assertRaises(ValueError):
                    set_resolution(num, "speed")

    def test_resolution_exception_negative_nums(self):
        nums = [-1, -1.5]
        for i, num in enumerate(nums):
            with self.subTest(i=i):
                with self.assertRaises(ValueError):
                    set_resolution(num, "speed")


class ConvertWind(TestCase):
    pass
