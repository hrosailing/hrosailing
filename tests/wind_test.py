"""

"""

# Author: Valentin F. Dannenberg


import unittest

import numpy as np

from hrosailing.wind import set_resolution, ResolutionSettingException


class ResolutionTest(unittest.TestCase):
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
                np.testing.assert_array_equal(
                    set_resolution(iter_, "angle"), np.asarray(iter_)
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
                with self.assertRaises(ResolutionSettingException):
                    set_resolution(iter_, "speed")
                    set_resolution(iter_, "angle")

    def test_resolution_exception_empty_iter(self):
        iters = [[], (), np.array([])]
        for i, iter_ in enumerate(iters):
            with self.subTest(i=i):
                with self.assertRaises(ResolutionSettingException):
                    set_resolution(iter_, "speed")
                    set_resolution(iter_, "angle")

    def test_resolution_exception_zero_nums(self):
        nums = [0, 0.0]
        for i, num in enumerate(nums):
            with self.subTest(i=i):
                with self.assertRaises(ResolutionSettingException):
                    set_resolution(num, "speed")
                    set_resolution(num, "angle")

    def test_resolution_exception_negative_nums(self):
        nums = [-1, -1.5]
        for i, num in enumerate(nums):
            with self.subTest(i=i):
                with self.assertRaises(ResolutionSettingException):
                    set_resolution(num, "speed")
                    set_resolution(num, "angle")


def set_resolution_suite():
    suite = unittest.TestSuite()
    suite.addTests(
        [
            ResolutionTest("test_resolution_None"),
            ResolutionTest("test_resolution_iters"),
            ResolutionTest("test_resolution_nums"),
            ResolutionTest("test_resolution_exception_set_dict"),
            ResolutionTest("test_resolution_exception_empty_iter"),
            ResolutionTest("test_resolution_exception_zero_nums"),
            ResolutionTest("test_resolution_exception_negative_nums"),
        ]
    )

    return suite


class ApparentWindTest(unittest.TestCase):
    pass


def apparent_wind_suite():
    suite = unittest.TestSuite()
    suite.addTests([])

    return suite


class TrueWindTest(unittest.TestCase):
    pass


def true_wind_suite():
    suite = unittest.TestSuite()
    suite.addTests([])

    return suite
