
import unittest
import numpy as np


from utils import (
    speed_resolution,
    angle_resolution,
)


def equal_arrays(a, b, msg=None):
    if np.all(a == b):
        return True
    raise AssertionError(msg)


class SpeedResolutionTest(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, equal_arrays)

    def test_speed_resolution_None(self):
        self.assertEqual(speed_resolution(None),
                         np.arange(2, 42, 2))

    def test_speed_resolution_iters(self):
        iters = [[1, 2, 3, 4],
                 (1, 2, 3, 4),
                 np.array([1, 2, 3, 4])]
        for i, iter_ in enumerate(iters):
            with self.subTest(i=i):
                self.assertEqual(speed_resolution(iter_),
                                 np.asarray(iter_))

    def test_speed_resolution_nums(self):
        nums = [1, 1.5]
        for i, num in enumerate(nums):
            with self.subTest(i=i):
                self.assertEqual(speed_resolution(num),
                                 np.arange(num, 40, num))

    def test_speed_resolution_exception_not_iter_int_float(self):
        with self.assertRaises(ValueError):
            pass

    def test_speed_resolution_exception_set_dict(self):
        iters = [{}, set()]
        for i, iter_ in enumerate(iters):
            with self.subTest(i=i):
                with self.assertRaises(ValueError):
                    speed_resolution(iter_)

    def test_speed_resolution_exception_empty_iter(self):
        iters = [[], (), np.array([])]
        for i, iter_ in enumerate(iters):
            with self.subTest(i=i):
                with self.assertRaises(ValueError):
                    speed_resolution(iter_)

    def test_speed_resolution_exception_zero_nums(self):
        nums = [0, 0.0]
        for i, num in enumerate(nums):
            with self.subTest(i=i):
                with self.assertRaises(ValueError):
                    speed_resolution(num)

    def test_speed_resolution_exception_negative_nums(self):
        nums = [-1, -1.5]
        for i, num in enumerate(nums):
            with self.subTest(i=i):
                with self.assertRaises(ValueError):
                    speed_resolution(num)


def speed_resolution_suite():
    suite = unittest.TestSuite()
    suite.addTests([
        SpeedResolutionTest('test_speed_resolution_None'),
        SpeedResolutionTest('test_speed_resolution_iters'),
        SpeedResolutionTest('test_speed_resolution_nums'),
        SpeedResolutionTest('test_speed_resolution_exception_set_dict'),
        SpeedResolutionTest('test_speed_resolution_exception_empty_iter'),
        SpeedResolutionTest('test_speed_resolution_exception_zero_nums'),
        SpeedResolutionTest('test_speed_resolution_exception_negative_nums'),
    ])

    return suite


class AngleResolutionTest(unittest.TestCase):
    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, equal_arrays)

    def test_angle_resolution_None(self):
        self.assertEqual(angle_resolution(None),
                         np.arange(0, 360, 5))

    def test_angle_resolution_iters(self):
        iters = [[1, 2, 3, 4],
                 (1, 2, 3, 4),
                 np.array([1, 2, 3, 4])]
        for i, iter_ in enumerate(iters):
            with self.subTest(i=i):
                self.assertEqual(angle_resolution(iter_),
                                 np.asarray(iter_))

    def test_angle_resolution_nums(self):
        nums = [1, 1.5]
        for i, num in enumerate(nums):
            with self.subTest(i=i):
                self.assertEqual(angle_resolution(num),
                                 np.arange(num, 360, num))

    def test_angle_resolution_exception_not_iter_int_float(self):
        with self.assertRaises(ValueError):
            pass

    def test_angle_resolution_exception_set_dict(self):
        iters = [{}, set()]
        for i, iter_ in enumerate(iters):
            with self.subTest(i=i):
                with self.assertRaises(ValueError):
                    angle_resolution(iter_)

    def test_angle_resolution_exception_empty_iter(self):
        iters = [[], (), np.array([])]
        for i, iter_ in enumerate(iters):
            with self.subTest(i=i):
                with self.assertRaises(ValueError):
                    angle_resolution(iter_)

    def test_angle_resolution_exception_zero_nums(self):
        nums = [0, 0.0]
        for i, num in enumerate(nums):
            with self.subTest(i=i):
                with self.assertRaises(ValueError):
                    angle_resolution(num)

    def test_angle_resolution_exception_negative_nums(self):
        nums = [-1, -1.5]
        for i, num in enumerate(nums):
            with self.subTest(i=i):
                with self.assertRaises(ValueError):
                    angle_resolution(num)


def angle_resolution_suite():
    suite = unittest.TestSuite()
    suite.addTests([
        AngleResolutionTest('test_angle_resolution_None'),
        AngleResolutionTest('test_angle_resolution_iters'),
        AngleResolutionTest('test_angle_resolution_nums'),
        AngleResolutionTest('test_angle_resolution_exception_set_dict'),
        AngleResolutionTest('test_angle_resolution_exception_empty_iter'),
        AngleResolutionTest('test_angle_resolution_exception_zero_nums'),
        AngleResolutionTest('test_angle_resolution_exception_negative_nums'),
    ])

    return suite
