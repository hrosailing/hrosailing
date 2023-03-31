# pylint: disable-all

from unittest import TestCase

import numpy as np


class hroTestCase(TestCase):
    def assert_list_almost_equal(self, result, expected_result, places, msg):
        for i, element in enumerate(expected_result):
            try:
                if result[i] is None:
                    self.assertIsNone(element)
                    continue
                if element is None:
                    self.assertIsNone(result[i])
                    continue
                self.assertAlmostEqual(result[i], element, places, msg)
            except AssertionError as exp:
                raise AssertionError(
                    f"{result} != {expected_result},\n"
                    "first different entry is "
                    f"{result[i]} != {expected_result[i]} at index {i}.\n\n"
                    f"{msg}"
                ) from exp

    def assert_time_list_equal(self, result, expected, msg=""):
        for i, (res, exp) in enumerate(zip(result, expected)):
            try:
                if result[i] is None:
                    self.assertIsNone(exp)
                    continue
                if exp is None:
                    self.assertIsNone(result[i])
                    continue
            except AssertionError as error:
                raise AssertionError(
                    f"{result} != {expected},\n"
                    "first different entry is "
                    f"{result[i]} != {expected[i]} at index {i}.\n\n"
                    f"{msg}"
                ) from error
            is_equal = (
                (res.hour == exp.hour)
                and (res.minute == exp.minute)
                and (res.second == exp.second)
                and (res.microsecond == exp.microsecond)
            )
            if not is_equal:
                raise AssertionError(
                    f"{result} != {expected},\n"
                    "first different entry is "
                    f"{result[i]} != {expected[i]} at index {i}.\n\n"
                    f"{msg}"
                )
