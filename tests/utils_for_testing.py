import numpy as np
from unittest import TestCase


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
                    f"first different entry is "
                    f"{result[i]} != {expected_result[i]} at index {i}.\n\n"
                    f"{msg}"
                ) from exp
