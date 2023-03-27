import numpy as np
from unittest import TestCase


class hroTestCase(TestCase):

    def assert_list_almost_equal(self, result, expected_result, places, msg):
        for i, element in enumerate(expected_result):
            self.assertAlmostEqual(result[i], element, places, msg)
