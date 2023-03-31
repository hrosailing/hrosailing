import unittest

from hrosailing.plotting.projections import _has_color_format


class TestHasColorFormat(unittest.TestCase):
    def test_string(self):
        # Input/Output
        self.assertTrue(_has_color_format("blue"))

    def test_3_tuple(self):
        # Input/Output
        self.assertTrue(_has_color_format((1, 0, 0)))

    def test_4_tuple(self):
        # Input/Output
        self.assertTrue(_has_color_format((1, 0, 0, 1)))

    def test_5_tuple(self):
        # Input/Output
        self.assertFalse(_has_color_format((1, 0, 0, 1, 0)))
