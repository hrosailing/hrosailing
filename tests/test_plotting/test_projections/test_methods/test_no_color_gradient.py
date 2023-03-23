import unittest

from hrosailing.plotting.projections import _no_color_gradient

class TestNoColorGradient(unittest.TestCase):
    def test_more_than_two_colors(self):
        # Input/Output
        self.assertTrue(
            _no_color_gradient(
                ("red", "green", "blue")
            )
        )

    def test_not_all_color_format(self):
        # Input/Output
        self.assertTrue(
            _no_color_gradient(
                ("red", "green", (1, 0))
            )
        )