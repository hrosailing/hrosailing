# pylint: disable-all
import unittest

from hrosailing.plotting.projections import _plot_with_color_gradient


class TestPlotWithColorGradients(unittest.TestCase):
    def test_true(self):
        # Input/Output
        self.assertTrue(_plot_with_color_gradient([1, 2, 3], ("red", "green")))

    def test_false(self):
        # Input/Output
        self.assertFalse(
            _plot_with_color_gradient([1, 2, 3], ("red", "green", "blue"))
        )
