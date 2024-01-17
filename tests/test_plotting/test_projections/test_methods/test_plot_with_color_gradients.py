# pylint: disable-all

import unittest

from hrosailing.plotting.projections import _plot_with_color_gradient


class TestPlotWithColorGradients(unittest.TestCase):
    def test_true(self):
        self.assertTrue(_plot_with_color_gradient([1, 2, 3], ("red", "green")))

    def test_false(self):
        self.assertFalse(
            _plot_with_color_gradient([1, 2, 3], ("red", "green", "blue"))
        )
