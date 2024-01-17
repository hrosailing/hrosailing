# pylint: disable-all

import unittest

from hrosailing.plotting.projections import _more_colors_than_plots


class TestMoreColorsThanPlots(unittest.TestCase):
    def test_true(self):
        self.assertTrue(
            _more_colors_than_plots([1, 2, 3], ["red", "blue", "green"])
        )

    def test_false(self):
        self.assertFalse(_more_colors_than_plots([1, 2, 3], ["red"]))
