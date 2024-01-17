# pylint: disable-all

import unittest

from hrosailing.plotting.projections import _check_plot_kw


class TestCheckPlotKW(unittest.TestCase):
    def test_with_linestyle(self):
        keywords = {"linestyle": ".."}
        _check_plot_kw(keywords)
        self.assertEqual(keywords, {"ls": ".."})

    def test_with_ls(self):
        keywords = {"ls": ".."}
        _check_plot_kw(keywords)
        self.assertEqual(keywords, {"ls": ".."})

    def test_without_linestyle_or_marker(self):
        keywords = {}
        _check_plot_kw(keywords)
        self.assertEqual(keywords, {"ls": "-"})

    def test_without_linestyle_or_marker_without_lines(self):
        keywords = {}
        _check_plot_kw(keywords, False)
        self.assertEqual(keywords, {"ls": "", "marker": "o"})
