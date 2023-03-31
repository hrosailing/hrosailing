import unittest

from hrosailing.plotting.projections import _check_plot_kw


class TestCheckPlotKW(unittest.TestCase):
    def test_with_linestyle(self):
        # Input/Output Test with kw "linestyle"
        keywords = {"linestyle": ".."}
        _check_plot_kw(keywords)
        self.assertEqual(keywords, {"ls": ".."})

    def test_with_ls(self):
        # Input/Output Test with kw "ls"
        keywords = {"ls": ".."}
        _check_plot_kw(keywords)
        self.assertEqual(keywords, {"ls": ".."})

    def test_without_linestyle_or_marker(self):
        # Input/Output Test without keywords
        keywords = {}
        _check_plot_kw(keywords)
        self.assertEqual(keywords, {"ls": "-"})

    def test_without_linestyle_or_marker_without_lines(self):
        # Input/Output Test without keywords, `lines=False`
        keywords = {}
        _check_plot_kw(keywords, False)
        self.assertEqual(keywords, {"ls": "", "marker": "o"})
