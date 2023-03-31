# pylint: disable-all
import unittest

import matplotlib.pyplot as plt

from hrosailing.plotting.projections import _configure_axes


class TestConfigureAxis(unittest.TestCase):
    def test_without_show_legend(self):
        # Execution test with `show_legend=False`
        _configure_axes(
            plt.subplot(), [1, 2, 3], ("green", "red"), False, None, marker="H"
        )

    def test_with_show_legend_without_keywords(self):
        # Execution test with `show_legend=True`, no keywords
        _configure_axes(
            plt.subplot(), [1, 2, 3], ("green", "red"), False, None
        )

    def test_with_show_legend_with_keywords(self):
        # Execution test with `show_legend=True` and keywords
        _configure_axes(
            plt.subplot(), [1, 2, 3], ("green", "red"), False, None, marker="H"
        )
