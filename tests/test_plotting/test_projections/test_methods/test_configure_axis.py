# pylint: disable-all

import unittest

import matplotlib.pyplot as plt

from hrosailing.plotting.projections import _configure_axes


class TestConfigureAxis(unittest.TestCase):
    def test_without_show_legend(self):
        _configure_axes(
            plt.subplot(), [1, 2, 3], ("green", "red"), False, None, marker="H"
        )

    def test_with_show_legend_without_keywords(self):
        _configure_axes(
            plt.subplot(), [1, 2, 3], ("green", "red"), False, None
        )

    def test_with_show_legend_with_keywords(self):
        _configure_axes(
            plt.subplot(), [1, 2, 3], ("green", "red"), False, None, marker="H"
        )
