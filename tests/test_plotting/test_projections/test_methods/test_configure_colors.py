import unittest

import matplotlib.pyplot as plt

from hrosailing.plotting.projections import _configure_colors


class TestConfigureColors(unittest.TestCase):
    def test_only_one_color(self):
        # Execution Test
        _configure_colors(plt.subplot(), [1, 2, 3], (1, 0, 0))

    def test_more_colors_than_plots(self):
        # Execution Test
        _configure_colors(
            plt.subplot(), [1, 2, 3], ["red", "green", "blue", "orange"]
        )

    def test_no_color_gradient(self):
        # Execution Test
        _configure_colors(
            plt.subplot(), [1, 2, 3, 4, 5], ("red", "green", "blue")
        )

    def test_color_gradient(self):
        # Execution Test
        _configure_colors(plt.subplot(), [1, 2, 3, 4, 5], ("red", "green"))
