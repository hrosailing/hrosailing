# pylint: disable-all

import unittest

import matplotlib.pyplot as plt

from hrosailing.plotting.projections import _configure_legend


class TestConfigureLegend(unittest.TestCase):
    def test_color_gradients(self):
        _configure_legend(
            plt.subplot(),
            [1, 2, 3],
            ("red", "green"),
            ["label 1", "label 2", "label 3"],
            location="left",
        )

    def test_tuples(self):
        _configure_legend(
            plt.subplot(),
            [1, 2, 3],
            ((1, "red"), (2, "green"), (3, "orange")),
            ["label 1", "label 2", "label 3"],
            loc="upper left",
        )

    def test_color_list(self):
        _configure_legend(
            plt.subplot(),
            [1, 2, 3],
            ["red", "green", "orange"],
            ["label 1", "label 2", "label 3"],
            loc="upper left",
        )
