# pylint: disable-all

import unittest

import matplotlib.pyplot as plt
import numpy as np

from hrosailing.polardiagram import PolarDiagramTable


class TestHROColorGradient(unittest.TestCase):
    def setUp(self):
        self.pd = PolarDiagramTable(
            [1, 2, 3], [0, 90, 180], [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
        )

    def test_plot_polar_diagram(self):
        ax = plt.subplot(projection="hro color gradient")
        ax.plot(
            self.pd,
            wind=(np.array([1, 2, 3, 4]), np.array([0, 5, 10, 15])),
            colors=("green", "red"),
            show_legend=True,
            legend_kw={"location": "left"},
            marker="H",
        )

    def test_plot_other(self):
        ax = plt.subplot(projection="hro color gradient")
        ax.plot([1, 2, 3], [1, 2, 3], ls="--", marker="H")

    def test_scatter_polar_diagram(self):
        ax = plt.subplot(projection="hro color gradient")
        ax.scatter(
            self.pd,
            wind=(np.array([1, 2, 3, 4]), np.array([0, 5, 10, 15])),
            colors=("green", "red"),
            show_legend=True,
            legend_kw={"location": "left"},
            marker="H",
        )

    def test_scatter_other(self):
        ax = plt.subplot(projection="hro color gradient")
        ax.scatter([1, 2, 3], [1, 2, 3], ls="--", marker="H")
