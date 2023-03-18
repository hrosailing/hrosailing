import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from tests.test_plotting.image_testcase import ImageTestcase

from hrosailing.plotting.projections import plot_color_gradient
from hrosailing.polardiagram import PolarDiagramTable

class TestPlotFlat(ImageTestcase):
    def test_regular_plot(self):
        # Is the data plotted at the right position?
        pd = PolarDiagramTable(
            [1, 2, 3],
            [0, 90, 180],
            [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
        )
        plot_color_gradient(
            pd,
            colors=("black", "black")
        )
        self.set_result_plot()

        ax = plt.subplot()
        ax.scatter(
            [1, 2, 3, 1, 2, 3, 1, 2, 3],
            [0, 0, 0, 90, 90, 90, 180, 180, 180],
            c=[(0,0,0), (0,0,0), (0,0,0), (0,0,0), (0,0,0), (0,0,0), (0,0,0), (0,0,0), (0,0,0)]
        )
        self.set_expected_plot()

        self.assertPlotsEqual()

    def test_with_keywords(self):
        # Is the data plotted at the right position (ignoring colors)?
        # With keywords
        self.debug = True
        keywords = {
            "marker": "H",
            "linestyle": "--"
        }
        pd = PolarDiagramTable(
            [1, 2, 3],
            [0, 90, 180],
            [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
        )
        plot_color_gradient(pd, colors=("black", "black"), **keywords)
        self.set_result_plot()

        ax = plt.subplot()
        ax.scatter(
            [1, 2, 3, 1, 2, 3, 1, 2, 3],
            [0, 0, 0, 90, 90, 90, 180, 180, 180],
            c=[(0,0,0), (0,0,0), (0,0,0), (0,0,0), (0,0,0), (0,0,0), (0,0,0), (0,0,0), (0,0,0)],
            **keywords
        )
        self.set_expected_plot()

        self.assertPlotsEqual()