import matplotlib.pyplot as plt
import numpy as np

from tests.test_plotting.image_testcase import ImageTestcase

from hrosailing.plotting.projections import plot_flat
from hrosailing.polardiagram import PolarDiagramTable

class TestPlotFlat(ImageTestcase):
    def test_regular_plot(self):
        # Input/Output
        pd = PolarDiagramTable(
            [1, 2, 3],
            [0, 90, 180],
            [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
        )
        plot_flat(pd)
        self.set_result_plot()

        ax = plt.subplot()
        ax.plot([0, 90, 180], [0, 1, 2], color=(0, 1, 0))
        ax.plot([0, 90, 180], [1, 2, 3], color=(0.5, 0.5, 0))
        ax.plot([0, 90, 180], [2, 3, 4], color=(1, 0, 0))
        self.set_expected_plot()

        self.assertPlotsEqual()

    def test_with_keywords(self):
        # Input/Output with keywords

        keywords = {
            "marker": "H",
            "linestyle": "--"
        }
        pd = PolarDiagramTable(
            [1, 2, 3],
            [0, 90, 180],
            [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
        )
        plot_flat(pd, **keywords)
        self.set_result_plot()

        ax = plt.subplot()
        ax.plot([0, 90, 180], [0, 1, 2], color=(0, 1, 0), **keywords)
        ax.plot([0, 90, 180], [1, 2, 3], color=(0.5, 0.5, 0), **keywords)
        ax.plot([0, 90, 180], [2, 3, 4], color=(1, 0, 0), **keywords)
        self.set_expected_plot()

        self.assertPlotsEqual()