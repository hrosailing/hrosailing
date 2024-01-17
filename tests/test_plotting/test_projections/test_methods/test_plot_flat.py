# pylint: disable-all

import unittest

from hrosailing.plotting.projections import plot_flat
from hrosailing.polardiagram import PolarDiagramTable


class TestPlotFlat(unittest.TestCase):
    def test_polar_diagram_plot(self):
        keywords = {"marker": "H", "linestyle": "--"}
        pd = PolarDiagramTable(
            [1, 2, 3], [0, 90, 180], [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
        )
        plot_flat(pd, **keywords)

    def test_regular_plot(self):
        plot_flat([1, 2, 3], [1, 2, 3])
