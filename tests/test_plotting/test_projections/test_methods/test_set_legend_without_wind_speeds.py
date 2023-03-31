# pylint: disable-all
import unittest

import matplotlib.pyplot as plt

from hrosailing.plotting.projections import _set_legend_without_wind_speeds


class TestSetLegendWithoutWindSpeeds(unittest.TestCase):
    def test_regular_input(self):
        # Execution Test
        _set_legend_without_wind_speeds(
            plt.subplot(),
            [(1, "red"), (2, "green"), (3, "blue")],
            {"loc": "upper left"},
        )
