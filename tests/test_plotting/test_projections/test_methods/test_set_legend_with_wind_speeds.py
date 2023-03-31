# pylint: disable-all
import unittest

import matplotlib.pyplot as plt

from hrosailing.plotting.projections import _set_legend_with_wind_speeds


class TestSetLegendWithWindSpeeds(unittest.TestCase):
    def test_regular_input(self):
        # Execution Test
        _set_legend_with_wind_speeds(
            plt.subplot(), ["red", "green", "blue"], [1, 2, 3], {"loc": "best"}
        )
