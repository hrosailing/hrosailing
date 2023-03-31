# pylint: disable-all
import unittest

import matplotlib.pyplot as plt

from hrosailing.plotting.projections import _set_colormap


class TestSetColormap(unittest.TestCase):
    def test_regular_input(self):
        # Execution Test
        _set_colormap(
            [1, 2, 3],
            ["red", "green", "blue"],
            plt.subplot(),
            ["red", "green", "blue"],
            location="left",
        )
