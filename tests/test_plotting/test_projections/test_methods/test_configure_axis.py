import unittest

import matplotlib.pyplot as plt

from hrosailing.plotting.projections import _configure_axes

class TestConfigureAxis(unittest.TestCase):
    def test_without_show_legend(self):
        # Execution test with `show_legend=False`
        _configure_axes(
            plt.subplot(),
            [1, 2, 3],
            ("green", "red"),
            False,
            None,
            marker="H"
        )