# pylint: disable-all

import os
import unittest

import matplotlib.pyplot as plt
from matplotlib.testing.compare import compare_images


def save_plot(path):
    plt.savefig(path)
    plt.close()


class ImageTestcase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug = False

    def set_expected_plot(self):
        """saves the plot currently stored in `matplotlib.pyplot`
        as expected plot and resets `matplotlib.pyplot`."""
        save_plot("expected.png")

    def set_result_plot(self):
        """saves the plot currently stored in `matplotlib.pyplot`
        as resulting plot and resets `matplotlib.pyplot`."""
        save_plot("result.png")

    def assertPlotsEqual(self, tol=10):
        """
        Checks if the last plot saved with `set_expected_plot` equals the
        last plot saved with `set_result_plot` up to an accuracy of `tol`.
        """
        res = compare_images("./expected.png", "./result.png", tol=tol)
        self.assertIsNone(res)

    def tearDown(self) -> None:
        super().tearDown()
        if self.debug:
            return
        for path in [
            "./expected.png",
            "./result.png",
            "./result-failed-diff.png",
        ]:
            if os.path.isfile(path):
                os.remove(path)
