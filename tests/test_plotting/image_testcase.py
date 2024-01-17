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
        save_plot("expected.png")

    def set_result_plot(self):
        save_plot("result.png")

    def assertPlotsEqual(self, tol=10):
        res = compare_images("./expected.png", "./result.png", tol=tol)
        self.assertIsNone(res)

    def tearDown(self):
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
