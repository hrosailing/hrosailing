from matplotlib.testing.compare import compare_images
import matplotlib.pyplot as plt
import unittest
import os


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

    def assertPlotsEqual(self, tol=0):
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
        os.remove("./expected.png")
        os.remove("./result.png")
        #os.remove("./result-failed-diff.png")
