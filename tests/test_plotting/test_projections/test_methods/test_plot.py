# pylint: disable-all

import matplotlib.pyplot as plt
import numpy as np

from hrosailing.plotting.projections import _plot
from tests.test_plotting.image_testcase import ImageTestcase


class TestPlot(ImageTestcase):
    def setUp(self):
        self.slices = [
            np.array(
                [
                    [1, 1, 1, 1, 1, 1],
                    [0, 45, 90, 180, 270, 315],
                    [1, 2, 2, 2, 2, 2],
                ]
            ),
            np.array(
                [
                    [2, 2, 2, 2, 2, 2],
                    [0, 45, 90, 180, 270, 315],
                    [10, 1, 2, 2, 2, 1],
                ]
            ),
        ]
        self.wa_rad = [
            0,
            np.pi / 4,
            np.pi / 2,
            np.pi,
            3 * np.pi / 2,
            7 * np.pi / 4,
        ]
        self.info = [
            ["A", "A", "B", "B", "B", "B"],
            ["A", "B", "A", "B", "A", "B"],
        ]

    def test_regular(self):
        ax = plt.subplot()
        _plot(ax, self.slices, None, False)
        self.set_result_plot()

        plt.plot([0, 45, 90, 180, 270, 315], [1, 2, 2, 2, 2, 2])
        plt.plot([0, 45, 90, 180, 270, 315], [10, 1, 2, 2, 2, 1])
        self.set_expected_plot()

        self.assertPlotsEqual()

    def test_different_axis(self):
        ax = plt.subplot(projection="polar")
        _plot(ax, self.slices, None, False)
        self.set_result_plot()

        ax = plt.subplot(projection="polar")
        ax.plot([0, 45, 90, 180, 270, 315], [1, 2, 2, 2, 2, 2])
        ax.plot([0, 45, 90, 180, 270, 315], [10, 1, 2, 2, 2, 1])
        self.set_expected_plot()

        self.assertPlotsEqual()

    def test_using_info(self):
        self.debug = True
        ax = plt.subplot()
        _plot(ax, self.slices, self.info, False)
        self.set_result_plot()

        ax = plt.subplot()
        ax.plot([90, 180, 270, 315], [2, 2, 2, 2])
        ax.plot([0, 45], [1, 2])
        ax.plot([0, 90, 270], [10, 2, 2])
        ax.plot([45, 180, 315], [1, 2, 1])
        self.set_expected_plot()

        self.assertPlotsEqual()

    def test_using_radians(self):
        ax = plt.subplot()
        _plot(ax, self.slices, None, True)
        self.set_result_plot()

        plt.plot(
            [0, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2, 7 * np.pi / 4],
            [1, 2, 2, 2, 2, 2],
        )
        plt.plot(
            [0, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2, 7 * np.pi / 4],
            [10, 1, 2, 2, 2, 1],
        )
        self.set_expected_plot()

        self.assertPlotsEqual()

    def test_using_convex_hull(self):
        ax = plt.subplot()
        _plot(ax, self.slices, None, False, use_convex_hull=True)
        self.set_result_plot()

        # creating expected plot
        plt.plot(
            [0, 45, 90, 180, 270, 315, 360],
            [np.sqrt(2), 2, 2, 2, 2, 2, np.sqrt(2)],
        )
        plt.plot([0, 90, 180, 270, 360], [10, 2, 2, 2, 10])
        self.set_expected_plot()

        self.assertPlotsEqual()

    def test_using_scatter(self):
        ax = plt.subplot()
        _plot(
            ax,
            self.slices,
            None,
            False,
            use_convex_hull=False,
            use_scatter=True,
        )
        self.set_result_plot()

        plt.scatter([0, 45, 90, 180, 270, 315], [1, 2, 2, 2, 2, 2])
        plt.scatter([0, 45, 90, 180, 270, 315], [10, 1, 2, 2, 2, 1])
        self.set_expected_plot()

        self.assertPlotsEqual()

    def test_with_keyword_arguments(self):
        ax = plt.subplot()
        _plot(
            ax,
            self.slices,
            None,
            False,
            alpha=0.1,
            color="blue",
            dashes=[0.1, 0.2, 0.1, 0.2],
            linewidth=10,
            marker="H",
        )
        self.set_result_plot()

        plt.plot(
            [0, 45, 90, 180, 270, 315],
            [1, 2, 2, 2, 2, 2],
            alpha=0.1,
            color="blue",
            dashes=[0.1, 0.2, 0.1, 0.2],
            linewidth=10,
            marker="H",
        )
        plt.plot(
            [0, 45, 90, 180, 270, 315],
            [10, 1, 2, 2, 2, 1],
            alpha=0.1,
            color="blue",
            dashes=[0.1, 0.2, 0.1, 0.2],
            linewidth=10,
            marker="H",
        )
        self.set_expected_plot()

        self.assertPlotsEqual()

    def test_edge_empty_slices(self):
        ax = plt.subplot()
        _plot(ax, [], None, False)
        self.set_result_plot()

        plt.plot([], [])
        self.set_expected_plot()

        self.assertPlotsEqual()
