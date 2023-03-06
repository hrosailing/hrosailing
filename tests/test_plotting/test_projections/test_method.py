import unittest
import os

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.testing.compare as plt_compare

from hrosailing.plotting.projections import _plot, _get_convex_hull
from ..image_testcase import ImageTestcase

class TestPlot(ImageTestcase):
    def setUp(self) -> None:
        self.slices = [
            np.array([
                [1, 1, 1, 1, 1, 1], [0, 45, 90, 180, 270, 315], [1, 2, 2, 2, 2, 2]
            ]),
            np.array([
                [2, 2, 2, 2, 2, 2], [0, 45, 90, 180, 270, 315], [10, 1, 2, 2, 2, 1]
            ])
        ]
        self.wa_rad = [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2, 7*np.pi/4]
        self.info = [
            ["A", "A", "B", "B", "B", "B"],
            ["A", "B", "A", "B", "A", "B"]
        ]

    def test_plot(self):
        #Input/Output Tests

        with self.subTest("regular plot"):
            #creating resulting plot
            ax = plt.subplot()
            _plot(ax, self.slices, None, False)
            self.set_result_plot()

            #creating expected plot
            plt.plot([0, 45, 90, 180, 270, 315], [1, 2, 2, 2, 2, 2])
            plt.plot([0, 45, 90, 180, 270, 315], [10, 1, 2, 2, 2, 1])
            self.set_expected_plot()

            self.assertPlotsEqual()

        with self.subTest("using info"):
            #creating resulting plot
            ax = plt.subplot()
            _plot(ax, self.slices, self.info, False)
            self.set_result_plot()

            #creating expected plot
            plt.plot([0, 45], [1, 2], color="C0")
            plt.plot([90, 180, 270, 315], [2, 2, 2, 2], color="C0")
            plt.plot([0, 90, 270], [10, 2, 2], color="C1")
            plt.plot([45, 180, 315], [1, 2, 1], color="C1")
            self.set_expected_plot()

            self.assertPlotsEqual()

        with self.subTest("using radians"):
            #creating resulting plot
            ax = plt.subplot()
            _plot(ax, self.slices, None, True)
            self.set_result_plot()

            #creating expected plot
            plt.plot([0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2, 7*np.pi/4], [1, 2, 2, 2, 2, 2])
            plt.plot([0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2, 7*np.pi/4], [10, 1, 2, 2, 2, 1])
            self.set_expected_plot()

            self.assertPlotsEqual()

        with self.subTest("use convex hull"):
            # creating resulting plot
            ax = plt.subplot()
            _plot(ax, self.slices, None, False, use_convex_hull=True)
            self.set_result_plot()

            # creating expected plot
            plt.plot([0, 45, 90, 180, 270, 315, 360], [2, 2, 2, 2, 2, 2, 2])
            plt.plot([0, 90, 180, 270, 360], [10, 2, 2, 2, 10])
            plt.plot([1,2,3], [1,2,3])
            self.set_expected_plot()

            self.assertPlotsEqual()


class TestGetConvexHull(unittest.TestCase):
    def setUp(self) -> None:
        self.slice_ = np.array([
            [1, 2, 3, 4, 5, 6],
            [0, 45, 90, 135, 180, 270],
            [1, 1, 0, 1, 1, 1]
        ])

    def assertOutputEqual(self, result, expected):
        ws, wa, bsp, info_ = result
        exp_ws, exp_wa, exp_bsp, exp_info_ = expected
        np.testing.assert_array_equal(
            ws, exp_ws,
            err_msg="Wind speeds not as expected!"
        )
        np.testing.assert_array_equal(
            wa, exp_wa,
            err_msg="Wind angles not as expected!"
        )
        np.testing.assert_array_equal(
            bsp, exp_bsp,
            err_msg="Boat speeds not as expected!"
        )
        self.assertEqual(
            info_, exp_info_,
            msg="Info not as expected!"
        )

    def test_get_convex_hull(self):
        # Input/Output Test

        with self.subTest("info is not None"):
            slice_ = np.array([
                [1, 2, 3, 4, 5, 6],
                [0, 45, 90, 135, 180, 270],
                [1, 1, 0, 1, 1, 1]
            ])
            info_ = ["A", "B", "A", "B", "A", "B"]
            result = _get_convex_hull(slice_, info_)
            expected = (
                np.array([1, 2, 4, 5, 6, 1]),
                np.array([0, 45, 135, 180, 270, 360]),
                np.array([1, 1, 1, 1, 1, 1]),
                ["A", "B", "B", "A", "B", "A"]
            )
            self.assertOutputEqual(result, expected)

        with self.subTest("wa given at 0 and 360"):
            slice_ = np.array([
                [1, 2, 3],
                [0, 315, 360],
                [1, 1, 2]
            ])
            result = _get_convex_hull(slice_, None)
            expected = (
                np.array([1, 2, 3]),
                np.array([0, 315, 360]),
                np.array([1, 1, 2]),
                None
            )
            self.assertOutputEqual(result, expected)

        with self.subTest("first and last wa < 180 apart"):
            slice_ = np.array([
                [1, 2, 3],
                [0, 45, 135],
                [1, 1, 1]
            ])
            result = _get_convex_hull(slice_, None)
            expected = (
                np.array([1, 2, 3]),
                np.array([0, 45, 135]),
                np.array([1, 1, 1]),
                None
            )
            self.assertOutputEqual(result, expected)