import unittest
import numpy as np
import matplotlib.pyplot as plt

import hrosailing.polardiagram as pol


class TablePlotTest(unittest.TestCase):
    def setUp(self):
        self.ws_res = np.array([2, 4, 6, 8])
        self.wa_res = np.array([10, 15, 20, 25])
        self.bsps = np.array(
            [
                [1, 2, 3, 4],
                [1.5, 2.4, 3.1, 4.1],
                [1.7, 2.6, 3.5, 4.4],
                [2, 3, 3.8, 4.6],
            ]
        )
        self.polar_diagram = pol.PolarDiagramTable(
            self.ws_res, self.wa_res, self.bsps
        )

    def test_plot_slice(self):
        f, ax = plt.subplots(subplot_kw={"projection": "polar"})
        self.polar_diagram.plot_polar_slice(2, ax=ax)
        x_plot, y_plot = ax.lines[0].get_xydata().T
        np.testing.assert_array_equal(x_plot, np.deg2rad(self.wa_res))
        np.testing.assert_array_equal(y_plot, self.bsps[:, 0].ravel())

    def test_plot_slices(self):
        f, ax = plt.subplots(subplot_kw={"projection": "polar"})
        self.polar_diagram.plot_polar([2, 4], ax=ax)
        x_plot_1, y_plot_1 = ax.lines[0].get_xydata().T
        x_plot_2, y_plot_2 = ax.lines[1].get_xydata().T
        x_plot = np.column_stack((x_plot_1, x_plot_2))
        y_plot = np.column_stack((y_plot_1, y_plot_2))
        wa = np.deg2rad(np.column_stack((self.wa_res, self.wa_res)))
        np.testing.assert_array_equal(x_plot, wa)
        np.testing.assert_array_equal(
            y_plot, self.bsps[:, [0, 1]].reshape(-1, 2)
        )

    def test_one_color_string(self):
        colors = [
            "green",
            "g",
            "#0f0f0f",
            "#0f0f0f80",
            "0.5",
            "xkcd:sky blue",
            "tab:blue",
            "C3",
        ]
        for i, c in enumerate(colors):
            with self.subTest(i=i):
                try:
                    self.polar_diagram.plot_polar_slice(2, c=c)
                except ValueError:
                    self.fail(f"String {c} didn't work")

        self.assertTrue(True)

    def test_one_color_rbg_tuple(self):
        colors = [(0.1, 0.2, 0.5), (0.1, 0.2, 0.5, 0.3)]
        for i, c in enumerate(colors):
            with self.subTest(i=i):
                try:
                    self.polar_diagram.plot_polar_slice(2, c=c)
                except ValueError:
                    self.fail(f"RGB-Tuple {c} didn't work")

        self.assertTrue(True)

    def test_multiple_colors_string(self):
        colors = [
            ("green", "red", "blue"),
            ("g", "r", "b"),
            ("#0f0f0f", "#0f0f0f80"),
            ("0.5", "0.3"),
            ("xkcd:sky blue", "xkcd:lime green"),
            ("tab:blue", "tab:pink", "tab:olive", "tab:cyan"),
        ]
        for i, c in enumerate(colors):
            with self.subTest(i=i):
                try:
                    self.polar_diagram.plot_polar([2, 4, 6, 8], colors=c)
                except ValueError:
                    self.fail(f"String-Tuple {c} didn't work")

        self.assertTrue(True)

    def test_multiple_colors_rbg_tuple(self):
        colors = [
            ((0.1, 0.2, 0.5), (0.5, 0.1, 0.2), (0.2, 0.5, 0.1)),
            ((0.1, 0.2, 0.5), (0.5, 0.1, 0.2)),
            (
                (0.1, 0.2, 0.5),
                (0.5, 0.1, 0.2),
                (0.2, 0.5, 0.1),
                (0.3, 0.6, 0.2),
            ),
            ((0.1, 0.2, 0.5, 0.3), (0.5, 0.1, 0.2, 0.3), (0.2, 0.5, 0.1, 0.3)),
            ((0.1, 0.2, 0.5, 0.3), (0.5, 0.1, 0.2, 0.3)),
            (
                (0.1, 0.2, 0.5, 0.3),
                (0.5, 0.1, 0.2, 0.3),
                (0.2, 0.5, 0.1, 0.3),
                (0.3, 0.6, 0.2, 0.3),
            ),
        ]
        for i, c in enumerate(colors):
            with self.subTest(i=i):
                try:
                    self.polar_diagram.plot_polar([2, 4, 6], colors=c)
                except ValueError:
                    self.fail(f"RBG-Tuples {c} didn't work")

        self.assertTrue(True)

    def test_plot_color_strings(self):
        colors = [
            ("green", "red"),
            ("g", "r"),
            ("#0f0f0f", "#0f0f0f80"),
            ("0.5", "0.3"),
            ("xkcd:sky blue", "xkcd:lime green"),
            ("tab:blue", "tab:pink"),
        ]
        for i, c in enumerate(colors):
            with self.subTest(i=i):
                try:
                    self.polar_diagram.plot_color_gradient(colors=c)
                except ValueError:
                    self.fail(f"String-Tuple {c} didn't work")

        self.assertTrue(True)

    def test_plot_color_tuples(self):
        colors = [
            ((0.1, 0.2, 0.5), (0.5, 0.1, 0.2)),
            ((0.1, 0.2, 0.5, 0.3), (0.5, 0.1, 0.2, 0.3)),
        ]
        for i, c in enumerate(colors):
            with self.subTest(i=i):
                try:
                    self.polar_diagram.plot_color_gradient(colors=c)
                except ValueError:
                    self.fail(f"RGBA-Tuple {c} didn't work")

        self.assertTrue(True)


def table_plot_suite():
    suite = unittest.TestSuite()
    suite.addTests(
        [
            TablePlotTest("test_plot_slice"),
            TablePlotTest("test_plot_slices"),
            TablePlotTest("test_one_color_string"),
            TablePlotTest("test_one_color_rbg_tuple"),
            TablePlotTest("test_multiple_colors_string"),
            TablePlotTest("test_multiple_colors_rbg_tuple"),
            TablePlotTest("test_plot_color_strings"),
            TablePlotTest("test_plot_color_tuples"),
        ]
    )

    return suite
