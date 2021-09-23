# pylint: disable=missing-docstring
# pylint: disable=too-many-public-methods
# pylint: disable=import-outside-toplevel

import unittest

import matplotlib.pyplot as plt
import numpy as np

import hrosailing.polardiagram as pol


def construct_convex_hull(wa, bsp):
    from scipy.spatial import ConvexHull

    if not isinstance(wa, list):
        wa = [wa]
    if not isinstance(bsp, list):
        bsp = [bsp]
    xs = ys = []
    for a, b in zip(wa, bsp):
        a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
        pts = np.column_stack((a, b))
        polar = np.column_stack(
            (pts[:, 1] * np.cos(pts[:, 0]), pts[:, 1] * np.sin(pts[:, 0]))
        )
        vert = sorted(ConvexHull(polar).vertices)
        x, y = list(zip(*([(a[i], b[i]) for i in vert])))
        x.append(x[0])
        y.append(y[0])
        xs.append(x)
        ys.append(ys)

    return xs, ys


class TablePlotTest(unittest.TestCase):
    def setUp(self):
        self.ws_res = np.arange(2, 42, 2)
        self.wa_res = np.arange(0, 360, 5)
        self.bsps = np.random.rand(72, 20)
        self.pd = pol.PolarDiagramTable(self.ws_res, self.wa_res, self.bsps)

    def test_plot_polar_slice(self):
        f, ax = plt.subplots(subplot_kw={"projection": "polar"})
        self.pd.plot_polar_slice(2, ax=ax)
        x_plot, y_plot = ax.lines[0].get_xydata().T
        np.testing.assert_array_equal(x_plot, np.deg2rad(self.wa_res))
        np.testing.assert_array_equal(y_plot, self.bsps[:, 0].ravel())

    def test_plot_polar(self):
        f, ax = plt.subplots(subplot_kw={"projection": "polar"})
        self.pd.plot_polar([2, 4], ax=ax)
        x_plot_1, y_plot_1 = ax.lines[0].get_xydata().T
        x_plot_2, y_plot_2 = ax.lines[1].get_xydata().T
        x_plot = np.column_stack((x_plot_1, x_plot_2))
        y_plot = np.column_stack((y_plot_1, y_plot_2))
        wa = np.deg2rad(np.column_stack((self.wa_res, self.wa_res)))
        np.testing.assert_array_equal(x_plot, wa)
        np.testing.assert_array_equal(
            y_plot, self.bsps[:, [0, 1]].reshape(-1, 2)
        )

    def test_flat_plot_slice(self):
        f, ax = plt.subplots()
        self.pd.plot_flat_slice(2, ax=ax)
        x_plot, y_plot = ax.lines[0].get_xydata().T
        np.testing.assert_array_equal(x_plot, self.wa_res)
        np.testing.assert_array_equal(y_plot, self.bsps[:, 0].ravel())

    def test_flat_plot(self):
        f, ax = plt.subplots()
        self.pd.plot_flat([2, 4], ax=ax)
        x_plot_1, y_plot_1 = ax.lines[0].get_xydata().T
        x_plot_2, y_plot_2 = ax.lines[1].get_xydata().T
        x_plot = np.column_stack((x_plot_1, x_plot_2))
        y_plot = np.column_stack((y_plot_1, y_plot_2))
        wa = np.column_stack((self.wa_res, self.wa_res))
        np.testing.assert_array_equal(x_plot, wa)
        np.testing.assert_array_equal(
            y_plot, self.bsps[:, [0, 1]].reshape(-1, 2)
        )

    def test_plot_convex_hull_slice(self):
        f, ax = plt.subplots(subplot_kw={"projection": "polar"})
        self.pd.plot_convex_hull_slice(2, ax=ax)
        x_plot, y_plot = ax.lines[0].get_xydata().T
        wa, bsp = construct_convex_hull(
            np.deg2rad(self.wa_res), self.bsps[:, 0]
        )
        np.testing.assert_array_equal(x_plot, wa[0])
        np.testing.assert_array_equal(y_plot, bsp[0])

    def test_plot_convex_hull(self):
        f, ax = plt.subplots(subplot_kw={"projection": "polar"})
        self.pd.plot_convex_hull([2, 4], ax=ax)
        x_plot_1, y_plot_1 = ax.lines[0].get_xydata().T
        x_plot_2, y_plot_2 = ax.lines[1].get_xydata().T
        wa, bsp = construct_convex_hull(
            np.deg2rad(np.column_stack((self.wa_res, self.wa_res))),
            self.bsps[:, [0, 1]],
        )
        np.testing.assert_array_equal(x_plot_1, wa[0])
        np.testing.assert_array_equal(x_plot_2, wa[1])
        np.testing.assert_array_equal(y_plot_1, bsp[0])
        np.testing.assert_array_equal(y_plot_2, bsp[1])

    def test_plot_color_gradient(self):
        f, ax = plt.subplots()
        x_plot, y_plot = ax.lines[0].get_xydata().T
        ws, wa = np.meshgrid(self.ws_res, self.wa_res)
        ws = ws.ravel()
        wa = wa.ravel()
        np.testing.assert_array_equal(x_plot, ws)
        np.testing.assert_array_equal(y_plot, wa)

    def test_one_color_string_works(self):
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
                    self.pd.plot_polar_slice(2, c=c)
                except ValueError:
                    self.fail(f"String {c} didn't work")

    def test_one_color_rbg_tuple(self):
        colors = [(0.1, 0.2, 0.5), (0.1, 0.2, 0.5, 0.3)]
        for i, c in enumerate(colors):
            with self.subTest(i=i):
                try:
                    self.pd.plot_polar_slice(2, c=c)
                except ValueError:
                    self.fail(f"RGB-Tuple {c} didn't work")

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
                    self.pd.plot_polar([2, 4, 6, 8], colors=c)
                except ValueError:
                    self.fail(f"String-Tuple {c} didn't work")

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
                    self.pd.plot_polar([2, 4, 6], colors=c)
                except ValueError:
                    self.fail(f"RBG-Tuples {c} didn't work")

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
                    self.pd.plot_color_gradient(colors=c)
                except ValueError:
                    self.fail(f"String-Tuple {c} didn't work")

    def test_plot_color_tuples(self):
        colors = [
            ((0.1, 0.2, 0.5), (0.5, 0.1, 0.2)),
            ((0.1, 0.2, 0.5, 0.3), (0.5, 0.1, 0.2, 0.3)),
        ]
        for i, c in enumerate(colors):
            with self.subTest(i=i):
                try:
                    self.pd.plot_color_gradient(colors=c)
                except ValueError:
                    self.fail(f"RGBA-Tuple {c} didn't work")


def table_plot_suite():
    suite = unittest.TestSuite()
    suite.addTests(
        [
            TablePlotTest("test_plot_polar_slice"),
            TablePlotTest("test_plot_polar"),
            TablePlotTest("test_plot_flat_slice"),
            TablePlotTest("test_plot_flat"),
            TablePlotTest("test_plot_convex_hull_slice"),
            TablePlotTest("test_plot_convex_hull"),
            TablePlotTest("test_plot_color_gradient"),
            TablePlotTest("test_one_color_string"),
            TablePlotTest("test_one_color_rbg_tuple"),
            TablePlotTest("test_multiple_colors_string"),
            TablePlotTest("test_multiple_colors_rbg_tuple"),
            TablePlotTest("test_plot_color_strings"),
            TablePlotTest("test_plot_color_tuples"),
        ]
    )

    return suite


def test_func(ws, wa, *params):
    return (
        params[0]
        + params[1] * ws
        - params[2] * np.power(ws, 2)
        + params[3] * wa
        + params[5] * np.square(wa - params[4])
        + params[6] * ws * wa
    )


class CurvePlotTest(unittest.TestCase):
    def setUp(self):
        self.curve = test_func
        self.radians = True
        self.params = 1, 1, 1, 1, 1, 1, 1
        self.pd = pol.PolarDiagramCurve(self.curve, self.params, self.radians)

    def test_plot_polar_slice(self):
        f, ax = plt.subplots(subplot_kw={"projection": "polar"})
        self.pd.plot_polar_slice(2, ax=ax)
        x_plot, y_plot = ax.lines[0].get_xydata().T
        wa = np.deg2rad(np.linspace(0, 360, 1000))
        bsp = test_func(np.array([2] * 1000), wa, *self.params).ravel()
        np.testing.assert_array_equal(x_plot, wa)
        np.testing.assert_array_equal(y_plot, bsp)

    def test_plot_polar(self):
        f, ax = plt.subplots(subplot_kw={"projection": "polar"})
        self.pd.plot_polar([2, 4], ax=ax)
        x_plot_1, y_plot_1 = ax.lines[0].get_xydata().T
        x_plot_2, y_plot_2 = ax.lines[1].get_xydata().T
        x_plot = np.column_stack((x_plot_1, x_plot_2))
        y_plot = np.column_stack((y_plot_1, y_plot_2))

    def test_flat_plot_slice(self):
        f, ax = plt.subplots()
        self.pd.plot_flat_slice(2, ax=ax)
        x_plot, y_plot = ax.lines[0].get_xydata().T
        wa = np.deg2rad(np.linspace(0, 360, 1000))
        bsp = test_func(np.array([2] * 1000), wa, *self.params).ravel()
        np.testing.assert_array_equal(x_plot, np.rad2deg(wa))
        np.testing.assert_array_equal(y_plot, bsp)

    def test_flat_plot(self):
        f, ax = plt.subplots()
        self.pd.plot_flat([2, 4], ax=ax)
        x_plot_1, y_plot_1 = ax.lines[0].get_xydata().T
        x_plot_2, y_plot_2 = ax.lines[1].get_xydata().T
        x_plot = np.column_stack((x_plot_1, x_plot_2))
        y_plot = np.column_stack((y_plot_1, y_plot_2))

    def test_plot_convex_hull_slice(self):
        f, ax = plt.subplots(subplot_kw={"projection": "polar"})
        self.pd.plot_convex_hull_slice(2, ax=ax)
        x_plot, y_plot = ax.lines[0].get_xydata().T

    def test_plot_convex_hull(self):
        f, ax = plt.subplots(subplot_kw={"projection": "polar"})
        self.pd.plot_convex_hull([2, 4], ax=ax)
        x_plot_1, y_plot_1 = ax.lines[0].get_xydata().T
        x_plot_2, y_plot_2 = ax.lines[1].get_xydata().T

    def test_plot_color_gradient(self):
        f, ax = plt.subplots()
        self.pd.plot_color_gradient(ax=ax)
        x_plot, y_plot = ax.lines[0].get_xydata().T

    def test_one_color_string_works(self):
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
                    self.pd.plot_polar_slice(2, c=c)
                except ValueError:
                    self.fail(f"String {c} didn't work")

    def test_one_color_rbg_tuple(self):
        colors = [(0.1, 0.2, 0.5), (0.1, 0.2, 0.5, 0.3)]
        for i, c in enumerate(colors):
            with self.subTest(i=i):
                try:
                    self.pd.plot_polar_slice(2, c=c)
                except ValueError:
                    self.fail(f"RGB-Tuple {c} didn't work")

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
                    self.pd.plot_polar([2, 4, 6, 8], colors=c)
                except ValueError:
                    self.fail(f"String-Tuple {c} didn't work")

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
                    self.pd.plot_polar([2, 4, 6], colors=c)
                except ValueError:
                    self.fail(f"RBG-Tuples {c} didn't work")

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
                    self.pd.plot_color_gradient(colors=c)
                except ValueError:
                    self.fail(f"String-Tuple {c} didn't work")

    def test_plot_color_tuples(self):
        colors = [
            ((0.1, 0.2, 0.5), (0.5, 0.1, 0.2)),
            ((0.1, 0.2, 0.5, 0.3), (0.5, 0.1, 0.2, 0.3)),
        ]
        for i, c in enumerate(colors):
            with self.subTest(i=i):
                try:
                    self.pd.plot_color_gradient(colors=c)
                except ValueError:
                    self.fail(f"RGBA-Tuple {c} didn't work")


def curve_plot_suite():
    suite = unittest.TestSuite()
    suite.addTests(
        [
            CurvePlotTest("test_plot_polar_slice"),
            CurvePlotTest("test_plot_polar"),
            CurvePlotTest("test_plot_flat_slice"),
            CurvePlotTest("test_plot_flat"),
            CurvePlotTest("test_plot_convex_hull_slice"),
            CurvePlotTest("test_plot_convex_hull"),
            CurvePlotTest("test_plot_color_gradient"),
            CurvePlotTest("test_one_color_string"),
            CurvePlotTest("test_one_color_rbg_tuple"),
            CurvePlotTest("test_multiple_colors_string"),
            CurvePlotTest("test_multiple_colors_rbg_tuple"),
            CurvePlotTest("test_plot_color_strings"),
            CurvePlotTest("test_plot_color_tuples"),
        ]
    )

    return suite


class PointcloudPlotTest(unittest.TestCase):
    def setUp(self):
        self.ws_res = np.arange(2, 42, 2)
        self.wa_res = np.arange(0, 360, 5)
        self.bsps = np.random.rand(72, 20)
        self.pd = pol.PolarDiagramTable(self.ws_res, self.wa_res, self.bsps)

    def test_plot_polar_slice(self):
        f, ax = plt.subplots(subplot_kw={"projection": "polar"})
        self.pd.plot_polar_slice(2, ax=ax)
        x_plot, y_plot = ax.lines[0].get_xydata().T
        np.testing.assert_array_equal(x_plot, np.deg2rad(self.wa_res))
        np.testing.assert_array_equal(y_plot, self.bsps[:, 0].ravel())

    def test_plot_polar(self):
        f, ax = plt.subplots(subplot_kw={"projection": "polar"})
        self.pd.plot_polar([2, 4], ax=ax)
        x_plot_1, y_plot_1 = ax.lines[0].get_xydata().T
        x_plot_2, y_plot_2 = ax.lines[1].get_xydata().T
        x_plot = np.column_stack((x_plot_1, x_plot_2))
        y_plot = np.column_stack((y_plot_1, y_plot_2))
        wa = np.deg2rad(np.column_stack((self.wa_res, self.wa_res)))
        np.testing.assert_array_equal(x_plot, wa)
        np.testing.assert_array_equal(
            y_plot, self.bsps[:, [0, 1]].reshape(-1, 2)
        )

    def test_flat_plot_slice(self):
        f, ax = plt.subplots()
        self.pd.plot_flat_slice(2, ax=ax)
        x_plot, y_plot = ax.lines[0].get_xydata().T
        np.testing.assert_array_equal(x_plot, self.wa_res)
        np.testing.assert_array_equal(y_plot, self.bsps[:, 0].ravel())

    def test_flat_plot(self):
        f, ax = plt.subplots()
        self.pd.plot_flat([2, 4], ax=ax)
        x_plot_1, y_plot_1 = ax.lines[0].get_xydata().T
        x_plot_2, y_plot_2 = ax.lines[1].get_xydata().T
        x_plot = np.column_stack((x_plot_1, x_plot_2))
        y_plot = np.column_stack((y_plot_1, y_plot_2))
        wa = np.column_stack((self.wa_res, self.wa_res))
        np.testing.assert_array_equal(x_plot, wa)
        np.testing.assert_array_equal(
            y_plot, self.bsps[:, [0, 1]].reshape(-1, 2)
        )

    def test_plot_convex_hull_slice(self):
        f, ax = plt.subplots(subplot_kw={"projection": "polar"})
        self.pd.plot_convex_hull_slice(2, ax=ax)
        x_plot, y_plot = ax.lines[0].get_xydata().T
        wa, bsp = construct_convex_hull(
            np.deg2rad(self.wa_res), self.bsps[:, 0]
        )
        np.testing.assert_array_equal(x_plot, wa[0])
        np.testing.assert_array_equal(y_plot, bsp[0])

    def test_plot_convex_hull(self):
        f, ax = plt.subplots(subplot_kw={"projection": "polar"})
        self.pd.plot_convex_hull([2, 4], ax=ax)
        x_plot_1, y_plot_1 = ax.lines[0].get_xydata().T
        x_plot_2, y_plot_2 = ax.lines[1].get_xydata().T
        wa, bsp = construct_convex_hull(
            np.deg2rad(np.column_stack((self.wa_res, self.wa_res))),
            self.bsps[:, [0, 1]],
        )
        np.testing.assert_array_equal(x_plot_1, wa[0])
        np.testing.assert_array_equal(x_plot_2, wa[1])
        np.testing.assert_array_equal(y_plot_1, bsp[0])
        np.testing.assert_array_equal(y_plot_2, bsp[1])

    def test_plot_color_gradient(self):
        f, ax = plt.subplots()
        x_plot, y_plot = ax.lines[0].get_xydata().T
        ws, wa = np.meshgrid(self.ws_res, self.wa_res)
        ws = ws.ravel()
        wa = wa.ravel()
        np.testing.assert_array_equal(x_plot, ws)
        np.testing.assert_array_equal(y_plot, wa)

    def test_one_color_string_works(self):
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
                    self.pd.plot_polar_slice(2, c=c)
                except ValueError:
                    self.fail(f"String {c} didn't work")

    def test_one_color_rbg_tuple(self):
        colors = [(0.1, 0.2, 0.5), (0.1, 0.2, 0.5, 0.3)]
        for i, c in enumerate(colors):
            with self.subTest(i=i):
                try:
                    self.pd.plot_polar_slice(2, c=c)
                except ValueError:
                    self.fail(f"RGB-Tuple {c} didn't work")

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
                    self.pd.plot_polar([2, 4, 6, 8], colors=c)
                except ValueError:
                    self.fail(f"String-Tuple {c} didn't work")

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
                    self.pd.plot_polar([2, 4, 6], colors=c)
                except ValueError:
                    self.fail(f"RBG-Tuples {c} didn't work")

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
                    self.pd.plot_color_gradient(colors=c)
                except ValueError:
                    self.fail(f"String-Tuple {c} didn't work")

    def test_plot_color_tuples(self):
        colors = [
            ((0.1, 0.2, 0.5), (0.5, 0.1, 0.2)),
            ((0.1, 0.2, 0.5, 0.3), (0.5, 0.1, 0.2, 0.3)),
        ]
        for i, c in enumerate(colors):
            with self.subTest(i=i):
                try:
                    self.pd.plot_color_gradient(colors=c)
                except ValueError:
                    self.fail(f"RGBA-Tuple {c} didn't work")


def pointcloud_plot_suite():
    suite = unittest.TestSuite()
    suite.addTests(
        [
            PointcloudPlotTest("test_plot_polar_slice"),
            PointcloudPlotTest("test_plot_polar"),
            PointcloudPlotTest("test_plot_flat_slice"),
            PointcloudPlotTest("test_plot_flat"),
            PointcloudPlotTest("test_plot_convex_hull_slice"),
            PointcloudPlotTest("test_plot_convex_hull"),
            PointcloudPlotTest("test_plot_color_gradient"),
            PointcloudPlotTest("test_one_color_string"),
            PointcloudPlotTest("test_one_color_rbg_tuple"),
            PointcloudPlotTest("test_multiple_colors_string"),
            PointcloudPlotTest("test_multiple_colors_rbg_tuple"),
            PointcloudPlotTest("test_plot_color_strings"),
            PointcloudPlotTest("test_plot_color_tuples"),
        ]
    )

    return suite
