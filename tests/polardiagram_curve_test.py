# pylint: disable=missing-docstring
# pylint: disable=too-many-public-methods
# pylint: disable=import-outside-toplevel
import itertools
import math
import unittest

import numpy as np

import matplotlib.pyplot as plt

import hrosailing.polardiagram as pol
from hrosailing.polardiagram._basepolardiagram import (
    PolarDiagramException,
    PolarDiagramInitializationException,
)
import _test_plot_functions as helper_functions


class PolarDiagramCurveTest(unittest.TestCase):
    def setUp(self):
        def func(ws, wa, *params):
            return params[0] * np.asarray(ws) * np.asarray(wa) + params[1]

        self.f = func
        self.params = 1, 2
        self.radians = False
        self.c = pol.PolarDiagramCurve(
            self.f, *self.params, radians=self.radians
        )

    def test_init(self):
        self.assertEqual(self.c.curve.__name__, "func")
        self.assertEqual(self.c.parameters, (1, 2))
        self.assertEqual(self.c.radians, False)

    def test_init_exception_not_callable(self):
        with self.assertRaises(PolarDiagramInitializationException):
            f = 5
            params = 1, 2
            pol.PolarDiagramCurve(f, params)

    def test_not_enough_params(self):
        with self.assertRaises(PolarDiagramInitializationException):
            pol.PolarDiagramCurve(self.f, radians=False)

        with self.assertRaises(PolarDiagramInitializationException):
            pol.PolarDiagramCurve(self.f, 1, radians=False)

    def test_more_params_then_needed(self):
        pol.PolarDiagramCurve(self.f, 1, 2, 3, radians=False)

    def test_curve(self):
        self.assertEqual(self.c.curve.__name__, "func")

    def test_parameters(self):
        self.assertEqual(self.c.parameters, (1, 2))

    def test_radians(self):
        self.assertEqual(self.c.radians, False)

    def test_call_scalar(self):
        import random

        for _ in range(500):
            ws = random.randrange(2, 40)
            wa = random.randrange(0, 360)
            self.assertEqual(self.c(ws, wa), ws * wa + 2)

    def test_call_array(self):
        for _ in range(500):
            ws = np.random.rand(100)
            wa = np.random.rand(100)
            np.testing.assert_array_equal(self.c(ws, wa), ws * wa + 2)

    def test_symmetrize(self):
        import random

        sym_c = self.c.symmetrize()
        for _ in range(500):
            ws = random.randrange(2, 40)
            wa = random.randrange(0, 360)
            np.testing.assert_array_equal(
                sym_c(ws, wa), 1 / 2 * (self.c(ws, wa) + self.c(ws, 360 - wa))
            )

    def test_get_slice(self):
        ws, wa, bsp = self.c.get_slices(10)
        self.assertEqual(ws, [10])
        np.testing.assert_array_equal(
            wa, np.deg2rad(np.linspace(0, 360, 1000))
        )
        np.testing.assert_array_equal(
            bsp[0], self.c(np.array(ws * 1000), np.linspace(0, 360, 1000))
        )

    def test_get_slices_list(self):
        ws, wa, bsp = self.c.get_slices([10, 12, 14])
        self.assertEqual(ws, [10, 12, 14])
        np.testing.assert_array_equal(
            wa, np.deg2rad(np.linspace(0, 360, 1000))
        )
        for i, w in enumerate(ws):
            np.testing.assert_array_equal(
                bsp[i], self.c(np.array([w] * 1000), np.linspace(0, 360, 1000))
            )

    def test_get_slices_tuple(self):
        ws, wa, bsp = self.c.get_slices((10, 15), n_steps=100)
        self.assertEqual(ws, list(np.linspace(10, 15, 100)))
        np.testing.assert_array_equal(
            wa, np.deg2rad(np.linspace(0, 360, 1000))
        )
        for i, w in enumerate(ws):
            np.testing.assert_array_equal(
                bsp[i], self.c(np.array([w] * 1000), np.linspace(0, 360, 1000))
            )

    def test_plot_polar(self):
        plt.close()
        self.c.plot_polar()
        ws, wa, bsp = self.c.get_slices(None)
        for i in range(20):
            with self.subTest(i=i):
                helper_functions.curve_table_plot_polar_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp)

    def test_plot_polar_single_ws(self):
        plt.close()
        self.c.plot_polar(ws=13)
        ws, wa, bsp = self.c.get_slices(ws=13)
        helper_functions.curve_table_plot_polar_comparing_x_plot_wa_y_plot_bsp(0, wa, bsp)

    def test_plot_polar_interval_ws(self):
        plt.close()
        self.c.plot_polar(ws=(10, 20))
        ws, wa, bsp = self.c.get_slices(ws=(10, 20))
        for i in range(10):
            with self.subTest(i=i):
                helper_functions.curve_table_plot_polar_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp)

    def test_plot_polar_iterable_list_ws(self):
        plt.close()
        self.c.plot_polar(ws=[5, 10, 15, 20])
        ws, wa, bsp = self.c.get_slices([5, 10, 15, 20])
        for i in range(4):
            with self.subTest(i=i):
                helper_functions.curve_table_plot_polar_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp)

    def test_plot_polar_iterable_tuple_ws(self):
        plt.close()
        self.c.plot_polar(ws=(5, 10, 15, 20))
        ws, wa, bsp = self.c.get_slices((5, 10, 15, 20))
        for i in range(4):
            with self.subTest(i=i):
                helper_functions.curve_table_plot_polar_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp)

    def test_plot_polar_iterable_set_ws(self):
        plt.close()
        self.c.plot_polar(ws={5, 10, 15, 20})
        ws, wa, bsp = self.c.get_slices({5, 10, 15, 20})
        for i in range(4):
            with self.subTest(i=i):
                helper_functions.curve_table_plot_polar_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp)

    def test_plot_polar_n_steps(self):
        plt.close()
        self.c.plot_polar(ws=(10, 20), n_steps=3)
        ws, wa, bsp = self.c.get_slices(ws=(10, 20), n_steps=3)
        for i in range(3):
            with self.subTest(i=i):
                helper_functions.curve_table_plot_polar_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp)

    # test for plot_polar with given axes:
    # is the axes that is saved at plt.gcf().axes the same that was given?
    # def test_plot_polar_axes_instance(self):

    def test_plot_polar_single_color(self):
        plt.close()
        self.c.plot_polar(colors="purple")
        for i in range(20):
            with self.subTest(i=i):
                self.assertEqual(plt.gca().lines[i].get_color(), "purple")

    def test_plot_polar_two_colors_passed_as_list(self):
        plt.close()
        self.c.plot_polar(ws=[10, 15, 20], colors=["red", "blue"], show_legend=True)
        helper_functions.comparing_colors_two_colors_passed()

    def test_plot_polar_two_colors_passed_as_tuple(self):
        plt.close()
        self.c.plot_polar(ws=[10, 15, 20], colors=("red", "blue"), show_legend=True)
        helper_functions.comparing_colors_two_colors_passed()

    def test_plot_polar_more_than_two_colors_passed(self):
        plt.close()
        self.c.plot_polar(ws=[5, 10, 15, 20], colors=["red", "yellow", "orange"])
        helper_functions.comparing_colors_more_than_two_colors_passed()

    def test_plot_polar_ws_color_pairs_passed(self):
        plt.close()
        self.c.plot_polar(ws=[5, 10, 15], colors=((5, "purple"), (10, "blue"), (15, "red")))
        helper_functions.comparing_colors_ws_color_pairs_passed()

    def test_plot_polar_ws_color_pairs_unsorted_passed(self):
        plt.close()
        self.c.plot_polar(ws=[5, 10, 15], colors=((5, "purple"), (15, "red"), (10, "blue")))
        helper_functions.comparing_colors_ws_color_pairs_passed()

    def test_plot_polar_show_legend(self):
        plt.close()
        self.c.plot_polar(ws=[5, 10, 15], colors=["red", "purple", "blue"], show_legend=True)
        helper_functions.test_curve_comparing_show_legend(self, plt.gca().get_legend())

    def test_plot_polar_legend_kw(self):
        plt.close()
        self.c.plot_polar(ws=[5, 10, 15], colors=["red", "purple", "blue"], show_legend=True,
                          legend_kw={'labels': ["ws 5", "ws 10", "ws 15"], 'loc': 'upper left'})
        helper_functions.test_curve_comparing_legend_keywords(self, plt.gca().get_legend())

    def test_plot_polar_show_colorbar(self):
        plt.close()
        self.c.plot_polar(ws=[2, 4, 6], colors=("red", "blue"), show_legend=True)
        helper_functions.test_comparing_show_colorbar(self, "True Wind Speed")

    # test for plot_polar with given legend keywords for colorbars
    # def test_plot_polar_colorbar_kw(self):

    def test_plot_polar_plot_kw(self):
        plt.close()
        self.c.plot_polar(ls=":", lw=1.5, marker="o")
        for i in range(20):
            with self.subTest(i=i):
                helper_functions.test_comparing_plot_kw(self, i)

    def test_plot_flat(self):
        plt.close()
        self.c.plot_flat()
        ws, wa, bsp = self.c.get_slices(None)
        for i in range(20):
            with self.subTest(i=i):
                helper_functions.curve_table_plot_flat_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp)

    def test_plot_flat_single_ws(self):
        plt.close()
        self.c.plot_flat(ws=13)
        ws, wa, bsp = self.c.get_slices(ws=13)
        helper_functions.curve_table_plot_flat_comparing_x_plot_wa_y_plot_bsp(0, wa, bsp)

    def test_plot_flat_interval_ws(self):
        plt.close()
        self.c.plot_flat(ws=(10, 20))
        ws, wa, bsp = self.c.get_slices(ws=(10, 20))
        for i in range(10):
            with self.subTest(i=i):
                helper_functions.curve_table_plot_flat_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp)

    def test_plot_flat_iterable_list_ws(self):
        plt.close()
        self.c.plot_flat(ws=[5, 10, 15, 20])
        ws, wa, bsp = self.c.get_slices([5, 10, 15, 20])
        for i in range(4):
            with self.subTest(i=i):
                helper_functions.curve_table_plot_flat_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp)

    def test_plot_flat_iterable_tuple_ws(self):
        plt.close()
        self.c.plot_flat(ws=(5, 10, 15, 20))
        ws, wa, bsp = self.c.get_slices((5, 10, 15, 20))
        for i in range(4):
            with self.subTest(i=i):
                helper_functions.curve_table_plot_flat_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp)

    def test_plot_flat_iterable_set_ws(self):
        plt.close()
        self.c.plot_flat(ws={5, 10, 15, 20})
        ws, wa, bsp = self.c.get_slices({5, 10, 15, 20})
        for i in range(4):
            with self.subTest(i=i):
                helper_functions.curve_table_plot_flat_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp)

    def test_plot_flat_n_steps(self):
        plt.close()
        self.c.plot_flat(ws=(10, 20), n_steps=3)
        ws, wa, bsp = self.c.get_slices(ws=(10, 20), n_steps=3)
        for i in range(3):
            with self.subTest(i=i):
                helper_functions.curve_table_plot_flat_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp)

    # test for plot_flat with given axes:
    # is the axes that is saved at plt.gcf().axes the same that was given?
    # def test_plot_flat_axes_instance(self):

    def test_plot_flat_single_color(self):
        plt.close()
        self.c.plot_flat(colors="purple")
        for i in range(20):
            with self.subTest(i=i):
                self.assertEqual(plt.gca().lines[i].get_color(), "purple")

    def test_plot_flat_two_colors_passed_as_list(self):
        plt.close()
        self.c.plot_flat(ws=[10, 15, 20], colors=["red", "blue"])
        helper_functions.comparing_colors_two_colors_passed()

    def test_plot_flat_two_colors_passed_as_tuple(self):
        plt.close()
        self.c.plot_flat(ws=[10, 15, 20], colors=("red", "blue"))
        helper_functions.comparing_colors_two_colors_passed()

    def test_plot_flat_more_than_two_colors_passed(self):
        plt.close()
        self.c.plot_flat(ws=[5, 10, 15, 20], colors=["red", "yellow", "orange"])
        helper_functions.comparing_colors_more_than_two_colors_passed()

    def test_plot_flat_ws_color_pairs_passed(self):
        plt.close()
        self.c.plot_flat(ws=[5, 10, 15], colors=((5, "purple"), (10, "blue"), (15, "red")))
        helper_functions.comparing_colors_ws_color_pairs_passed()

    def test_plot_flat_ws_color_pairs_unsorted_passed(self):
        plt.close()
        self.c.plot_flat(ws=[5, 10, 15], colors=((5, "purple"), (15, "red"), (10, "blue")))
        helper_functions.comparing_colors_ws_color_pairs_passed()

    def test_plot_flat_show_legend(self):
        plt.close()
        self.c.plot_flat(ws=[5, 10, 15], colors=["red", "purple", "blue"], show_legend=True)
        helper_functions.test_curve_comparing_show_legend(self, plt.gca().get_legend())

    def test_plot_flat_legend_kw(self):
        plt.close()
        self.c.plot_flat(ws=[5, 10, 15], colors=["red", "purple", "blue"], show_legend=True,
                         legend_kw={'labels': ["ws 5", "ws 10", "ws 15"], 'loc': 'upper left'})
        helper_functions.test_curve_comparing_legend_keywords(self, plt.gca().get_legend())

    def test_plot_flat_show_colorbar(self):
        plt.close()
        self.c.plot_flat(ws=[2, 4, 6], colors=("red", "blue"), show_legend=True)
        helper_functions.test_comparing_show_colorbar(self, "True Wind Speed")

    # test for plot_flat with given legend keywords for colorbars
    # def test_plot_flat_colorbar_kw(self):

    def test_plot_flat_plot_kw(self):
        plt.close()
        self.c.plot_flat(ls=":", lw=1.5, marker="o")
        for i in range(20):
            with self.subTest(i=i):
                helper_functions.test_comparing_plot_kw(self, i)

    def test_plot_3d(self):
        # test not finished yet
        plt.close()
        self.c.plot_3d()
        wind_speeds = plt.gca().collections[0]._vec[0]
        bsp_sinus_wa = plt.gca().collections[0]._vec[1]
        bsp_cosinus_wa = plt.gca().collections[0]._vec[2]
        print(wind_speeds)
        print(bsp_sinus_wa)
        print(bsp_cosinus_wa)
        wss, was, bsps = self.c.get_slices()
        bsp_sin_wa_results = []
        bsp_cos_wa_results = []
        print(len(wind_speeds))
        for i in range(len(wss)):
            for bsp, wa in zip(bsps[i], was):
                bsp_sin_wa_results.append(bsp * math.sin(wa))
                bsp_cos_wa_results.append(bsp * math.cos(wa))

        for triple in zip(bsp_sinus_wa, bsp_cosinus_wa):
            self.assertIn(triple, [tuple(item) for item in zip(bsp_sin_wa_results, bsp_cos_wa_results)])
        for result in zip(bsp_sin_wa_results, bsp_cos_wa_results):
            self.assertIn(result, [tuple(item) for item in zip(bsp_sinus_wa, bsp_cosinus_wa)])

    def test_plot_3d_axes_instance(self):
        plt.close()
        ax = plt.axes(projection="3d", label="axes label")
        self.c.plot_3d(ax=ax)
        ax.set_xlabel("ws")
        axes = plt.gcf().axes[0]
        self.assertEqual(axes.get_label(), "axes label")
        self.assertEqual(axes.get_xlabel(), "ws")
        self.assertEqual(axes.get_ylabel(), "Polar plane: TWA / BSP ")

    def test_plot_color_gradient(self):
        plt.close()
        self.c.plot_color_gradient()
        _, _, bsp = self.c.get_slices()
        colors = [item[:-1] for item in plt.gca().collections[0]._facecolors]
        helper_functions.curve_plot_color_gradient_calculations(self, bsp, colors)

    def test_plot_color_gradient_ws(self):
        plt.close()
        self.c.plot_color_gradient(ws=(5, 10))
        _, _, bsp = self.c.get_slices(ws=(5, 10))
        colors = [item[:-1] for item in plt.gca().collections[0]._facecolors]
        helper_functions.curve_plot_color_gradient_calculations(self, bsp, colors)

    def test_plot_color_gradient_n_steps(self):
        plt.close()
        self.c.plot_color_gradient(ws=(5, 10), n_steps=10)
        _, _, bsp = self.c.get_slices(ws=(5, 10), n_steps=10)
        colors = [item[:-1] for item in plt.gca().collections[0]._facecolors]
        helper_functions.curve_plot_color_gradient_calculations(self, bsp, colors)

    # test for plot_color_gradient with given axes:
    # is the axes that is saved at plt.gcf().axes the same that was given?
    # def test_plot_color_gradient_axes_instance(self):

    def test_plot_color_gradient_color_pair(self):
        plt.close()
        self.c.plot_color_gradient(colors=("red", "blue"))
        _, _, bsp = self.c.get_slices()
        colors = [item[:-1] for item in plt.gca().collections[0]._facecolors]
        helper_functions.curve_plot_color_gradient_calculations(self, bsp, colors)

    # test for plot_color_gradient when the marker and the marker size are given
    # def test_plot_color_gradient_marker_ms(self):

    def test_plot_color_gradient_show_colorbar(self):
        plt.close()
        self.c.plot_color_gradient(show_legend=True)
        helper_functions.test_comparing_show_colorbar(self, "Boat Speed")

    def test_plot_color_gradient_colorbar_kw(self):
        # test not finished yet
        plt.close()
        self.c.plot_color_gradient(show_legend=True, orientation="horizontal", ticklocation="top")
        colorbar_axes = None
        for axes in plt.gcf().axes:
            if axes.get_label() == "<colorbar>":
                colorbar_axes = axes
        print(colorbar_axes.__dict__)

    def test_plot_convex_hull(self):
        # test not finished yet
        plt.close()
        self.c.plot_convex_hull()

    def test_plot_convex_hull_single_element_ws(self):
        # test not finished yet
        plt.close()
        self.c.plot_convex_hull(ws=10)

    def test_plot_convex_hull_interval_ws(self):
        # test not finished yet
        plt.close()
        self.c.plot_convex_hull(ws=(5, 15))

    def test_plot_convex_hull_iterable_list_ws(self):
        # test not finished yet
        plt.close()
        self.c.plot_convex_hull(ws=[5, 10, 15])

    def test_plot_convex_hull_iterable_tuple_ws(self):
        # test not finished yet
        plt.close()
        self.c.plot_convex_hull(ws=(5, 10, 15))

    def test_plot_convex_hull_iterable_set_ws(self):
        # test not finished yet
        plt.close()
        self.c.plot_convex_hull(ws={5, 10, 15})

    # test for plot_convex_hull with given parameter `n_steps`
    # def test_plot_convex_hull_n_steps(self):

    # test for plot_convex_hull with given axes:
    # is the axes that is saved at plt.gcf().axes the same that was given?
    # def test_plot_convex_hull_axes_instance(self):

    def test_plot_convex_hull_single_color(self):
        plt.close()
        self.c.plot_convex_hull(colors="purple")
        for i in range(20):
            with self.subTest(i=i):
                self.assertEqual(plt.gca().lines[i].get_color(), "purple")

    def test_plot_convex_hull_two_colors_passed_as_list(self):
        plt.close()
        self.c.plot_convex_hull(ws=[10, 15, 20], colors=["red", "blue"])
        helper_functions.comparing_colors_two_colors_passed()

    def test_plot_convex_hull_two_colors_passed_as_tuple(self):
        plt.close()
        self.c.plot_convex_hull(ws=[10, 15, 20], colors=("red", "blue"))
        helper_functions.comparing_colors_two_colors_passed()

    def test_plot_convex_hull_more_than_two_colors_passed(self):
        plt.close()
        self.c.plot_convex_hull(ws=[5, 10, 15, 20], colors=["red", "yellow", "orange"])
        helper_functions.comparing_colors_more_than_two_colors_passed()

    def test_plot_convex_hull_ws_color_pairs_passed(self):
        plt.close()
        self.c.plot_convex_hull(ws=[5, 10, 15], colors=((5, "purple"), (10, "blue"), (15, "red")))
        helper_functions.comparing_colors_ws_color_pairs_passed()

    def test_plot_convex_hull_ws_color_pairs_unsorted_passed(self):
        plt.close()
        self.c.plot_convex_hull(ws=[5, 10, 15], colors=((5, "purple"), (15, "red"), (10, "blue")))
        helper_functions.comparing_colors_ws_color_pairs_passed()

    def test_plot_convex_hull_show_legend(self):
        plt.close()
        self.c.plot_convex_hull(ws=[5, 10, 15], colors=["red", "purple", "blue"], show_legend=True)
        helper_functions.test_curve_comparing_show_legend(self, plt.gca().get_legend())

    def test_plot_convex_hull_legend_kw(self):
        plt.close()
        self.c.plot_convex_hull(ws=[5, 10, 15], colors=["red", "purple", "blue"], show_legend=True,
                                legend_kw={'labels': ["ws 5", "ws 10", "ws 15"], 'loc': 'upper left'})
        helper_functions.test_curve_comparing_legend_keywords(self, plt.gca().get_legend())

    def test_plot_convex_hull_show_colorbar(self):
        plt.close()
        self.c.plot_convex_hull(ws=[2, 4, 6], colors=("red", "blue"), show_legend=True)
        helper_functions.test_comparing_show_colorbar(self, "True Wind Speed")

    # test for plot_convex_hull with given legend keywords for colorbars
    # def test_plot_convex_hull_colorbar_kw(self):

    def test_plot_convex_hull_plot_kw(self):
        plt.close()
        self.c.plot_convex_hull(ls=":", lw=1.5, marker="o")
        for i in range(20):
            with self.subTest(i=i):
                helper_functions.test_comparing_plot_kw(self, i)
