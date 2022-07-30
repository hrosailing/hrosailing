import numpy as np

import matplotlib.pyplot as plt

import unittest


# functions for more than two PolarDiagram Subclass:
def comparing_colors_two_colors_passed():
    np.testing.assert_array_equal(plt.gca().lines[0].get_color(), [1, 0, 0])
    np.testing.assert_array_equal(plt.gca().lines[1].get_color(), [0.5, 0, 0.5])
    np.testing.assert_array_equal(plt.gca().lines[2].get_color(), [0, 0, 1])


def comparing_colors_more_than_two_colors_passed():
    np.testing.assert_array_equal(plt.gca().lines[0].get_color(), "red")
    np.testing.assert_array_equal(plt.gca().lines[1].get_color(), "yellow")
    np.testing.assert_array_equal(plt.gca().lines[2].get_color(), "orange")
    np.testing.assert_array_equal(plt.gca().lines[3].get_color(), "blue")


def comparing_colors_ws_color_pairs_passed():
    np.testing.assert_array_equal(plt.gca().lines[0].get_color(), "purple")
    np.testing.assert_array_equal(plt.gca().lines[1].get_color(), "blue")
    np.testing.assert_array_equal(plt.gca().lines[2].get_color(), "red")


# functions for Curve and Table:
def curve_table_plot_polar_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp):
    x_plot = plt.gca().lines[i].get_xdata()
    y_plot = plt.gca().lines[i].get_ydata()
    np.testing.assert_array_equal(x_plot, wa)
    np.testing.assert_array_equal(y_plot, bsp[i])


def curve_table_plot_flat_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp):
    x_plot = plt.gca().lines[i].get_xdata()
    y_plot = plt.gca().lines[i].get_ydata()
    np.testing.assert_array_equal(x_plot, np.rad2deg(wa))
    np.testing.assert_array_equal(y_plot, bsp[i])


# functions for MultiSails:
def multisails_plot_polar_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp):
    x_plot = plt.gca().lines[i].get_xdata()
    y_plot = plt.gca().lines[i].get_ydata()
    np.testing.assert_array_equal(x_plot, np.deg2rad(wa))
    np.testing.assert_array_equal(y_plot, bsp[i])


def multisails_plot_flat_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp):
    x_plot = plt.gca().lines[i].get_xdata()
    y_plot = plt.gca().lines[i].get_ydata()
    np.testing.assert_array_equal(x_plot, wa)
    np.testing.assert_array_equal(y_plot, bsp[i])


def multisails_comparing_colors_two_colors_passed():
    np.testing.assert_array_equal(plt.gca().lines[0].get_color(), [1, 0, 0])
    np.testing.assert_array_equal(plt.gca().lines[1].get_color(), [0.5, 0, 0.5])
    np.testing.assert_array_equal(plt.gca().lines[2].get_color(), [0, 0, 1])
    np.testing.assert_array_equal(plt.gca().lines[3].get_color(), [1, 0, 0])
    np.testing.assert_array_equal(plt.gca().lines[4].get_color(), [0.5, 0, 0.5])
    np.testing.assert_array_equal(plt.gca().lines[5].get_color(), [0, 0, 1])


def multisails_comparing_colors_more_than_two_colors_passed():
    np.testing.assert_array_equal(plt.gca().lines[0].get_color(), "red")
    np.testing.assert_array_equal(plt.gca().lines[1].get_color(), "yellow")
    np.testing.assert_array_equal(plt.gca().lines[2].get_color(), "orange")
    np.testing.assert_array_equal(plt.gca().lines[3].get_color(), "blue")
    np.testing.assert_array_equal(plt.gca().lines[4].get_color(), "red")
    np.testing.assert_array_equal(plt.gca().lines[5].get_color(), "yellow")
    np.testing.assert_array_equal(plt.gca().lines[6].get_color(), "orange")
    np.testing.assert_array_equal(plt.gca().lines[7].get_color(), "blue")


def multisails_comparing_colors_ws_color_pairs_passed():
    np.testing.assert_array_equal(plt.gca().lines[0].get_color(), "purple")
    np.testing.assert_array_equal(plt.gca().lines[1].get_color(), "blue")
    np.testing.assert_array_equal(plt.gca().lines[2].get_color(), "red")
    np.testing.assert_array_equal(plt.gca().lines[0].get_color(), "purple")
    np.testing.assert_array_equal(plt.gca().lines[1].get_color(), "blue")
    np.testing.assert_array_equal(plt.gca().lines[2].get_color(), "red")


# functions for Pointcloud:
def cloud_plot_polar_comparing_x_plot_wa_y_plot_bsp_single_ws(wa, bsp):
    x_plot = plt.gca().lines[0].get_xdata()
    y_plot = plt.gca().lines[0].get_ydata()
    np.testing.assert_array_equal(x_plot, np.asarray(wa).flat)
    np.testing.assert_array_equal(y_plot, np.asarray(bsp).flat)


def cloud_plot_polar_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp):
    x_plot = plt.gca().lines[i].get_xdata()
    y_plot = plt.gca().lines[i].get_ydata()
    np.testing.assert_array_equal(x_plot, wa[i])
    np.testing.assert_array_equal(y_plot, bsp[i])


def cloud_plot_flat_comparing_x_plot_wa_y_plot_bsp_single_ws(wa, bsp):
    x_plot = plt.gca().lines[0].get_xdata()
    y_plot = plt.gca().lines[0].get_ydata()
    np.testing.assert_array_equal(x_plot, np.rad2deg(np.asarray(wa).flat))
    np.testing.assert_array_equal(y_plot, np.asarray(bsp).flat)


def cloud_plot_flat_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp):
    x_plot = plt.gca().lines[i].get_xdata()
    y_plot = plt.gca().lines[i].get_ydata()
    np.testing.assert_array_equal(x_plot, np.rad2deg(wa[i]))
    np.testing.assert_array_equal(y_plot, bsp[i])


class Testfunctions(unittest.TestCase):
    def test_comparing_plot_kw(self, i):
        line = plt.gca().lines[i]
        self.assertEqual(line.get_linestyle(), ':')
        self.assertEqual(line.get_linewidth(), 1.5)
        self.assertEqual(line.get_marker(), 'o')

    def test_curve_comparing_show_legend(self, legend):
        self.assertNotEqual(None, legend)
        handles = legend.__dict__["legendHandles"]
        self.assertEqual(handles[0].get_label(), 'TWS 5')
        self.assertEqual(handles[0].get_color(), 'red')
        self.assertEqual(handles[1].get_label(), 'TWS 10')
        self.assertEqual(handles[1].get_color(), 'purple')
        self.assertEqual(handles[2].get_label(), 'TWS 15')
        self.assertEqual(handles[2].get_color(), 'blue')

    def test_multisails_comparing_show_legend(self, legend):
        self.assertNotEqual(None, legend)
        handles = legend.__dict__["legendHandles"]
        self.assertEqual(handles[0].get_label(), 'TWS 42.0')
        self.assertEqual(handles[0].get_color(), 'red')
        self.assertEqual(handles[1].get_label(), 'TWS 44.0')
        self.assertEqual(handles[1].get_color(), 'purple')
        self.assertEqual(handles[2].get_label(), 'TWS 46.0')
        self.assertEqual(handles[2].get_color(), 'blue')

    def test_cloud_table_comparing_show_legend(self, legend):
        self.assertNotEqual(None, legend)
        handles = legend.__dict__["legendHandles"]
        self.assertEqual(handles[0].get_label(), 'TWS 2')
        self.assertEqual(handles[0].get_color(), 'red')
        self.assertEqual(handles[1].get_label(), 'TWS 4')
        self.assertEqual(handles[1].get_color(), 'purple')
        self.assertEqual(handles[2].get_label(), 'TWS 6')
        self.assertEqual(handles[2].get_color(), 'blue')