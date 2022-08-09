import numpy as np

import matplotlib.pyplot as plt

import unittest


# functions for more than one PolarDiagram Subclass:
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


def test_comparing_plot_kw(self, i):
    line = plt.gca().lines[i]
    self.assertEqual(line.get_linestyle(), ':')
    self.assertEqual(line.get_linewidth(), 1.5)
    self.assertEqual(line.get_marker(), 'o')


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


def test_cloud_table_comparing_show_legend(self, legend):
    self.assertNotEqual(None, legend)
    labels1 = ['TWS 2', 'TWS 4', 'TWS 6']
    labels2 = ['TWS 2.0', 'TWS 4.0', 'TWS 6.0']
    colors = ['red', 'purple', 'blue']
    handles = legend.__dict__["legendHandles"]
    for i in range(3):
        with self.subTest(i=i):
            self.assertIn(handles[i].get_label(), [labels1[i], labels2[i]])
            self.assertEqual(handles[i].get_color(), colors[i])


def test_cloud_table_comparing_legend_keywords(self, legend):
    texts = legend.__dict__["texts"]
    labels = ['ws 2', 'ws 4', 'ws 6']
    self.assertEqual(legend.__dict__["_loc_real"], 2)
    for i in range(3):
        with self.subTest(i=i):
            self.assertEqual(str(texts[i]), "Text(0, 0, '" + labels[i] + "')")


# functions for Curve:
def test_curve_comparing_show_legend(self, legend):
    self.assertNotEqual(None, legend)
    labels1 = ['TWS 5', 'TWS 10', 'TWS 15']
    labels2 = ['TWS 5.0', 'TWS 10.0', 'TWS 15.0']
    colors = ['red', 'purple', 'blue']
    handles = legend.__dict__["legendHandles"]
    for i in range(3):
        with self.subTest(i=i):
            self.assertIn(handles[i].get_label(), [labels1[i], labels2[i]])
            self.assertEqual(handles[i].get_color(), colors[i])

def test_curve_comparing_legend_keywords(self, legend):
    texts = legend.__dict__["texts"]
    labels = ['ws 5', 'ws 10', 'ws 15']
    self.assertEqual(legend.__dict__["_loc_real"], 2)
    for i in range(3):
        with self.subTest(i=i):
            self.assertEqual(str(texts[i]), "Text(0, 0, '" + labels[i] + "')")



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


def test_multisails_comparing_show_legend(self, legend):
    self.assertNotEqual(None, legend)
    labels1 = ['TWS 42', 'TWS 44', 'TWS 46']
    labels2 = ['TWS 42.0', 'TWS 44.0', 'TWS 46.0']
    colors = ['red', 'purple', 'blue']
    handles = legend.__dict__["legendHandles"]
    for i in range(3):
        with self.subTest(i=i):
            self.assertIn(handles[i].get_label(), [labels1[i], labels2[i]])
            self.assertEqual(handles[i].get_color(), colors[i])


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