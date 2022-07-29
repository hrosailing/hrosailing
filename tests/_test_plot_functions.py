import numpy as np

import matplotlib.pyplot as plt


# Tests for Curve:
def plot_polar_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp):
    x_plot = plt.gca().lines[i].get_xdata()
    y_plot = plt.gca().lines[i].get_ydata()
    np.testing.assert_array_equal(x_plot, wa)
    np.testing.assert_array_equal(y_plot, bsp[i])


def plot_flat_comparing_x_plot_wa_y_plot_bsp(i, wa, bsp):
    x_plot = plt.gca().lines[i].get_xdata()
    y_plot = plt.gca().lines[i].get_ydata()
    np.testing.assert_array_equal(x_plot, np.rad2deg(wa))
    np.testing.assert_array_equal(y_plot, bsp[i])


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