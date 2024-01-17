# pylint: disable-all

import unittest

import numpy as np

from hrosailing.plotting.projections import _get_convex_hull


class TestGetConvexHull(unittest.TestCase):
    def setUp(self):
        self.slice_ = np.array(
            [
                [1, 2, 3, 4, 5, 6],
                [0, 45, 90, 135, 180, 270],
                [1, 1, 0, 1, 1, 1],
            ]
        )

    def assertOutputEqual(self, result, expected):
        ws, wa, bsp, info_ = result
        exp_ws, exp_wa, exp_bsp, exp_info_ = expected

        np.testing.assert_array_equal(ws, exp_ws)
        np.testing.assert_array_equal(wa, exp_wa)
        np.testing.assert_array_equal(bsp, exp_bsp)

        self.assertEqual(info_, exp_info_, msg="Info not as expected!")

    def test_ValueError_in_ConvexHull(self):
        slice_ = np.array([[], [], []])
        result = _get_convex_hull(slice_, None)
        expected = (np.array([]), np.array([]), np.array([]), None)

        self.assertOutputEqual(result, expected)

    def test_QHullError_in_ConvexHull(self):
        slice_ = np.array([[1, 2], [0, 315], [1, 1]])
        result = _get_convex_hull(slice_, None)
        expected = (
            np.array([1, 2]),
            np.array([0, 315]),
            np.array([1, 1]),
            None,
        )

        self.assertOutputEqual(result, expected)

    def test_wa_given_at_0_and_360(self):
        slice_ = np.array([[1, 2, 3], [0, 315, 360], [1, 1, 2]])
        result = _get_convex_hull(slice_, None)
        expected = (
            np.array([1, 2, 3]),
            np.array([0, 315, 360]),
            np.array([1, 1, 2]),
            None,
        )

        self.assertOutputEqual(result, expected)

    def test_first_and_last_wa_less_than_180_apart(self):
        slice_ = np.array([[1, 2, 3], [0, 45, 135], [1, 1, 1]])
        result = _get_convex_hull(slice_, None)
        expected = (
            np.array([1, 2, 3]),
            np.array([0, 45, 135]),
            np.array([1, 1, 1]),
            None,
        )

        self.assertOutputEqual(result, expected)

    def test_wa_given_at_0(self):
        slice_ = np.array(
            [
                [1, 2, 3, 4, 5, 6],
                [0, 45, 90, 135, 180, 270],
                [1, 1, 0, 1, 1, 1],
            ]
        )
        result = _get_convex_hull(slice_, None)
        expected = (
            np.array([1, 2, 4, 5, 6, 1]),
            np.array([0, 45, 135, 180, 270, 360]),
            np.array([1, 1, 1, 1, 1, 1]),
            None,
        )

        self.assertOutputEqual(result, expected)

    def test_wa_given_at_0_info_not_None(self):
        slice_ = np.array(
            [
                [1, 2, 3, 4, 5, 6],
                [0, 45, 90, 135, 180, 270],
                [1, 1, 0, 1, 1, 1],
            ]
        )
        info_ = ["A", "B", "A", "B", "A", "B"]
        result = _get_convex_hull(slice_, info_)
        expected = (
            np.array([1, 2, 4, 5, 6, 1]),
            np.array([0, 45, 135, 180, 270, 360]),
            np.array([1, 1, 1, 1, 1, 1]),
            ["A", "B", "B", "A", "B", "A"],
        )

        self.assertOutputEqual(result, expected)

    def test_wa_given_at_360(self):
        slice_ = np.array(
            [
                [2, 3, 4, 5, 6, 1],
                [45, 90, 135, 180, 270, 360],
                [1, 0, 1, 1, 1, 1],
            ]
        )
        result = _get_convex_hull(slice_, None)
        expected = (
            np.array([1, 2, 4, 5, 6, 1]),
            np.array([0, 45, 135, 180, 270, 360]),
            np.array([1, 1, 1, 1, 1, 1]),
            None,
        )

        self.assertOutputEqual(result, expected)

    def test_wa_given_at_360_info_not_None(self):
        slice_ = np.array(
            [
                [2, 3, 4, 5, 6, 1],
                [45, 90, 135, 180, 270, 360],
                [1, 0, 1, 1, 1, 1],
            ]
        )
        info_ = ["A", "B", "A", "B", "A", "B"]
        result = _get_convex_hull(slice_, info_)
        expected = (
            np.array([1, 2, 4, 5, 6, 1]),
            np.array([0, 45, 135, 180, 270, 360]),
            np.array([1, 1, 1, 1, 1, 1]),
            ["B", "A", "A", "B", "A", "B"],
        )

        self.assertOutputEqual(result, expected)

    def test_no_wa_at_0_no_wa_at_360_more_than_180_apart(self):
        slice_ = np.array(
            [
                [1, 2, 3, 4, 5, 6],
                [0, 45, 90, 180, 270, 315],
                [0, 1, 1, 1, 1, 1],
            ]
        )
        result = _get_convex_hull(slice_, None)
        expected = (
            np.array([4, 2, 3, 4, 5, 6, 4]),
            np.array([0, 45, 90, 180, 270, 315, 360]),
            np.array([1 / np.sqrt(2), 1, 1, 1, 1, 1, 1 / np.sqrt(2)]),
            None,
        )

        self.assertOutputEqual(result, expected)

    def test_no_wa_at_0_no_wa_at_360_more_than_180_apart_info_not_None(self):
        slice_ = np.array(
            [
                [1, 2, 3, 4, 5, 6],
                [0, 45, 90, 180, 270, 315],
                [0, 1, 1, 1, 1, 1],
            ]
        )
        info_ = ["A", "B", "A", "B", "A", "B"]
        result = _get_convex_hull(slice_, info_)
        expected = (
            np.array([4, 2, 3, 4, 5, 6, 4]),
            np.array([0, 45, 90, 180, 270, 315, 360]),
            np.array([1 / np.sqrt(2), 1, 1, 1, 1, 1, 1 / np.sqrt(2)]),
            [None, "B", "A", "B", "A", "B", None],
        )

        self.assertOutputEqual(result, expected)
