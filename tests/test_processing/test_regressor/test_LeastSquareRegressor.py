"""
Tests
"""

from unittest import TestCase

import numpy as np

import hrosailing.processing.regressor as reg


class TestLeastSquareRegressor(TestCase):
    def setUp(self) -> None:
        self.model_func = lambda ws, wa, *params: params[0] * ws
        self.init_vals = np.array([1.26])
        self.opt_param = 1.3
        self.data = np.array([[10, 30, 13], [12, 33, 15.6], [14, 36, 18.2]])

    def test_property_model_func(self):
        """
        Input/Output-Test.
        """

        result = reg.LeastSquareRegressor(
            self.model_func, self.init_vals
        ).model_func(1, 2, 1)
        expected_result = 1
        self.assertEqual(
            result,
            expected_result,
            f"Expected {expected_result} but got {result}!",
        )

    def test_property_optimal_params(self):
        """
        Input/Output-Test.
        """

        result = reg.LeastSquareRegressor(
            self.model_func, self.init_vals
        ).optimal_params
        expected_result = None
        self.assertEqual(
            result,
            expected_result,
            f"Expected {expected_result} but got {result}!",
        )

    def test_fit(self):
        """
        Input/Output-Test.
        Testing optimal params after fitting.
        """

        # fitting the regressor
        lsr = reg.LeastSquareRegressor(self.model_func, self.init_vals)
        lsr.fit(self.data)

        # testing optimal params
        result = lsr.optimal_params
        expected_result = self.opt_param
        self.assertEqual(
            result,
            expected_result,
            msg=f"Expected {expected_result} but got {result}!",
        )

    def test_fit_edge_empty_data(self):
        """
        EdgeCase: empty fitting data.
        """

        lsr = reg.LeastSquareRegressor(self.model_func, self.init_vals)
        with self.assertRaises(ValueError):
            lsr.fit(np.empty((0, 3)))

    def test_fit_edge_wrong_size_data(self):
        """
        EdgeCase: fitting data has wrong size.
        """

        lsr = reg.LeastSquareRegressor(self.model_func, self.init_vals)
        with self.assertRaises(IndexError):
            lsr.fit(np.array([[1], [1, 2]], dtype=object))

    def test__get_optimal_parameters(self):
        """
        Execution-Test.
        """
        reg.LeastSquareRegressor(self.model_func)._get_optimal_parameters(
            self.data[:, :2], self.data[:, 2]
        )
