from unittest import TestCase
import numpy as np

import hrosailing.processing.regressor as reg


class TestODRegressor(TestCase):
    def setUp(self) -> None:
        self.model_func = lambda ws, wa, *params: params[0] * ws
        self.init_vals = np.array([1.26])
        self.opt_param = 1.3
        self.max_it = 10
        self.data = np.array([[10, 30, 13], [12, 33, 15.6], [14, 36, 18.2]])

    def test_property_model_func(self):
        """
        Input/Output-Test.
        """

        result = reg.ODRegressor(self.model_func, self.init_vals).model_func(1, 2, 1)
        expected_result = 1
        self.assertEqual(result, expected_result,
                         f"Expected {expected_result} but got {result}!")

    def test_property_optimal_params(self):
        """
        Input/Output-Test.
        """

        result = reg.ODRegressor(self.model_func, self.init_vals).optimal_params
        expected_result = None
        self.assertEqual(result, expected_result,
                         f"Expected {expected_result} but got {result}!")

    def test_fit(self):
        """
        Input/Output-Test.
        Testing optimal params after fitting.
        """

        # fitting the regressor
        odr = reg.ODRegressor(self.model_func, self.init_vals)
        odr.fit(self.data)

        # testing optimal params
        result = odr.optimal_params
        expected_result = self.opt_param
        self.assertEqual(result, expected_result,
                         msg=f"Expected {expected_result} but got {result}!")

    def test_fit_custom_max_it(self):
        """
        Input/Output-Test.
        Testing optimal params after fitting.
        """

        # fitting the regressor
        odr = reg.ODRegressor(self.model_func, self.init_vals, self.max_it)
        odr.fit(self.data)

        # testing optimal params
        result = odr.optimal_params
        expected_result = self.opt_param
        self.assertEqual(result, expected_result,
                         msg=f"Expected {expected_result} but got {result}!")

    def test_fit_edge_empty_data(self):
        """
        Test for ValueError
        """

        # fitting the regressor
        odr = reg.ODRegressor(self.model_func, self.init_vals)
        with self.assertRaises(ValueError):
            odr.fit(np.empty((3, 0)))
