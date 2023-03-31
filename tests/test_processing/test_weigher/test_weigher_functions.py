"""
Tests
"""

from unittest import TestCase

import numpy as np

import hrosailing.processing.weigher as wgh
from hrosailing.core.data import Data


class TestWeigherFunctions(TestCase):
    def setUp(self) -> None:
        self.dimensions = ["TWS", "TWA", "BSP"]
        self.pts = np.array([[1, 1, 1, 1], [1, 2, 3, 4], [3, 8, 3, 8]])
        self.attrs = ["TWS", "TWA", "BSP", "WVHGT"]
        self.dict = {
            "TWS": [1.0, 1.0, 3.0],
            "TWA": [1.0, 2.0, 8.0],
            "BSP": [1.0, 3.0, 3.0],
            "WVHGT": [1.0, 4.0, 8.0],
        }
        self.data_bsp = Data().from_dict(self.dict)
        self.data_sog = Data().from_dict(
            {
                "TWS": [1.0, 1.0, 3.0],
                "TWA": [1.0, 2.0, 8.0],
                "SOG": [1.0, 3.0, 3.0],
                "WVHGT": [1.0, 4.0, 8.0],
            }
        )

    def test_hrosailing_standard_scaled_euclidean_norm_default(self):
        """
        Input/Output-Test.
        """
        result = wgh.hrosailing_standard_scaled_euclidean_norm()(
            self.pts[:, :2]
        )
        expected_result = [
            0.0500771010481109624044336287,
            0.0503076952118745368142,
            0.151637156266179,
        ]
        np.testing.assert_array_almost_equal(
            result,
            expected_result,
            decimal=4,
            err_msg=f"Expected {expected_result} but got {result}!",
        )

    def test_hrosailing_standard_scaled_euclidean_norm_custom_2dimensions(
        self,
    ):
        """
        Input/Output-Test.
        """
        result = wgh.hrosailing_standard_scaled_euclidean_norm(
            ["TWS", "sth_else"]
        )(self.pts[:, :2])
        expected_result = [
            1.0012492197250,
            2.000624902374255662,
            8.001406126425529714,
        ]
        np.testing.assert_array_almost_equal(
            result,
            expected_result,
            decimal=4,
            err_msg=f"Expected {expected_result} but got {result}!",
        )

    def test_hrosailing_standard_scaled_euclidean_norm_custom_4dimensions(
        self,
    ):
        """
        Input/Output-Test.
        """
        result = wgh.hrosailing_standard_scaled_euclidean_norm(
            ["TWS", "TWA", "sth_else", "sth_else2"]
        )(self.pts)
        expected_result = [1.4150998961, 5.0002530800148, 8.5453492513273]
        np.testing.assert_array_almost_equal(
            result,
            expected_result,
            decimal=4,
            err_msg=f"Expected {expected_result} but got {result}!",
        )

    def test__standard_deviation_of(self):
        """
        Execution-Test.
        """
        wgh._standard_deviation_of(self.pts)

    def test__mean_of(self):
        """
        Execution-Test.
        """
        wgh._mean_of(self.pts)

    def test__normalize_empty_weights(self):
        """
        Input/Output-Test.
        """
        result = wgh._normalize([], lambda x: x)
        expected_result = []
        np.testing.assert_array_equal(
            result,
            expected_result,
            f"Expected {expected_result} but got {result}!",
        )

    def test__normalize_normalizer_0(self):
        """
        Input/Output-Test.
        """
        result = wgh._normalize(self.pts[:, 0], lambda x: 0)
        expected_result = [1, 1, 3]
        np.testing.assert_array_equal(
            result,
            expected_result,
            f"Expected {expected_result} but got {result}!",
        )

    def test__normalize(self):
        """
        Input/Output-Test.
        """
        result = wgh._normalize(self.pts[:, 0], lambda x: max(x))
        expected_result = [1 / 3, 1 / 3, 1]
        np.testing.assert_array_equal(
            result,
            expected_result,
            f"Expected {expected_result} but got {result}!",
        )

    def test__set_points_from_data_nparr_reduce(self):
        """
        Input/Output-Test.
        """
        result = wgh._set_points_from_data(self.pts[:, :3], self.dimensions)
        expected_result = self.dimensions, self.pts[:, :2], self.pts[:, 2]
        np.testing.assert_array_equal(
            result[0],
            expected_result[0],
            f"Expected {expected_result} but got {result}!",
        )
        np.testing.assert_array_equal(
            result[1],
            expected_result[1],
            f"Expected {expected_result} but got {result}!",
        )
        np.testing.assert_array_equal(
            result[2],
            expected_result[2],
            f"Expected {expected_result} but got {result}!",
        )

    def test__set_points_from_data_nparr_reduce_False(self):
        """
        Input/Output-Test.
        """
        result = wgh._set_points_from_data(
            self.pts[:, :3], self.dimensions, False
        )
        expected_result = self.dimensions, self.pts[:, :3]
        np.testing.assert_array_equal(
            result[0],
            expected_result[0],
            f"Expected {expected_result} but got {result}!",
        )
        np.testing.assert_array_equal(
            result[1],
            expected_result[1],
            f"Expected {expected_result} but got {result}!",
        )

    def test__set_points_from_data_data_bsp_reduce_no_attrs(self):
        """
        Input/Output-Test.
        """

        result = wgh._set_points_from_data(self.data_bsp, None)
        expected_result = ["TWS", "TWA", "WVHGT"], np.array(
            [[1, 1, 1], [1, 2, 4], [3, 8, 8]]
        )
        np.testing.assert_array_equal(
            result[0],
            expected_result[0],
            f"Expected {expected_result} but got {result}!",
        )
        np.testing.assert_array_equal(
            result[1],
            expected_result[1],
            f"Expected {expected_result} but got {result}!",
        )

    def test__set_points_from_data_data_sog_reduce_no_attrs(self):
        """
        Input/Output-Test.
        """

        result = wgh._set_points_from_data(self.data_sog, None)
        expected_result = ["TWS", "TWA", "WVHGT"], np.array(
            [[1, 1, 1], [1, 2, 4], [3, 8, 8]]
        )
        np.testing.assert_array_equal(
            result[0],
            expected_result[0],
            f"Expected {expected_result} but got {result}!",
        )
        np.testing.assert_array_equal(
            result[1],
            expected_result[1],
            f"Expected {expected_result} but got {result}!",
        )

    def test__set_points_from_data_Error(self):
        """
        ValueError if data does not contain key "BSP" or "SOG".
        """
        with self.assertRaises(ValueError):
            wgh._set_points_from_data(
                Data().from_dict({"not_speed": [1, 2, 3]}), None
            )

    def test__set_points_from_data_attributes_no_reduce(self):
        """
        Input/Output-Test.
        """
        result = wgh._set_points_from_data(self.dict, self.attrs, False)
        expected_result = self.attrs, self.pts
        np.testing.assert_array_equal(
            result[0],
            expected_result[0],
            f"Expected {expected_result} but got {result}!",
        )
        np.testing.assert_array_equal(
            result[1],
            expected_result[1],
            f"Expected {expected_result} but got {result}!",
        )
