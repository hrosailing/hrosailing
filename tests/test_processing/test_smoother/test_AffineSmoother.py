# pylint: disable-all

from datetime import datetime, timedelta
from unittest import TestCase

import numpy as np

import hrosailing.core.data as dt
import hrosailing.processing.smoother as smt


class TestAffineSmoother(TestCase):
    def setUp(self) -> None:
        self.timespan = timedelta(seconds=15)
        self.data = dt.Data().from_dict(
            {
                "datetime": [
                    datetime(2023, 3, 29, 11, 0, 0),
                    datetime(2023, 3, 29, 11, 0, 20),
                    datetime(2023, 3, 29, 11, 0, 40),
                    datetime(2023, 3, 29, 11, 1, 0),
                    datetime(2023, 3, 29, 11, 1, 20),
                    datetime(2023, 3, 29, 11, 1, 40),
                ],
                "TWS": [12.0, 12.0, 12.0, 16.0, 16.0, 16.0],
                "TWA": [32.0, 34.0, 34.0, 38.0, 38.0, 38.0],
            }
        )
        self.sm_data = {
            "datetime": [
                datetime(2023, 3, 29, 11, 0, 0),
                datetime(2023, 3, 29, 11, 0, 20),
                datetime(2023, 3, 29, 11, 0, 40),
                datetime(2023, 3, 29, 11, 1, 0),
                datetime(2023, 3, 29, 11, 1, 20),
                datetime(2023, 3, 29, 11, 1, 40),
            ],
            "TWS": [12.0, 12.0, 13.2, 14.8, 16.0, 16.0],
            "TWA": [32, 33.5, 35, 36.8, 38, 38],
        }

    def test_smooth_Error(self):
        """
        ValueError if "datetime" is not chronologically ordered.
        """
        with self.assertRaises(ValueError):
            smt.AffineSmoother().smooth(
                dt.Data().from_dict(
                    {"datetime": [datetime(2023, 1, 1), datetime(2000, 1, 1)]}
                )
            )

    def test_smooth_default(self):
        """
        Input/Output-Test.
        """
        result = smt.AffineSmoother().smooth(self.data)._data
        expected_result = self.sm_data
        self.assertEqual(result["datetime"], expected_result["datetime"])
        np.testing.assert_array_almost_equal(
            result["TWS"],
            expected_result["TWS"],
            err_msg="TWS not as expected!",
        )
        np.testing.assert_array_almost_equal(
            result["TWA"],
            expected_result["TWA"],
            err_msg="TWA not as expected!",
        )

    def test_smooth_custom_timespan(self):
        """
        Input/Output-Test.
        """
        result = smt.AffineSmoother(self.timespan).smooth(self.data)._data
        expected_result = {
            "datetime": [
                datetime(2023, 3, 29, 11, 0, 0),
                datetime(2023, 3, 29, 11, 0, 20),
                datetime(2023, 3, 29, 11, 0, 40),
                datetime(2023, 3, 29, 11, 1, 0),
                datetime(2023, 3, 29, 11, 1, 20),
                datetime(2023, 3, 29, 11, 1, 40),
            ],
            "TWS": [12.0, 12.0, 12.6666666, 15.3333333, 16.0, 16.0],
            "TWA": [32, 33.6666666, 34.6666666, 37.3333333, 38, 38],
        }
        self.assertEqual(result["datetime"], expected_result["datetime"])
        np.testing.assert_array_almost_equal(
            result["TWS"],
            expected_result["TWS"],
            err_msg="TWS not as expected!",
        )
        np.testing.assert_array_almost_equal(
            result["TWA"],
            expected_result["TWA"],
            err_msg="TWA not as expected!",
        )

    def test_smooth_edge_timespan_0(self):
        """
        EdgeCase: timespan is 0.
        """
        result = (
            smt.AffineSmoother(timedelta(seconds=0)).smooth(self.data)._data
        )
        expected_result = self.data._data
        self.assertDictEqual(
            result,
            expected_result,
            f"Expected {expected_result} but got {result}!",
        )
