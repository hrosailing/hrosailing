from unittest import TestCase
from datetime import timedelta, datetime

import hrosailing.processing.smoother as smt
import hrosailing.core.data as dt


class TestAffineSmoother(TestCase):
    def setUp(self) -> None:
        self.timespan = timedelta(minutes=1)
        self.data = dt.Data().from_dict({
            "datetime": [datetime(2023, 3, 14, 11), datetime(2023, 3, 14, 11, 0), datetime(2023, 3, 14, 11, 0, 30),
                         datetime(2023, 3, 14, 11, 1),  datetime(2023, 3, 14, 11, 1),
                         datetime(2023, 3, 14, 11, 1)],
            "TWS": [12.0, 13.0, 12.0, 16.0, 17.0, 18.0],
            "TWA": [31.0, 34.0, 34.0, 39.0, 39.0, 39.0]
        })
        self.sm_data = {
            "datetime": [datetime(2023, 3, 14, 11), datetime(2023, 3, 14, 11, 0), datetime(2023, 3, 14, 11, 0, 30),
                         datetime(2023, 3, 14, 11, 1),  datetime(2023, 3, 14, 11, 1),
                         datetime(2023, 3, 14, 11, 1)],
            "TWS": [12.0, 13.0, 16.2, 16.0, 17.0, 18.0],
            "TWA": [31.0, 34.0, 36.4, 39.0, 39.0, 39.0]
        }

    def test_smooth_Error(self):
        """
        ValueError if "datetime" is not chronologically ordered.
        """
        with self.assertRaises(ValueError):
            smt.AffineSmoother().smooth(dt.Data().from_dict({"datetime": [datetime(2023, 1, 1), datetime(2000, 1, 1)]}))

    def test_smooth_default(self):
        """
        Input/Output-Test.
        """
        result = smt.AffineSmoother().smooth(self.data)._data
        expected_result = self.sm_data
        self.assertDictEqual(result, expected_result,
                             f"Expected {expected_result} but got {result}!")

    def test_smooth_custom_timespan(self):
        """
        Input/Output-Test.
        """
        # TODO: this does not change the data for some reason
        #  consider changing datetimes for a different result
        result = smt.AffineSmoother(self.timespan).smooth(self.data)._data
        expected_result = dt.Data().from_dict({
            "datetime": [datetime(2023, 3, 14, 11), datetime(2023, 3, 14, 11, 0, 30), datetime(2023, 3, 14, 11, 1),
                         datetime(2023, 3, 14, 11, 1, 30), datetime(2023, 3, 14, 11, 2),
                         datetime(2023, 3, 14, 11, 2, 30)],
            "TWS": [12.0, 12.0, 12.0, 14.4, 18.0, 18.0],
            "TWA": [31.0, 34.0, 34.0, 36.5, 39.0, 39.0]
        })._data
        self.assertDictEqual(result, expected_result,
                             f"Expected {expected_result} but got {result}!")

    def test_smooth_edge_timespan_0(self):
        """
        EdgeCase: timespan is 0.
        """
        result = smt.AffineSmoother(timedelta(seconds=0)).smooth(self.data)._data
        expected_result = self.data._data
        self.assertDictEqual(result, expected_result,
                             f"Expected {expected_result} but got {result}!")
