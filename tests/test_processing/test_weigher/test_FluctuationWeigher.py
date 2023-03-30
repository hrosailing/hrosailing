from unittest import TestCase
import numpy as np
from datetime import timedelta, datetime

from hrosailing.processing.weigher import FluctuationWeigher
from hrosailing.core.data import Data


class TestFluctuationWeigher(TestCase):
    def setUp(self) -> None:
        self.dimensions = ["TWS", "TWA"]
        self.time_single = timedelta(minutes=3)
        self.time_tuple = (timedelta(seconds=0), timedelta(minutes=2))
        self.u_b = [2, 1]
        self.data = Data().from_dict({"datetime": [datetime(2023, 3, 28, 11, i) for i in range(3)],
                                      "TWS": [10., 8., 12.], "TWA": [33., 39., 37.]})

    def test_weigh_single_time_single_dim(self):
        """
        Input/Output-Test.
        """
        # TODO: finish once _set_points_from_data is debugged
        result = FluctuationWeigher(self.dimensions[0], self.time_single, self.u_b[0]).weigh(self.data)
        expected_result = [1, 1 / 2, 0.18350341907227397]
        np.testing.assert_array_equal(result, expected_result,
                                      f"Expected {expected_result} but got {result}!")

    def test_weigh_tuple_time_more_dim(self):
        """
        Input/Output-Test.
        """

        result = FluctuationWeigher(self.dimensions, self.time_tuple, self.u_b).weigh(self.data)
        expected_result = [0, 0, 1]
        np.testing.assert_array_equal(result, expected_result,
                                      f"Expected {expected_result} but got {result}!")

    def test_weigh_edge_empty_data(self):
        """
        EdgeCase: Empty array.
        """

        result = FluctuationWeigher(self.dimensions, self.time_single, self.u_b).weigh(Data().from_dict({
            "datetime": [],
            "TWS": [],
            "TWA": []})
        )
        expected_result = []
        np.testing.assert_array_equal(result, expected_result,
                                      f"Expected {expected_result} but got {result}!")
