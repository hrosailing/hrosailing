from unittest import TestCase
import numpy as np

from hrosailing.processing.weigher import FuzzyWeigher, FuzzyBool
from hrosailing.core.data import Data


class TestFuzzyWeigher(TestCase):
    def setUp(self) -> None:
        self.fuzzy = FuzzyBool(lambda x: 1/((1+np.exp(-10 * x[0]))*(1+np.exp(-10 * x[1]))))
        self.fuzzy_keys = FuzzyBool(lambda x: 1/((1+np.exp(-10 * x["TWS"]))*(1+np.exp(-10 * x["TWA"]))))
        self.np_arr = np.array([[1, 1], [2, 3], [.25, .1]])
        self.data = Data().from_dict({"TWS": [1., 2., .25], "TWA": [1., 3., .1]})

    def test_weigh_Error(self):
        """
        TypeError if points not np.array or hrosailing.core.data.Data instance.
        """
        with self.assertRaises(TypeError):
            FuzzyWeigher.weigh([1, 2])

    def test_weigh_nparr(self):
        """
        Input/Output-Test.
        """

        result = FuzzyWeigher(self.fuzzy).weigh(self.np_arr)
        expected_result = [0.9999092063235617, 0.9999999979387528, 0.6756018053662156]
        np.testing.assert_array_almost_equal(result, expected_result,
                                             err_msg=f"Expected {expected_result} but got {result}!")

    def test_weigh_data(self):
        """
        Input/Output-Test.
        """

        result = FuzzyWeigher(self.fuzzy_keys).weigh(self.data)
        expected_result = [0.9999092063235617, 0.9999999979387528, 0.6756018053662156]
        np.testing.assert_array_almost_equal(result, expected_result,
                                             err_msg=f"Expected {expected_result} but got {result}!")

    def test_weigh_edge_empty_array(self):
        """
        EdgeCase: Empty array.
        """

        result = FuzzyWeigher(self.fuzzy).weigh(np.array([]))
        expected_result = []
        np.testing.assert_array_equal(result, expected_result,
                                      f"Expected {expected_result} but got {result}!")

    def test_weigh_edge_empty_data(self):
        """
        EdgeCase: Empty array.
        """

        result = FuzzyWeigher(self.fuzzy_keys).weigh(Data())
        expected_result = []
        np.testing.assert_array_equal(result, expected_result,
                                      f"Expected {expected_result} but got {result}!")
