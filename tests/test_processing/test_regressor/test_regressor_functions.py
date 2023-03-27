from unittest import TestCase

import hrosailing.processing.regressor as reg


class TestRegressorFunctions(TestCase):
    def setUp(self) -> None:
        self.func = lambda x, *params: params[0] + params[1] * x
        self.params = [1, 2, 3, 4]

    def test__determine_params(self):
        """
        Input/Output-Test.
        """
        result = reg._determine_params(self.func)
        expected_result = [1, 1]
        self.assertEqual(result, expected_result,
                                      f"Expected {expected_result} but got {result}!")