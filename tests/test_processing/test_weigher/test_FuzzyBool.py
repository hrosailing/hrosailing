from unittest import TestCase
import numpy as np

from hrosailing.processing.weigher import FuzzyBool


class TestFuzzyBool(TestCase):
    def setUp(self) -> None:
        self.ev_fun = lambda x: 1/(1+np.exp(-10 * x))
        self.ev_fun2 = lambda x: 1/(1+np.exp(-5 * x))

    def test_call(self):
        """
        Input/Output-Test.
        """

        result = FuzzyBool(self.ev_fun)(0.25)
        expected_result = 0.924142
        self.assertAlmostEqual(result, expected_result, places=4,
                               msg=f"Expected {expected_result} but got {result}!")

    def test_str(self):
        """
        Input/Output-Test.
        """

        result = str(FuzzyBool(self.ev_fun))
        expected_result = "Fuzzy-Bool"
        self.assertEqual(result, expected_result, f"Expected {expected_result} but got {result}!")

    def test_fuzzy_and(self):
        """
        Input/Output-Test.
        """

        result = FuzzyBool.fuzzy_and(FuzzyBool(self.ev_fun), FuzzyBool(self.ev_fun2))(0.25)
        expected_result = 0.7773
        self.assertAlmostEqual(result, expected_result, places=4,
                               msg=f"Expected {expected_result} but got {result}!")

    def test_fuzzy_or(self):
        """
        Input/Output-Test.
        """

        result = FuzzyBool.fuzzy_or(FuzzyBool(self.ev_fun), FuzzyBool(self.ev_fun2))(0.25)
        expected_result = 0.924142
        self.assertAlmostEqual(result, expected_result, places=4,
                               msg=f"Expected {expected_result} but got {result}!")

    def test_fuzzy_not(self):
        """
        Input/Output-Test.
        """

        result = FuzzyBool.fuzzy_not(FuzzyBool(self.ev_fun2))(0.25)
        expected_result = 1 - .7773
        self.assertAlmostEqual(result, expected_result, places=4,
                               msg=f"Expected {expected_result} but got {result}!")

    def test_sigmoid(self):
        """
        Input/Output-Test.
        """

        result = FuzzyBool.sigmoid(0, 10, -1)(.25)
        expected_result = FuzzyBool(self.ev_fun)(.25)
        self.assertEqual(result, expected_result, f"Expected {expected_result} but got {result}!")

    def test__concat_repr(self):
        """
        Input/Output-Test.
        """
        fuzz = FuzzyBool.fuzzy_and(FuzzyBool(self.ev_fun), FuzzyBool(self.ev_fun2))
        fuzz._concat_repr(FuzzyBool(self.ev_fun), FuzzyBool(self.ev_fun2), "and")

        result = str(fuzz)
        expected_result = "Fuzzy-Bool and Fuzzy-Bool"
        self.assertEqual(result, expected_result, f"Expected {expected_result} but got {result}!")

