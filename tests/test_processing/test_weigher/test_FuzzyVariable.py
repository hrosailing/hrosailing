from unittest import TestCase
import numpy as np

from hrosailing.processing.weigher import FuzzyVariable
from hrosailing.core.data import Data


class TestFuzzyVariable(TestCase):
    def setUp(self) -> None:
        self.sharpness = 5
        self.key = "TWS"
        self.center = 10
        self.data = Data().from_dict({"TWS": [10], "TWA": [33.]})

    def test_sharpness(self):
        """
        Input/Output-Test.
        """

        result = FuzzyVariable(self.sharpness, self.key).sharpness
        expected_result = 5
        self.assertEqual(result, expected_result, f"Expected {expected_result} but got {result}!")

    def test_truth(self):
        """
        Input/Output-Test.
        """

        result = FuzzyVariable(self.sharpness)._truth(self.center, -1)(10.25)
        expected_result = 0.7773
        self.assertAlmostEqual(result, expected_result, places=4,
                               msg=f"Expected {expected_result} but got {result}!")

    def test_gt(self):
        """
        Input/Output-Test.
        """

        fuzz_var = FuzzyVariable(self.sharpness)
        fuzz_boo = fuzz_var > 0
        result = fuzz_boo(.25)
        expected_result = 0.7773
        self.assertAlmostEqual(result, expected_result, places=4,
                               msg=f"Expected {expected_result} but got {result}!")

    def test_lt(self):
        """
        Input/Output-Test.
        """

        fuzz_var = FuzzyVariable(self.sharpness)
        fuzz_boo = fuzz_var < 0
        result = fuzz_boo(-.25)
        expected_result = 0.7773
        self.assertAlmostEqual(result, expected_result, places=4,
                               msg=f"Expected {expected_result} but got {result}!")

    def test_ge(self):
        """
        Input/Output-Test.
        """

        fuzz_var = FuzzyVariable(self.sharpness)
        fuzz_boo = fuzz_var >= 0
        result = fuzz_boo(.25)
        expected_result = 0.7773
        self.assertAlmostEqual(result, expected_result, places=4,
                               msg=f"Expected {expected_result} but got {result}!")

    def test_le(self):
        """
        Input/Output-Test.
        """

        fuzz_var = FuzzyVariable(self.sharpness)
        fuzz_boo = fuzz_var <= 0
        result = fuzz_boo(-.25)
        expected_result = 0.7773
        self.assertAlmostEqual(result, expected_result, places=4,
                               msg=f"Expected {expected_result} but got {result}!")

    def test_eq(self):
        """
        Input/Output-Test.
        """

        fuzz_var = FuzzyVariable(self.sharpness)
        fuzz_boo = fuzz_var == 0
        result = fuzz_boo(.25)
        expected_result = 1 - 0.7773
        self.assertAlmostEqual(result, expected_result, places=4,
                               msg=f"Expected {expected_result} but got {result}!")

    def test_getitem(self):
        """
        Input/Output-Test.
        """

        result = FuzzyVariable(self.sharpness)[self.key](10.25)
        expected_result = 0.7773
        self.assertAlmostEqual(result, expected_result, places=4,
                               msg=f"Expected {expected_result} but got {result}!")

    def test_call(self):
        """
        Input/Output-Test.
        """

        result = FuzzyVariable()(self.sharpness)
        expected_sharpness = 10
        expected_new_sharpness = self.sharpness
        self.assertEqual(result._next_sharpness, expected_new_sharpness,
                         f"Expected {expected_new_sharpness} but got {result._next_sharpness}!")
        self.assertEqual(result._sharpness, expected_sharpness,
                         f"Expected {expected_sharpness} but got {result._sharpness}!")

    def test_str_key_is_None(self):
        """
        Input/Output-Test.
        """

        result = str(FuzzyVariable())
        expected_result = "x"
        self.assertEqual(result, expected_result, f"Expected {expected_result} but got {result}!")

    def test_str_key_is_not_None(self):
        """
        Input/Output-Test.
        """

        result = str(FuzzyVariable(key=self.key))
        expected_result = f"x[{self.key}]"
        self.assertEqual(result, expected_result, f"Expected {expected_result} but got {result}!")

    def test_repr_key_is_None(self):
        """
        Input/Output-Test.
        """

        result = repr(FuzzyVariable(self.sharpness))
        expected_result = f"x({self.sharpness})"
        self.assertEqual(result, expected_result, f"Expected {expected_result} but got {result}!")

    def test_repr_key_is_not_None(self):
        """
        Input/Output-Test.
        """

        result = repr(FuzzyVariable(self.sharpness, self.key))
        expected_result = f"x({self.sharpness})[{self.key}]"
        self.assertEqual(result, expected_result, f"Expected {expected_result} but got {result}!")


