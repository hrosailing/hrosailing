# pylint: disable-all

from unittest import TestCase

import numpy as np

from hrosailing.core.data import Data
from hrosailing.processing.weigher import FuzzyVariable


class TestFuzzyVariable(TestCase):
    def setUp(self) -> None:
        self.sharpness = 5
        self.key = "TWS"
        self.center = 10
        self.data = Data().from_dict({"TWS": [0.25], "TWA": [33.0]})

    def test_sharpness(self):
        """
        Input/Output-Test.
        """

        result = FuzzyVariable(self.sharpness, self.key).sharpness
        expected_result = 5
        self.assertEqual(
            result,
            expected_result,
            f"Expected {expected_result} but got {result}!",
        )

    def test_truth(self):
        """
        Input/Output-Test.
        """

        result = [
            FuzzyVariable(self.sharpness)._truth(self.center, -1)(10.25),
            FuzzyVariable(self.sharpness)._truth(self.center, -1)(9.5),
            FuzzyVariable(self.sharpness)._truth(self.center, -1)(12),
        ]
        expected_result = [0.7773, 0.07585818002124355, 0.9999546021313]
        np.testing.assert_array_almost_equal(
            result,
            expected_result,
            decimal=4,
            err_msg=f"Expected {expected_result} but got {result}!",
        )

    def test_truth_with_key(self):
        """
        Input/Output-Test.
        """
        result = FuzzyVariable(self.sharpness, "TWS")._truth(self.center, -1)(
            {"TWS": 10.25}
        )
        expected_result = 0.7773
        self.assertAlmostEqual(
            result,
            expected_result,
            places=4,
            msg=f"Expected {expected_result} but got {result}!",
        )

    def test_gt(self):
        """
        Input/Output-Test.
        """

        fuzz_var = FuzzyVariable(self.sharpness)
        fuzz_boo = fuzz_var > 0
        result = fuzz_boo(0.25)
        expected_result = 0.7773
        self.assertAlmostEqual(
            result,
            expected_result,
            places=4,
            msg=f"Expected {expected_result} but got {result}!",
        )

    def test_lt(self):
        """
        Input/Output-Test.
        """

        fuzz_var = FuzzyVariable(self.sharpness)
        fuzz_boo = fuzz_var < 0
        result = fuzz_boo(-0.25)
        expected_result = 0.7773
        self.assertAlmostEqual(
            result,
            expected_result,
            places=4,
            msg=f"Expected {expected_result} but got {result}!",
        )

    def test_ge(self):
        """
        Input/Output-Test.
        """

        fuzz_var = FuzzyVariable(self.sharpness)
        fuzz_boo = fuzz_var >= 0
        result = fuzz_boo(0.25)
        expected_result = 0.7773
        self.assertAlmostEqual(
            result,
            expected_result,
            places=4,
            msg=f"Expected {expected_result} but got {result}!",
        )

    def test_le(self):
        """
        Input/Output-Test.
        """

        fuzz_var = FuzzyVariable(self.sharpness)
        fuzz_boo = fuzz_var <= 0
        result = fuzz_boo(-0.25)
        expected_result = 0.7773
        self.assertAlmostEqual(
            result,
            expected_result,
            places=4,
            msg=f"Expected {expected_result} but got {result}!",
        )

    def test_eq(self):
        """
        Input/Output-Test.
        """

        fuzz_var = FuzzyVariable(self.sharpness)
        fuzz_boo = fuzz_var == 0
        result = fuzz_boo(0.25)
        expected_result = 1 - 0.7773
        self.assertAlmostEqual(
            result,
            expected_result,
            places=4,
            msg=f"Expected {expected_result} but got {result}!",
        )

    def test_getitem(self):
        """
        Input/Output-Test.
        """

        result = FuzzyVariable(self.sharpness)[self.key](self.data)
        expected_result = 0.7773
        self.assertAlmostEqual(
            result,
            expected_result,
            places=4,
            msg=f"Expected {expected_result} but got {result}!",
        )

    def test_call(self):
        """
        Input/Output-Test.
        """

        result = FuzzyVariable()(self.sharpness)
        expected_sharpness = 10
        expected_new_sharpness = self.sharpness
        self.assertEqual(
            result._next_sharpness,
            expected_new_sharpness,
            f"Expected {expected_new_sharpness} but got"
            f" {result._next_sharpness}!",
        )
        self.assertEqual(
            result._sharpness,
            expected_sharpness,
            f"Expected {expected_sharpness} but got {result._sharpness}!",
        )

    def test_str_key_is_None(self):
        """
        Input/Output-Test.
        """

        result = str(FuzzyVariable())
        expected_result = "x"
        self.assertEqual(
            result,
            expected_result,
            f"Expected {expected_result} but got {result}!",
        )

    def test_str_key_is_not_None(self):
        """
        Input/Output-Test.
        """

        result = str(FuzzyVariable(key=self.key))
        expected_result = f"x[{self.key}]"
        self.assertEqual(
            result,
            expected_result,
            f"Expected {expected_result} but got {result}!",
        )

    def test_repr_key_is_None(self):
        """
        Input/Output-Test.
        """

        result = repr(FuzzyVariable(self.sharpness))
        expected_result = f"x({self.sharpness})"
        self.assertEqual(
            result,
            expected_result,
            f"Expected {expected_result} but got {result}!",
        )

    def test_repr_key_is_not_None(self):
        """
        Input/Output-Test.
        """

        result = repr(FuzzyVariable(self.sharpness, self.key))
        expected_result = f"x({self.sharpness})[{self.key}]"
        self.assertEqual(
            result,
            expected_result,
            f"Expected {expected_result} but got {result}!",
        )
