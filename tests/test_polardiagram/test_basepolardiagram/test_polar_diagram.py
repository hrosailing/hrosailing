import unittest
import numpy as np

from hrosailing.polardiagram import PolarDiagram

class DummyPolarDiagram(PolarDiagram):
    def to_csv(self, csv_path):
        pass

    @classmethod
    def __from_csv__(cls, file):
        pass

    def __call__(self, ws, wa):
        pass

    def symmetrize(self):
        pass

    @property
    def default_points(self):
        pass

    @property
    def default_slices(self):
        pass

    def ws_to_slices(self, ws, **kwargs):
        return ws



class TestPolarDiagram(unittest.TestCase):
    def setUp(self) -> None:
        self.pd = DummyPolarDiagram()

    def test_get_slices_without_full_info(self):
        # Input/Output
        ws, slices = self.pd.get_slices(
            ws=[1,2,3],
            n_steps=1,
            full_info=False,
            test_key="test"
        )
        np.testing.assert_array_equal(ws, np.array([1, 1.5, 2, 2.5, 3]))
        np.testing.assert_array_equal(
            slices,
            np.array([1, 1.5, 2, 2.5, 3])
        )

    def test_get_slices_with_full_info(self):
        # Input/Output
        ws, slices, info = self.pd.get_slices(
            ws=[1, 2, 3],
            n_steps=1,
            full_info=True,
            test_key="test"
        )
        np.testing.assert_array_equal(ws, np.array([1, 1.5, 2, 2.5, 3]))
        np.testing.assert_array_equal(
            slices,
            np.array([1, 1.5, 2, 2.5, 3])
        )
        self.assertIsNone(info)
