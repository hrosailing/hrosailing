import unittest
import numpy as np
import os

from hrosailing.polardiagram import PolarDiagramMultiSails
from tests.test_polardiagram.dummy_classes import DummyPolarDiagram


class TestPolarDiagramMultisails(unittest.TestCase):
    def setUp(self) -> None:
        self.pd = PolarDiagramMultiSails(
            [DummyPolarDiagram(), DummyPolarDiagram()]
        )
        self.pd_named = PolarDiagramMultiSails(
            [DummyPolarDiagram(), DummyPolarDiagram()],
            ["Testsail1", "Testsail2"]
        )

        with open("read_example.csv", "w", encoding="utf-8") as file:
            file.write(
                "Sails:Test1:Test2\n"
            )

        with open("read_example_sail_0.csv", "w", encoding="utf-8") as file:
            file.write(
                "DummyPolarDiagram"
            )

        with open("read_example_sail_1.csv", "w", encoding="utf-8") as file:
            file.write(
                "DummyPolarDiagram"
            )

    def tearDown(self) -> None:
        if os.path.isfile("write_example.csv"):
            os.remove("write_example.csv")
        if os.path.isfile("write_example_sail_0.csv"):
            os.remove("write_example_sail_0.csv")
        if os.path.isfile("write_example_sail_1.csv"):
            os.remove("write_example_sail_1.csv")
        if os.path.isfile("read_example.csv"):
            os.remove("read_example.csv")
        if os.path.isfile("read_example_sail_0.csv"):
            os.remove("read_example_sail_0.csv")
        if os.path.isfile("read_example_sail_1.csv"):
            os.remove("read_example_sail_1.csv")

    def test_init_sails_None(self):
        # Input/Output
        self.assertEqual(
            self.pd._sails, ["Sail 0", "Sail 1"]
        )

    def test_init_not_enough_sails(self):
        # Input/Output
        pd = PolarDiagramMultiSails(
            [DummyPolarDiagram(), DummyPolarDiagram(), DummyPolarDiagram()],
            ["Testsail"]
        )
        self.assertEqual(
            pd._sails, ["Testsail", "Sail 1", "Sail 2"]
        )

    def test_init_too_many_sails(self):
        # Input/Output
        pd = PolarDiagramMultiSails(
            [DummyPolarDiagram()],
            ["Testsail1", "Testsail2", "Testsail3"]
        )
        self.assertEqual(
            pd._sails, ["Testsail1"]
        )

    def test_init_regular(self):
        # Input/Output
        self.assertEqual(
            self.pd_named._sails,
            ["Testsail1", "Testsail2"]
        )

    def test_sails(self):
        # Input/Output
        self.assertEqual(
            self.pd.sails,
            ["Sail 0", "Sail 1"]
        )

    def test_diagrams(self):
        # Test if diagrams is the right length and has the right type
        self.assertEqual(len(self.pd.diagrams), 2)
        self.assertTrue(
            all(
                isinstance(diagram, DummyPolarDiagram) for diagram in self.pd.diagrams
            )
        )

    def test_getitem_illegal_name(self):
        # Exception Test
        with self.assertRaises(ValueError):
            return self.pd["NonExistentSail"]

    def test_getitem_legal_name(self):
        # Execution Test
        return self.pd["Sail 0"]

    def test_str(self):
        # Input/Output
        self.assertEqual(
            str(self.pd),
            "Sail 0\nDummy\n\nSail 1\nDummy\n\n"
        )

    def test_repr(self):
        # Input/Output
        self.assertEqual(
            repr(self.pd),
            "PolarDiagramMultiSails(['Dummy()', 'Dummy()'], ['Sail 0', 'Sail 1'])"
        )

    def test_call(self):
        # Input/Output
        self.assertEqual(
            self.pd(1, 1),
            1
        )

    def test_default_points(self):
        # Input/Output
        np.testing.assert_array_equal(
            self.pd.default_points,
            [[1, 1, 1], [1, 1, 1]]
        )

    def test_get_slices(self):
        # Execution Test
        return self.pd.get_slices([1, 2, 3], 3, False)

    def test_ws_to_slices(self):
        # Input/Output Test
        result = self.pd.ws_to_slices([1, 2, 3])
        np.testing.assert_array_equal(
            result,
            [[[1, 1], [1, 1], [1, 1]]]
        )

    def test_get_slice_info(self):
        # Input/Output Test
        result = self.pd.get_slice_info(
            [1, 2, 3], [np.array([[1, 1], [1, 1], [1, 1]])]
        )
        expected_result = [
            ["Sail 0", "Sail 1"]
        ]
        self.assertEqual(result, expected_result)

    def test_to_csv(self):
        # Input/Output Test, testing the file and the subfiles
        self.pd.to_csv("write_example.csv")
        with open("write_example.csv", "r", encoding="utf-8") as file:
            content = file.read()

        self.assertEqual(
            content,
            "PolarDiagramMultiSails\nSails:Sail 0:Sail 1\n"
        )
        with open("write_example_sail_0.csv", "r", encoding="utf-8") as file:
            content = file.read()

        self.assertEqual(
            content,
            "DummyPolarDiagram"
        )
        with open("write_example_sail_1.csv", "r", encoding="utf-8") as file:
            content = file.read()

        self.assertEqual(
            content,
            "DummyPolarDiagram"
        )

    def test_from_csv(self):
        # Test if sails have been read correctly.
        # Test if diagrams is the right length and has the right type
        with open("read_example.csv", "r", encoding="utf-8") as file:
            pd = PolarDiagramMultiSails.__from_csv__(file)

        self.assertEqual(
            pd.sails,
            ["Test1", "Test2"]
        )

        self.assertEqual(
            len(pd.diagrams),
            2
        )

        self.assertTrue(
            all(
                isinstance(diagram, DummyPolarDiagram) for diagram in self.pd.diagrams
            )
        )

    def test_get_sub_csv_path(self):
        # Input/Output
        result = PolarDiagramMultiSails._get_sub_csv_path(
            "this.is.a.test.path.csv",
            20
        )
        self.assertEqual(
            result,
            "this_sail_20.csv"
        )

    def test_symmetrize(self):
        # Execution test
        return self.pd.symmetrize()

    def test_default_slices(self):
        # Test shape, max and min
        self.assertEqual(
            self.pd.default_slices.shape,
            (20,)
        )
        self.assertEqual(
            self.pd.default_slices.min(),
            1
        )
        self.assertEqual(
            self.pd.default_slices.max(),
            3
        )