# pylint: disable-all

import os
import unittest
import tempfile
import numpy as np

from hrosailing.polardiagram._incompletedatahandler import (
    has_missing_values,
    interpolate_missing_values,
)
from hrosailing.polardiagram._reading import from_csv
from hrosailing.processing import ArithmeticMeanInterpolator, IDWInterpolator


class TestHasMissingValues(unittest.TestCase):
    """Test the has_missing_values function."""
    
    def test_no_missing_values(self):
        """Test data with no missing values."""
        bsps = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ]
        self.assertFalse(has_missing_values(bsps))
    
    def test_with_missing_values(self):
        """Test data with missing values (NaN)."""
        bsps = [
            [1.0, 2.0, np.nan],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ]
        self.assertTrue(has_missing_values(bsps))


class TestInterpolateMissingValues(unittest.TestCase):
    """Test the interpolate_missing_values function."""
    
    def setUp(self):
        self.ws_res = [6, 8, 10, 12, 14]
        self.wa_res = [30, 45, 60, 75, 90]
    
    def test_no_missing_values(self):
        """Test interpolation when no values are missing."""
        bsps = [
            [3.0, 3.2, 3.4, 3.6, 3.8],
            [4.0, 4.2, 4.4, 4.6, 4.8],
            [4.5, 4.7, 4.9, 5.1, 5.3],
            [4.8, 5.0, 5.2, 5.4, 5.6],
            [4.6, 4.8, 5.0, 5.2, 5.4]
        ]
        filled_bsps, interpolation_performed = interpolate_missing_values(
            self.ws_res, self.wa_res, bsps
        )
        
        self.assertFalse(interpolation_performed)
        self.assertEqual(filled_bsps, bsps)
    
    def test_with_missing_values(self):
        """Test interpolation when values are missing."""
        bsps = [
            [3.0, np.nan, 3.4, 3.6, 3.8],
            [4.0, 4.2, np.nan, 4.6, 4.8],
            [4.5, 4.7, 4.9, np.nan, 5.3],
            [4.8, 5.0, 5.2, 5.4, 5.6],
            [4.6, 4.8, 5.0, 5.2, np.nan]
        ]
        filled_bsps, interpolation_performed = interpolate_missing_values(
            self.ws_res, self.wa_res, bsps
        )
        
        self.assertTrue(interpolation_performed)
        # Check that no NaN values remain
        for row in filled_bsps:
            for val in row:
                self.assertFalse(np.isnan(val))
                self.assertGreaterEqual(val, 0.0)  # All values should be non-negative
    
    def test_custom_interpolator(self):
        """Test interpolation with custom interpolator."""
        bsps = [
            [3.0, np.nan, 3.4, 3.6, 3.8],
            [4.0, 4.2, np.nan, 4.6, 4.8],
            [4.5, 4.7, 4.9, np.nan, 5.3],
            [4.8, 5.0, 5.2, 5.4, 5.6],
            [4.6, 4.8, 5.0, 5.2, np.nan]
        ]
        
        custom_interpolator = ArithmeticMeanInterpolator(params=(25,))
        filled_bsps, interpolation_performed = interpolate_missing_values(
            self.ws_res, self.wa_res, bsps, custom_interpolator
        )
        
        self.assertTrue(interpolation_performed)
        # Check that no NaN values remain
        for row in filled_bsps:
            for val in row:
                self.assertFalse(np.isnan(val))
                self.assertGreaterEqual(val, 0.0)


class TestFromCSVWithInterpolation(unittest.TestCase):
    """Test the from_csv function with interpolation features."""
    
    def setUp(self):
        """Create temporary incomplete CSV files for testing."""
        # Create incomplete ORC format file
        self.incomplete_orc_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        orc_content = [
            "twa/tws;6;8;10;12;14\n",
            "0;0;0;0;0;0\n",
            "30;;4.2;4.5;4.8;\n",
            "45;3.8;;4.1;4.4;4.7\n",
            "60;4.0;4.3;;4.9;5.2\n",
            "90;4.2;4.5;;5.3;5.6\n",
            "120;;4.1;4.4;;5.0\n"
        ]
        self.incomplete_orc_file.writelines(orc_content)
        self.incomplete_orc_file.close()
        
        # Create complete ORC format file for comparison
        self.complete_orc_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        complete_content = [
            "twa/tws;6;8;10;12;14\n",
            "0;0;0;0;0;0\n",
            "30;4.0;4.2;4.5;4.8;5.1\n",
            "45;3.8;4.0;4.1;4.4;4.7\n",
            "60;4.0;4.3;4.6;4.9;5.2\n",
            "90;4.2;4.5;4.8;5.3;5.6\n",
            "120;3.9;4.1;4.4;4.7;5.0\n"
        ]
        self.complete_orc_file.writelines(complete_content)
        self.complete_orc_file.close()
    
    def test_load_incomplete_without_interpolation_uses_zeros(self):
        """Test loading incomplete file without interpolation flag uses zeros for missing values."""
        pd = from_csv(self.incomplete_orc_file.name, fmt="orc", interpolate_missing=False)
        
        # Should not have performed interpolation
        self.assertFalse(pd.interpolation_performed)
        
        # Should have zeros where data was missing (backward compatible behavior)
        has_zeros = False
        for row in pd.boat_speeds:
            for val in row:
                if val == 0.0:
                    has_zeros = True
                    break
            if has_zeros:
                break
        
        self.assertTrue(has_zeros, "Should contain zeros for missing values when interpolation is disabled")
    
    def test_load_incomplete_with_interpolation(self):
        """Test loading incomplete file with interpolation enabled."""
        pd = from_csv(self.incomplete_orc_file.name, fmt="orc", interpolate_missing=True)
        
        # Should have no NaN values in the boat speeds
        has_nan = False
        for row in pd.boat_speeds:
            for val in row:
                if np.isnan(val):
                    has_nan = True
                    break
            if has_nan:
                break
        
        self.assertFalse(has_nan, "Should not contain NaN values when interpolation is enabled")
        self.assertTrue(pd.interpolation_performed, "Interpolation flag should be True")
        
        # All values should be non-negative
        for row in pd.boat_speeds:
            for val in row:
                self.assertGreaterEqual(val, 0.0)
    
    def test_interpolation_preserves_original_values(self):
        """Test that interpolation preserves existing values."""
        pd = from_csv(self.incomplete_orc_file.name, fmt="orc", interpolate_missing=True)
        
        # Check that known values are preserved (within tolerance due to floating point)
        # Row 1 (45°): original values at positions [0]=3.8, [2]=4.1, [3]=4.4, [4]=4.7
        self.assertAlmostEqual(pd.boat_speeds[1][0], 3.8, places=5)
        self.assertAlmostEqual(pd.boat_speeds[1][2], 4.1, places=5)
        self.assertAlmostEqual(pd.boat_speeds[1][3], 4.4, places=5)
        self.assertAlmostEqual(pd.boat_speeds[1][4], 4.7, places=5)
    
    def test_symmetrization_preserves_interpolation_flag(self):
        """Test that symmetrization preserves the interpolation flag."""
        pd = from_csv(self.incomplete_orc_file.name, fmt="orc", interpolate_missing=True)
        self.assertTrue(pd.interpolation_performed)
        
        # Symmetrize the polar diagram
        symmetric_pd = pd.symmetrize()  
        
        # Interpolation flag should be preserved
        self.assertTrue(symmetric_pd.interpolation_performed)
        self.assertTrue(symmetric_pd.symmetrization_performed)
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.incomplete_orc_file.name):
            os.unlink(self.incomplete_orc_file.name)
        if os.path.exists(self.complete_orc_file.name):
            os.unlink(self.complete_orc_file.name)


class TestIncompleteDataExample(unittest.TestCase):
    """Test using the provided incomplete_orc_format_example.csv file."""
    
    def setUp(self):
        """Set up path to the example incomplete file."""
        test_dir = os.path.dirname(__file__)
        self.example_file = os.path.join(os.path.dirname(test_dir), "incomplete_orc_format_example.csv")
    
    def test_example_file_exists(self):
        """Test that the example incomplete file exists."""
        self.assertTrue(os.path.exists(self.example_file), 
                       f"Example file should exist at {self.example_file}")
    
    def test_load_example_file_with_interpolation(self):
        """Test loading the example incomplete file with interpolation."""
        if not os.path.exists(self.example_file):
            self.skipTest("Example file not found")
        
        pd = from_csv(self.example_file, fmt="orc", interpolate_missing=True)
        
        # Should have interpolated missing values
        self.assertTrue(pd.interpolation_performed)
        
        # Should have no NaN values
        has_nan = False
        for row in pd.boat_speeds:
            for val in row:
                if np.isnan(val):
                    has_nan = True
                    break
            if has_nan:
                break
        
        self.assertFalse(has_nan, "Example file should have no NaN values after interpolation")
        
        # Check that we have expected dimensions (10 rows, 7 wind speeds)
        self.assertEqual(len(pd.boat_speeds), 10, "Should have 10 wind angle rows")
        self.assertEqual(len(pd.boat_speeds[0]), 7, "Should have 7 wind speed columns")
        
        # All interpolated values should be non-negative
        for row in pd.boat_speeds:
            for val in row:
                self.assertGreaterEqual(val, 0.0, "All boat speeds should be non-negative")
    
    def test_load_example_file_without_interpolation_uses_zeros(self):
        """Test loading the example incomplete file without interpolation uses zeros."""
        if not os.path.exists(self.example_file):
            self.skipTest("Example file not found")
        
        pd = from_csv(self.example_file, fmt="orc", interpolate_missing=False)
        
        # Should not have performed interpolation
        self.assertFalse(pd.interpolation_performed)
        
        # Should have zeros where data was missing (backward compatible behavior)
        has_zeros = False
        for row in pd.boat_speeds:
            for val in row:
                if val == 0.0:
                    has_zeros = True
                    break
            if has_zeros:
                break
        
        self.assertTrue(has_zeros, "Example file should contain zeros for missing values when interpolation is disabled")


if __name__ == '__main__':
    unittest.main()