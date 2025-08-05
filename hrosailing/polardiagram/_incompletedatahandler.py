"""
Methods for handling incomplete polar diagram data with missing values.
"""

import numpy as np


def has_missing_values(bsps):
    """Check if the boat speeds array contains any missing values (NaN).
    
    Parameters
    ----------
    bsps : array-like
        Boat speeds array to check for missing values.
        
    Returns
    -------
    bool
        True if any missing values (NaN) are found, False otherwise.
    """
    for row in bsps:
        for val in row:
            if np.isnan(val):
                return True
    return False


def interpolate_missing_values(ws_res, wa_res, bsps, interpolator=None):
    """Interpolate missing values in boat speeds using nearby valid points.
    
    Parameters
    ----------
    ws_res : array-like
        Wind speed resolution array.
    wa_res : array-like  
        Wind angle resolution array.
    bsps : array-like
        Boat speeds array that may contain missing values (NaN).
    interpolator : Interpolator, optional
        Custom interpolator to use for missing value interpolation.
        If None, uses ArithmeticMeanInterpolator(params=(50,)) as default.
    
    Returns
    -------
    tuple
        (filled_bsps, interpolation_performed)
        - filled_bsps: boat speeds array with interpolated values
        - interpolation_performed: bool indicating if any interpolation was done
    """
    if interpolator is None:
        # Import here to avoid circular imports
        from hrosailing.processing import ArithmeticMeanInterpolator
        # Use ArithmeticMeanInterpolator with params=(50,) as the reliable default
        interpolator = ArithmeticMeanInterpolator(params=(50,))
    
    # Extract valid data points for interpolation
    valid_points = []
    for wa_idx, wa in enumerate(wa_res):
        for ws_idx, ws in enumerate(ws_res):
            bsp = bsps[wa_idx][ws_idx]
            if not np.isnan(bsp):
                valid_points.append((ws, wa, bsp))
    
    # Check if we actually have missing values to interpolate
    has_missing = has_missing_values(bsps)
    
    if not has_missing:
        # No missing values, return original data unchanged
        return bsps, False
    
    if len(valid_points) < 3:
        # Not enough valid points for interpolation, convert NaN to 0 and return
        filled_bsps = [[0.0 if np.isnan(val) else val for val in row] for row in bsps]
        return filled_bsps, True  # We did modify the data (NaN -> 0)
    
    # Convert valid points to the format expected by the library's interpolation
    from hrosailing.core.data import WeightedPoints
    import hrosailing.processing.neighbourhood as nbh
    
    # Create points array: each row is [ws, wa, bsp]
    points_data = []
    for ws, wa, bsp in valid_points:
        points_data.append([ws, wa, bsp])
    
    if not points_data:
        # No valid points, return zeros for missing values
        filled_bsps = [[0.0 if np.isnan(val) else val for val in row] for row in bsps]
        return filled_bsps, True
    
    points_array = np.array(points_data)
    weighted_points = WeightedPoints(points_array, weights=1.0)
    
    # Use a reasonable neighborhood for interpolation
    neighbourhood = nbh.Ball(radius=2.0)
    
    # Fill missing values using the library's interpolation infrastructure
    filled_bsps = []
    for wa_idx, wa in enumerate(wa_res):
        filled_row = []
        for ws_idx, ws in enumerate(ws_res):
            original_val = bsps[wa_idx][ws_idx]
            
            if not np.isnan(original_val):
                # Use original value if available (including 0)
                filled_row.append(original_val)
            else:
                # Use library's interpolation method
                try:
                    grid_point = np.array([ws_res[ws_idx], wa])
                    
                    # Find points in neighborhood
                    point_coords = points_array[:, :2]  # Just ws, wa coordinates
                    considered_points = neighbourhood.is_contained_in(point_coords - grid_point)
                    
                    if np.any(considered_points):
                        # Use library interpolator
                        interpolated_val = interpolator.interpolate(
                            weighted_points[considered_points], grid_point
                        )
                        filled_row.append(max(0.0, interpolated_val))  # Ensure non-negative
                    else:
                        filled_row.append(0.0)  # No nearby points
                except:
                    filled_row.append(0.0)  # Fallback
        
        filled_bsps.append(filled_row)
    
    return filled_bsps, True  # Interpolation was performed