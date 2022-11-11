"""
Contains the abstract base class `QualityAssurance` and some ready-to-use implementations aimed to measure the quality
of a polar diagram against test data.
"""

from abc import ABC, abstractmethod

import numpy as np


class QualityAssurance(ABC):
    """
    Base class for all quality assurance classes.
    """

    @abstractmethod
    def check(self, polar_diagram, test_data):
        """Method that should test if a given polar diagram works with given
        preprocessed test data and returns a dict containing relevant
        statistics."""


class MinimalQualityAssurance(QualityAssurance):
    """
    Quality assurance which evaluates the quality of a polar diagram using a small set of well established
    statistical metrics.
    """

    def check(self, polar_diagram, test_data):
        """
        Returns
        -------
        statistics: dict
            Dictionary containing the keys

            - 'max_error' : the maximal absolute difference between the boat speed of a test point the corresponding polar diagram output,
            - 'min_error' : the minimal absolute difference between the boat speed of a test point the corresponding polar diagram output,
            - 'average_error' : the average absolute difference between the boat speed of a test point the corresponding polar diagram output,
            - 'average_quadratic_error' : the average quadratic difference between the boat speed of a test point the corresponding polar diagram output.

        See also
        ---------
        `QualityAssurance.check`
        """
        diffs = [
            abs(bsp - polar_diagram(ws, wa))
            for ws, wa, bsp in test_data
        ]
        statistics = {
            "max_error": max(diffs),
            "min_error": min(diffs),
            "average_error": np.mean(diffs),
            "average_quadratic_error": np.mean([diff**2 for diff in diffs])
        }
        return statistics


class ComformingQualityAssurance(QualityAssurance):
    """
    Quality assurance which evaluates the quality of a polar diagram using a small set of well established
    statistical metrics and additionally quality features derived from the theory of polar diagrams.
    """
    def check(self, polar_diagram, test_data):
        """
        Returns
        -------
        statistics: dict
            Dictionary containing the keys

            - 'max_error' : the maximal absolute difference between the boat speed of a test point the corresponding polar diagram output,
            - 'min_error' : the minimal absolute difference between the boat speed of a test point the corresponding polar diagram output,
            - 'average_error' : the average absolute difference between the boat speed of a test point the corresponding polar diagram output,
            - 'average_quadratic_error' : the average quadratic difference between the boat speed of a test point the corresponding polar diagram output.
            - 'max_zero_val' : the maximal value of the polar diagram for wind angles of 0 or 360 degree and wind speed between 0 and 20,
            - 'min_zero_val': the minimal value of the polar diagram for wind angles of 0 or 360 degree and wind speed between 0 and 20,
            - 'average_zero_val': the average value of the polar diagram for wind angles of 0 or 360 degree and wind speed between 0 and 20,
            - 'average_quadratic_zero_val': the average squared value of the polar diagram for wind angles of 0 or 360 degree and wind speed between 0 and 20,

        See also
        ---------
        `QualityAssurance.check`
        """
        diffs = [
            abs(bsp - polar_diagram(ws, wa))
            for ws, wa, bsp in test_data
        ]
        zero_diffs = [
            abs(polar_diagram(ws, wa))
            for ws in np.linspace(0, 20, 20)
            for wa in [0, 360]
        ]
        statistics = {
            "max_error": max(diffs),
            "min_error": min(diffs),
            "average_error": np.mean(diffs),
            "average_quadratic_error": np.mean([diff**2 for diff in diffs]),
            "max_zero_val": max(zero_diffs),
            "min_zero_val": min(zero_diffs),
            "average_zero_val": np.mean(zero_diffs),
            "average_quadratic_zero_val": np.mean([diff**2 for diff in zero_diffs]),
        }
        return statistics
