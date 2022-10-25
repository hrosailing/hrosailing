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
        preprocessed test data and return a dict containing relevant
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

            - 'max_error' : The maximal absolute difference between the boat speed of a test point the corresponding polar diagram output,
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
