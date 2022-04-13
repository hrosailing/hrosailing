from abc import ABC, abstractmethod

import numpy as np

class QualityAssurance(ABC):

    @abstractmethod
    def check(self, polar_diagram, test_data):
        """Method that should test if a given polar_diagram works with given
        preprocessed test data and return a dict containing relevant
        statistics"""


class MinimalQualityAssurance(QualityAssurance):

    def check(self, polar_diagram, test_data):
        """WIP"""
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
