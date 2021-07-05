"""
Small utility functions used throughout the module
"""

# Author: Valentin F. Dannenberg / Ente

# TODO Move to Pipelinecomponents or Remove entirely

import numpy as np


def euclidean_norm(vec):
    return np.linalg.norm(vec, axis=1)
