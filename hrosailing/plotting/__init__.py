"""
Functions for plotting support of instances of `hrosailing` objects.
Currently, plots of `PolarDiagram` instances are supported.
"""

import hrosailing.plotting.projections
from hrosailing.plotting.projections import (
    HROColorGradient,
    HROFlat,
    HROPolar,
    plot_3d,
    plot_color_gradient,
    plot_flat,
    plot_polar,
    scatter_3d,
    scatter_flat,
    scatter_polar,
)

__all__ = [
    "plot_polar",
    "plot_flat",
    "plot_3d",
    "plot_color_gradient",
    "scatter_flat",
    "scatter_polar",
    "scatter_3d",
    "HROPolar",
    "HROFlat",
    "HROColorGradient",
]
