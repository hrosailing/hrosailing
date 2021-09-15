"""
The Python package hrosailing provides classes and functions ....
polar diagrams .... sailing .... from data ... pipeline ... machine learning
... modular ....


Getting Started
---------------


References
----------
"""

# Tell users if hard depencencies are missing
hard_dependencies = ("numpy", "matplotlib", "scipy")
missing_dependencies = []

for depencency in hard_dependencies:
    try:
        __import__(depencency)
    except ImportError as ie:
        missing_dependencies.append(f"{depencency}: {ie}")

if missing_dependencies:
    raise ImportError(
        "Unable to import required depencencies:\n"
        + "\n".join(missing_dependencies)
    )
del hard_dependencies, depencency, missing_dependencies

from ._version import __version__

version = __version__


import hrosailing.cruising
import hrosailing.polardiagram
import hrosailing.pipeline
from .wind import true_wind_to_apparent, apparent_wind_to_true

__pdoc__ = {
    "hrosailing.wind.set_resolution": False,
    "hrosailing.wind.convert_wind": False,
    "hrosailing.polardiagram.PolarDiagramTable.__getitem__": True,
    "hrosailing.polardiagram.PolarDiagramMultiSails.__getitem__": True,
    "hrosailing.polardiagram.PolarDiagramCurve.__call__": True,
    "hrosailing.pipeline.pipeline.PolarPipeline.__call__": True,
    "hrosailing.pipeline.pipelinecomponents.interpolator.gauss_potential": False,
    "hrosailing.pipeline.pipelinecomponents.interpolator.scaled": False,
    "hrosailing.pipeline.pipelinecomponents.interpolator.euclidean_norm": False,
    "hrosailing.pipeline.pipelinecomponents.neighbourhood.euclidean_norm": False,
    "hrosailing.pipeline.pipelinecomponents.neighbourhood.scaled": False,
    "hrosailing.pipeline.pipelinecomponents.sampler.make_circle": False,
    "hrosailing.pipeline.pipelinecomponents.weigher.euclidean_norm": False,
    "hrosailing.pipeline.pipelinecomponents.weigher.scaled": False,
}
