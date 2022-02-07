# pylint: disable=wrong-import-order
# pylint: disable=wrong-import-position
# pylint: disable=unused-import

from ._doc import doc
from ._pdoc import pdoc
from ._version import __version__ as version

__doc__ = doc
__pdoc__ = pdoc


# Tell users if and which hard depencencies are missing
hard_dependencies = ("numpy", "matplotlib", "scipy")
# soft_dependencies = ("pandas", "pynmea2")
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


import hrosailing.cruising
import hrosailing.pipeline
import hrosailing.pipelinecomponents
import hrosailing.polardiagram
import hrosailing.wind

__all__ = [
        "cruising",
        "pipeline",
        "pipelinecomponents",
        "polardiagram",
        "wind",
        "version",
        "__doc__",
]
