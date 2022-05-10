"""The hrosailing package is a Python library that provides assistance for the
scientific aspects of sailing.
Currently, it implements the data processing framework described
[here](https://www.mdpi.com/2076-3417/12/6/3085) and will be extended in the
future.

In particular, hrosailing provides:

- four kinds of representations of polar diagrams,
- serialization and visualization of polar diagrams,
- creation of polar diagrams from measurement data using
a modular pipeline model and suitable data science methods,
- wind conversion,
- suggesting optimal tacks and jibes,
- calculating the costs of a sailing trip with respect to the
weather on the way,
- calculating isocrone points with respect to the weather along the way.

Installation
------------
The recommended way to install `hrosailing` is with
[pip](http://pypi.python.org/pypi/pip)

    pip install hrosailing

[![PyPi version](https://badge.fury.io/py/hrosailing.svg)](https://badge.\
        fury.io/py/hrosailing)

Contributing
------------
In alphabetical order:

- Valentin Dannenberg : design and implementation
- Julia Garbe : quality control
- Robert Schüler : design and implementation


License
-------

The `hrosailing` package is published under the [Apache 2.0 License](https://\
        choosealicense.com/licenses/apache-2.0/)
"""

# pylint: disable=wrong-import-order
# pylint: disable=wrong-import-position
# pylint: disable=unused-import

from ._pdoc import pdoc
from ._version import __version__ as version

__pdoc__ = pdoc

# Tell users if and which hard dependencies are missing
hard_dependencies = ("numpy", "matplotlib", "scipy")
# soft_dependencies = ("pandas", "pynmea2")
missing_dependencies = []

for dependency in hard_dependencies:
    try:
        __import__(dependency)
    except ImportError as ie:
        missing_dependencies.append(f"{dependency}: {ie}")

if missing_dependencies:
    raise ImportError(
        "Unable to import required dependencies:\n"
        + "\n".join(missing_dependencies)
    )
del hard_dependencies, dependency, missing_dependencies


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
]
