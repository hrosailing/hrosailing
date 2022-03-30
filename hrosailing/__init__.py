"""The hrosailing package provides classes and functions ....
polar diagrams .... sailing .... from data ... pipeline ... machine learning
... modular ....

Installation
------------
The recommended way to install `hrosailing` is with
[pip](http://pypi.python.org/pypi/pip)

    pip install hrosailing

[![PyPi version](https://badge.fury.io/py/hrosailing.svg)](https://badge.\
        fury.io/py/hrosailing)


Getting Started
---------------

Contributing
------------

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
