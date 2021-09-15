"""
The Python package hrosailing provides classes and functions ....
polar diagrams .... sailing .... from data ... pipeline ... machine learning
... modular ....

Installation
------------
hrosailing can be installed using `pip install hrosailing`. It requires
Python ..., aswell as numpy ..., matplotlib ... and scipy ... to run.

Getting Started
---------------

Contributing
------------

License
-------

Authors
-------
Valentin Dannenberg (valentin.dannenberg2@uni-rostock.de)

Robert Schueler (robert.schueler2@uni-rostock.de)
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

from ._pdoc import pdoc

__pdoc__ = pdoc


import hrosailing.cruising
import hrosailing.pipeline
import hrosailing.pipelinecomponents
import hrosailing.polardiagram
from .wind import true_wind_to_apparent, apparent_wind_to_true
