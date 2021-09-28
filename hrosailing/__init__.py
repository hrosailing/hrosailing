# pylint: disable=wrong-import-order, wrong-import-position

from ._doc import doc
from ._version import __version__
from ._pdoc import pdoc

__doc__ = doc
version = __version__
__pdoc__ = pdoc


# create folder for logging, where script is located, if it doesn't exist
import hrosailing._logfolder as log

path = log.get_script_path()
log.set_log_folder(path)
log.create_log_folder()

del log


# Tell users if hard depencencies are missing
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
import hrosailing.pipelinecomponents
import hrosailing.pipeline
import hrosailing.polardiagram
import hrosailing.wind
