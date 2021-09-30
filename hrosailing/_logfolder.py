# pylint: disable=missing-docstring
# pylint: disable=global-statement,global-variable-not-assigned

import sys
from pathlib import Path

log_folder = ""


def get_script_path():
    return sys.path[0]


def set_log_folder(path):
    global log_folder
    log_folder = str(path) + "/logging"


def create_log_folder():
    global log_folder
    Path(log_folder).mkdir(parents=True, exist_ok=True)
