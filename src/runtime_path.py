"""Resolve data-file paths for both dev and PyInstaller-bundled modes."""

import sys
from pathlib import Path


def data_path(relative: str) -> Path:
    """Return the absolute path to a bundled data file.

    When running from a PyInstaller bundle, files are extracted to a
    temporary directory stored in ``sys._MEIPASS``.  In development mode
    this simply returns the path relative to the project root.
    """
    base = getattr(sys, "_MEIPASS", None)
    if base is not None:
        return Path(base) / relative
    return Path(relative)
