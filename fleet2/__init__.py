"""fleet-v2 worker package: the agent plus the models and backtester it runs.

Shipped to workers as a tarball from the coordinator (/dl/worker.tar.gz). Runs on the
system python3 of Debian; besides the standard library it may use only psutil and
numpy, both installed from Debian packages (python3-psutil, python3-numpy), never pip.
"""

from __future__ import annotations

import os


def get_version() -> str:
    """Return the code version from fleet2/VERSION, or "dev" when the file is absent."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "VERSION")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            value = fh.read().strip()
    except OSError:
        return "dev"
    return value or "dev"


__version__ = get_version()
