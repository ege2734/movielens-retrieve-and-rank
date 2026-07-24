"""Put the flat `src/` modules on the import path for the test suite.

This repo is deliberately not a pip-installable package - the modules live directly
under `src/` and are read top to bottom. Tests import them by their bare names
(`import model`, `from dataset import ...`), so we prepend `src/` to `sys.path` here.
pytest imports the rootdir conftest before collecting tests, so this runs first.
"""

import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC))
