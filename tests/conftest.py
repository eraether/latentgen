"""Make `latentgen` importable when the package is not pip-installed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
