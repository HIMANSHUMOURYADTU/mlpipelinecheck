"""Allow ``python -m mlpipelinecheck``."""

from __future__ import annotations

import sys

from mlpipelinecheck.cli import main

if __name__ == "__main__":
    sys.exit(main())
