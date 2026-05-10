"""Deprecated compatibility entrypoint for the quality-overfit sanity test.

Use ``scripts/run_quality_overfit_test.py`` directly. This wrapper stays in
the repo so older notebooks or shell commands that still invoke the old script
name do not break during the transition.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_quality_overfit_test import main

if __name__ == "__main__":
    raise SystemExit(main())
