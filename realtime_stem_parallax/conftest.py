"""Put the repo root on sys.path so `import realtime_stem` / `tools` / `examples` work when
running the tests from a checkout without installing the package (`pytest` from the repo root)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
