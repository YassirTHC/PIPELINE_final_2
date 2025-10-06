import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

SRC_DIR = ROOT_DIR / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(1, str(SRC_DIR))

import os

os.environ.setdefault("PIPELINE_FAST_TESTS", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


