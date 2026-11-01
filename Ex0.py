#!/usr/bin/env python3
"""Project Root CLI Launcher for Ex0 (delegates to src/Ex0.py).

CLI Usage:
    python Ex0.py <file.nrrd> <file.csv> <file.mp4>
"""

import os
from pathlib import Path
import sys

# Ensure src/ is on sys.path
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from src.Ex0 import main

if __name__ == "__main__":
    main()
