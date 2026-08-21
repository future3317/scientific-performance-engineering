#!/usr/bin/env python3
"""CLI wrapper for the formal evolution-validation authority."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.formal.evolution_validation import main


if __name__ == "__main__":
    main()
