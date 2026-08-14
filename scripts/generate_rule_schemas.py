#!/usr/bin/env python3
"""Write the JSON Schema projections of the canonical typed rule models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from core.schema import schemas


def main() -> None:
    destination = ROOT / "assets"
    for name, value in schemas().items():
        (destination / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(destination / name)


if __name__ == "__main__":
    main()
