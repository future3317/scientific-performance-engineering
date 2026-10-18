from __future__ import annotations

import argparse
from pathlib import Path

from experiments.external_real.upstream_provenance import write_provenance_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit train-derived SkillGen artifacts before native reuse.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(write_provenance_audit(args.source_root, args.output))


if __name__ == "__main__":
    main()
