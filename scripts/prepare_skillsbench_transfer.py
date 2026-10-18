from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from experiments.external_real.skillsbench_transfer import write_sb0_sb2_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    archive_digest = None
    if args.archive:
        digest = hashlib.sha256()
        with args.archive.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        archive_digest = digest.hexdigest()
    result = write_sb0_sb2_artifacts(args.benchmark_root, args.output_dir, archive_sha256=archive_digest)
    print({"active_tasks": len(result["tasks"]), "skills": len(result["skills"]), "selected": len(result["selected"]), "pairs": len(result["target_rows"])})


if __name__ == "__main__":
    main()
