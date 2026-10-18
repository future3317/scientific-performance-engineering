from __future__ import annotations

import argparse
from pathlib import Path

from experiments.external_real.upstream_reference import build_release_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute metrics from the pinned SkillGen ScienceWorld release logs.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-url", default="https://github.com/ruomengd/SkillGen")
    args = parser.parse_args()
    result = build_release_artifacts(
        args.source_root,
        args.output_dir,
        source_commit=args.source_commit,
        source_url=args.source_url,
    )
    print(result["summary"]["canonical_qwen_20_step_sampling6"])


if __name__ == "__main__":
    main()
