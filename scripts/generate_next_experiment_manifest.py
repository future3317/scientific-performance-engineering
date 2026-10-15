"""Record a new study plan; this command cannot launch a provider or overwrite outputs."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", default=True)
    args = parser.parse_args()
    raw = args.config.read_bytes()
    config = json.loads(raw)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("commit study code before generating its execution identity")
    required = ("experiment_id", "model", "split", "resource_contract", "budget", "seeds")
    if any(key not in config for key in required):
        raise ValueError(f"config requires {required}")
    value = {"schema_version": 2, "status": "DRY_RUN_NOT_EXECUTED", "code_revision": revision,
             "config_sha256": hashlib.sha256(raw).hexdigest(), "config": config,
             "provider_calls": 0, "results_claimed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key:value[key] for key in ("status", "code_revision", "config_sha256")}))


if __name__ == "__main__":
    main()
