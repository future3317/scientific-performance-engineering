"""Small receipt and identity helpers shared by the two V5 studies."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def require_final_access(spec):
    if spec["mode"] in {"qualification", "compile_qualification"} or spec["split"] != "final":
        return
    release_path = Path(spec["final_release_path"])
    if not release_path.is_file():
        raise PermissionError("final access before study release")
    release = read_json(release_path)
    root = release_path.parent
    if (release.get("study_id") != spec["study_id"]
            or release.get("policy_sha256") != digest(read_json(root / "POLICY_FREEZE.json"))
            or release.get("manifest_sha256") != digest(read_json(root / "FROZEN_MANIFEST.json"))):
        raise PermissionError("final release differs from frozen policy/study identity")
