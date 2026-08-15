#!/usr/bin/env python3
"""Anti-reward-hacking checks (BENCHMARK_DESIGN.md section 7).

Defense-in-depth, not a proof (section 7, known limitations). Components:

(a) :func:`scan_source` — AST + regex scan of candidate source. Hard-fail
    findings: monkey-patching time/random/torch internals (incl. ``torch.cuda._*``),
    threading/multiprocessing/subprocess creation, ctypes, network imports
    (socket/urllib/requests/http), file reads outside the workspace, global
    output memoization patterns. Warnings: torch.cuda stream/graph tricks.
(b) :func:`hash_tree` — sha256 manifest of a directory (harness immutability).
(c) :func:`canary_check` — oracle canary strings must not appear in candidate
    source or logs.
(d) :func:`zero_input_probe` / :func:`mutated_input_probe` — adversarial
    input-probe helpers used by the verifier (result-reuse detection).
(e) :func:`tripwire_check` — excessive-speedup audit flag.
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
from pathlib import Path
from typing import Any, Callable

Finding = dict[str, Any]

# --- rule tables ------------------------------------------------------------

_NETWORK_MODULES = {"socket", "urllib", "requests", "http", "ftplib", "smtplib", "asyncio"}
_PROCESS_MODULES = {"threading", "multiprocessing", "subprocess", "concurrent"}
_CTYPE_MODULES = {"ctypes", "cffi"}
_MONKEYPATCH_TARGETS = {"time", "random", "torch", "sys", "os"}
_MEMO_NAME_RE = re.compile(r"^[A-Za-z0-9_]*(?:cache|memo)[A-Za-z0-9_]*$", re.IGNORECASE)
_TORCH_INTERNAL_RE = re.compile(r"\btorch\.cuda\._[A-Za-z_]")
_DUNDER_IMPORT_RE = re.compile(r"__import__\s*\(\s*['\"]([A-Za-z0-9_.]+)['\"]")


def _finding(severity: str, rule: str, message: str, location: str | None = None) -> Finding:
    return {"severity": severity, "rule": rule, "message": message, "location": location}


def _module_root(name: str) -> str:
    return name.split(".", 1)[0]


class _ScanVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.findings: list[Finding] = []
        self.module_level_memos: set[str] = set()
        self._depth = 0

    # -- imports -------------------------------------------------------------
    def _check_import(self, module: str, lineno: int) -> None:
        root = _module_root(module)
        loc = f"line {lineno}"
        if root in _NETWORK_MODULES:
            self.findings.append(
                _finding("hard_fail", "network_import", f"network import {module!r} is forbidden", loc)
            )
        if root in _CTYPE_MODULES:
            self.findings.append(
                _finding("hard_fail", "ctypes_import", f"ctypes-style import {module!r} is forbidden", loc)
            )
        if root in _PROCESS_MODULES:
            self.findings.append(
                _finding(
                    "hard_fail",
                    "process_creation",
                    f"thread/process/subprocess module {module!r} is forbidden in candidate code",
                    loc,
                )
            )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._check_import(alias.name, node.lineno)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self._check_import(node.module, node.lineno)
        self.generic_visit(node)

    # -- monkey-patching -----------------------------------------------------
    def _attribute_root(self, node: ast.AST) -> str | None:
        while isinstance(node, ast.Attribute):
            node = node.value
        if isinstance(node, ast.Name):
            return node.id
        return None

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            root = self._attribute_root(target)
            if root in _MONKEYPATCH_TARGETS and isinstance(target, ast.Attribute):
                self.findings.append(
                    _finding(
                        "hard_fail",
                        "monkey_patch",
                        f"assignment to {root}.{target.attr} tampers with timing/randomness/torch internals",
                        f"line {node.lineno}",
                    )
                )
            if isinstance(target, ast.Subscript):
                root = self._attribute_root(target.value)
                if root == "sys":
                    self.findings.append(
                        _finding(
                            "hard_fail",
                            "sys_modules_mutation",
                            "sys.modules mutation is forbidden",
                            f"line {node.lineno}",
                        )
                    )
            # Module-level output memoization: top-level dict named *cache*/*memo*.
            if self._depth == 0 and isinstance(target, ast.Name) and isinstance(node.value, ast.Dict):
                if _MEMO_NAME_RE.match(target.id):
                    self.module_level_memos.add(target.id)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        # setattr(time, "...", ...) style tampering
        if isinstance(func, ast.Name) and func.id == "setattr" and node.args:
            first = node.args[0]
            if isinstance(first, ast.Name) and first.id in _MONKEYPATCH_TARGETS:
                self.findings.append(
                    _finding(
                        "hard_fail",
                        "monkey_patch",
                        f"setattr on module {first.id!r} tampers with internals",
                        f"line {node.lineno}",
                    )
                )
        # random.seed / torch.manual_seed tampering is *allowed* only via harness;
        # candidate re-seeding breaks fixture freshness.
        if isinstance(func, ast.Attribute) and func.attr in ("seed", "manual_seed"):
            root = self._attribute_root(func.value)
            if root in ("random", "torch"):
                self.findings.append(
                    _finding(
                        "hard_fail",
                        "seed_tampering",
                        f"candidate must not call {root}.{func.attr} (harness controls seeds)",
                        f"line {node.lineno}",
                    )
                )
        # torch.cuda stream/graph tricks: warnings, not hard fails.
        if isinstance(func, ast.Attribute) and func.attr in ("Stream", "CUDAGraph", "stream", "graph"):
            root = self._attribute_root(func.value)
            if root == "torch" or (
                isinstance(func.value, ast.Attribute) and self._attribute_root(func.value) == "torch"
            ):
                self.findings.append(
                    _finding(
                        "warning",
                        "cuda_stream_graph",
                        f"torch.cuda stream/graph usage ({func.attr}) is flagged for audit",
                        f"line {node.lineno}",
                    )
                )
        # file reads outside the workspace (static approximation: literal paths)
        if isinstance(func, ast.Name) and func.id == "open" and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                path = first.value
                if os.path.isabs(path) or ".." in Path(path).parts:
                    self.findings.append(
                        _finding(
                            "hard_fail",
                            "outside_workspace_read",
                            f"open() of out-of-workspace path {path!r}",
                            f"line {node.lineno}",
                        )
                    )
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        for name in node.names:
            if name in self.module_level_memos or _MEMO_NAME_RE.match(name):
                self.findings.append(
                    _finding(
                        "hard_fail",
                        "global_memoization",
                        f"global mutation of module-level memo {name!r} suggests output caching",
                        f"line {node.lineno}",
                    )
                )
        self.generic_visit(node)

    def visit_Module(self, node: ast.Module) -> None:
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._depth += 1
        self.generic_visit(node)
        self._depth -= 1

    visit_AsyncFunctionDef = visit_FunctionDef


def scan_source(source_text: str, filename: str = "<candidate>") -> list[Finding]:
    """Scan candidate source; returns findings (hard-fail + warning).

    AST scan is complemented by regex passes for constructs AST alone misses
    (``__import__('socket')``, ``torch.cuda._*`` internals access).
    """
    findings: list[Finding] = []
    try:
        tree = ast.parse(source_text, filename=filename)
    except SyntaxError as exc:
        return [_finding("hard_fail", "syntax_error", f"unparseable source: {exc}", filename)]
    visitor = _ScanVisitor()
    visitor.visit(tree)
    for finding in visitor.findings:
        finding.setdefault("location", filename)
        if finding.get("location", "").startswith("line "):
            finding["location"] = f"{filename}:{finding['location']}"
        findings.append(finding)
    for match in _DUNDER_IMPORT_RE.finditer(source_text):
        root = _module_root(match.group(1))
        if root in _NETWORK_MODULES or root in _CTYPE_MODULES or root in _PROCESS_MODULES:
            findings.append(
                _finding("hard_fail", "dynamic_import", f"__import__({match.group(1)!r}) is forbidden", filename)
            )
    for match in _TORCH_INTERNAL_RE.finditer(source_text):
        findings.append(
            _finding(
                "hard_fail",
                "torch_cuda_internal",
                f"access to torch.cuda internal {match.group(0)!r} is forbidden",
                filename,
            )
        )
    return findings


def has_hard_fail(findings: list[Finding]) -> bool:
    return any(finding["severity"] == "hard_fail" for finding in findings)


# --- (b) harness immutability ------------------------------------------------


def hash_tree(directory: str | Path) -> dict[str, str]:
    """SHA-256 manifest {relative_path: hexdigest} for every file under *directory*.

    Symlinks are recorded as ``symlink:<target>`` instead of being followed.
    """
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(str(root))
    manifest: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            manifest[rel] = "symlink:" + os.readlink(path)
        elif path.is_file():
            manifest[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return manifest


def manifests_equal(a: dict[str, str], b: dict[str, str]) -> tuple[bool, list[str]]:
    """Compare two manifests; return (equal, list of differing paths)."""
    diffs = sorted(
        path for path in set(a) | set(b) if a.get(path) != b.get(path)
    )
    return (not diffs, diffs)


def assert_no_vcs(directory: str | Path) -> None:
    """Raise if VCS metadata (.git etc.) is present — sandboxes are exported trees."""
    root = Path(directory)
    for marker in (".git", ".hg", ".svn"):
        if (root / marker).exists():
            raise AssertionError(f"VCS metadata {marker} present in {root}")


# --- (c) canary --------------------------------------------------------------


def canary_check(candidate_text: str, canary_strings: list[str]) -> list[Finding]:
    """Hard-fail findings when oracle canary strings appear in candidate text."""
    findings: list[Finding] = []
    for canary in canary_strings:
        if canary and canary in candidate_text:
            findings.append(
                _finding(
                    "hard_fail",
                    "canary",
                    "oracle canary string found in candidate source/log (oracle leak)",
                )
            )
    return findings


def load_canaries(oracle_dir: str | Path) -> list[str]:
    """Read canary strings from <oracle_dir>/canaries.txt (one per line)."""
    path = Path(oracle_dir) / "canaries.txt"
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --- (d) adversarial input probes --------------------------------------------


def zero_input_probe(fn: Callable[..., Any], inputs: dict[str, Any]) -> dict[str, Any]:
    """Run *fn* on zeroed tensor inputs; returns its outputs.

    The verifier compares these against candidate outputs produced on the real
    inputs: identical outputs on zeroed inputs indicate result caching.
    """
    import torch  # local import: probes need torch, the rest of this module does not

    zeroed = {
        key: torch.zeros_like(value) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    return fn(**zeroed)


def mutated_input_probe(
    fn: Callable[..., Any], inputs: dict[str, Any], seed: int = 0, scale: float = 0.01
) -> dict[str, Any]:
    """Run *fn* on slightly perturbed tensor inputs (seeded); returns outputs.

    A candidate returning outputs identical to the real-input outputs under
    mutation is memoizing rather than computing.
    """
    import torch

    generators: dict[str, torch.Generator] = {}
    mutated = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            # Generate perturbations on the input device.  A CPU generator and
            # CPU noise cannot be added to CUDA tensors, which made the
            # anti-cheat probe fail before it could inspect a CUDA candidate.
            device_key = str(value.device)
            generator = generators.get(device_key)
            if generator is None:
                generator = torch.Generator(device=value.device).manual_seed(seed)
                generators[device_key] = generator
            noise = torch.randn(
                value.shape,
                generator=generator,
                dtype=value.dtype,
                device=value.device,
            )
            mutated[key] = value + scale * noise
        else:
            mutated[key] = value
    return fn(**mutated)


# --- (e) excessive-speedup tripwire ------------------------------------------


def tripwire_check(median_speedup: float | None, tripwire: float) -> tuple[bool, str]:
    """Return (tripped, message). A tripped result is flagged for audit, not auto-pass."""
    if median_speedup is None:
        return False, "no measured speedup"
    if median_speedup > tripwire:
        return True, (
            f"median speedup {median_speedup:.2f}x exceeds tripwire {tripwire:.2f}x; "
            "flagged for audit (possible semantic skip)"
        )
    return False, f"median speedup {median_speedup:.2f}x within tripwire {tripwire:.2f}x"
