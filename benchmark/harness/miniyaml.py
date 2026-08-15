#!/usr/bin/env python3
"""Restricted-subset YAML parser/dumper for SPE-EvoBench configuration files.

PyYAML is not guaranteed in the benchmark environment, so the harness ships this
small parser. It intentionally supports ONLY the subset that ``task.yaml``,
``split/sequential.yaml`` and episode manifests need (BENCHMARK_DESIGN.md section 11).

Supported subset
----------------
- Block mappings via indentation: ``key: value`` and ``key:`` followed by a
  more-indented block. Nested to arbitrary depth; indentation must be spaces
  (tabs are rejected) and consistent per nesting level.
- Block sequences: ``- item`` lines. Items may be scalars, flow collections, or
  an inline mapping start (``- key: value``) continued by more-indented keys.
- Flow collections on a single line: ``[a, b, c]`` and ``{k: v, k2: v2}``,
  nestable, with quoted scalars allowed inside.
- Scalars: plain strings, single/double-quoted strings (with the usual double
  -quote escapes ``\\n \\t \\\\ \\"``), integers, floats (incl. ``1.0e-5``),
  booleans (``true``/``false``, any case), null (``null``, ``~``, or empty value).
- Comments: ``#`` starts a comment outside of quotes, anywhere on a line.
- Document separators ``---`` / ``...`` are tolerated (single document only).

NOT supported (rejected with a MiniYAMLError): anchors/aliases, tags, block
scalars (``|``/``>``), multi-line flow collections, multi-document streams,
complex mapping keys.

The dumper :func:`dump` emits only this subset, so ``parse(dump(x)) == x`` for
dict/list/scalar trees built from str/int/float/bool/None.
"""

from __future__ import annotations

import re
from typing import Any


class MiniYAMLError(ValueError):
    """Raised when input falls outside the supported YAML subset."""


# ---------------------------------------------------------------------------
# Scalars
# ---------------------------------------------------------------------------

_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?$")
_PLAIN_FORBIDDEN = set("[]{}#,&*!|>'\"%@`")


def _unescape_double(text: str) -> str:
    escapes = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"', "0": "\0"}
    out: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text):
            nxt = text[i + 1]
            if nxt in escapes:
                out.append(escapes[nxt])
                i += 2
                continue
            raise MiniYAMLError(f"unsupported escape sequence: \\{nxt}")
        out.append(ch)
        i += 1
    return "".join(out)


def parse_scalar(text: str) -> Any:
    """Parse one scalar token from the supported subset."""
    text = text.strip()
    if text == "":
        return None
    if text[0] == '"':
        if len(text) < 2 or not text.endswith('"'):
            raise MiniYAMLError(f"unterminated double-quoted scalar: {text!r}")
        return _unescape_double(text[1:-1])
    if text[0] == "'":
        if len(text) < 2 or not text.endswith("'"):
            raise MiniYAMLError(f"unterminated single-quoted scalar: {text!r}")
        return text[1:-1].replace("''", "'")
    lowered = text.lower()
    if lowered in ("null", "~"):
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if _INT_RE.match(text):
        return int(text)
    if _FLOAT_RE.match(text) and any(c.isdigit() for c in text):
        # _FLOAT_RE also matches plain ints; keep ints as ints (handled above).
        return float(text)
    if text[0] in _PLAIN_FORBIDDEN:
        raise MiniYAMLError(f"plain scalar may not start with {text[0]!r}: {text!r}")
    return text


# ---------------------------------------------------------------------------
# Flow collections
# ---------------------------------------------------------------------------


def _split_flow_items(body: str) -> list[str]:
    """Split a flow-collection body on top-level commas, respecting nesting/quotes."""
    items: list[str] = []
    depth = 0
    quote: str | None = None
    start = 0
    i = 0
    while i < len(body):
        ch = body[i]
        if quote is not None:
            if ch == "\\" and quote == '"':
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
            if depth < 0:
                raise MiniYAMLError("unbalanced flow collection")
        elif ch == "," and depth == 0:
            items.append(body[start:i].strip())
            start = i + 1
        i += 1
    if quote is not None:
        raise MiniYAMLError("unterminated quote in flow collection")
    if depth != 0:
        raise MiniYAMLError("unbalanced flow collection")
    tail = body[start:].strip()
    if tail:
        items.append(tail)
    return items


def _split_key_value(text: str) -> tuple[str, str] | None:
    """Split ``key: value`` at the first top-level colon; None if no such colon."""
    quote: str | None = None
    i = 0
    while i < len(text):
        ch = text[i]
        if quote is not None:
            if ch == "\\" and quote == '"':
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
        elif ch == ":" and (i + 1 >= len(text) or text[i + 1] in (" ", "")):
            return text[:i].strip(), text[i + 1:].strip()
        i += 1
    if text.rstrip().endswith(":"):
        return text.rstrip()[:-1].strip(), ""
    return None


def parse_flow(text: str) -> Any:
    """Parse a flow list ``[...]`` or flow map ``{...}`` (or a bare scalar)."""
    text = text.strip()
    if text.startswith("["):
        if not text.endswith("]"):
            raise MiniYAMLError(f"unterminated flow list: {text!r}")
        body = text[1:-1].strip()
        if not body:
            return []
        return [parse_flow(item) for item in _split_flow_items(body)]
    if text.startswith("{"):
        if not text.endswith("}"):
            raise MiniYAMLError(f"unterminated flow map: {text!r}")
        body = text[1:-1].strip()
        if not body:
            return {}
        result: dict[str, Any] = {}
        for item in _split_flow_items(body):
            split = _split_key_value(item)
            if split is None:
                raise MiniYAMLError(f"flow map entry lacks a key: {item!r}")
            key, value = split
            parsed_key = parse_scalar(key)
            if not isinstance(parsed_key, str):
                parsed_key = str(parsed_key)
            result[parsed_key] = parse_flow(value) if value != "" else None
        return result
    return parse_scalar(text)


# ---------------------------------------------------------------------------
# Block structure
# ---------------------------------------------------------------------------


def _strip_comment(line: str) -> str:
    quote: str | None = None
    i = 0
    while i < len(line):
        ch = line[i]
        if quote is not None:
            if ch == "\\" and quote == '"':
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1] in (" ", "\t")):
            return line[:i]
        i += 1
    return line


def _logical_lines(text: str) -> list[tuple[int, str, int]]:
    """Return (indent, content, line_number) for non-empty, comment-free lines."""
    lines: list[tuple[int, str, int]] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        if "\t" in raw[: len(raw) - len(raw.lstrip())]:
            raise MiniYAMLError(f"line {lineno}: tabs are not allowed for indentation")
        stripped = _strip_comment(raw).rstrip()
        if not stripped.strip():
            continue
        content = stripped.lstrip(" ")
        if content in ("---", "..."):
            continue
        lines.append((len(stripped) - len(content), content, lineno))
    return lines


class _Parser:
    def __init__(self, lines: list[tuple[int, str, int]]):
        self.lines = lines
        self.pos = 0

    def parse(self) -> Any:
        if not self.lines:
            return None
        value = self._parse_block(self.lines[0][0])
        if self.pos != len(self.lines):
            _, _, lineno = self.lines[self.pos]
            raise MiniYAMLError(f"line {lineno}: trailing content after document end")
        return value

    def _parse_block(self, indent: int) -> Any:
        if self.pos >= len(self.lines):
            return None
        cur_indent, content, lineno = self.lines[self.pos]
        if cur_indent < indent:
            return None
        if cur_indent > indent:
            raise MiniYAMLError(f"line {lineno}: unexpected indentation")
        if content.startswith("- ") or content == "-":
            return self._parse_list(indent)
        return self._parse_map(indent)

    def _parse_list(self, indent: int) -> list[Any]:
        items: list[Any] = []
        while self.pos < len(self.lines):
            cur_indent, content, lineno = self.lines[self.pos]
            if cur_indent != indent or not (content.startswith("- ") or content == "-"):
                break
            rest = content[1:].strip()
            self.pos += 1
            if rest == "":
                # Item is a nested block on the following lines.
                if self.pos < len(self.lines) and self.lines[self.pos][0] > indent:
                    items.append(self._parse_block(self.lines[self.pos][0]))
                else:
                    items.append(None)
                continue
            split = _split_key_value(rest)
            if split is not None and not rest.startswith(("[", "{", "'", '"')):
                # Inline mapping start: `- key: value` with continuation keys.
                key, value = split
                item: dict[str, Any] = {key: self._value_or_block(value, indent, lineno)}
                if self.pos < len(self.lines) and self.lines[self.pos][0] > indent:
                    extra_indent = self.lines[self.pos][0]
                    extra = self._parse_block(extra_indent)
                    if not isinstance(extra, dict):
                        raise MiniYAMLError(
                            f"line {lineno}: sequence item mapping continuation must be a mapping"
                        )
                    item.update(extra)
                items.append(item)
            else:
                items.append(parse_flow(rest))
        return items

    def _parse_map(self, indent: int) -> dict[str, Any]:
        result: dict[str, Any] = {}
        while self.pos < len(self.lines):
            cur_indent, content, lineno = self.lines[self.pos]
            if cur_indent != indent or content.startswith("- ") or content == "-":
                break
            split = _split_key_value(content)
            if split is None:
                raise MiniYAMLError(f"line {lineno}: expected 'key: value' entry, got {content!r}")
            key, value = split
            parsed_key = parse_scalar(key)
            if not isinstance(parsed_key, str):
                parsed_key = str(parsed_key)
            self.pos += 1
            result[parsed_key] = self._value_or_block(value, indent, lineno)
        return result

    def _value_or_block(self, value: str, indent: int, lineno: int) -> Any:
        if value != "":
            return parse_flow(value)
        # Empty value: either null, or a nested block on following lines.
        if self.pos < len(self.lines) and self.lines[self.pos][0] > indent:
            return self._parse_block(self.lines[self.pos][0])
        return None


def parse(text: str) -> Any:
    """Parse a YAML document from the supported subset into Python objects."""
    return _Parser(_logical_lines(text)).parse()


def load(path: str) -> Any:
    """Parse a YAML file (UTF-8) from the supported subset."""
    with open(path, "r", encoding="utf-8") as handle:
        return parse(handle.read())


# ---------------------------------------------------------------------------
# Dumper
# ---------------------------------------------------------------------------

_NEEDS_QUOTES_RE = re.compile(
    r"^$|^(true|false|null|~|[+-]?[\d.])|[:#\[\]{},&*!|>'\"%@`\s]",
    re.IGNORECASE,
)


def _dump_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        if _NEEDS_QUOTES_RE.search(value):
            return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
        return value
    raise MiniYAMLError(f"cannot dump value of type {type(value).__name__}")


def _dump_flow(value: Any) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(_dump_flow(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{_dump_scalar(k)}: {_dump_flow(v)}" for k, v in value.items()) + "}"
    return _dump_scalar(value)


def _is_flat(value: Any) -> bool:
    """True when a value renders cleanly as a single-line flow collection."""
    if isinstance(value, list):
        return all(not isinstance(item, (list, dict)) for item in value)
    return False


def _dump_block(value: Any, indent: int, out: list[str]) -> None:
    pad = " " * indent
    if isinstance(value, dict):
        if not value:
            out.append(pad + "{}")
            return
        for key, item in value.items():
            key_text = _dump_scalar(key)
            if isinstance(item, dict) and item:
                out.append(f"{pad}{key_text}:")
                _dump_block(item, indent + 2, out)
            elif isinstance(item, list) and item and not _is_flat(item):
                out.append(f"{pad}{key_text}:")
                _dump_block(item, indent + 2, out)
            else:
                out.append(f"{pad}{key_text}: {_dump_flow(item)}")
    elif isinstance(value, list):
        if not value:
            out.append(pad + "[]")
            return
        for item in value:
            if isinstance(item, (dict, list)):
                rendered: list[str] = []
                _dump_block(item, indent + 2, rendered)
                first = rendered[0].lstrip()
                out.append(f"{pad}- {first}")
                out.extend(rendered[1:])
            else:
                out.append(f"{pad}- {_dump_flow(item)}")
    else:
        out.append(pad + _dump_scalar(value))


def dump(value: Any) -> str:
    """Dump a dict/list/scalar tree into the supported YAML subset."""
    lines: list[str] = []
    _dump_block(value, 0, lines)
    return "\n".join(lines) + "\n"


def save(value: Any, path: str) -> None:
    """Dump *value* to a UTF-8 YAML file."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(dump(value))
