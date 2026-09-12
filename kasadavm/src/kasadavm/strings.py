"""Optional string-table support.

String operand cells are ``(length, offset)`` references into the VM's string
table (the interpreter's ``P.r``), which is spliced out of the instruction
stream at build time.  A real sample may therefore arrive without it.

The frontend stays complete either way:

* with a table, string operands decode to their text;
* without one, they decode to an explicit :class:`~kasadavm.decoder.StringRef`
  ``(offset, length)`` pair, which is the whole decoded operand - nothing is
  fabricated and nothing is dropped.

A table is supplied by the caller.  Deriving one from a particular interpreter
build is a separate static-analysis concern and lives in ``tools/``, so the
shipped frontend never depends on a specific interpreter revision.
"""

from __future__ import annotations

import json
from pathlib import Path


def normalize_string_table(raw: str) -> str:
    """Accept either the raw concatenated table or a JSON list of characters."""

    text = raw.strip()
    if text.startswith("["):
        try:
            parts = json.loads(text)
        except json.JSONDecodeError as exc:  # pragma: no cover - malformed input
            raise ValueError(f"invalid JSON string table: {exc}") from exc
        if not isinstance(parts, list) or not all(isinstance(p, str) for p in parts):
            raise ValueError("JSON string table must be a list of single-character strings")
        return "".join(parts)
    return raw


def load_string_table(path: str | Path) -> str:
    """Read a string table from a text or JSON file.

    The table is a byte-exact concatenation and contains control characters,
    so it is read without universal-newline translation: ``Path.read_text``
    would fold a ``\r\n`` pair into one character and shift every later
    ``(offset, length)`` reference by one.
    """

    with open(path, encoding="utf-8", newline="") as handle:
        return normalize_string_table(handle.read())
