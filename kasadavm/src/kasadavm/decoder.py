"""Deterministic Kasada VM bytecode decoder.

The artifact is a flat little-endian ``int32`` word stream.  Word ``k`` is the
VM program counter value ``k``; the dispatch loop executes
``i = u[t[e.E[0]++]]`` so the word *is* the console opcode index.

Decoding is layered:

1. :func:`read_value` / :func:`read_register` implement the interpreter's value
   reader (``g``) and register reader (``j``/``B``) cell encodings.
2. :func:`decode_instruction` turns one opcode plus its operand cells into a
   :class:`~kasadavm.model.KasadaInstruction`.
3. :func:`discover_boundaries` walks explicit successors only - fallthrough,
   immediate branch targets, and closure entry cells - never address order.
4. :func:`decode_program` groups the discovered instructions into function
   roots taken from the entry, closure cells, and nothing else.

The decoder never executes the VM and never constructs control-flow graphs,
blocks, or statements.  Successor discovery only decides *where instructions
begin*, which is a property of the bytecode format.
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass

from unidecompiler.plugins import FrontendDecodeError

from .strings import normalize_string_table

_DATA_RANGE = re.compile(r"word range (?:starting at |\[)(\d+)")

from .model import (
    KasadaFunction,
    KasadaInstruction,
    KasadaOperand,
    KasadaProgram,
    KasadaValue,
)
from .opcodes import BRANCHES, BY_CODE, CLOSURE_OPCODES, TERMINATORS, operands_of

WORD_SIZE = 4

#: Value-reader tags (interpreter ``g`` uses ``y = [6,46,40,4,24,50]``).
TAG_FLOAT64 = 6
TAG_UNDEFINED = 46
TAG_NULL = 40
TAG_STRING = 4
TAG_TRUE = 24
TAG_FALSE = 50

#: Entry program counter.  ``h()`` builds the root frame as
#: ``{E: [1, scope, undefined]}`` so ``E[0]`` starts at 1; word 0 is the
#: HALT sentinel that an empty frame would execute.
ENTRY_PC = 1

_SENTINEL_PC = 0


@dataclass(frozen=True)
class StringRef:
    """A string operand whose bytes live in the external string table."""

    offset: int
    length: int

    def __str__(self) -> str:  # pragma: no cover - presentation helper
        return f"str[{self.offset}:{self.length}]"


class _Cursor:
    """A bounds-checked reader over the decoded word stream."""

    __slots__ = ("words", "pc")

    def __init__(self, words: tuple[int, ...], pc: int) -> None:
        self.words = words
        self.pc = pc

    def take(self, count: int, what: str) -> list[int]:
        if self.pc + count > len(self.words):
            raise FrontendDecodeError(
                f"truncated {what} at word {self.pc}: need {count} word(s), "
                f"artifact has {len(self.words)}"
            )
        out = list(self.words[self.pc : self.pc + count])
        self.pc += count
        return out

    def take_one(self, what: str) -> int:
        return self.take(1, what)[0]


def _as_signed32(value: int) -> int:
    return value - 0x100000000 if value & 0x80000000 else value


def read_value(
    words: tuple[int, ...],
    pc: int,
    *,
    limit: int,
    string_table: str | None = None,
) -> KasadaValue:
    """Decode one value cell, mirroring the interpreter's ``g()``.

    ``limit`` is the exclusive end of the enclosing instruction, so a value
    cell can never run past the instruction that owns it.  When
    ``string_table`` is supplied the string operand resolves to its text;
    otherwise it stays an explicit ``(offset, length)`` reference.
    """

    cursor = _Cursor(words, pc)
    tag = _as_signed32(cursor.take_one("value tag"))

    if tag & 1:
        return KasadaValue("int", tag >> 1, 1, str(tag >> 1))
    if tag == TAG_TRUE:
        return KasadaValue("bool", True, 1, "true")
    if tag == TAG_FALSE:
        return KasadaValue("bool", False, 1, "false")
    if tag == TAG_NULL:
        return KasadaValue("null", None, 1, "null")
    if tag == TAG_UNDEFINED:
        return KasadaValue("undefined", None, 1, "undefined")
    if tag == TAG_FLOAT64:
        high, low = cursor.take(2, "float64 payload")
        return KasadaValue("float", _float64(high, low), 3, repr(_float64(high, low)))
    if tag == TAG_STRING:
        length, offset = cursor.take(2, "string reference")
        ref = StringRef(offset=_as_signed32(offset), length=_as_signed32(length))
        if string_table is not None:
            text = string_table[ref.offset : ref.offset + ref.length]
            if len(text) == ref.length:
                return KasadaValue("string", text, 3, json.dumps(text, ensure_ascii=False))
        return KasadaValue("string", ref, 3, str(ref))

    index = tag >> 5
    if index < 0:
        raise FrontendDecodeError(f"negative register index {index} at word {pc}")
    if cursor.pc > limit:
        raise FrontendDecodeError(f"value cell at word {pc} overruns its instruction")
    return KasadaValue("register", index, 1, f"r{index}")


def _float64(high: int, low: int) -> float:
    """Reassemble the interpreter's two-word IEEE-754 payload (``g``)."""

    sign = -1.0 if high & 0x80000000 else 1.0
    exponent = _as_signed32(high & 0x7FF00000) >> 20
    mantissa = _as_signed32(high & 0xFFFFF) * (2**32) + (low + 2**32 if low < 0 else low)
    if exponent == 2047:
        return float("nan") if mantissa else sign * float("inf")
    if exponent != 0:
        mantissa += 2**52
    else:
        exponent += 1
    return sign * mantissa * (2.0 ** (exponent - 1075))


def read_register(words: tuple[int, ...], pc: int, *, limit: int) -> tuple[int, int]:
    """Decode one register cell (``j`` reads ``word >> 5``)."""

    if pc >= limit:
        raise FrontendDecodeError(f"register cell at word {pc} overruns its instruction")
    raw = _as_signed32(words[pc])
    index = raw >> 5
    if index < 0:
        raise FrontendDecodeError(f"negative register index {index} at word {pc}")
    return index, pc + 1


def decode_instruction(
    words: tuple[int, ...], pc: int, *, string_table: str | None = None
) -> KasadaInstruction:
    """Decode exactly one instruction starting at word ``pc``."""

    if pc < 0 or pc >= len(words):
        raise FrontendDecodeError(f"instruction offset {pc} outside artifact")
    opcode = _as_signed32(words[pc])
    spec = BY_CODE.get(opcode)
    if spec is None:
        raise FrontendDecodeError(
            f"unknown opcode {opcode} at word {pc}; the console defines 86 opcodes (0..85)"
        )

    operands: list[KasadaOperand] = []
    cursor = pc + 1
    for position, kind in enumerate(operands_of(opcode)):
        if kind == "V":
            value = read_value(words, cursor, limit=len(words), string_table=string_table)
            operands.append(
                KasadaOperand(role="value", value=value, words=value.words, text=value.text)
            )
            cursor += value.words
        elif kind == "R":
            index, cursor = read_register(words, cursor, limit=len(words))
            operands.append(
                KasadaOperand(role="register", value=index, words=1, text=f"r{index}")
            )
        elif kind == "S":
            index, cursor = read_register(words, cursor, limit=len(words))
            operands.append(
                KasadaOperand(role="store", value=index, words=1, text=f"->r{index}")
            )
        else:  # pragma: no cover - table invariant
            raise FrontendDecodeError(f"bad operand kind {kind!r} for opcode {opcode}")

    size = cursor - pc
    span = " ".join(str(_as_signed32(w)) for w in words[pc:cursor])
    return KasadaInstruction(
        offset=pc,
        opcode=opcode,
        mnemonic=spec.mnemonic,
        operands=tuple(operands),
        size=size,
        raw=f"{spec.mnemonic} {span}",
        category=spec.category,
        artifact_offset=pc * WORD_SIZE,
        artifact_size=size * WORD_SIZE,
    )


def immediate_target(instruction: KasadaInstruction) -> int | None:
    """Return the static target of a branch, when the target cell is an integer."""

    if instruction.opcode not in BRANCHES:
        return None
    values = instruction.value_operands()
    if instruction.opcode == 79:
        candidate = values[0] if values else None
    else:
        candidate = values[1] if len(values) > 1 else None
    if candidate is not None and candidate.kind == "int":
        return int(candidate.value)
    return None


def handler_target(instruction: KasadaInstruction) -> int | None:
    """Return the handler offset named by ``SET_HANDLER``.

    ``SET_HANDLER`` writes the value into ``scope.C``, and the unwinder ``D``
    resumes at exactly ``scope.C`` when it finds that frame, so the operand is
    a proven control-flow entry even though no branch reaches it.
    """

    if instruction.opcode != 75:
        return None
    values = instruction.value_operands()
    if values and values[0].kind == "int":
        return int(values[0].value)
    return None


def closure_entry(instruction: KasadaInstruction) -> int | None:
    """Return the closure body entry named by a ``MAKE_CLOSURE`` instruction."""

    if instruction.opcode not in CLOSURE_OPCODES:
        return None
    values = instruction.value_operands()
    if values and values[0].kind == "int":
        return int(values[0].value)
    return None


def discover_boundaries(
    words: tuple[int, ...], *, string_table: str | None = None
) -> tuple[dict[int, KasadaInstruction], list[str]]:
    """Decode every instruction reachable through explicit successors."""

    decoded: dict[int, KasadaInstruction] = {}
    diagnostics: list[str] = []
    worklist = [_SENTINEL_PC, ENTRY_PC]

    while worklist:
        pc = worklist.pop()
        while pc not in decoded:
            if pc < 0 or pc >= len(words):
                diagnostics.append(f"successor {pc} outside artifact")
                break
            try:
                instruction = decode_instruction(words, pc, string_table=string_table)
            except FrontendDecodeError as exc:
                diagnostics.append(f"decode stopped at word {pc}: {exc}")
                break
            decoded[pc] = instruction

            target = immediate_target(instruction)
            if target is not None:
                worklist.append(target)
            entry = closure_entry(instruction)
            if entry is not None:
                worklist.append(entry)
            handler = handler_target(instruction)
            if handler is not None:
                worklist.append(handler)
            if instruction.opcode in TERMINATORS:
                break
            pc = pc + instruction.size

    return decoded, diagnostics


def _walk_function(
    entry: int,
    decoded: dict[int, KasadaInstruction],
    other_entries: frozenset[int],
) -> tuple[KasadaInstruction, ...]:
    """Collect one function's instructions by following its own successors."""

    collected: dict[int, KasadaInstruction] = {}
    worklist = [entry]
    while worklist:
        pc = worklist.pop()
        while pc in decoded and pc not in collected:
            if pc != entry and pc in other_entries:
                break
            instruction = decoded[pc]
            collected[pc] = instruction
            target = immediate_target(instruction)
            if target is not None:
                worklist.append(target)
            handler = handler_target(instruction)
            if handler is not None:
                worklist.append(handler)
            if instruction.opcode in TERMINATORS:
                break
            pc = pc + instruction.size
    return tuple(collected[offset] for offset in sorted(collected))


def _residual_root(
    words: tuple[int, ...],
    lo: int,
    real_owned: frozenset[int],
    collected: dict[int, KasadaInstruction],
    *,
    string_table: str | None,
) -> tuple[list[KasadaInstruction], list[int], str | None]:
    """Follow explicit successors from ``lo`` through words no function owns.

    Residual words can still reference each other: a stub that jumps into a
    later residual region is following a proven target, so the walk crosses
    region boundaries.  The walk stops at terminators and never enters words a
    real function already owns.
    """

    worklist = [lo]
    ordered: list[KasadaInstruction] = []
    external: list[int] = []
    while worklist:
        pc = worklist.pop()
        while pc not in collected:
            if pc in real_owned:
                external.append(pc)
                break
            if pc < 0 or pc >= len(words):
                break
            try:
                instruction = decode_instruction(words, pc, string_table=string_table)
            except FrontendDecodeError as exc:
                return ordered, external, str(exc)
            span = range(instruction.offset, instruction.offset + instruction.size)
            if any(word in real_owned for word in span):
                return ordered, external, f"instruction at word {pc} overlaps a decoded function"
            collected[instruction.offset] = instruction
            ordered.append(instruction)
            for successor in (
                immediate_target(instruction),
                handler_target(instruction),
                closure_entry(instruction),
            ):
                if successor is not None:
                    worklist.append(successor)
            if instruction.opcode in TERMINATORS:
                break
            pc = pc + instruction.size
    return ordered, external, None


def _region_gaps(
    words: tuple[int, ...], claimed: frozenset[int]
) -> list[tuple[int, int]]:
    """Contiguous word ranges that no decoded instruction owns."""

    gaps: list[tuple[int, int]] = []
    start: int | None = None
    for index in range(len(words)):
        if index not in claimed:
            if start is None:
                start = index
        elif start is not None:
            gaps.append((start, index - start))
            start = None
    if start is not None:
        gaps.append((start, len(words) - start))
    return gaps


def parse_words(data: bytes) -> tuple[int, ...]:
    """Decode the artifact's little-endian int32 word stream."""

    if not data:
        raise FrontendDecodeError("empty kasadavm artifact")
    if len(data) % WORD_SIZE:
        raise FrontendDecodeError(
            f"kasadavm artifact size {len(data)} is not a multiple of {WORD_SIZE}"
        )
    return struct.unpack(f"<{len(data) // WORD_SIZE}i", data)


def decode_program(
    data: bytes,
    filename: str | None = None,
    *,
    string_table: str | None = None,
    include_unreachable: bool = True,
) -> KasadaProgram:
    """Decode a kasadavm artifact into the frontend-private program model.

    ``include_unreachable`` sweeps word ranges that no statically-provable
    successor reaches.  A range that decodes into well-formed instructions
    ending exactly on its boundary becomes a labelled ``unreachable`` root so
    those instructions still submit thin IR; a range that does not is reported
    as data rather than guessed at.
    """

    words = parse_words(data)
    table = normalize_string_table(string_table) if string_table is not None else None
    decoded, diagnostics = discover_boundaries(words, string_table=table)

    entries: list[tuple[int, str]] = [
        # An empty frame is created with ``E:[0]``, so word 0 is a real
        # executable entry even though no static successor reaches it.
        (_SENTINEL_PC, "sentinel"),
        (ENTRY_PC, "entry"),
    ]
    for offset in sorted(decoded):
        instruction = decoded[offset]
        closure = closure_entry(instruction)
        if closure is not None:
            entries.append((closure, "closure"))

    seen: set[int] = set()
    roots: list[tuple[int, str]] = []
    for offset, origin in entries:
        if offset in seen or offset not in decoded:
            continue
        seen.add(offset)
        roots.append((offset, origin))

    other_entries = frozenset(seen)
    functions: list[KasadaFunction] = []
    for offset, origin in roots:
        instructions = _walk_function(offset, decoded, other_entries)
        if not instructions:
            continue
        name = {
            "entry": "entry",
            "sentinel": "<empty-frame>",
        }.get(origin, f"sub_{offset:x}")
        functions.append(
            KasadaFunction(name=name, entry=offset, origin=origin, instructions=instructions)
        )

    # Residual regions: words between decoded instructions that no successor
    # reaches.  A region that decodes into well-formed instructions ending
    # exactly on its boundary is residual code, not data, so it is surfaced as
    # a labelled root instead of being silently dropped.
    residual_roots: list[KasadaFunction] = []
    residual_data: list[str] = []
    if include_unreachable:
        owned = frozenset(
            word
            for function in functions
            for instruction in function.instructions
            for word in range(instruction.offset, instruction.offset + instruction.size)
        )
        collected: dict[int, KasadaInstruction] = {}
        for lo, _size in _region_gaps(words, owned):
            if lo in collected:
                continue
            instructions, external, failure = _residual_root(
                words, lo, owned, collected, string_table=table
            )
            if failure is None and instructions and external:
                # A stub whose only unresolved successor enters a decoded
                # function is an alternate entry trampoline for that function,
                # so its instructions belong to it rather than to a root of
                # their own.
                owners = {
                    owner
                    for target in external
                    for owner in (next((f for f in functions if target in f.offsets), None),)
                    if owner is not None
                }
                if len(owners) == 1:
                    owner = owners.pop()
                    merged = sorted(
                        (*owner.instructions, *instructions), key=lambda i: i.offset
                    )
                    functions[functions.index(owner)] = KasadaFunction(
                        name=owner.name,
                        entry=owner.entry,
                        origin=owner.origin,
                        instructions=tuple(merged),
                    )
                    for instruction in instructions:
                        collected[instruction.offset] = instruction
                        owned = owned | frozenset(
                            range(instruction.offset, instruction.offset + instruction.size)
                        )
                    continue
            if failure is not None or not instructions:
                residual_data.append(
                    f"word range starting at {lo} is data, not code"
                    + (f": {failure}" if failure else "")
                )
                continue
            residual_roots.append(
                KasadaFunction(
                    name=f"<unreachable_{lo:x}>",
                    entry=lo,
                    origin="unreachable",
                    instructions=tuple(sorted(instructions, key=lambda i: i.offset)),
                )
            )
            for instruction in instructions:
                owned = owned | frozenset(
                    range(instruction.offset, instruction.offset + instruction.size)
                )
    reported = {
        int(match.group(1))
        for message in residual_data
        for match in (_DATA_RANGE.match(message),)
        if match is not None
    }
    for lo, size in _region_gaps(
        words,
        frozenset(
            word
            for function in (*functions, *residual_roots)
            for instruction in function.instructions
            for word in range(instruction.offset, instruction.offset + instruction.size)
        ),
    ):
        if lo in reported:
            continue
        residual_data.append(f"word range [{lo}, {lo + size}) is data, not code")

    functions.extend(residual_roots)

    diagnostics.extend(residual_data)
    unreached = sorted(set(decoded) - {offset for fn in functions for offset in fn.offsets})
    if unreached:
        diagnostics.append(
            f"{len(unreached)} decoded instruction(s) belong to no function root: "
            f"{unreached[:8]}{'...' if len(unreached) > 8 else ''}"
        )

    return KasadaProgram(
        words=words,
        functions=tuple(functions),
        filename=filename,
        string_table=table,
        entry=ENTRY_PC,
        diagnostics=tuple(diagnostics),
        claimed_words=frozenset(
            word
            for function in functions
            for instruction in function.instructions
            for word in range(instruction.offset, instruction.offset + instruction.size)
        ),
    )


def looks_like_input(data: bytes, filename: str | None = None) -> bool:
    """Recognize a kasadavm artifact without executing external programs."""

    if filename and filename.lower().endswith((".kasada",)):
        return True
    if len(data) < 2 * WORD_SIZE or len(data) % WORD_SIZE:
        return False
    words = struct.unpack(f"<{len(data) // WORD_SIZE}i", data)
    # Word 0 is the HALT sentinel an empty frame executes.  Word 1 must be the
    # first executable instruction and must decode completely, operands
    # included; that structural check rejects unrelated binary data.
    if words[0] != 6:
        return False
    try:
        decode_instruction(words, ENTRY_PC)
    except FrontendDecodeError:
        return False
    return True


def decode_input(data: bytes, filename: str | None = None, *, string_table: str | None = None):
    """Legacy entry point kept for the plugin facade."""

    return decode_program(data, filename, string_table=string_table)
