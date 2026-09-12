"""Deterministic decoder for the jdvm bytecode family.

The jdvm image is a flat array of signed 32-bit words. Each word is either an
opcode or an operand, decided by the opcode table of the function that owns the
word's position. There is no embedded magic number and no header: the format is
recognized structurally, by decoding the first function region cleanly.

Decoding is a pure function of ``(data)``. Nothing here executes the VM, builds
control flow, or guesses at an instruction the tables do not describe.
"""
from __future__ import annotations

import struct

from unidecompiler.plugins import FrontendDecodeError

from .model import (
    FRONTEND_ID,
    JdvmDiagnostic,
    JdvmFunction,
    JdvmImage,
    JdvmInstruction,
    JdvmSwitchCase,
)
from .vmdef import FunctionDefinition, VmDefinition, vm_definition

WORD_BYTES = 4
_WORD = struct.Struct("<i")

#: Semantics that carry one relative branch operand at ``pc + 1``.
_RELATIVE_TARGET = frozenset(
    {
        "jump",
        "branch-false-pop",
        "branch-true-pop",
        "branch-false-keep",
        "branch-true-keep",
    }
)

#: Semantics whose single operand selects a string-pool entry at ``base + delta``.
_POOL_DELTA = frozenset(
    {
        "dup-load-member-raw",
        "load-member-raw",
        "load-member",
        "dup-load-member",
        "store-member",
        "store-member-dup",
        "push-pool",
        "push-this-member",
        "construct-pool1",
    }
)

#: Semantics whose single operand is a plain immediate value.
_IMMEDIATE = frozenset({"push-immediate", "construct-count"})

#: Table switch operand block, relative to the opcode word:
#: ``+1`` default offset, ``+2`` case count, then ``count`` (pool, offset) pairs.
_SWITCH = "switch-multiway"
#: Two consecutive pool strings starting at ``base + delta``.
_POOL_PAIR = "construct-pool2"


def looks_like_input(data: bytes, filename: str | None = None) -> bool:
    """Recognize a jdvm image without executing anything.

    The format has no magic; recognition is the structural claim that the image
    has the supported word count and that its first function region decodes
    cleanly. A filename suffix alone is never treated as proof.
    """
    if not data or len(data) % WORD_BYTES:
        return False
    definition = vm_definition()
    if len(data) != definition.image_bytes:
        return False
    words = _unpack(data)
    first = definition.functions[0]
    _instructions, issues = _decode_region(definition, words, first)
    return not issues


def decode_input(data: bytes, filename: str | None = None) -> JdvmImage:
    """Decode ``data`` into the frontend-private jdvm model."""
    definition = vm_definition()
    if not data:
        raise FrontendDecodeError("jdvm input is empty")
    if len(data) % WORD_BYTES:
        raise FrontendDecodeError(
            f"jdvm input length {len(data)} is not a multiple of {WORD_BYTES}"
        )
    if len(data) != definition.image_bytes:
        raise FrontendDecodeError(
            f"jdvm input is {len(data)} bytes but the supported VM build "
            f"({definition.family}) uses {definition.image_bytes} bytes"
        )

    words = _unpack(data)
    functions: list[JdvmFunction] = []
    diagnostics: list[JdvmDiagnostic] = []
    for function in definition.functions:
        instructions, issues = _decode_region(definition, words, function)
        diagnostics.extend(issues)
        functions.append(
            JdvmFunction(
                entry=function.entry,
                end=function.end,
                name=function.name,
                locals=function.locals,
                instructions=tuple(instructions),
            )
        )

    return JdvmImage(
        words=words,
        pool=definition.pool,
        functions=tuple(functions),
        diagnostics=tuple(diagnostics),
        metadata={
            "frontend": FRONTEND_ID,
            "bytecode_format": "jdvm",
            "word_bytes": definition.word_bytes,
            "image_bytes": definition.image_bytes,
            "function_count": len(functions),
            "instruction_count": sum(len(f.instructions) for f in functions),
            "pool_size": len(definition.pool),
            "decoder": definition.decoder_name,
            "decoder_xor": definition.decoder_xor,
        },
    )


def _unpack(data: bytes) -> tuple[int, ...]:
    return tuple(value for (value,) in _WORD.iter_unpack(data))


def _decode_region(
    definition: VmDefinition,
    words: tuple[int, ...],
    function: FunctionDefinition,
) -> tuple[list[JdvmInstruction], list[JdvmDiagnostic]]:
    instructions: list[JdvmInstruction] = []
    diagnostics: list[JdvmDiagnostic] = []
    offset = function.entry
    while offset < function.end:
        instruction, issues = _decode_one(definition, words, function, offset)
        instructions.append(instruction)
        diagnostics.extend(issues)
        offset += instruction.size
    return instructions, diagnostics


def _decode_one(
    definition: VmDefinition,
    words: tuple[int, ...],
    function: FunctionDefinition,
    offset: int,
) -> tuple[JdvmInstruction, list[JdvmDiagnostic]]:
    opcode = words[offset]
    cell = function.cell(opcode)
    if cell is None:
        diagnostic = JdvmDiagnostic(
            code="jdvm.unknown-opcode",
            offset=offset,
            message=(
                f"opcode {opcode} is not described by the opcode table of "
                f"function {function.name} (entry {function.entry})"
            ),
            raw=f"word[{offset}]={opcode}",
        )
        return (
            JdvmInstruction(
                offset=offset,
                size=1,
                opcode=opcode,
                semantic="unknown",
                canonical_index=-1,
                function=function.name,
                diagnostics=(diagnostic,),
            ),
            [diagnostic],
        )

    semantic = cell.semantic
    size = 1 + max(cell.operand_words, 0)
    issues: list[JdvmDiagnostic] = []
    operands: tuple[int, ...] = ()
    pool_index: int | None = None
    pool_value: str | None = None
    pool_secondary: str | None = None
    target: int | None = None
    cases: tuple[JdvmSwitchCase, ...] = ()
    default_target: int | None = None

    if semantic == _SWITCH:
        count = words[offset + 2] if offset + 2 < function.end else 0
        if count < 0 or offset + 3 + 2 * count > function.end:
            diagnostic = JdvmDiagnostic(
                code="jdvm.invalid-switch",
                offset=offset,
                message=(
                    f"table switch declares {count} cases but its operand block "
                    f"does not fit before the end of function {function.name}"
                ),
                raw=_raw_text(function, offset, semantic, (count,)),
            )
            issues.append(diagnostic)
            return (
                JdvmInstruction(
                    offset=offset,
                    size=max(size, 3),
                    opcode=opcode,
                    semantic=semantic,
                    canonical_index=cell.canonical_index,
                    operands=(count,),
                    function=function.name,
                    diagnostics=(diagnostic,),
                ),
                issues,
            )
        size = 3 + 2 * count
        operands = tuple(words[offset + 1 : offset + size])
        arms: list[JdvmSwitchCase] = []
        for index in range(count):
            delta = words[offset + 3 + 2 * index]
            arm_target = offset + 1 + words[offset + 4 + 2 * index]
            pool = definition.pool_at(cell.pool_base, delta)
            if pool is None:
                issues.append(
                    _pool_diagnostic(
                        function, offset, semantic, cell.pool_base + delta
                    )
                )
            arms.append(
                JdvmSwitchCase(
                    pool_index=cell.pool_base + delta,
                    pool_value=pool,
                    target=arm_target,
                )
            )
        cases = tuple(arms)
        default_target = offset + 1 + words[offset + 1]
    elif semantic in _RELATIVE_TARGET:
        delta = words[offset + 1]
        operands = (delta,)
        target = offset + 1 + delta
    elif semantic in _POOL_DELTA:
        delta = words[offset + 1]
        operands = (delta,)
        pool_index = cell.pool_base + delta
        pool_value = definition.pool_at(cell.pool_base, delta)
        if pool_value is None:
            issues.append(
                _pool_diagnostic(function, offset, semantic, pool_index)
            )
    elif semantic == _POOL_PAIR:
        delta = words[offset + 1]
        operands = (delta,)
        pool_index = cell.pool_base + delta
        pool_value = definition.pool_at(cell.pool_base, delta)
        pool_secondary = definition.pool_at(cell.pool_base, delta + 1)
        if pool_value is None or pool_secondary is None:
            issues.append(
                _pool_diagnostic(function, offset, semantic, pool_index)
            )
    elif semantic in _IMMEDIATE:
        operands = (words[offset + 1],)

    if target is not None and not (function.entry <= target < function.end):
        issues.append(
            JdvmDiagnostic(
                code="jdvm.target-outside-function",
                offset=offset,
                message=(
                    f"branch target {target} leaves the region of function "
                    f"{function.name} [{function.entry}, {function.end})"
                ),
                raw=_raw_text(function, offset, semantic, operands),
            )
        )
    for arm in cases:
        if not (function.entry <= arm.target < function.end):
            issues.append(
                JdvmDiagnostic(
                    code="jdvm.target-outside-function",
                    offset=offset,
                    message=(
                        f"switch arm target {arm.target} leaves the region of "
                        f"function {function.name} "
                        f"[{function.entry}, {function.end})"
                    ),
                    raw=_raw_text(function, offset, semantic, operands),
                )
            )
    if default_target is not None and not (
        function.entry <= default_target < function.end
    ):
        issues.append(
            JdvmDiagnostic(
                code="jdvm.target-outside-function",
                offset=offset,
                message=(
                    f"switch default target {default_target} leaves the region "
                    f"of function {function.name} "
                    f"[{function.entry}, {function.end})"
                ),
                raw=_raw_text(function, offset, semantic, operands),
            )
        )

    instruction = JdvmInstruction(
        offset=offset,
        size=size,
        opcode=opcode,
        semantic=semantic,
        canonical_index=cell.canonical_index,
        operands=operands,
        pool_index=pool_index,
        pool_value=pool_value,
        pool_secondary=pool_secondary,
        closure_entry=cell.closure_entry,
        symbols=cell.symbols,
        target=target,
        cases=cases,
        default_target=default_target,
        function=function.name,
        diagnostics=tuple(issues),
    )
    return instruction, issues


def _pool_diagnostic(
    function: FunctionDefinition,
    offset: int,
    semantic: str,
    index: int,
) -> JdvmDiagnostic:
    return JdvmDiagnostic(
        code="jdvm.pool-out-of-range",
        offset=offset,
        message=(
            f"string-pool index {index} is outside the {len(vm_definition().pool)}"
            f"-entry pool for function {function.name}"
        ),
        raw=_raw_text(function, offset, semantic, ()),
    )


def _raw_text(
    function: FunctionDefinition,
    offset: int,
    semantic: str,
    operands: tuple[int, ...],
) -> str:
    suffix = " ".join(str(operand) for operand in operands)
    text = f"{function.name}+{offset} {semantic}"
    return f"{text} {suffix}".strip()
