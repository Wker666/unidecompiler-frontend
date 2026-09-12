"""Deterministic decoder for the mtgsigvm bytecode container.

Two layers are decoded:

1. The container: a section table plus function, constant and string sections.
   This mirrors ``aS``'s parser in ``analysis_inputs/interpreter/H5guard.js``
   exactly, including its big-endian reads, its reversed section order and its
   XOR + UTF-16BE string pool.
2. The instruction stream of each function: one-byte opcode, signed LEB128
   operands, 4-byte big-endian relative jumps and the inline switch table.

Nothing here executes VM bytecode. The decoder only reads bytes and maps
operands; branches are recorded as targets, never followed.
"""

from __future__ import annotations

import struct
from dataclasses import replace

from unidecompiler.plugins import FrontendDecodeError

from .model import (
    MAX_CONSTANTS,
    MAX_FUNCTIONS,
    MAX_INSTRUCTIONS_PER_FUNCTION,
    MAX_SECTIONS,
    MAX_STRINGS,
    SECTION_CONSTANTS,
    SECTION_FUNCTION,
    SECTION_STRINGS,
    SECTION_TYPES,
    MtgsigDiagnostic,
    MtgsigFunction,
    MtgsigInstruction,
    MtgsigModule,
    MtgsigOperand,
)
from .opcodes import CALL_OPCODES, OPCODES, OpcodeInfo
from .support import DEFAULT_STRING_XOR_KEY, derive_string_xor_key

#: The interpreter's outermost opcode is ``CALL``: its dispatch ends in an
#: ``else`` that treats every opcode it has not matched as an apply call.
CANONICAL_CALL_OPCODE = 58
MAX_CALL_OPCODE = 255

#: Literal-push opcodes whose pushed value is available at decode time.
_LITERAL_PUSH_OPCODES = frozenset({39, 40, 43, 44, 45, 46})


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------


def _shl32(value: int, shift: int) -> int:
    return (value << shift) & 0xFFFFFFFF


def _asr32(value: int, shift: int) -> int:
    value &= 0xFFFFFFFF
    if value & 0x80000000:
        value -= 0x100000000
    return value >> shift


def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    """Signed LEB128, byte-for-byte identical to the interpreter's ``k4()``."""
    if pos >= len(buf):
        raise FrontendDecodeError("mtgsigvm: truncated varint operand")
    byte = buf[pos]
    pos += 1
    if byte <= 0x7F:
        return _asr32(_shl32(byte, 25), 25), pos
    value = byte & 0x7F
    if pos >= len(buf):
        raise FrontendDecodeError("mtgsigvm: truncated varint operand")
    byte = buf[pos]
    pos += 1
    value |= (byte & 0x7F) << 7
    if byte <= 0x7F:
        return _asr32(_shl32(value, 18), 18), pos
    if pos >= len(buf):
        raise FrontendDecodeError("mtgsigvm: truncated varint operand")
    byte = buf[pos]
    pos += 1
    value |= (byte & 0x7F) << 14
    if byte <= 0x7F:
        return _asr32(_shl32(value, 11), 11), pos
    if pos >= len(buf):
        raise FrontendDecodeError("mtgsigvm: truncated varint operand")
    byte = buf[pos]
    pos += 1
    value |= (byte & 0x7F) << 21
    if byte <= 0x7F:
        return _asr32(_shl32(value, 4), 4), pos
    if pos >= len(buf):
        raise FrontendDecodeError("mtgsigvm: truncated varint operand")
    byte = buf[pos]
    pos += 1
    return _asr32(value | _shl32(byte, 28), 0), pos


def _u32(buf: bytes, pos: int) -> int:
    if pos + 4 > len(buf):
        raise FrontendDecodeError("mtgsigvm: truncated container field")
    return struct.unpack_from(">I", buf, pos)[0]


def _f64(buf: bytes, pos: int) -> float:
    if pos + 8 > len(buf):
        raise FrontendDecodeError("mtgsigvm: truncated float64 constant")
    return struct.unpack_from(">d", buf, pos)[0]


def _decode_string_pool_entry(raw: bytes, key: str) -> str:
    """Reproduce the interpreter's byte-XOR plus UTF-16BE string decoding."""
    if not key:
        return raw.decode("utf-16-be", "surrogatepass")
    keyed = bytes(
        byte ^ ord(key[index % len(key)]) for index, byte in enumerate(raw)
    )
    if len(keyed) % 2:
        keyed += b"\x00"
    return keyed.decode("utf-16-be", "surrogatepass")


# --------------------------------------------------------------------------
# instruction decoding
# --------------------------------------------------------------------------


def _stack_effect(instruction: MtgsigInstruction) -> tuple[int, int]:
    """Operand-stack arity of a decoded instruction.

    Decoding metadata only: the frontend never maintains a value stack. It is
    used to keep emitted effect shapes exact and to run the local scope-key
    window rule.
    """
    info = OPCODES.get(instruction.opcode)
    if info is None:
        # The interpreter treats every unmatched opcode as CALL.
        return 3, 1
    pops, pushes = info.pops, info.pushes
    if instruction.opcode in (30, 34, 41, 42, 56):
        count = _operand_count(instruction)
        if instruction.opcode == 34:  # TRY: optional finally function
            pops = 4 if count else 3
        elif instruction.opcode == 56:  # SWITCH: case values plus the subject
            pops = count + 1
        elif instruction.opcode == 41:  # BUILDOBJ: key and value per property
            pops = 2 * count
        else:  # CONCAT / BUILDARR: one input per element
            pops = count
    return pops, pushes


def _operand_count(instruction: MtgsigInstruction) -> int:
    """First operand as a count, ignoring non-numeric operand roles."""
    if not instruction.operands:
        return 0
    value = instruction.operands[0].value
    return value if isinstance(value, int) else 0


def _decode_stream(code: bytes, artifact_base: int, function_name: str):
    instructions: list[MtgsigInstruction] = []
    diagnostics: list[MtgsigDiagnostic] = []
    pos = 0
    limit = len(code)
    while pos < limit:
        if len(instructions) >= MAX_INSTRUCTIONS_PER_FUNCTION:
            raise FrontendDecodeError(
                "mtgsigvm: function %r exceeds the instruction limit" % function_name
            )
        start = pos
        opcode = code[pos]
        pos += 1
        info: OpcodeInfo | None = OPCODES.get(opcode)
        if info is None and opcode > MAX_CALL_OPCODE:
            raise FrontendDecodeError(
                "mtgsigvm: invalid opcode byte %d at %d in function %r"
                % (opcode, start, function_name)
            )
        if info is None:
            diagnostics.append(
                MtgsigDiagnostic(
                    code="vm.non-canonical-call-opcode",
                    message=(
                        "opcode byte %d is handled as CALL by the interpreter's "
                        "final dispatch branch" % opcode
                    ),
                    function=function_name,
                    offset=start,
                )
            )
        operands: list[MtgsigOperand] = []
        targets: list[int] = []
        fallthrough: int | None = None

        if info is None:
            kind = "none"
        else:
            kind = info.operand

        if kind == "varint":
            value, pos = _read_varint(code, pos)
            operands.append(MtgsigOperand(role="operand", value=value, text=str(value)))
        elif kind == "rel32":
            if pos + 4 > limit:
                raise FrontendDecodeError(
                    "mtgsigvm: truncated jump operand at %d in %r" % (start, function_name)
                )
            relative = struct.unpack_from(">i", code, pos)[0]
            pos += 4
            operands.append(
                MtgsigOperand(role="target", value=relative, text="%+d" % relative)
            )
            targets.append(pos + relative)
        elif kind == "cases":
            count, pos = _read_varint(code, pos)
            count = int(count)
            if count < 0:
                raise FrontendDecodeError(
                    "mtgsigvm: negative switch case count at %d in %r"
                    % (start, function_name)
                )
            operands.append(MtgsigOperand(role="count", value=count, text=str(count)))
            table_start = pos
            if table_start + 4 * count > limit:
                raise FrontendDecodeError(
                    "mtgsigvm: truncated switch table at %d in %r" % (start, function_name)
                )
            for index in range(count):
                entry = struct.unpack_from(">i", code, table_start + 4 * index)[0]
                target = table_start + 4 * index + 4 + entry
                operands.append(
                    MtgsigOperand(role="case", value=target, text="case[%d]=%d" % (index, target))
                )
                targets.append(target)
            pos = table_start + 4 * count
            fallthrough = table_start + 4 * count
            operands.append(
                MtgsigOperand(
                    role="target",
                    value=fallthrough,
                    text="default=%d" % fallthrough,
                )
            )
            targets.append(fallthrough)

        decoded_operands = tuple(operands)
        if kind in ("varint",) and opcode == 45:
            decoded_operands = (
                MtgsigOperand(
                    role="constant",
                    value=decoded_operands[0].value,
                    text="const[%d]" % decoded_operands[0].value,
                ),
            )
        elif kind == "varint" and opcode == 46:
            decoded_operands = (
                MtgsigOperand(
                    role="string",
                    value=decoded_operands[0].value,
                    text="string[%d]" % decoded_operands[0].value,
                ),
            )

        if info is not None and info.terminates:
            # RETURN/RET/THROW and the unconditional JMP have no linear
            # successor; JMPIF/JMPNOT and SWITCH fall through.
            fallthrough = None
        elif fallthrough is None:
            fallthrough = pos

        mnemonic = info.mnemonic if info is not None else "CALL"
        raw = "%05d  %s" % (start, mnemonic)
        if decoded_operands:
            raw += "  " + " ".join(operand.text for operand in decoded_operands)

        instructions.append(
            MtgsigInstruction(
                offset=start,
                opcode=opcode,
                mnemonic=mnemonic,
                operands=decoded_operands,
                size=pos - start,
                raw=raw,
                artifact_offset=artifact_base + start,
                targets=tuple(targets),
                fallthrough=fallthrough,
            )
        )
    return tuple(instructions), tuple(diagnostics)


#: Bound on the straight-line window used to resolve a scope key.
MAX_SCOPE_WINDOW = 64

#: Bound on the walk that proves a produced value is never read.
MAX_DEAD_WINDOW = 4096

#: Opcodes that leave a value the compiled corpus routinely discards: the call
#: result and the ``delete`` boolean.
_DEAD_RESULT_OPCODES = frozenset({0, 4, 29, 37, 58})


def resolve_stream_literals(
    instructions: tuple[MtgsigInstruction, ...],
    strings: tuple[str, ...],
    function_name: str,
) -> tuple[tuple[MtgsigInstruction, ...], tuple[MtgsigDiagnostic, ...]]:
    """Name stack-supplied literals when a straight-line window proves them.

    ``GETSCOPE``/``SETSCOPE``/``SETGLOBAL`` take their key from the operand
    stack rather than the instruction stream. Recovering the slot name is pure
    operand mapping: the decoder walks backwards over straight-line code only,
    counting operand-stack slots, until it reaches the literal push that
    produced the key. It evaluates no values, follows no branch and builds no
    frames. The walk stops at join points, at terminating instructions, at the
    function start and after ``MAX_SCOPE_WINDOW`` instructions; anything not
    proven this way stays unnamed and is reported as a diagnostic.
    """
    join_points: set[int] = set()
    for instruction in instructions:
        join_points.update(instruction.targets)

    resolved: list[MtgsigInstruction] = []
    diagnostics: list[MtgsigDiagnostic] = []

    def entered_by_fallthrough(index: int) -> bool:
        """True when instruction ``index`` is entered from its predecessor."""
        return index > 0 and instructions[index - 1].fallthrough == instructions[index].offset

    def literal_slot(index: int) -> str | None:
        instruction = instructions[index]
        if instruction.opcode != 46 or not instruction.operands:
            return None
        operand = instruction.operands[0]
        value = operand.value
        if operand.role == "string" and isinstance(value, int) and 0 <= value < len(strings):
            return strings[value]
        return None

    def key_slot(consumer: int, depth: int) -> str | None:
        """Literal push that produced the value ``depth`` slots below the top.

        Walks backwards along the fallthrough chain, undoing each instruction's
        operand-stack arity, until the slot turns out to be one of an
        instruction's pushed results. The walk aborts at any join point, because
        a second incoming edge can leave a different value in that slot, and at
        the window bound. Only a string push names a slot.
        """
        if instructions[consumer].offset in join_points:
            return None
        if not entered_by_fallthrough(consumer):
            return None
        index = consumer - 1
        hops = 0
        while index >= 0 and hops < MAX_SCOPE_WINDOW:
            if instructions[index].offset in join_points:
                return None
            if index > 0 and not entered_by_fallthrough(index):
                return None
            pops, pushes = _stack_effect(instructions[index])
            if 1 <= depth <= pushes:
                return literal_slot(index) if (pushes == 1 and pops == 0) else None
            below = depth - pushes
            if below < 1:
                return None
            depth = below + pops
            index -= 1
            hops += 1
        return None

    by_offset = {
        instruction.offset: position for position, instruction in enumerate(instructions)
    }

    def dead_result(index: int) -> bool:
        """True when an instruction's produced value is provably never read.

        The interpreter pushes a result for every call and for ``delete``, but
        this compiler emits expression statements and leaves those results on
        the operand stack, which makes the operand-stack depth inconsistent at
        every merge a discarded value reaches.

        The walk proves the slot is unread: it explores the reachable
        continuations (fallthrough, jump target and conditional target),
        tracking how deep the slot sits, and stops a path as soon as a pop would
        consume it or a path reaches the same offset with the slot no shallower
        than before (a deeper slot is strictly safer, so the earlier result
        dominates). It only concludes at a function-ending transfer, and never
        exhausts the budget into a positive answer. Values that flow into a
        branch merge are therefore still seen being consumed.
        """
        call = instructions[index]
        work: list[tuple[int, int]] = [(index + 1, 1)]
        deepest: dict[int, int] = {}
        budget = MAX_DEAD_WINDOW
        while work:
            if budget <= 0:
                return False
            budget -= 1
            position, depth = work.pop()
            if position < 0 or position >= len(instructions):
                continue
            instruction = instructions[position]
            previous = deepest.get(instruction.offset)
            if previous is not None and previous <= depth:
                continue
            deepest[instruction.offset] = depth
            pops, pushes = _stack_effect(instruction)
            if depth - pops < 1:
                return False
            next_depth = depth - pops + pushes
            if instruction.fallthrough is None:
                if instruction.mnemonic in ("RET", "RETURN", "THROW"):
                    continue
                if instruction.mnemonic == "JMP" and instruction.targets:
                    target = by_offset.get(instruction.targets[0])
                    if target is not None:
                        work.append((target, next_depth))
                continue
            work.append((position + 1, next_depth))
            if instruction.mnemonic in ("JMPIF", "JMPNOT") and instruction.targets:
                target = by_offset.get(instruction.targets[0])
                if target is not None:
                    work.append((target, next_depth))
        return True

    for index, instruction in enumerate(instructions):
        if instruction.opcode in _DEAD_RESULT_OPCODES and dead_result(index):
            resolved.append(replace(instruction, dead_result=True))
            continue
        if instruction.opcode == 57:
            # CLOSURE takes the referenced VM function name from the operand
            # stack; record it when the producer is provable.
            name = key_slot(index, 1)
            if name is not None:
                instruction = replace(
                    instruction,
                    closure_name=name,
                    raw="%s  closure=%r" % (instruction.raw, name),
                )
            resolved.append(instruction)
            continue
        if instruction.opcode not in (50, 51, 52):
            resolved.append(instruction)
            continue
        depth = 1 if instruction.opcode == 51 else 2
        slot = key_slot(index, depth)
        if slot is None:
            diagnostics.append(
                MtgsigDiagnostic(
                    code="vm.unresolved-scope-key",
                    message=(
                        "%s key is supplied by the operand stack and is not "
                        "provable from a straight-line window; the slot stays "
                        "unnamed" % instruction.mnemonic
                    ),
                    function=function_name,
                    offset=instruction.offset,
                )
            )
            resolved.append(instruction)
            continue
        key_operand = MtgsigOperand(role="scope-key", value=slot, text="key=%r" % slot)
        resolved.append(
            MtgsigInstruction(
                offset=instruction.offset,
                opcode=instruction.opcode,
                mnemonic=instruction.mnemonic,
                operands=(key_operand,),
                size=instruction.size,
                raw="%s  key=%r" % (instruction.raw, slot),
                artifact_offset=instruction.artifact_offset,
                targets=instruction.targets,
                fallthrough=instruction.fallthrough,
                scope_slot=slot,
                scope_key=slot,
            )
        )
    return tuple(resolved), tuple(diagnostics)


# --------------------------------------------------------------------------
# container decoding
# --------------------------------------------------------------------------


def _section_table(data: bytes) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    if len(data) < 9:
        raise FrontendDecodeError("mtgsigvm: artifact is shorter than the header")
    section_count = _u32(data, 0)
    declared_size = _u32(data, 4)
    debug_tables = data[8]
    if not 1 <= section_count <= MAX_SECTIONS:
        raise FrontendDecodeError(
            "mtgsigvm: implausible section count %d" % section_count
        )
    if declared_size != len(data):
        raise FrontendDecodeError(
            "mtgsigvm: header length %d does not match artifact size %d"
            % (declared_size, len(data))
        )
    pos = 9
    entries: list[tuple[int, int]] = []
    for _ in range(section_count):
        section_type = _u32(data, pos)
        offset = _u32(data, pos + 4)
        pos += 8
        if section_type not in SECTION_TYPES:
            raise FrontendDecodeError(
                "mtgsigvm: unknown section type %d" % section_type
            )
        if offset >= len(data):
            raise FrontendDecodeError(
                "mtgsigvm: section %d offset %d is out of range" % (section_type, offset)
            )
        entries.append((section_type, offset))
    if pos > len(data):
        raise FrontendDecodeError("mtgsigvm: truncated section table")
    return section_count, debug_tables, tuple(entries)


def parse_container(
    data: bytes,
    *,
    string_xor_key: str | None = None,
    name: str = "<mtgsigvm>",
) -> MtgsigModule:
    """Decode a mtgsigvm container into the frontend-private model."""
    key = DEFAULT_STRING_XOR_KEY if string_xor_key is None else string_xor_key
    section_count, debug_tables, sections = _section_table(data)

    function_entries: list[tuple[int, int, int, int]] = []
    constants: list[float] = []
    strings: list[str] = []
    diagnostics: list[MtgsigDiagnostic] = []

    # The interpreter walks the section table backwards so the string pool is
    # populated before function sections ask it for their names.
    for section_type, offset in reversed(sections):
        if section_type == SECTION_STRINGS:
            count = _u32(data, offset)
            if count > MAX_STRINGS:
                raise FrontendDecodeError("mtgsigvm: implausible string count %d" % count)
            pos = offset + 4
            base = pos + 8 * count
            for _ in range(count):
                relative = _u32(data, pos)
                length = _u32(data, pos + 4)
                pos += 8
                start = base + relative
                end = start + length
                if end > len(data):
                    raise FrontendDecodeError("mtgsigvm: string entry is out of range")
                strings.append(_decode_string_pool_entry(data[start:end], key))
        elif section_type == SECTION_CONSTANTS:
            count = _u32(data, offset)
            if count > MAX_CONSTANTS:
                raise FrontendDecodeError("mtgsigvm: implausible constant count %d" % count)
            pos = offset + 4
            base = pos + 4 * count
            for _ in range(count):
                relative = _u32(data, pos)
                pos += 4
                constants.append(_f64(data, base + relative))
        else:
            count = _u32(data, offset)
            if count > MAX_FUNCTIONS:
                raise FrontendDecodeError("mtgsigvm: implausible function count %d" % count)
            pos = offset + 4
            base = pos + 16 * count
            for _ in range(count):
                descriptor = _u32(data, pos)
                name_index = _u32(data, pos + 4)
                code_relative = _u32(data, pos + 8)
                code_length = _u32(data, pos + 12)
                pos += 16
                function_entries.append(
                    (descriptor, name_index, base + code_relative, code_length)
                )

    functions: list[MtgsigFunction] = []
    for index, (descriptor, name_index, code_offset, code_length) in enumerate(
        function_entries
    ):
        if name_index >= len(strings):
            function_name = "<function_%d>" % index
        else:
            function_name = strings[name_index]
        end = code_offset + code_length
        if end > len(data):
            raise FrontendDecodeError(
                "mtgsigvm: function %r code range is out of bounds" % function_name
            )
        code = data[code_offset:end]
        instructions, stream_diagnostics = _decode_stream(code, code_offset, function_name)
        instructions, scope_diagnostics = resolve_stream_literals(
            instructions, tuple(strings), function_name
        )
        diagnostics.extend(stream_diagnostics)
        diagnostics.extend(scope_diagnostics)
        referenced = _closure_targets(instructions, tuple(strings))
        functions.append(
            MtgsigFunction(
                name=function_name,
                index=index,
                flags=descriptor,
                code=code,
                code_offset=code_offset,
                instructions=instructions,
                referenced_names=referenced,
            )
        )

    return MtgsigModule(
        name=name,
        artifact_size=len(data),
        section_count=section_count,
        debug_tables=bool(debug_tables),
        sections=sections,
        functions=tuple(functions),
        constants=tuple(constants),
        strings=tuple(strings),
        string_xor_key=key,
        diagnostics=tuple(diagnostics),
    )


def _closure_targets(
    instructions: tuple[MtgsigInstruction, ...], strings: tuple[str, ...]
) -> tuple[str, ...]:
    """VM function names captured by CLOSURE instructions, in first-use order."""
    names: list[str] = []
    for instruction in instructions:
        if instruction.opcode != 57 or instruction.closure_name is None:
            continue
        if instruction.closure_name not in names:
            names.append(instruction.closure_name)
    return tuple(names)


# --------------------------------------------------------------------------
# public entry points
# --------------------------------------------------------------------------


def _probe(data: bytes) -> bool:
    try:
        _section_table(data)
    except FrontendDecodeError:
        return False
    except Exception:  # pragma: no cover - defensive
        return False
    return True


def looks_like_input(data: bytes, filename: str | None = None) -> bool:
    """Recognize a mtgsigvm artifact without executing anything."""
    if _probe(data):
        return True
    if not filename:
        return False
    lowered = filename.lower()
    return lowered.endswith((".mtgsig",))


def decode_input(data: bytes, filename: str | None = None) -> MtgsigModule:
    """Decode raw bytes into the frontend-private model."""
    if not data:
        raise FrontendDecodeError("mtgsigvm: empty input")
    name = filename or "<mtgsigvm>"
    return parse_container(data, name=name)


__all__ = [
    "decode_input",
    "derive_string_xor_key",
    "looks_like_input",
    "parse_container",
]
