"""Deterministic decoder for the bdvm packed payload format.

The implementation performs pure data parsing. It never executes, imports, or
sends an input artifact anywhere.

Unpacking chain:

1. base64 text (``atob``) or already-decoded packed bytes;
2. ``key = sum(bytes[4:8]) % 256``;
3. ``body[i] = packed[8 + i] ^ ((key + key % 10 * i) % 256)``;
4. raw-deflate inflate (``wbits = -15``);
5. a 7-bit little-endian varint stream: string pool ``Z`` then function table
   ``z``.
"""

from __future__ import annotations

import base64
import binascii
import re
import zlib

from unidecompiler.plugins import FrontendDecodeError

from .model import (
    BRANCH_OPCODES,
    OPERAND_COUNT,
    OPCODE_MNEMONICS,
    BdvmFormatError,
    BdvmFunction,
    BdvmInstruction,
    BdvmModule,
    BdvmRegion,
)

PACKED_MAGIC = b"PK\x02\x00"
"""First four bytes of the packed stream; the next four bytes carry the key."""

_BASE64_RE = re.compile(rb"^[A-Za-z0-9+/=\s]+$")

SUPPORTED_SUFFIXES = (".bd",)

# The three module entry points selected by the caller in the interpreter
# source (``J(232, ...)`` at line 4399, ``J(728, ...)`` at line 4454 and
# ``J(731, ...)`` at line 4463).
KNOWN_ENTRY_INDICES: tuple[int, ...] = (232, 728, 731)


def looks_like_input(data: bytes, filename: str | None = None) -> bool:
    """Recognize a bdvm packed payload without executing anything."""

    if not data:
        return bool(filename and _has_supported_suffix(filename))
    if data.startswith(PACKED_MAGIC):
        return True
    if filename and _has_supported_suffix(filename):
        return True
    stripped = _strip_whitespace(data)
    if len(stripped) < 8 or not _BASE64_RE.match(stripped):
        return False
    try:
        head = base64.b64decode(stripped[:16], validate=True)
    except (binascii.Error, ValueError):
        return False
    return head.startswith(PACKED_MAGIC)


def _has_supported_suffix(filename: str) -> bool:
    lowered = filename.lower()
    return any(lowered.endswith(suffix) for suffix in SUPPORTED_SUFFIXES)


def _strip_whitespace(data: bytes) -> bytes:
    return b"".join(data.split())


def decode_input(data: bytes, filename: str | None = None) -> BdvmModule:
    """Decode a packed bdvm payload into the frontend-private model."""

    packed, xor_key = _unpack_bytes(data, filename)
    inflated = _inflate(packed)
    return _parse_payload(
        inflated, packed_size=len(packed), xor_key=xor_key
    )


def _unpack_bytes(data: bytes, filename: str | None) -> tuple[bytes, int]:
    if data.startswith(PACKED_MAGIC):
        packed = data
    else:
        stripped = _strip_whitespace(data)
        if not stripped:
            raise FrontendDecodeError("bdvm input is empty")
        try:
            packed = base64.b64decode(stripped, validate=True)
        except (binascii.Error, ValueError) as error:
            raise FrontendDecodeError(f"bdvm input is not valid base64: {error}") from error
        if not packed.startswith(PACKED_MAGIC):
            raise FrontendDecodeError(
                "bdvm packed stream does not start with the expected magic "
                f"{PACKED_MAGIC!r}"
            )
        if len(packed) <= 8:
            raise FrontendDecodeError("bdvm packed stream is truncated before its body")
    return packed, sum(packed[4:8]) % 256


def _inflate(packed: bytes) -> bytes:
    key = sum(packed[4:8]) % 256
    body = bytes(
        (value ^ ((key + (key % 10) * index) % 256)) & 0xFF
        for index, value in enumerate(packed[8:])
    )
    try:
        return zlib.decompress(body, -15)
    except zlib.error as error:
        raise FrontendDecodeError(f"bdvm payload is not valid raw-deflate data: {error}") from error


class _Reader:
    """Cursor over the inflated varint stream (``W``/``K`` semantics)."""

    __slots__ = ("data", "index")

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.index = 0

    def read_int(self) -> int:
        result = 0
        shift = 0
        while True:
            if self.index >= len(self.data):
                raise BdvmFormatError("truncated varint in bdvm payload")
            byte = self.data[self.index]
            self.index += 1
            result |= (byte & 127) << shift
            shift += 7
            if not byte & 128:
                if shift < 32 and byte & 64:
                    return result | (-1 << shift)
                return result

    def read_string(self) -> str:
        accumulator = -1
        codepoints: list[int] = []
        while True:
            if self.index >= len(self.data):
                raise BdvmFormatError("unterminated string in bdvm payload")
            byte = self.data[self.index]
            self.index += 1
            if 128 <= byte < 192:
                accumulator = (accumulator << 6) + (byte & 63)
            else:
                if accumulator >= 0:
                    codepoints.append(accumulator)
                if byte < 128:
                    accumulator = byte
                elif byte < 224:
                    accumulator = byte & 31
                elif byte < 240:
                    accumulator = byte & 15
                elif byte < 248:
                    accumulator = byte & 7
                else:
                    break
        try:
            return "".join(chr(codepoint) for codepoint in codepoints)
        except ValueError as error:
            raise BdvmFormatError(f"invalid code point in bdvm string pool: {error}") from error

    @property
    def remaining(self) -> int:
        return len(self.data) - self.index


def _parse_payload(inflated: bytes, *, packed_size: int, xor_key: int) -> BdvmModule:
    reader = _Reader(inflated)
    try:
        string_count = reader.read_int()
        if string_count < 0:
            raise BdvmFormatError("negative string-pool count")
        strings = tuple(reader.read_string() for _ in range(string_count))
        function_count = reader.read_int()
        if function_count < 0:
            raise BdvmFormatError("negative function count")
        functions = tuple(
            _parse_function(reader, index) for index in range(function_count)
        )
    except BdvmFormatError as error:
        raise FrontendDecodeError(f"malformed bdvm payload: {error}") from error
    if reader.remaining:
        raise FrontendDecodeError(
            f"malformed bdvm payload: {reader.remaining} trailing bytes after the "
            "function table"
        )
    entries = tuple(index for index in KNOWN_ENTRY_INDICES if index < len(functions))
    return BdvmModule(
        strings=strings,
        functions=functions,
        entry_indices=entries,
        packed_size=packed_size,
        inflated_size=len(inflated),
        xor_key=xor_key,
        format_metadata={
            "format": "bdvm-packed",
            "string_count": len(strings),
            "function_count": len(functions),
            "instruction_count": sum(
                function.instruction_count for function in functions
            ),
        },
    )


def _parse_function(reader: _Reader, index: int) -> BdvmFunction:
    try:
        params = reader.read_int()
        is_global = bool(reader.read_int())
        region_count = reader.read_int()
        if region_count < 0:
            raise BdvmFormatError("negative exception-region count")
        regions = tuple(
            BdvmRegion(
                try_start=reader.read_int(),
                handler_start=reader.read_int(),
                finally_start=reader.read_int(),
                finally_end=reader.read_int(),
            )
            for _ in range(region_count)
        )
        code_length = reader.read_int()
        if code_length < 0:
            raise BdvmFormatError("negative instruction-cell count")
        code = tuple(reader.read_int() for _ in range(code_length))
    except BdvmFormatError as error:
        raise FrontendDecodeError(
            f"malformed bdvm function {index}: {error}"
        ) from error
    instructions = _decode_instructions(code, index)
    return BdvmFunction(
        index=index,
        name=f"func_{index}",
        params=params,
        is_global=is_global,
        code=code,
        regions=regions,
        instructions=instructions,
    )


def _decode_instructions(code: tuple[int, ...], function_index: int) -> tuple[BdvmInstruction, ...]:
    instructions: list[BdvmInstruction] = []
    offset = 0
    total = len(code)
    while offset < total:
        opcode = code[offset]
        if opcode not in OPERAND_COUNT:
            raise FrontendDecodeError(
                f"bdvm function {function_index} cell {offset}: unknown opcode "
                f"{opcode} (supported range 0..{max(OPCODE_MNEMONICS)})"
            )
        count = OPERAND_COUNT[opcode]
        operands = code[offset + 1 : offset + 1 + count]
        if len(operands) != count:
            raise FrontendDecodeError(
                f"bdvm function {function_index} cell {offset}: opcode {opcode} "
                f"({OPCODE_MNEMONICS[opcode]}) is truncated; expected {count} "
                f"operand cell(s)"
            )
        target = (
            offset + 2 + operands[0] if opcode in BRANCH_OPCODES else None
        )
        instructions.append(
            BdvmInstruction(
                offset=offset,
                opcode=opcode,
                mnemonic=OPCODE_MNEMONICS[opcode],
                operands=tuple(operands),
                target=target,
                size=1 + count,
                raw=_raw_text(opcode, operands, target),
            )
        )
        offset += 1 + count
    return tuple(instructions)


def _raw_text(opcode: int, operands: tuple[int, ...], target: int | None) -> str:
    parts = [OPCODE_MNEMONICS[opcode]]
    parts.extend(str(operand) for operand in operands)
    text = " ".join(parts)
    if target is not None:
        text = f"{text} -> {target}"
    return text
