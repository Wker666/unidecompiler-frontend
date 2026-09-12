"""Frontend-private decoded model for the mtgsigvm bytecode family.

These classes belong to the frontend only. Core must never see them; the
lifter converts them into neutral ``VMBytecodeStep`` values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MtgsigOperand:
    """One decoded operand value."""

    role: str          # "constant" | "string" | "count" | "target" | "case" | "flag"
    value: Any
    text: str = ""

    def describe(self) -> str:
        return self.text or str(self.value)


@dataclass(frozen=True)
class MtgsigInstruction:
    """One decoded VM instruction."""

    offset: int                    # VM PC, function-relative
    opcode: int
    mnemonic: str
    operands: tuple[MtgsigOperand, ...] = ()
    size: int = 1
    raw: str = ""
    #: Absolute, read-only provenance inside the input artifact.
    artifact_offset: int | None = None
    #: Resolved branch / case targets (function-relative VM PCs).
    targets: tuple[int, ...] = ()
    #: Fallthrough target, when the instruction has one.
    fallthrough: int | None = None
    #: Resolved literal slot name for the scope-access opcodes when provable.
    scope_slot: str | None = None
    scope_key: str | None = None
    #: Resolved VM function name for CLOSURE when provable.
    closure_name: str | None = None
    #: True when a CALL result is provably never read (an expression statement).
    dead_result: bool = False

    @property
    def ends_block(self) -> bool:
        return self.fallthrough is None


@dataclass(frozen=True)
class MtgsigFunction:
    """One VM function."""

    name: str
    index: int
    #: Section entry descriptor word. Unused by the interpreter's parser; kept
    #: as provenance.
    flags: int
    code: bytes
    code_offset: int               # absolute offset of ``code`` in the artifact
    instructions: tuple[MtgsigInstruction, ...]
    #: Function names referenced by CLOSURE instructions, in first-use order.
    referenced_names: tuple[str, ...] = ()

    @property
    def size(self) -> int:
        return len(self.code)


@dataclass(frozen=True)
class MtgsigDiagnostic:
    """Deterministic, analyzable frontend diagnostic."""

    code: str
    message: str
    function: str | None = None
    offset: int | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.function is not None:
            payload["function"] = self.function
        if self.offset is not None:
            payload["offset"] = self.offset
        return payload


@dataclass(frozen=True)
class MtgsigModule:
    """Decoded container: sections, functions, constants and strings."""

    name: str
    artifact_size: int
    section_count: int
    debug_tables: bool                     # the interpreter's ``jG`` header byte
    sections: tuple[tuple[int, int], ...]  # (type, offset) as stored, in file order
    functions: tuple[MtgsigFunction, ...]
    constants: tuple[float, ...]
    strings: tuple[str, ...]
    string_xor_key: str
    diagnostics: tuple[MtgsigDiagnostic, ...] = field(default=())

    def function(self, name: str) -> MtgsigFunction | None:
        for candidate in self.functions:
            if candidate.name == name:
                return candidate
        return None

    @property
    def entry_function(self) -> MtgsigFunction | None:
        """The VM bootstrap function the interpreter runs first.

        The interpreter's ``run`` bootstraps with the literal name ``"@0"``
        before dispatching the caller-supplied entry.
        """
        return self.function(BOOTSTRAP_FUNCTION)


#: The function the interpreter executes first (``b(680)`` -> ``"@0"``).
BOOTSTRAP_FUNCTION = "@0"

#: Section type tags.
SECTION_FUNCTION = 1
SECTION_CONSTANTS = 2
SECTION_STRINGS = 3
SECTION_TYPES = frozenset({SECTION_FUNCTION, SECTION_CONSTANTS, SECTION_STRINGS})

#: Guard rails so malformed input is rejected instead of allocating wildly.
MAX_SECTIONS = 64
MAX_FUNCTIONS = 4096
MAX_STRINGS = 65536
MAX_CONSTANTS = 65536
MAX_INSTRUCTIONS_PER_FUNCTION = 1 << 20
