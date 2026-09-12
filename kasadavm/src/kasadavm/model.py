"""Frontend-private decoded model for the Kasada VM.

Nothing in this module is visible to core.  Core only ever sees the neutral
``VMBytecodeStep`` stream that :mod:`kasadavm.lifter` builds from these objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ValueKind = Literal["int", "float", "string", "bool", "null", "undefined", "register"]
OperandRole = Literal["value", "register", "store"]

#: Registers 0..3 of a frame are structural, not general purpose.
#:   E[0] program counter     E[1] scope object
#:   E[2] return-value cell   E[3] arguments object
RESERVED_REGISTERS: dict[int, str] = {
    0: "pc",
    1: "scope",
    2: "ret",
    3: "args",
}


@dataclass(frozen=True)
class KasadaValue:
    """One decoded *value* operand cell.

    The value reader is self-describing: the tag word selects the encoding and
    also determines how many further words belong to this cell.
    """

    kind: ValueKind
    value: object
    words: int
    text: str

    @property
    def is_constant(self) -> bool:
        return self.kind != "register"

    @property
    def register(self) -> int:
        assert self.kind == "register"
        return int(self.value)  # type: ignore[arg-type]


@dataclass(frozen=True)
class KasadaOperand:
    """One decoded operand cell in submission order."""

    role: OperandRole
    value: KasadaValue | int
    words: int
    text: str

    @property
    def is_store(self) -> bool:
        return self.role == "store"

    def as_value(self) -> KasadaValue:
        assert self.role == "value"
        assert isinstance(self.value, KasadaValue)
        return self.value

    def as_register(self) -> int:
        assert self.role in {"register", "store"}
        assert isinstance(self.value, int)
        return self.value


@dataclass(frozen=True)
class KasadaInstruction:
    """A fully decoded instruction with exact artifact provenance."""

    offset: int
    opcode: int
    mnemonic: str
    operands: tuple[KasadaOperand, ...]
    size: int
    raw: str
    category: str
    artifact_offset: int
    artifact_size: int

    def value_operands(self) -> tuple[KasadaValue, ...]:
        return tuple(op.as_value() for op in self.operands if op.role == "value")

    def store_register(self) -> int | None:
        for op in self.operands:
            if op.role == "store":
                return op.as_register()
        return None

    def register_operands(self) -> tuple[int, ...]:
        return tuple(op.as_register() for op in self.operands if op.role == "register")


@dataclass(frozen=True)
class KasadaFunction:
    """One function root discovered from the entry, a closure, or a call target."""

    name: str
    entry: int
    origin: str
    instructions: tuple[KasadaInstruction, ...]

    @property
    def offsets(self) -> tuple[int, ...]:
        return tuple(ins.offset for ins in self.instructions)


@dataclass(frozen=True)
class KasadaProgram:
    """The decoded artifact."""

    words: tuple[int, ...]
    functions: tuple[KasadaFunction, ...]
    filename: str | None = None
    string_table: str | None = None
    entry: int = 1
    diagnostics: tuple[str, ...] = ()
    claimed_words: frozenset[int] = field(default_factory=frozenset)

    @property
    def instructions_by_offset(self) -> dict[int, KasadaInstruction]:
        out: dict[int, KasadaInstruction] = {}
        for function in self.functions:
            for instruction in function.instructions:
                out.setdefault(instruction.offset, instruction)
        return out


def register_name(index: int) -> str:
    """Stable, VM-neutral name for a frame register cell."""

    reserved = RESERVED_REGISTERS.get(index)
    return reserved if reserved is not None else f"r{index}"


def scope_slot_name(key: object) -> str:
    """Stable name for a scope variable slot."""

    return f"v{key}"


#: Frontend-local names for VM-internal frame/scope state the bytecode really
#: reads and writes.  They are modelled as ordinary locals so that every
#: decoded mutation survives into generic IR instead of being dropped.
PENDING_EXCEPTION = "<pending-exception>"
RESUME_CELL = "<resume-cell>"
HANDLER_CELL = "<handler-cell>"
CALL_RECORD = "<call-target>"
CALL_STATE = "<call-state>"
CALL_DEPTH = "<call-depth>"
SAVED_CALL = "<saved-call>"
PARENT_SCOPE = "<parent-scope>"
SCOPE_GLOBAL = "<scope-global>"
SCOPE_TABLE = "<scope-table>"
HOST_GLOBAL = "<host-global>"
