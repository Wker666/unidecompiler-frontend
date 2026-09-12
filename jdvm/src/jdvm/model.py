"""Frontend-private decoded model for the jdvm bytecode family.

These dataclasses are the jdvm frontend's own view of a decoded image. They are
deliberately *not* core IR: no blocks, no control-flow graph, no statements.
The lifter converts each :class:`JdvmInstruction` into a core
``VMBytecodeStep``; everything above that is core's responsibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

FRONTEND_ID = "jdvm"


@dataclass(frozen=True)
class JdvmDiagnostic:
    """An explicit, contextual decoding problem.

    Diagnosed instructions are still submitted to core through the lifter's
    explicit fallback path; the diagnostic explains why the frontend could not
    prove their semantics.
    """

    code: str
    offset: int
    message: str
    raw: str = ""


@dataclass(frozen=True)
class JdvmSwitchCase:
    """One table-switch arm: pool value to match, and its target PC."""

    pool_index: int
    pool_value: str | None
    target: int


@dataclass(frozen=True)
class JdvmInstruction:
    """One decoded instruction, positioned by its public VM PC.

    No ``ByteRange`` is attached: the instruction's absolute artifact position
    would have to be derived from its VM PC, and the contract forbids deriving
    a range that way. The VM PC stays in ``SourceRef.offset``.
    """

    offset: int
    size: int
    opcode: int
    semantic: str
    canonical_index: int
    operands: tuple[int, ...] = ()
    pool_index: int | None = None
    pool_value: str | None = None
    pool_secondary: str | None = None
    closure_entry: int | None = None
    symbols: tuple[str, ...] = ()
    target: int | None = None
    cases: tuple[JdvmSwitchCase, ...] = ()
    default_target: int | None = None
    function: str = ""
    diagnostics: tuple[JdvmDiagnostic, ...] = ()

    @property
    def end(self) -> int:
        return self.offset + self.size

    @property
    def flow(self) -> str | None:
        """VM-neutral control-flow class for this instruction, if any."""
        if self.semantic == "jump":
            return "unconditional"
        if self.cases:
            return "multiway"
        if self.target is not None:
            return "conditional"
        return None


@dataclass(frozen=True)
class JdvmFunction:
    """One VM function root and the instructions decoded from its region."""

    entry: int
    end: int
    name: str
    locals: tuple[str, ...]
    instructions: tuple[JdvmInstruction, ...]

    @property
    def offsets(self) -> tuple[int, ...]:
        return tuple(instruction.offset for instruction in self.instructions)

    def instruction_at(self, offset: int) -> JdvmInstruction | None:
        for instruction in self.instructions:
            if instruction.offset == offset:
                return instruction
        return None


@dataclass(frozen=True)
class JdvmImage:
    """A decoded jdvm bytecode image."""

    words: tuple[int, ...]
    pool: tuple[str, ...]
    functions: tuple[JdvmFunction, ...]
    diagnostics: tuple[JdvmDiagnostic, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.diagnostics

    def function_named(self, name: str) -> JdvmFunction | None:
        for function in self.functions:
            if function.name == name:
                return function
        return None
