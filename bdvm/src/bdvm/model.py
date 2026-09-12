"""Frontend-private decoded model for the bdvm packed bytecode format.

Everything in this module is decoder-owned data.  Core never imports it and
never inspects the decoded shapes; only :mod:`bdvm.lifter` converts them into
VM-neutral thin IR.

The decoded artifacts contain:

* ``Z`` -- the string pool.
* ``z`` -- the function table, each record ``[code, params, is_global, desc]``
  in packed order.
* ``code`` -- a flat integer array addressed by the interpreter's program
  counter. Instruction cells are opcode-then-operands, so a decoded
  instruction's public offset is its opcode cell index.
* ``desc`` -- the exception-region table. Each entry is
  ``[try_start, handler_start, finally_start, finally_end]``.

Nothing here executes, imports, or evaluates the interpreter or the payload.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Opcode mnemonics are decoder-local labels and are not part of any language
# contract.
OPCODE_MNEMONICS: dict[int, str] = {
    0: "call",
    1: "le",
    2: "gt",
    3: "forin-prep",
    4: "forin-next",
    5: "import-global",
    6: "neq",
    7: "newobj",
    8: "propget",
    9: "true",
    10: "top=undef",
    11: "mod",
    12: "bitand",
    13: "instanceof",
    14: "propset",
    15: "globalset",
    16: "propset3",
    17: "jif-pop",
    18: "dup",
    19: "ushr",
    20: "cpropset",
    21: "sub",
    22: "guard",
    23: "jeq-pop",
    24: "typeof",
    25: "delete",
    26: "pop",
    27: "false",
    28: "NaN",
    29: "not",
    30: "cpropget",
    31: "jtrue-pop",
    32: "lt",
    33: "push-undef",
    34: "thisctx",
    35: "shr",
    36: "uplus",
    37: "bitnot",
    38: "pushint",
    39: "array",
    40: "preinc",
    41: "jfalse",
    42: "div",
    43: "neg",
    44: "predec",
    45: "mul",
    46: "pushnum",
    47: "defgetter",
    48: "defsetter",
    49: "throw",
    50: "postinc",
    51: "bitor",
    52: "ret",
    53: "jump",
    54: "storescope",
    55: "in",
    56: "shl",
    57: "seq",
    58: "eq",
    59: "new",
    60: "loadglobal",
    61: "ref",
    62: "nseq",
    63: "mkclosure",
    64: "ge",
    65: "Infinity",
    66: "postdec",
    67: "defprop",
    68: "add",
    69: "typeofglobal",
    70: "xor",
    71: "jtrue",
    72: "declareglobal",
    73: "pushstr",
    74: "loadscope",
    75: "null",
    76: "finish",
}

# Operand-cell counts proven by the dispatch behavior for each opcode.
OPERAND_COUNT: dict[int, int] = {
    0: 1, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0,
    12: 0, 13: 0, 14: 0, 15: 1, 16: 0, 17: 1, 18: 0, 19: 0, 20: 1, 21: 0, 22: 0,
    23: 1, 24: 0, 25: 0, 26: 0, 27: 0, 28: 0, 29: 0, 30: 1, 31: 1, 32: 0, 33: 0,
    34: 0, 35: 0, 36: 0, 37: 0, 38: 1, 39: 1, 40: 0, 41: 1, 42: 0, 43: 0, 44: 0,
    45: 0, 46: 1, 47: 1, 48: 1, 49: 0, 50: 0, 51: 0, 52: 1, 53: 1, 54: 2, 55: 0,
    56: 0, 57: 0, 58: 0, 59: 1, 60: 1, 61: 2, 62: 0, 63: 1, 64: 0, 65: 0, 66: 0,
    67: 1, 68: 0, 69: 1, 70: 0, 71: 1, 72: 1, 73: 1, 74: 2, 75: 0, 76: 0,
}

# Relative-offset branch opcodes.  Their target is ``opcode_index + 2 + operand``
# (``a`` has already advanced past the operand when the interpreter adds the
# offset; see ``d`` lines 4258-4263, 4272, 4297-4299, 4329).
BRANCH_OPCODES: frozenset[int] = frozenset({17, 23, 31, 41, 52, 53, 71})
CONDITIONAL_BRANCH_OPCODES: frozenset[int] = frozenset({17, 23, 31, 41, 71})
UNCONDITIONAL_BRANCH_OPCODES: frozenset[int] = frozenset({52, 53})

# Operand roles per opcode, used for neutral ``VMOperand`` facts.
_STRING_POOL_OPERAND_OPS: frozenset[int] = frozenset(
    {5, 15, 20, 30, 46, 47, 48, 60, 67, 69, 72, 73}
)
_CLOSURE_OPERAND_OPS: frozenset[int] = frozenset({63})
_SCOPE_OPERAND_OPS: frozenset[int] = frozenset({54, 61, 74})
_FORIN_OPERAND_OPS: frozenset[int] = frozenset({3, 4})

MAX_OPCODE = max(OPCODE_MNEMONICS)


class BdvmFormatError(ValueError):
    """A structurally invalid bdvm payload (not a semantic gap)."""


@dataclass(frozen=True)
class BdvmRegion:
    """One protected interval from the function's ``desc`` table.

    ``try_start < pc <= handler_start`` is protected, ``finally_start`` is the
    cleanup entry (equal to ``finally_end`` when the region has no finally), and
    ``finally_end`` closes the region.  Proven by ``y`` (lines 4342-4365).
    """

    try_start: int
    handler_start: int
    finally_start: int
    finally_end: int

    @property
    def end(self) -> int:
        return self.handler_start

    @property
    def has_handler(self) -> bool:
        return self.handler_start != self.finally_start

    @property
    def has_finally(self) -> bool:
        return self.finally_start != self.finally_end


@dataclass(frozen=True)
class BdvmInstruction:
    """One decoded instruction cell group."""

    offset: int
    opcode: int
    mnemonic: str
    operands: tuple[int, ...]
    target: int | None
    size: int
    raw: str


@dataclass(frozen=True)
class BdvmFunction:
    """One decoded function record."""

    index: int
    name: str
    params: int
    is_global: bool
    code: tuple[int, ...]
    regions: tuple[BdvmRegion, ...]
    instructions: tuple[BdvmInstruction, ...]

    @property
    def instruction_count(self) -> int:
        return len(self.instructions)


@dataclass(frozen=True)
class BdvmModule:
    """The complete decoded payload."""

    strings: tuple[str, ...]
    functions: tuple[BdvmFunction, ...]
    entry_indices: tuple[int, ...] = ()
    packed_size: int = 0
    inflated_size: int = 0
    xor_key: int = 0
    format_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def string_pool(self) -> tuple[str, ...]:
        return self.strings

    def function(self, index: int) -> BdvmFunction | None:
        if 0 <= index < len(self.functions):
            return self.functions[index]
        return None

    def function_name(self, index: int) -> str:
        function = self.function(index)
        if function is not None:
            return function.name
        return f"func_{index}"
