"""Instruction set of the mtgsigvm bytecode family.

Every entry is derived from the interpreter's dispatch chain in
``analysis_inputs/interpreter/H5guard.js`` (function ``aS``). The dispatch is a
single-line ``if``/``else`` cascade on the fetched opcode; ``_OPERAND`` records
how the operand bytes are consumed after the one-byte opcode, and
``_STACK`` records the operand-stack arity the handler expects.

``_STACK`` is *decoding metadata* only. The frontend never runs a stack machine:
it is used to keep the emitted thin-IR effect shapes exact, and to resolve the
literal operand of the three scope-access opcodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Operand encodings, in the order the interpreter consumes them.
#   "none"   : no operand bytes
#   "varint" : one signed LEB128 value
#   "rel32"  : one 4-byte big-endian signed relative offset
#   "cases"  : one varint case count followed by ``count`` 4-byte big-endian
#              relative offsets (the inline switch table)
OperandKind = Literal["none", "varint", "rel32", "cases"]


@dataclass(frozen=True)
class OpcodeInfo:
    """Decoded facts about one opcode."""

    code: int
    mnemonic: str
    operand: OperandKind = "none"
    pops: int = 0
    pushes: int = 0
    #: True when the instruction ends straight-line decoding (no fallthrough).
    terminates: bool = False
    #: Inline table stride in bytes for the ``cases`` operand kind.
    table_stride: int = 4

    @property
    def stack_delta(self) -> int:
        return self.pushes - self.pops


def _build() -> dict[int, OpcodeInfo]:
    table: dict[int, OpcodeInfo] = {}

    def add(code: int, name: str, **kw) -> None:
        table[code] = OpcodeInfo(code=code, mnemonic=name, **kw)

    # --- construction / calls -------------------------------------------------
    add(0, "NEW", pops=2, pushes=1)          # new ctor(...argsArray)
    add(37, "NEWARGS", pops=2, pushes=1)     # new ctor(a0..a5) from an args array
    add(58, "CALL", pops=3, pushes=1)        # callee.apply(thisArg, argsArray)

    # --- returns / control ----------------------------------------------------
    add(1, "RETURN", pops=1, terminates=True)
    add(35, "RET", pops=1, terminates=True)
    add(33, "THROW", pops=1, terminates=True)
    add(34, "TRY", operand="varint")         # pops 1 (finally fn) + 3 body fns
    add(53, "JMP", operand="rel32", terminates=True)
    add(54, "JMPIF", operand="rel32", pops=1)
    add(55, "JMPNOT", operand="rel32", pops=1)
    add(56, "SWITCH", operand="cases")       # pops count case values + the subject
    add(32, "FORIN", pops=3)

    # --- property / item access ----------------------------------------------
    add(2, "SETPROP", pops=3)                # obj[key] = value
    add(3, "GETPROP", pops=2, pushes=1)      # obj[key]
    add(4, "DELETE", pops=2, pushes=1)       # delete obj[key]

    # --- comparison -----------------------------------------------------------
    for code, name in (
        (5, "EQ"), (6, "NEQ"), (7, "SEQ"), (8, "SNEQ"),
        (9, "LT"), (10, "LE"), (11, "GT"), (12, "GE"),
    ):
        add(code, name, pops=2, pushes=1)
    add(27, "INSTANCEOF", pops=2, pushes=1)
    add(31, "IN", pops=2, pushes=1)

    # --- arithmetic / bitwise -------------------------------------------------
    for code, name in (
        (13, "ADD"), (14, "SUB"), (15, "MUL"), (16, "POW"),
        (17, "DIV"), (18, "MOD"), (21, "BOR"), (22, "BXOR"), (23, "BAND"),
        (24, "SHL"), (25, "SHR"), (26, "USHR"),
    ):
        add(code, name, pops=2, pushes=1)
    add(19, "LNOT", pops=1, pushes=1)
    add(20, "BNOT", pops=1, pushes=1)
    add(28, "TYPEOF", pops=1, pushes=1)
    add(29, "REGEXP", pops=2, pushes=1)

    # --- literals -------------------------------------------------------------
    add(39, "UNDEFINED", pushes=1)
    add(40, "NULL", pushes=1)
    add(43, "TRUE", pushes=1)
    add(44, "FALSE", pushes=1)
    add(38, "NOP")
    add(45, "CONST", operand="varint", pushes=1)
    add(46, "STRING", operand="varint", pushes=1)

    # --- aggregates -----------------------------------------------------------
    add(30, "CONCAT", operand="varint", pushes=1)
    add(41, "BUILDOBJ", operand="varint", pushes=1)
    add(42, "BUILDARR", operand="varint", pushes=1)

    # --- operand-stack shuffles ----------------------------------------------
    add(47, "POP", pops=1)
    add(48, "SWAP")
    add(49, "DUP", pushes=1)

    # --- bindings -------------------------------------------------------------
    add(50, "SETSCOPE", pops=2)
    add(51, "GETSCOPE", pops=1, pushes=1)
    add(52, "SETGLOBAL", pops=2)

    # --- closures -------------------------------------------------------------
    add(36, "BIND", pops=1, pushes=1)
    add(57, "CLOSURE", pops=1, pushes=1)
    return table


OPCODES: dict[int, OpcodeInfo] = _build()

#: Every opcode the interpreter treats as a call (the dispatch's final ``else``).
CALL_OPCODES = frozenset({58})

#: Opcodes whose operand is supplied by the operand stack rather than the
#: instruction stream.
SCOPE_OPCODES = frozenset({50, 51, 52})

#: Control-flow opcodes, as required by ``VMRegionOpcodeClasses``.
#:
#: ``JMP`` is the only unconditional transfer, and it is direction-agnostic: the
#: region profile decides forward or backward from the decoded target. The
#: conditional transfers are ``JMPIF`` (jump when the popped value is truthy)
#: and ``JMPNOT`` (jump when it is falsy); neither is a plain jump, because core
#: recovers their condition from the branch callbacks rather than from a target
#: alone.
JUMP_OPCODES = frozenset({"JMP"})
CONDITIONAL_JUMP_OPCODES = frozenset({"JMPIF", "JMPNOT"})
SWITCH_OPCODES = frozenset({"SWITCH"})
#: Terminating opcodes are deliberately NOT region control points: core wants
#: the terminator to come from the lifted slice, exactly as the Python frontend
#: treats RETURN_VALUE. Marking them as control made the region walker stop just
#: before every return and emit an unsupported tail.
#: ``TRY`` and ``FORIN`` are deliberately absent as well. They carry no branch
#: target -- ``TRY`` consumes three or four closure operands and ``FORIN`` one
#: callback -- so core's region walker must lift them inside a slice instead of
#: stopping on them.
CONTROL_OPCODES = (
    JUMP_OPCODES | CONDITIONAL_JUMP_OPCODES | SWITCH_OPCODES
)

#: Instructions that do not change the recovered shape.
NOISE_OPCODES = frozenset({"NOP"})


def info(code: int) -> OpcodeInfo | None:
    return OPCODES.get(code)


def mnemonic(code: int) -> str:
    entry = OPCODES.get(code)
    return entry.mnemonic if entry is not None else "OP_%d" % code
