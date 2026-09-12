"""Kasada VM opcode table (decoder-private).

Derived by static analysis of ``analysis_inputs/interpreter/vmp.js``:
the 86 handler bodies of ``u = new Proxy([...])`` (file line 75) define the
console opcode set, and the bytecode operand sequence of each opcode is the
left-to-right order in which its handler consumes operand cells.

Operand cell kinds
------------------
``V``  a *value* cell, read by ``G()`` -> ``g()`` (1 or 3 words, see decoder)
``R``  a *register* cell, read by ``j()`` (``word >> 5``), 1 word
``S``  a *store destination* cell, consumed by ``F()`` (``word >> 5``), 1 word

``S`` is consumed *after* every ``V``/``R`` cell of the same instruction
because the handler evaluates its value expression first and only then calls
``a(n, value)``, and ``a`` is ``F`` which advances the cursor itself.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OpcodeSpec:
    code: int
    mnemonic: str
    operands: str
    category: str
    description: str


def _o(code: int, mnemonic: str, operands: str, category: str, description: str) -> OpcodeSpec:
    return OpcodeSpec(code, mnemonic, operands, category, description)


OPCODE_SPECS: tuple[OpcodeSpec, ...] = (
    _o(0, "CALL_APPLY", "VVV", "call", "callee.apply(this, args)"),
    _o(1, "OR", "RRS", "bitwise", "dst = a | b"),
    _o(2, "XOR", "RRS", "bitwise", "dst = a ^ b"),
    _o(3, "THROW", "V", "exception", "raise value"),
    _o(4, "GET_ITEM", "VVS", "member", "dst = obj[key]"),
    _o(5, "ADD", "VRS", "arith", "dst = value + reg"),
    _o(6, "HALT", "", "control", "end of frame; returns the ret cell"),
    _o(7, "IN", "RRS", "compare", "dst = a in b"),
    _o(8, "NEW_REGEXP", "VVS", "build", "dst = new RegExp(pattern, flags)"),
    _o(9, "OR_VV", "VVS", "bitwise", "dst = value | value"),
    _o(10, "NEW_CALLEE", "VVS", "call", "dst = construct callee with expanded args"),
    _o(11, "DIV", "VRS", "arith", "dst = value / reg"),
    _o(12, "CALL1", "VVS", "call", "dst = callee(arg)"),
    _o(13, "LT", "RVS", "compare", "dst = reg < value"),
    _o(14, "GE", "RRS", "compare", "dst = a >= b"),
    _o(15, "SCOPE_SET", "VV", "scope", "scope slot[key] = value"),
    _o(16, "AND", "RVS", "bitwise", "dst = reg & value"),
    _o(17, "SET_RESUME", "V", "scope", "scope resume cell = value"),
    _o(18, "PUSH_PROMISE", "S", "build", "dst = Promise"),
    _o(19, "SCOPE_LOOKUP", "VS", "scope", "dst = variable lookup along scope chain"),
    _o(20, "JUMP_IF_TRUE", "VV", "control", "if condition then pc = target"),
    _o(21, "SNEQ", "RRS", "compare", "dst = a !== b"),
    _o(22, "MOD", "RRS", "arith", "dst = a % b"),
    _o(23, "GT", "RRS", "compare", "dst = a > b"),
    _o(24, "DIV_RR", "RRS", "arith", "dst = a / b"),
    _o(25, "MOD_RV", "RVS", "arith", "dst = reg % value"),
    _o(26, "SCOPE_ASSIGN", "VV", "scope", "assign variable along scope chain"),
    _o(27, "ADD_VV", "VVS", "arith", "dst = value + value"),
    _o(28, "UPLUS", "VS", "arith", "dst = +value"),
    _o(29, "EQ", "VRS", "compare", "dst = value == reg"),
    _o(30, "CLEAR_EXCEPTION", "", "exception", "frame pending-exception cell = undefined"),
    _o(31, "GE_RV", "RVS", "compare", "dst = reg >= value"),
    _o(32, "SUB", "RRS", "arith", "dst = a - b"),
    _o(33, "USHR", "RVS", "bitwise", "dst = reg >>> value"),
    _o(34, "XOR_RV", "VRS", "bitwise", "dst = value ^ reg"),
    _o(35, "LE", "RVS", "compare", "dst = reg <= value"),
    _o(36, "BITNOT", "VS", "bitwise", "dst = ~value"),
    _o(37, "IN_VR", "VRS", "compare", "dst = value in reg"),
    _o(38, "SNEQ_VR", "VRS", "compare", "dst = value != reg"),
    _o(39, "TYPEOF", "VS", "compare", "dst = typeof value"),
    _o(40, "SHL", "RVS", "bitwise", "dst = reg << value"),
    _o(41, "SET_ITEM", "VVV", "member", "obj[key] = value"),
    _o(42, "ADD_RR", "RRS", "arith", "dst = a + b"),
    _o(43, "SEQ", "RVS", "compare", "dst = reg === value"),
    _o(44, "FRAME_EPILOGUE", "V", "control", "frame teardown / rethrow / trampoline resume"),
    _o(45, "NEW_ARRAY", "VS", "build", "dst = new Array(size)"),
    _o(46, "NEW_OBJECT", "S", "build", "dst = {}"),
    _o(47, "SHL_RR", "RRS", "bitwise", "dst = a << b"),
    _o(48, "ADD_RV", "RVS", "arith", "dst = reg + value"),
    _o(49, "PUSH_REGENERATOR", "S", "build", "dst = regenerator runtime"),
    _o(50, "DELETE_ITEM", "VVS", "member", "delete obj[key]"),
    _o(51, "SNEQ_VV", "VRS", "compare", "dst = value !== reg"),
    _o(52, "GET_EXCEPTION", "S", "exception", "dst = frame pending exception"),
    _o(53, "RESTORE_CALL_STATE", "V", "scope", "restore saved call state from slot"),
    _o(54, "INSTANCEOF", "RRS", "compare", "dst = a instanceof b"),
    _o(55, "CALL0", "VS", "call", "dst = callee()"),
    _o(56, "LT_RR", "RRS", "compare", "dst = a < b"),
    _o(57, "GET_GLOBAL", "VS", "member", "dst = host_global[name]"),
    _o(58, "SEQ_VR", "VRS", "compare", "dst = value === reg"),
    _o(59, "MAKE_CLOSURE", "VVVS", "build", "dst = closure(entry, name, arity)"),
    _o(60, "ARRAY_LITERAL", "S", "build", "dst = []"),
    _o(61, "CATCH_BIND", "V", "exception", "scope slot[key] = pending exception; clear pending"),
    _o(62, "JUMP_IF_FALSE", "VV", "control", "if not condition then pc = target"),
    _o(63, "PUSH_PARENT_SCOPE", "V", "scope", "scope slot[key] = parent scope"),
    _o(64, "OR_RV", "VRS", "bitwise", "dst = value | reg"),
    _o(65, "LOOSE_EQ", "VVS", "compare", "dst = value == value"),
    _o(66, "SNEQ_RV", "RVS", "compare", "dst = reg !== value"),
    _o(67, "SUB_VV", "VVS", "arith", "dst = value - value"),
    _o(68, "AND_RR", "RRS", "bitwise", "dst = a & b"),
    _o(69, "CALL_FRAME", "V", "call", "invoke frame target"),
    _o(70, "SUB_RV", "RVS", "arith", "dst = reg - value"),
    _o(71, "SAVE_CALL_STATE", "V", "scope", "scope slot[key] = saved call state"),
    _o(72, "SEQ_RR", "RRS", "compare", "dst = a === b"),
    _o(73, "NOT", "VS", "compare", "dst = !value"),
    _o(74, "DECLARE_VAR", "V", "scope", "scope slot[key] = undefined"),
    _o(75, "SET_HANDLER", "V", "exception", "scope handler cell = target"),
    _o(76, "MUL_RV", "RVS", "arith", "dst = reg * value"),
    _o(77, "CALL_UNDEFINED_TARGET", "", "call", "invoke frame target undefined"),
    _o(78, "PUSH_SCOPE_GLOBAL", "S", "build", "dst = scope global object"),
    _o(79, "JUMP", "V", "control", "pc = target"),
    _o(80, "DIV_RV", "RVS", "arith", "dst = reg / value"),
    _o(81, "LOAD", "VS", "move", "dst = value"),
    _o(82, "CALL3", "VVVVS", "call", "dst = callee(a, b, c)"),
    _o(83, "CALL2", "VVVS", "call", "dst = callee(a, b)"),
    _o(84, "GT_RV", "RVS", "compare", "dst = reg > value"),
    _o(85, "MUL_RR", "RRS", "arith", "dst = a * b"),
)

OPCODE_COUNT = len(OPCODE_SPECS)
BY_CODE: dict[int, OpcodeSpec] = {spec.code: spec for spec in OPCODE_SPECS}

#: Opcodes whose handler mutates the frame program counter.
PC_WRITING: frozenset[int] = frozenset({6, 20, 44, 62, 79})

#: Opcodes that end the current frame's linear run and therefore have no
#: fallthrough successor.
#:
#: ``3``  THROW           ``D`` unwinds to a handler or re-raises
#: ``6``  HALT            ``U`` breaks out of the dispatch loop
#: ``44`` FRAME_EPILOGUE  resumes at ``scope.o`` / ``scope._`` or re-raises
#: ``69`` CALL_FRAME      ``X`` jumps to ``scope._`` or replaces the frame
#: ``77`` CALL_UNDEF_TGT  same frame transfer with an undefined target
#: ``79`` JUMP            ``E[0] = target``
#:
#: Treating ``69``/``77`` as fallthrough opcodes makes 43 of the closure
#: bodies run on into the shared top-level code; excluding them yields a
#: partition in which every decoded instruction belongs to exactly one
#: function root (433 roots, multiplicity 1).
TERMINATORS: frozenset[int] = frozenset({3, 6, 44, 69, 77, 79})

#: Frame-transfer opcodes: no static successor offset is encoded in the
#: instruction, so they never produce a branch hint.
FRAME_TRANSFERS: frozenset[int] = frozenset({69, 77})

#: Immediate-target branch opcodes.
BRANCHES: frozenset[int] = frozenset({20, 62, 79})
CONDITIONAL_BRANCHES: frozenset[int] = frozenset({20, 62})

#: Opcodes that construct a closure and therefore name a nested function root.
CLOSURE_OPCODES: frozenset[int] = frozenset({59})

#: Branch polarity: True when the target is taken because the condition holds.
TARGET_IF_TRUE: frozenset[int] = frozenset({20})
TARGET_IF_FALSE: frozenset[int] = frozenset({62})


def mnemonic(code: int) -> str:
    spec = BY_CODE.get(code)
    return spec.mnemonic if spec is not None else f"OP_{code}"


def operands_of(code: int) -> str:
    spec = BY_CODE.get(code)
    return spec.operands if spec is not None else ""


def is_known(code: int) -> bool:
    return code in BY_CODE
