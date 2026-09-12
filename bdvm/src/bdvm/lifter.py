"""Thin-IR lifting for bdvm.

The frontend submits one :class:`VMBytecodeStep` per decoded instruction.  It
never builds blocks, CFG edges, loops, or source structures: control-flow
recovery, stack recovery, exception edges, and structuring stay in core.

Scope model
-----------
The interpreter keeps variables in a scope chain (``g`` at line 4184:
``s = [outer_world, arguments, param0.., local..]``) and reaches a cell with
``loadscope depth cell`` / ``storescope depth cell`` / ``ref depth cell``
(``d`` lines 4300-4316, 4333-4335).  The frontend names cell ``c`` of frame
``d`` as ``scope{d}_{c}`` for direct reads/writes, and exposes the raw frame
array as ``scope{d}`` for the reference-pair opcodes (``ref`` and the
member/increment opcodes that consume it), which the VM itself performs as
``frame[cell]``.

Value model
-----------
Calls use the VM's ``[this, callee, args...]`` layout; the receiver is rotated
off with stack-only effects so the callee stays the attribute read that
produced it and a method call keeps its receiver.  Branches that retain a value
on one successor (``jif-pop``/``jtrue-pop``/``jeq-pop``) are modelled with a
stack duplicate plus a deferred edge effect.  The exception-region table is
flattened into disjoint handler intervals before submission.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from unidecompiler.core.effects import (
    Binary,
    BuildArray,
    BuildCall,
    Compare,
    Copy,
    DeleteItem,
    DuplicateTop,
    DuplicateTopBelow,
    Emit,
    Invoke,
    LoadAttr,
    LoadIndirect,
    LoadItem,
    LoadItemAddress,
    LoadLocal,
    Pop,
    Push,
    RaiseTop,
    ReturnTop,
    StoreAttr,
    StoreGlobal,
    StoreIndirect,
    StoreItemEffect,
    StoreLocal,
    Swap,
    Unary,
)
from unidecompiler.core.ir import (
    BinaryOp,
    Call,
    Const,
    ExprStmt,
    GetItem,
    Global,
    ObjectLiteral,
    SourceRef,
    UnaryOp,
    UndefinedLiteral,
    Var,
)
from unidecompiler.core.vm_bytecode import VMBytecodeStep
from unidecompiler.core.vm_effect_table import VMEffectTable
from unidecompiler.core.vm_function import (
    VMFunctionSpec,
    lift_steps,
    lift_vm_step_function,
    recover_vm_function,
)
from unidecompiler.core.vm_hints import VMHint
from unidecompiler.core.vm_module import assemble_vm_module
from unidecompiler.core.vm_operands import VMDecodedInstruction, VMOperand
from unidecompiler.core.vm_region import (
    VMRegionOpcodeClasses,
    VMStatefulCallbacks,
    VMLinearState,
    build_hint_region_profile,
)
from unidecompiler.plugins import FrontendModule
from unidecompiler.progress import ProgressReporter, report_progress

from .model import (
    CONDITIONAL_BRANCH_OPCODES,
    BdvmFunction,
    BdvmInstruction,
    BdvmModule,
)

FRONTEND_ID = "bdvm"

# Neutral helper names used for VM behaviours that generic IR cannot spell
# directly.  They are data-only markers, exactly like core's ``call_ex`` or
# ``new_array`` helpers; nothing executes them.
_HELPER_CONSTRUCT = "construct"
_HELPER_FORIN_NEXT = "forin_next"
_HELPER_DECLARE_GLOBAL = "declare_global"
_HELPER_IMPORT_GLOBAL = "import_global"

_BITWISE_OPERATORS = {
    "bitand": "&",
    "bitor": "|",
    "xor": "^",
    "shl": "<<",
    "shr": ">>",
}

_ARITHMETIC_OPERATORS = {
    "sub": "-",
    "mul": "*",
    "div": "/",
    "mod": "%",
}

_BINARY_OPERATORS = {
    "add": "+",
    "in": "in",
    "instanceof": "instanceof",
}

_COMPARE_OPERATORS = {
    "lt": "<",
    "le": "<=",
    "gt": ">",
    "ge": ">=",
    "eq": "==",
    "neq": "!==",
    "seq": "===",
    "nseq": "!=",
}

_UNARY_OPERATORS = {
    "not": "!",
    "bitnot": "~",
    "neg": "-",
    "uplus": "+",
    "typeof": "typeof ",
}

_INCREMENT_OPERATORS = {
    "preinc": "++ ",
    "predec": "-- ",
    "postinc": "++ ",
    "postdec": "-- ",
}

# Opcodes whose value is fully described by a constant or a scope read; used
# only for operand role labelling.
_STRING_OPERAND_OPS = frozenset(
    {
        "import-global",
        "globalset",
        "cpropset",
        "cpropget",
        "pushnum",
        "defgetter",
        "defsetter",
        "loadglobal",
        "defprop",
        "typeofglobal",
        "declareglobal",
        "pushstr",
    }
)
_CLOSURE_OPERAND_OPS = frozenset({"mkclosure"})
_SCOPE_OPERAND_OPS = frozenset({"storescope", "ref", "loadscope"})
_FORIN_OPERAND_OPS = frozenset({"forin-prep", "forin-next"})

BDVM_REGION_OPCODE_CLASSES = VMRegionOpcodeClasses(
    noise=frozenset({"guard"}),
    control=frozenset(
        {"jif-pop", "jeq-pop", "jtrue-pop", "jfalse", "jtrue", "ret", "jump"}
    ),
    jumps=frozenset({"ret", "jump"}),
    forward_jumps=frozenset({"jif-pop", "jeq-pop", "jtrue-pop", "jfalse", "jtrue", "ret", "jump"}),
    backward_jumps=frozenset({"jif-pop", "jeq-pop", "jtrue-pop", "jfalse", "jtrue", "ret", "jump"}),
    conditional_jumps=frozenset({"jif-pop", "jeq-pop", "jtrue-pop", "jfalse", "jtrue"}),
)

# Branch polarity: which side of the tested value the jump target is on.
_BRANCH_DETAIL = {
    "jif-pop": "target-if-false",
    "jfalse": "target-if-false",
    "jeq-pop": "target-if-true",
    "jtrue-pop": "target-if-true",
    "jtrue": "target-if-true",
}

_BRANCH_WIDTH = {"jeq-pop": 2}

# Branches that leave their tested value on the operand stack for one
# successor instead of consuming it on both paths.
VALUE_RETAINING_BRANCHES = frozenset({"jif-pop", "jtrue-pop", "jeq-pop"})


@dataclass(frozen=True)
class BdvmEffectContext:
    """Decoder-owned facts the effect table needs."""

    module: BdvmModule
    function: BdvmFunction


def _scope_cell_name(depth: int, cell: int) -> str:
    return f"scope{depth}_{cell}"


def _frame_name(depth: int) -> str:
    return f"scope{depth}"


def _cell_value_expr(depth: int, cell: int, source: SourceRef) -> Var:
    return Var(name=_scope_cell_name(depth, cell), source=source)


def _frame_expr(depth: int, source: SourceRef) -> Var:
    return Var(name=_frame_name(depth), source=source)


def _push_helper_call(name: str, args: tuple, source: SourceRef) -> Push:
    return Push(
        source=source,
        value=Call(
            source=source,
            callee=Global(name=name, source=source),
            args=args,
            returns=1,
        ),
    )


# --------------------------------------------------------------------------
# effect factories
# --------------------------------------------------------------------------


def _no_effect(_context: BdvmEffectContext, _instruction: BdvmInstruction, _source: SourceRef) -> tuple:
    return ()


def _push_const(value: object):
    def factory(_context: BdvmEffectContext, _instruction: BdvmInstruction, source: SourceRef) -> tuple:
        return (Push(source=source, value=Const(value=value, source=source)),)

    return factory


def _push_undefined(_context: BdvmEffectContext, _instruction: BdvmInstruction, source: SourceRef) -> tuple:
    return (Push(source=source, value=UndefinedLiteral(source=source)),)


def _push_global_name(context: BdvmEffectContext, instruction: BdvmInstruction, source: SourceRef) -> tuple:
    return (Push(source=source, value=Global(name=_string(context, instruction.operands[0]), source=source)),)


def _store_global_name(context: BdvmEffectContext, instruction: BdvmInstruction, source: SourceRef) -> tuple:
    return (StoreGlobal(source=source, name=_string(context, instruction.operands[0])),)


def _typeof_global(context: BdvmEffectContext, instruction: BdvmInstruction, source: SourceRef) -> tuple:
    name = _string(context, instruction.operands[0])
    return (
        Push(
            source=source,
            value=UnaryOp(source=source, op="typeof ", value=Global(name=name, source=source)),
        ),
    )


def _declare_global(context: BdvmEffectContext, instruction: BdvmInstruction, source: SourceRef) -> tuple:
    name = _string(context, instruction.operands[0])
    return (
        Emit(
            source=source,
            statement=ExprStmt(
                source=source,
                value=Call(
                    source=source,
                    callee=Global(name=_HELPER_DECLARE_GLOBAL, source=source),
                    args=(Const(value=name, source=source),),
                    returns=0,
                ),
            ),
        ),
    )


def _import_global(context: BdvmEffectContext, instruction: BdvmInstruction, source: SourceRef) -> tuple:
    name = _string(context, instruction.operands[0])
    return (
        _push_helper_call(_HELPER_IMPORT_GLOBAL, (Const(value=name, source=source),), source),
        Push(source=source, value=Const(value=name, source=source)),
    )


def _push_string(context: BdvmEffectContext, instruction: BdvmInstruction, source: SourceRef) -> tuple:
    return (Push(source=source, value=Const(value=_string(context, instruction.operands[0]), source=source)),)


def _push_number(context: BdvmEffectContext, instruction: BdvmInstruction, source: SourceRef) -> tuple:
    # ``pushnum`` pushes ``+Z[x]`` (unary plus), which is ToNumber, not a
    # numeric literal.  Keeping the unary operator preserves that exactly.
    return (
        Push(
            source=source,
            value=UnaryOp(
                source=source,
                op="+",
                value=Const(value=_string(context, instruction.operands[0]), source=source),
            ),
        ),
    )


def _push_int(_context: BdvmEffectContext, instruction: BdvmInstruction, source: SourceRef) -> tuple:
    return (Push(source=source, value=Const(value=instruction.operands[0], source=source)),)


def _push_this(_context: BdvmEffectContext, _instruction: BdvmInstruction, source: SourceRef) -> tuple:
    return (Push(source=source, value=Var(name="this", source=source)),)


def _push_object(_context: BdvmEffectContext, _instruction: BdvmInstruction, source: SourceRef) -> tuple:
    return (Push(source=source, value=ObjectLiteral(source=source)),)


def _load_scope(_context: BdvmEffectContext, instruction: BdvmInstruction, source: SourceRef) -> tuple:
    depth, cell = instruction.operands
    name = _scope_cell_name(depth, cell)
    return (LoadLocal(source=source, name=name, fallback=_cell_value_expr(depth, cell, source)),)


def _store_scope(_context: BdvmEffectContext, instruction: BdvmInstruction, source: SourceRef) -> tuple:
    depth, cell = instruction.operands
    name = _scope_cell_name(depth, cell)
    return (
        StoreLocal(
            source=source,
            name=name,
            target=_cell_value_expr(depth, cell, source),
            materialize=True,
        ),
    )


def _push_reference(_context: BdvmEffectContext, instruction: BdvmInstruction, source: SourceRef) -> tuple:
    depth, cell = instruction.operands
    return (
        Push(source=source, value=_frame_expr(depth, source)),
        Push(source=source, value=Const(value=cell, source=source)),
    )


def _load_attr(context: BdvmEffectContext, instruction: BdvmInstruction, source: SourceRef) -> tuple:
    return (LoadAttr(source=source, attr=_string(context, instruction.operands[0])),)


def _store_attr(context: BdvmEffectContext, instruction: BdvmInstruction, source: SourceRef) -> tuple:
    return (StoreAttr(source=source, attr=_string(context, instruction.operands[0]), order="obj-value"),)


def _define_property(context: BdvmEffectContext, instruction: BdvmInstruction, source: SourceRef) -> tuple:
    """``defprop``/``defgetter``/``defsetter``: pop the value, keep the object.

    ``Object.defineProperty(v[p], Z[x], ...)`` consumes only the definition
    value (``d`` lines 4275-4290, 4320-4325).  Generic ``StoreAttr`` consumes
    both operands, so the object is duplicated below the stored value first.
    A data property definition (``defprop``) has the same observable effect as
    a plain member store; the accessor forms (``defgetter``/``defsetter``) are
    mapped to the same neutral store because the format does not carry an
    accessor-definition node.  Neither accessor opcode occurs in the sample.
    """

    return (
        Copy(source=source, depth=2),
        Swap(source=source, depth=3),
        Swap(source=source, depth=2),
        StoreAttr(source=source, attr=_string(context, instruction.operands[0]), order="obj-value"),
    )


def _binary(op: str, **kwargs):
    def factory(_context: BdvmEffectContext, _instruction: BdvmInstruction, source: SourceRef) -> tuple:
        return (Binary(source=source, op=op, **kwargs),)

    return factory


def _compare(op: str, **kwargs):
    def factory(_context: BdvmEffectContext, _instruction: BdvmInstruction, source: SourceRef) -> tuple:
        return (Compare(source=source, op=op, **kwargs),)

    return factory


def _unary(op: str):
    def factory(_context: BdvmEffectContext, _instruction: BdvmInstruction, source: SourceRef) -> tuple:
        return (Unary(source=source, op=op),)

    return factory


def _call(_context: BdvmEffectContext, instruction: BdvmInstruction, source: SourceRef) -> tuple:
    """``call n``: stack is ``[this, callee, arg0..arg(n-1)]``.

    The VM pops the arguments, the callee and the receiver, then applies
    (``d`` lines 4214-4223).  Generic ``Invoke`` consumes only ``callee`` and
    the arguments, so the receiver is rotated off the bottom of that window with
    stack-only effects.  Rotating instead of binding temporaries keeps the
    callee as the attribute read that produced it, so a method call keeps its
    receiver (``obj.m(args)``) and every value is evaluated exactly once.
    """

    count = instruction.operands[0]
    effects: list = [Swap(source=source, depth=count + 2)]
    # A receiver that is a deferred call still has to run.
    effects.append(Pop(source=source, count=1, emit_calls=True))
    for depth in range(2, count + 2):
        effects.append(Swap(source=source, depth=depth))
    effects.append(Invoke(source=source, arg_count=count, returns=1))
    return tuple(effects)


def _new(_context: BdvmEffectContext, instruction: BdvmInstruction, source: SourceRef) -> tuple:
    """``new n``: stack is ``[callee, arg0..arg(n-1)]`` (``d`` lines 4304-4308).

    Generic IR has no dynamic-constructor node, so the neutral ``construct``
    helper carries the callee and arguments explicitly.
    """

    count = instruction.operands[0]
    return (
        BuildCall(
            source=source,
            callee=Global(name=_HELPER_CONSTRUCT, source=source),
            arg_count=count + 1,
            returns=1,
        ),
    )


def _propset_keep_value(_context: BdvmEffectContext, _instruction: BdvmInstruction, source: SourceRef) -> tuple:
    """``propset3``: ``obj[key] = value`` while keeping ``value`` on the stack.

    ``(v[p--])[v[p--]] = v[p]`` (``d`` line 4255-4257).  The address is formed
    once, the stored value is copied below it, and ``StoreIndirect`` assigns
    through the address while the copy stays on the stack.
    """

    return (
        LoadItemAddress(source=source),
        Swap(source=source, depth=2),
        DuplicateTopBelow(source=source, below_count=1),
        StoreIndirect(source=source),
    )


def _push_increment(op: str):
    """``++obj[k]`` / ``obj[k]++`` / ``--obj[k]`` / ``obj[k]--``.

    The VM reads the addressed slot, applies the increment operator, writes it
    back, and pushes either the new value (prefix) or the pre-update value
    (postfix).  The address is formed once so the read and the write observe the
    same object and key.
    """

    unary = _INCREMENT_OPERATORS[op]
    postfix = op in {"postinc", "postdec"}

    def factory(_context: BdvmEffectContext, _instruction: BdvmInstruction, source: SourceRef) -> tuple:
        effects: list = [
            LoadItemAddress(source=source),
            DuplicateTop(source=source),
            LoadIndirect(source=source),
        ]
        if postfix:
            # [old, ref, old] -> increment the top -> [old, ref, new]
            effects.append(DuplicateTopBelow(source=source, below_count=1))
            effects.append(Unary(source=source, op=unary))
        else:
            # [ref, new] -> keep the new value below the store
            effects.append(Unary(source=source, op=unary))
            effects.append(DuplicateTopBelow(source=source, below_count=1))
        effects.append(StoreIndirect(source=source))
        return tuple(effects)

    return factory


def _delete(_context: BdvmEffectContext, _instruction: BdvmInstruction, source: SourceRef) -> tuple:
    """``delete obj[key]``.

    The VM pushes the JS delete result (``d`` lines 4263-4265).  Generic IR's
    ``Delete`` statement has no result value, so the statement is emitted and a
    boolean result is pushed for the VM's stack slot.  Every observed use
    discards that value immediately.
    """

    return (
        DeleteItem(source=source),
        Push(source=source, value=Const(value=True, source=source)),
    )


def _array(_context: BdvmEffectContext, instruction: BdvmInstruction, source: SourceRef) -> tuple:
    return (BuildArray(source=source, kind="list", count=instruction.operands[0]),)


def _forin_prepare(_context: BdvmEffectContext, instruction: BdvmInstruction, source: SourceRef) -> tuple:
    """``forin-prep slot``: snapshot the object's keys into the iteration state."""

    slot = instruction.operands[0]
    state_name = f"forin_state_{slot}"
    return (
        StoreLocal(
            source=source,
            name=state_name,
            target=Var(name=state_name, source=source),
            materialize=True,
        ),
    )


def _forin_next(context: BdvmEffectContext, instruction: BdvmInstruction, source: SourceRef) -> tuple:
    """``forin-next slot``: advance the iteration and bind the produced key.

    The VM pops the reference pair, writes the next key into that cell and
    pushes whether a key was produced (``d`` lines 4231-4238).  The neutral
    ``forin_next(state, target)`` helper carries both facts: it advances the
    state, binds the produced key to the target and returns the key or
    ``undefined``.  The pushed condition is ``key !== undefined`` so the
    recovered loop advances on every iteration and exits when the state is
    exhausted.
    """

    slot = instruction.operands[0]
    state = Var(name=f"forin_state_{slot}", source=source)
    binding = _forin_bindings(context.function).get(instruction.offset)
    target: object
    if binding is None:
        target = UndefinedLiteral(source=source)
    else:
        depth, cell = binding
        target = GetItem(
            source=source,
            obj=_frame_expr(depth, source),
            key=Const(value=cell, source=source),
        )
    return (
        Pop(source=source, count=2),
        Push(
            source=source,
            value=Call(
                source=source,
                callee=Global(name=_HELPER_FORIN_NEXT, source=source),
                args=(state, target),
                returns=1,
            ),
        ),
        Push(source=source, value=UndefinedLiteral(source=source)),
        Compare(source=source, op="!==", numeric_domain="default"),
    )


@lru_cache(maxsize=128)
def _forin_bindings(function: BdvmFunction) -> dict[int, tuple[int, int]]:
    """Map each ``forin-next`` cell to the ``ref`` that names its loop variable.

    The format always emits ``ref depth cell`` immediately before
    ``forin-next`` (verified for every occurrence in the sample), so the pair is
    a fixed VM idiom: ``ref`` forms the address and ``forin-next`` writes the
    produced key through it.
    """

    bindings: dict[int, tuple[int, int]] = {}
    instructions = function.instructions
    for index, instruction in enumerate(instructions):
        if instruction.mnemonic != "forin-next" or index == 0:
            continue
        previous = instructions[index - 1]
        if previous.mnemonic == "ref":
            bindings[instruction.offset] = (previous.operands[0], previous.operands[1])
    return bindings


def _mkclosure(context: BdvmEffectContext, instruction: BdvmInstruction, source: SourceRef) -> tuple:
    target = context.module.function_name(instruction.operands[0])
    return (Push(source=source, value=Global(name=f"<function {target}>", source=source)),)


def _top_undefined(_context: BdvmEffectContext, _instruction: BdvmInstruction, source: SourceRef) -> tuple:
    return (
        Pop(source=source, count=1, emit_calls=True),
        Push(source=source, value=UndefinedLiteral(source=source)),
    )


def _string(context: BdvmEffectContext, index: int) -> str:
    strings = context.module.strings
    if 0 <= index < len(strings):
        return strings[index]
    return f"<string {index}>"


def _unknown_opcode(_context: BdvmEffectContext, instruction: BdvmInstruction, source: SourceRef) -> tuple:
    """Contextual fallback for an opcode the decoder accepted but cannot map."""

    from unidecompiler.core.effects import UnknownOpcode

    return (UnknownOpcode(source=source, opcode=instruction.mnemonic, raw=instruction.raw),)


_BDVM_EXACT_EFFECTS: dict[str, object] = {
    "call": _call,
    "new": _new,
    "forin-prep": _forin_prepare,
    "forin-next": _forin_next,
    "import-global": _import_global,
    "newobj": _push_object,
    "propget": lambda _context, _instruction, source: (LoadItem(source=source),),
    "true": _push_const(True),
    "false": _push_const(False),
    "null": _push_const(None),
    "NaN": lambda _context, _instruction, source: (
        Push(source=source, value=Global(name="NaN", source=source)),
    ),
    "Infinity": lambda _context, _instruction, source: (
        Push(source=source, value=Global(name="Infinity", source=source)),
    ),
    "top=undef": _top_undefined,
    "instanceof": _binary("instanceof"),
    "propset": lambda _context, _instruction, source: (
        StoreItemEffect(source=source, order="obj-key-value"),
    ),
    "globalset": _store_global_name,
    "propset3": _propset_keep_value,
    "dup": lambda _context, _instruction, source: (DuplicateTop(source=source),),
    "cpropset": _store_attr,
    "guard": _no_effect,
    "jif-pop": _no_effect,
    "jeq-pop": _no_effect,
    "jtrue-pop": _no_effect,
    "jfalse": _no_effect,
    "jtrue": _no_effect,
    "jump": _no_effect,
    "ret": _no_effect,
    "delete": _delete,
    # ``pop`` discards the value, but a discarded deferred call still has to
    # run, so core is asked to emit call expressions as statements.
    "pop": lambda _context, _instruction, source: (
        Pop(source=source, count=1, emit_calls=True),
    ),
    "cpropget": _load_attr,
    "push-undef": _push_undefined,
    "thisctx": _push_this,
    "pushint": _push_int,
    "array": _array,
    "preinc": _push_increment("preinc"),
    "predec": _push_increment("predec"),
    "postinc": _push_increment("postinc"),
    "postdec": _push_increment("postdec"),
    "pushnum": _push_number,
    "defgetter": _define_property,
    "defsetter": _define_property,
    "defprop": _define_property,
    "throw": lambda _context, _instruction, source: (RaiseTop(source=source),),
    "storescope": _store_scope,
    "in": _binary("in"),
    "loadglobal": _push_global_name,
    "ref": _push_reference,
    "mkclosure": _mkclosure,
    "typeofglobal": _typeof_global,
    "declareglobal": _declare_global,
    "pushstr": _push_string,
    "loadscope": _load_scope,
    "finish": lambda _context, _instruction, source: (
        ReturnTop(source=source, empty_is_void=False),
    ),
    **{mnemonic: _binary(op) for mnemonic, op in _BINARY_OPERATORS.items()},
    **{
        mnemonic: _binary(op, semantics="dynamic", numeric_domain="float")
        for mnemonic, op in _ARITHMETIC_OPERATORS.items()
    },
    **{
        mnemonic: _binary(op, semantics="static", numeric_domain="signed", bit_width=32)
        for mnemonic, op in _BITWISE_OPERATORS.items()
    },
    "ushr": _binary(">>>", semantics="static", numeric_domain="unsigned", bit_width=32),
    **{mnemonic: _compare(op) for mnemonic, op in _COMPARE_OPERATORS.items()},
    **{mnemonic: _unary(op) for mnemonic, op in _UNARY_OPERATORS.items()},
}

BDVM_EFFECT_TABLE: VMEffectTable[BdvmEffectContext, BdvmInstruction] = VMEffectTable(
    opcode_attr="mnemonic",
    ignored=frozenset({"guard"}),
    exact=_BDVM_EXACT_EFFECTS,
    rules=(),
    fallback=_unknown_opcode,
)


# --------------------------------------------------------------------------
# thin-IR submission
# --------------------------------------------------------------------------


def _function_spec(function: BdvmFunction) -> VMFunctionSpec:
    return VMFunctionSpec(
        name=function.name,
        params=tuple(
            _scope_cell_name(0, 2 + index) for index in range(max(0, function.params))
        ),
        frontend=FRONTEND_ID,
        instruction_count=function.instruction_count,
    )


def _operand_role(instruction: BdvmInstruction, index: int) -> str:
    if instruction.mnemonic in _SCOPE_OPERAND_OPS:
        return "local" if index == 1 else "immediate"
    if instruction.mnemonic in _FORIN_OPERAND_OPS:
        return "local"
    if instruction.mnemonic in _STRING_OPERAND_OPS:
        return "constant"
    if instruction.mnemonic in _CLOSURE_OPERAND_OPS:
        return "target"
    if instruction.target is not None and index == 0:
        return "target"
    return "immediate"


def _operand_text(context: BdvmEffectContext, instruction: BdvmInstruction, index: int) -> str:
    value = instruction.operands[index]
    if instruction.mnemonic in _STRING_OPERAND_OPS:
        return f"{value}:{_string(context, value)}"
    if instruction.mnemonic in _CLOSURE_OPERAND_OPS:
        return f"{value}:{context.module.function_name(value)}"
    if instruction.mnemonic in _SCOPE_OPERAND_OPS:
        return f"{value}"
    if instruction.target is not None and index == 0:
        return f"{value}->{instruction.target}"
    return str(value)


def _decoded_instruction(
    context: BdvmEffectContext,
    instruction: BdvmInstruction,
    source: SourceRef,
) -> VMDecodedInstruction:
    return VMDecodedInstruction(
        opcode=instruction.mnemonic,
        source=source,
        operands=tuple(
            VMOperand(
                role=_operand_role(instruction, index),
                value=value,
                text=_operand_text(context, instruction, index),
            )
            for index, value in enumerate(instruction.operands)
        ),
        raw=instruction.raw,
        # The instruction cells live in the inflated varint stream, which is a
        # derived buffer: the packed input is base64 text over raw-deflate data,
        # so a cell has no provable position in the input artifact.  The range
        # stays unset rather than being derived from a compressed offset.
        artifact_range=None,
    )


def _effective_target(function: BdvmFunction, instruction: BdvmInstruction) -> int | None:
    """Return the control-flow target the interpreter actually uses.

    ``ret`` computes its resume position from the operand, but ``y`` first
    redirects a return that sits inside a protected range with a finally block
    to that finally's entry. The hint must describe the executed edge, not the
    raw arithmetic.
    """

    if instruction.mnemonic != "ret" or instruction.target is None:
        return instruction.target
    pc = instruction.offset + 2
    for region in reversed(function.regions):
        if region.try_start < pc <= region.finally_end:
            if pc <= region.finally_start and region.has_finally:
                return region.finally_start
            return instruction.target
    return instruction.target


def _handler_intervals(function: BdvmFunction) -> tuple[tuple[int, int, int, bool], ...]:
    """Flatten the region table into disjoint protected intervals.

    ``y`` scans the table backwards, so a later entry overrides an earlier one
    inside its own range: a protected instruction reaches the innermost catch
    (line 4356) and otherwise the innermost finally (line 4357).  Flattening
    reproduces that priority exactly and yields disjoint intervals, which is
    what core's protected-range recovery needs to keep nested try blocks
    recoverable.  Each interval is ``(start, end, handler, push_exception)``.
    """

    assignment: dict[int, tuple[int, bool]] = {}
    for region in function.regions:
        if region.has_handler:
            for cell in range(region.try_start, region.handler_start):
                assignment[cell] = (region.handler_start, True)
        if region.has_finally:
            start = region.handler_start if region.has_handler else region.try_start
            for cell in range(start, region.finally_start):
                assignment[cell] = (region.finally_start, False)
    intervals: list[tuple[int, int, int, bool]] = []
    for cell in sorted(assignment):
        handler, push = assignment[cell]
        if (
            intervals
            and intervals[-1][1] == cell
            and intervals[-1][2] == handler
            and intervals[-1][3] == push
        ):
            start, _end, _handler, _push = intervals[-1]
            intervals[-1] = (start, cell + 1, handler, push)
        else:
            intervals.append((cell, cell + 1, handler, push))
    return tuple(intervals)


def _hints(
    function: BdvmFunction,
    instruction: BdvmInstruction,
    source: SourceRef,
) -> tuple[VMHint, ...]:
    hints: list[VMHint] = []
    target = _effective_target(function, instruction)
    if target is not None:
        kind = "loop-backedge" if target <= instruction.offset else "branch-target"
        detail = _BRANCH_DETAIL.get(instruction.mnemonic)
        flow = (
            "conditional"
            if instruction.opcode in CONDITIONAL_BRANCH_OPCODES
            else "unconditional"
        )
        hints.append(
            VMHint(
                kind=kind,
                source=source,
                target=target,
                label=instruction.mnemonic,
                detail=detail,
                flow=flow,
            )
        )
    if instruction.mnemonic in VALUE_RETAINING_BRANCHES:
        # This branch leaves its tested value on the operand stack for one
        # successor.  The retained value is expressed with a stack duplicate
        # plus a deferred drop, which only the exact low-level CFG preserves.
        hints.append(
            VMHint(
                kind="materialized-condition",
                source=source,
                label=instruction.mnemonic,
            )
        )
    if instruction.offset == 0:
        for start, end, handler, push in _handler_intervals(function):
            value = {
                "start": start,
                "end": end,
                "target": handler,
                # ``y`` pushes exactly the raised value at a catch entry
                # (line 4356) and only redirects the program counter at a
                # finally entry (lines 4345, 4349, 4357).  The protected
                # instructions are stack-balanced at every throw point, so the
                # entry stack below that value is empty.
                "stack_depth": 0,
            }
            if push:
                value["stack_suffix"] = ["exception"]
            hints.append(
                VMHint(
                    kind="exception-region",
                    source=source,
                    target=handler,
                    value=value,
                    label="desc",
                )
            )
    return tuple(hints)


def _bytecode_step(
    context: BdvmEffectContext,
    function: BdvmFunction,
    instruction: BdvmInstruction,
) -> VMBytecodeStep:
    source = SourceRef(
        frontend=FRONTEND_ID,
        offset=instruction.offset,
        detail=f"cell={instruction.offset}",
    )
    decoded = _decoded_instruction(context, instruction, source)
    effects = BDVM_EFFECT_TABLE.effects_for(context, instruction, source)
    extras, deferred_pops, deferred_loads = _branch_value_facts(function)
    load_name = deferred_loads.get(instruction.offset)
    if load_name:
        # The branch consumed this value before the successor instruction ran,
        # so the deferred restore precedes the instruction's own effects.
        effects = (
            LoadLocal(
                source=source,
                name=load_name,
                fallback=Var(name=load_name, source=source),
            ),
            *effects,
        )
    drop_count = deferred_pops.get(instruction.offset, 0)
    if drop_count:
        effects = (*(Pop(source=source, count=1) for _ in range(drop_count)), *effects)
    extra_effects = extras.get(instruction.offset)
    if extra_effects:
        effects = (*effects, *extra_effects)
    return VMBytecodeStep(
        opcode=instruction.mnemonic,
        source=source,
        effects=effects,
        raw=instruction.raw,
        decoded=decoded,
        hints=_hints(function, instruction, source),
    )


@lru_cache(maxsize=128)
def _branch_value_facts(
    function: BdvmFunction,
) -> tuple[dict[int, tuple], dict[int, int], dict[int, str]]:
    """Describe the value a branch retains on one of its successors.

    ``jif-pop``/``jtrue-pop`` keep the tested value on the jump edge and pop it
    on the fallthrough edge (``d`` lines 4258-4261); ``jeq-pop`` keeps the
    compared value on the fallthrough edge and drops it on the jump edge (line
    4263).  Core removes a branch's condition from both successors, so the
    frontend restores the retained value and submits the matching edge effect on
    the instruction that owns that edge.  Both facts are plain stack effects;
    no control-flow structure is built here.
    """

    extras: dict[int, tuple] = {}
    deferred_pops: dict[int, int] = {}
    deferred_loads: dict[int, str] = {}
    instructions = function.instructions
    for index, instruction in enumerate(instructions):
        if instruction.mnemonic in {"jif-pop", "jtrue-pop"}:
            extras[instruction.offset] = (DuplicateTop(source=None),)
            if index + 1 < len(instructions):
                fallthrough = instructions[index + 1]
                if instruction.target != fallthrough.offset:
                    deferred_pops[fallthrough.offset] = deferred_pops.get(fallthrough.offset, 0) + 1
        elif instruction.mnemonic == "jeq-pop":
            name = f"case_subject_{instruction.offset}"
            extras[instruction.offset] = (
                Copy(source=None, depth=2),
                StoreLocal(
                    source=None,
                    name=name,
                    target=Var(name=name, source=None),
                    materialize=True,
                ),
            )
            if index + 1 < len(instructions):
                deferred_loads[instructions[index + 1].offset] = name
    return extras, deferred_pops, deferred_loads


def _raw_window(function: BdvmFunction, index: int) -> tuple[str, ...]:
    start = max(0, index - 2)
    end = min(len(function.instructions), index + 3)
    return tuple(
        f"{instruction.offset}: {instruction.raw}"
        for instruction in function.instructions[start:end]
    )


def _branch_stack_width(branch: VMBytecodeStep) -> int:
    return _BRANCH_WIDTH.get(branch.opcode, 1)


def _branch_condition(branch: VMBytecodeStep, stack: tuple) -> object | None:
    """Return the tested value; its truthiness selects the hint's polarity.

    ``jfalse``/``jif-pop`` jump when the value is falsey, so their hint says
    ``target-if-false``; ``jtrue-pop``/``jtrue`` jump when it is truthy.  For
    ``jeq-pop`` the VM pops the expected value and compares it with the value
    below it (``d`` line 4263), so the tested value is the equality.
    """

    source = branch.source
    if branch.opcode == "jeq-pop":
        if len(stack) < 2:
            return None
        return _equality(source, stack[-2], stack[-1])
    if not stack:
        return None
    return stack[-1]


def _equality(source: SourceRef, left: object, right: object) -> object:
    return BinaryOp(
        source=source,
        op="===",
        left=left,
        right=right,
        semantics="dynamic",
    )


def _linear_state(
    context: BdvmEffectContext,
    function: BdvmFunction,
    steps: tuple[VMBytecodeStep, ...],
    start: int,
    end: int,
    initial_locals: dict,
    initial_stack: tuple,
) -> VMLinearState | None:
    if start >= end:
        return VMLinearState(
            locals=dict(initial_locals),
            stack=tuple(initial_stack),
        )
    result = lift_steps(
        steps[start:end],
        initial_locals=dict(initial_locals),
        initial_stack=initial_stack,
    )
    if result.state.diagnostics:
        return None
    if result.stopped_at is not None and result.state.terminator is None:
        return None
    return VMLinearState(
        locals=dict(result.state.locals),
        stack=tuple(result.state.stack),
        statements=tuple(result.state.statements),
        terminator=result.state.terminator,
    )


def _region_profile(
    steps: tuple[VMBytecodeStep, ...],
    function: BdvmFunction,
) -> object:
    return build_hint_region_profile(
        steps,
        frontend=FRONTEND_ID,
        opcode_classes=BDVM_REGION_OPCODE_CLASSES,
        raw_window=lambda index: _raw_window(function, index),
    )


def lift_function(module: BdvmModule, function: BdvmFunction):
    """Lift one decoded function into generic IR through the core pipeline."""

    context = BdvmEffectContext(module=module, function=function)
    steps = tuple(
        _bytecode_step(context, function, instruction)
        for instruction in function.instructions
    )
    return lift_vm_step_function(
        _function_spec(function),
        steps,
        profile=_region_profile(steps, function),
        stateful_callbacks=VMStatefulCallbacks(
            initial_locals=lambda: {},
            lift_linear=lambda start, end, locals, stack: _linear_state(
                context, function, steps, start, end, locals, stack
            ),
            branch_condition=_branch_condition,
            branch_stack_width=_branch_stack_width,
        ),
        raw_window=lambda index: _raw_window(function, index),
    )


def lift_module(module: FrontendModule, *, reporter: ProgressReporter | None = None):
    """Submit complete VMBytecodeStep streams to core from this module's model."""

    if module.frontend_id != FRONTEND_ID:
        raise TypeError(f"bdvm frontend cannot lift module from {module.frontend_id!r}")
    decoded: BdvmModule = module.payload
    total = len(decoded.functions)
    report_progress(
        reporter,
        phase="lift",
        status="started",
        completed=0,
        total=total,
        unit="function",
        message=f"lifting {total} bdvm functions",
    )
    functions: list = []
    for index, function in enumerate(decoded.functions, start=1):
        item_label = function.name or f"function@{index - 1}"
        report_progress(
            reporter,
            phase="lift",
            completed=index - 1,
            total=total,
            unit="function",
            item_label=item_label,
            message=f"lifting {item_label}",
        )
        functions.append(
            recover_vm_function(
                _function_spec(function),
                lambda function=function: lift_function(decoded, function),
                raw=tuple(instruction.raw for instruction in function.instructions),
            )
        )
        report_progress(
            reporter,
            phase="lift",
            completed=index,
            total=total,
            unit="function",
            item_label=item_label,
            message=f"lifted {item_label}",
        )
    report_progress(
        reporter,
        phase="lift",
        status="completed",
        completed=total,
        total=total,
        unit="function",
        message=f"lifted {total} bdvm functions",
    )
    return assemble_vm_module(
        name=_module_name(module),
        source_language="javascript",
        functions=tuple(functions),
        metadata={
            "frontend": module.metadata,
            "bytecode_format": "bdvm-packed",
            "string_pool_size": len(decoded.strings),
            "entry_indices": decoded.entry_indices,
            "packed_size": decoded.packed_size,
            "inflated_size": decoded.inflated_size,
        },
    )


def _module_name(module: FrontendModule) -> str:
    """Return a display name that never embeds a directory path."""

    filename = None
    if isinstance(module.metadata, dict):
        filename = module.metadata.get("filename")
    if not filename:
        return "<bdvm-module>"
    return Path(str(filename)).name or "<bdvm-module>"
