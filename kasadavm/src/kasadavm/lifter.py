"""Thin-IR lifting for the Kasada VM.

Every decoded instruction becomes exactly one :class:`VMBytecodeStep` carrying
neutral effects and neutral hints.  This module never builds blocks, CFG, AST,
regions, or source structures, and it never executes the VM.

Register model
--------------
A frame is the interpreter's ``E`` array.  ``E[0]`` is the program counter,
``E[1]`` the scope object, ``E[2]`` the return-value cell and ``E[3]`` the
arguments object; everything above is a general-purpose cell.  Cells are
modelled as ordinary locals (``r4``, ``r5``, ...) so that core owns stack and
value recovery.  Scope slots (``scope.x[key]``) are modelled as locals named
``v<key>`` when the slot key is a literal, and as item stores on the scope
table when the key is dynamic.

Every effect sequence is stack-balanced per instruction: sources are loaded,
the operation pushes exactly one value, and the store consumes it.  The
operand stack is therefore empty at every instruction boundary, which lets the
frontend state exact exceptional-edge state without guessing.
"""

from __future__ import annotations

from typing import Callable

from unidecompiler.core.effects import (
    Binary,
    BuildCall,
    BuildArray,
    BuildArrayCall,
    BuildMap,
    CallStackArgs,
    Effect,
    Invoke,
    InvokeExpanded,
    LoadAttr,
    LoadItem,
    LoadLocal,
    MakeFunctionValue,
    Push,
    RaiseTop,
    ReturnTop,
    ReturnVoid,
    StoreItemEffect,
    StoreLocal,
    Unary,
)
from unidecompiler.core.ir import (
    BinaryOp,
    Const,
    Expr,
    GetAttr,
    Global,
    SourceRef,
    UndefinedLiteral,
    Var,
)
from unidecompiler.core.vm_bytecode import VMBytecodeStep
from unidecompiler.core.vm_function import (
    VMFunctionSpec,
    lift_vm_step_function,
    recover_vm_function,
)
from unidecompiler.core.vm_hints import VMHint
from unidecompiler.core.vm_module import assemble_vm_module
from unidecompiler.core.vm_operands import VMDecodedInstruction, VMOperand
from unidecompiler.core.vm_region import (
    VMRegionOpcodeClasses,
    VMStatefulCallbacks,
    build_hint_region_profile,
)
from unidecompiler.core.vm_function import lift_steps
from unidecompiler.core.vm_region import VMLinearState

from .decoder import StringRef
from .model import (
    CALL_DEPTH,
    CALL_RECORD,
    CALL_STATE,
    HANDLER_CELL,
    HOST_GLOBAL,
    PENDING_EXCEPTION,
    RESUME_CELL,
    SAVED_CALL,
    SCOPE_TABLE,
    KasadaFunction,
    KasadaInstruction,
    KasadaProgram,
    register_name,
    scope_slot_name,
)
from .opcodes import BRANCHES, CONDITIONAL_BRANCHES

FRONTEND_ID = "kasadavm"

#: JavaScript bitwise operators coerce to 32-bit signed integers.
_INT32 = {"semantics": "fixed", "numeric_domain": "signed", "bit_width": 32, "overflow_policy": "wrap"}

#: ``dst = a <op> b`` with two register cells.
_BINARY_RRS: dict[int, str] = {
    1: "|", 2: "^", 7: "in", 14: ">=", 21: "!==", 22: "%", 23: ">", 24: "/",
    32: "-", 42: "+", 47: "<<", 54: "instanceof", 56: "<", 68: "&",
    72: "===", 85: "*",
}
#: ``dst = reg <op> value``.
_BINARY_RVS: dict[int, str] = {
    13: "<", 16: "&", 25: "%", 31: ">=", 33: ">>>", 35: "<=", 40: "<<",
    43: "===", 48: "+", 66: "!==", 70: "-", 76: "*", 80: "/", 84: ">",
}
#: ``dst = value <op> reg``.
_BINARY_VRS: dict[int, str] = {
    5: "+", 11: "/", 29: "==", 34: "^", 37: "in", 38: "!=", 51: "!==",
    58: "===", 64: "|",
}
#: ``dst = value <op> value``.
_BINARY_VVS: dict[int, str] = {9: "|", 27: "+", 65: "==", 67: "-"}

_BITWISE = frozenset({"|", "^", "&", "<<", ">>>", "~"})

_UNARY: dict[int, str] = {28: "+", 36: "~", 39: "typeof ", 73: "not "}

_CONTROL_OPCODES = frozenset(
    {3, 6, 20, 44, 62, 69, 77, 79}
)  # THROW HALT JUMP_IF_TRUE FRAME_EPILOGUE JUMP_IF_FALSE CALL_FRAME CALL_UNDEFINED_TARGET JUMP

REGION_CLASSES = VMRegionOpcodeClasses(
    noise=frozenset(),
    control=frozenset(
        {
            "THROW",
            "HALT",
            "FRAME_EPILOGUE",
            "JUMP",
            "JUMP_IF_TRUE",
            "JUMP_IF_FALSE",
            "CALL_FRAME",
            "CALL_UNDEFINED_TARGET",
        }
    ),
    jumps=frozenset({"JUMP", "JUMP_IF_TRUE", "JUMP_IF_FALSE"}),
    forward_jumps=frozenset({"JUMP", "JUMP_IF_TRUE", "JUMP_IF_FALSE"}),
    backward_jumps=frozenset({"JUMP", "JUMP_IF_TRUE", "JUMP_IF_FALSE"}),
    conditional_jumps=frozenset({"JUMP_IF_TRUE", "JUMP_IF_FALSE"}),
)


# --------------------------------------------------------------------------
# operand helpers
# --------------------------------------------------------------------------


def _var(name: str, source: SourceRef) -> Var:
    return Var(name=name, source=source)


def _load_fallback(name: str, fallback: Expr, source: SourceRef) -> Effect:
    """Load a local with an explicit, purpose-built fallback expression."""

    return LoadLocal(source=source, name=name, fallback=fallback)


def _load(name: str, source: SourceRef) -> Effect:
    """Load a local, always carrying an explicit fallback value.

    ``LoadLocal`` without a fallback resolves an unknown name to an opaque
    placeholder expression, which renders as an unsupported node.  A frame
    cell that has not been written yet holds ``undefined`` in the VM, so that
    is the fallback: the read is defined, and core's value-invariant pass does
    not have to treat it as an unbound variable.
    """

    return LoadLocal(source=source, name=name, fallback=UndefinedLiteral(source=source))


def _const(value: object, source: SourceRef) -> Const:
    return Const(value=value, source=source)


def _push_value(instruction: KasadaInstruction, index: int, source: SourceRef) -> list[Effect]:
    """Effects that put one *value* cell on the operand stack."""

    value = instruction.value_operands()[index]
    if value.kind == "register":
        return [_load(register_name(value.register), source)]
    if value.kind == "string" and isinstance(value.value, StringRef):
        # The bytes live in an external string table that this artifact does
        # not carry; the reference itself is the decoded constant.
        return [Push(source=source, value=_const(value.value, source))]
    return [Push(source=source, value=_const(value.value, source))]


def _operand_expr(instruction: KasadaInstruction, index: int, source: SourceRef) -> Expr:
    """Build the IR expression for one operand cell without touching the stack."""

    value = instruction.value_operands()[index]
    if value.kind == "register":
        return Var(name=register_name(value.register), source=source)
    return _const(value.value, source)


def _push_register(index: int, source: SourceRef) -> list[Effect]:
    return [_load(register_name(index), source)]


def _store(destination: int | None, source: SourceRef) -> list[Effect]:
    if destination is None:
        return []
    return [StoreLocal(source=source, name=register_name(destination))]


def _slot_read_effects(instruction: KasadaInstruction, index: int, source: SourceRef) -> list[Effect]:
    """Read a scope slot named by a value cell (literal key or dynamic key)."""

    value = instruction.value_operands()[index]
    if value.kind == "int":
        return [_load(scope_slot_name(value.value), source)]
    return [
        _load(register_name(1), source),
        LoadAttr(source=source, attr="x"),
        *_push_value(instruction, index, source),
        LoadItem(source=source),
    ]


def _slot_write_effects(
    instruction: KasadaInstruction, key_index: int, value_index: int, source: SourceRef
) -> list[Effect]:
    """Write a scope slot named by one value cell from another value cell."""

    key = instruction.value_operands()[key_index]
    if key.kind == "int":
        return [
            *_push_value(instruction, value_index, source),
            StoreLocal(source=source, name=scope_slot_name(key.value)),
        ]
    return [
        _load(register_name(1), source),
        LoadAttr(source=source, attr="x"),
        *_push_value(instruction, key_index, source),
        *_push_value(instruction, value_index, source),
        StoreItemEffect(source=source, order="obj-key-value"),
    ]


def _clear_local(name: str, source: SourceRef) -> list[Effect]:
    return [Push(source=source, value=_const(None, source)), StoreLocal(source=source, name=name)]


def _scope_field(attr: str, source: SourceRef) -> list[Effect]:
    return [
        _load(register_name(1), source),
        LoadAttr(source=source, attr=attr),
    ]


# --------------------------------------------------------------------------
# per-opcode effect builders
# --------------------------------------------------------------------------


def _binary_effects(op: str, left: list[Effect], right: list[Effect], destination, source) -> list[Effect]:
    kwargs = _INT32 if op in _BITWISE else {}
    return [
        *left,
        *right,
        Binary(source=source, op=op, **kwargs),
        *_store(destination, source),
    ]


def _effects_for(instruction: KasadaInstruction, program: KasadaProgram, source: SourceRef) -> tuple[Effect, ...]:
    opcode = instruction.opcode
    destination = instruction.store_register()

    if opcode in _BINARY_RRS:
        op = _BINARY_RRS[opcode]
        a, b = instruction.register_operands()
        return tuple(
            _binary_effects(op, _push_register(a, source), _push_register(b, source), destination, source)
        )
    if opcode in _BINARY_RVS:
        op = _BINARY_RVS[opcode]
        (a,) = instruction.register_operands()
        return tuple(
            _binary_effects(
                op, _push_register(a, source), _push_value(instruction, 0, source), destination, source
            )
        )
    if opcode in _BINARY_VRS:
        op = _BINARY_VRS[opcode]
        (b,) = instruction.register_operands()
        return tuple(
            _binary_effects(
                op, _push_value(instruction, 0, source), _push_register(b, source), destination, source
            )
        )
    if opcode in _BINARY_VVS:
        op = _BINARY_VVS[opcode]
        return tuple(
            _binary_effects(
                op,
                _push_value(instruction, 0, source),
                _push_value(instruction, 1, source),
                destination,
                source,
            )
        )
    if opcode in _UNARY:
        return tuple(
            [
                *_push_value(instruction, 0, source),
                Unary(source=source, op=_UNARY[opcode]),
                *_store(destination, source),
            ]
        )

    builder = _SPECIAL.get(opcode)
    if builder is None:
        raise AssertionError(f"no effect builder for opcode {opcode} ({instruction.mnemonic})")
    return tuple(builder(instruction, program, source, destination))


def _build_call_apply(instruction, program, source, destination):
    # CALL_APPLY: callee.apply(this_value, args_array).  The result lands in the
    # return cell (E[2]) because the closure returns its own E[2].
    callee = _operand_expr(instruction, 1, source)
    return [
        *_push_value(instruction, 0, source),
        *_push_value(instruction, 2, source),
        BuildCall(
            source=source,
            callee=GetAttr(source=source, obj=callee, attr="apply"),
            arg_count=2,
            returns=1,
        ),
        # CALL_APPLY has no store cell: the handler writes the result into the
        # frame's return cell E[2] for the native path.
        StoreLocal(source=source, name=register_name(2)),
    ]


def _build_get_item(instruction, program, source, destination):
    return [
        *_push_value(instruction, 0, source),
        *_push_value(instruction, 1, source),
        LoadItem(source=source),
        *_store(destination, source),
    ]


def _build_set_item(instruction, program, source, destination):
    return [
        *_push_value(instruction, 0, source),
        *_push_value(instruction, 1, source),
        *_push_value(instruction, 2, source),
        StoreItemEffect(source=source, order="obj-key-value"),
    ]


def _build_delete_item(instruction, program, source, destination):
    # The handler stores ``delete obj[key]`` into a register, but generic IR has
    # no value-producing delete effect.  The shared stack-arguments call
    # primitive keeps both decoded operands and the destination instead of
    # dropping the mutation or inventing a boolean result.
    return [
        *_push_value(instruction, 0, source),
        *_push_value(instruction, 1, source),
        CallStackArgs(source=source, callee_name="delete_item", arg_count=2, returns=1),
        *_store(destination, source),
    ]


def _build_throw(instruction, program, source, destination):
    return [*_push_value(instruction, 0, source), RaiseTop(source=source)]


def _build_halt(instruction, program, source, destination):
    # HALT ends the frame; the frame's return cell already holds the value.
    return [
        _load_fallback(register_name(2), UndefinedLiteral(source=source), source),
        ReturnTop(source=source, empty_is_void=True),
    ]


def _build_jump(instruction, program, source, destination):
    return []  # control-only instruction; the successor is a branch-target hint


def _build_frame_epilogue(instruction, program, source, destination):
    # The saved value is taken out of its scope slot before the frame ends.
    key = instruction.value_operands()[0]
    slot = scope_slot_name(key.value) if key.kind == "int" else SAVED_CALL
    return [
        _load(slot, source),
        *_clear_local(slot, source),
        ReturnTop(source=source, empty_is_void=True),
    ]


def _build_call_frame(instruction, program, source, destination):
    # X() records the call target and then either resumes at the scope resume
    # cell or replaces the frame, so this function's linear run ends here.
    return [
        *_push_value(instruction, 0, source),
        Invoke(source=source, arg_count=0, returns=0),
        ReturnVoid(source=source),
    ]


def _build_call_undefined_target(instruction, program, source, destination):
    return [ReturnVoid(source=source)]


def _build_call1(instruction, program, source, destination):
    return [
        *_push_value(instruction, 0, source),
        *_push_value(instruction, 1, source),
        Invoke(source=source, arg_count=1, returns=1),
        *_store(destination, source),
    ]


def _build_call2(instruction, program, source, destination):
    return [
        *_push_value(instruction, 0, source),
        *_push_value(instruction, 1, source),
        *_push_value(instruction, 2, source),
        Invoke(source=source, arg_count=2, returns=1),
        *_store(destination, source),
    ]


def _build_call3(instruction, program, source, destination):
    return [
        *_push_value(instruction, 0, source),
        *_push_value(instruction, 1, source),
        *_push_value(instruction, 2, source),
        *_push_value(instruction, 3, source),
        Invoke(source=source, arg_count=3, returns=1),
        *_store(destination, source),
    ]


def _build_call0(instruction, program, source, destination):
    return [
        *_push_value(instruction, 0, source),
        Invoke(source=source, arg_count=0, returns=1),
        *_store(destination, source),
    ]


def _build_new_callee(instruction, program, source, destination):
    # NEW_CALLEE constructs the callee with expanded arguments.  Generic IR has
    # no construct-with-expanded-arguments effect, so the shared expand-call
    # primitive carries the two decoded operands without dropping either.
    return [
        *_push_value(instruction, 0, source),
        *_push_value(instruction, 1, source),
        InvokeExpanded(source=source, has_keywords=False, returns=1),
        *_store(destination, source),
    ]


def _build_new_regexp(instruction, program, source, destination):
    return [
        *_push_value(instruction, 0, source),
        *_push_value(instruction, 1, source),
        CallStackArgs(source=source, callee_name="RegExp", arg_count=2, returns=1),
        *_store(destination, source),
    ]


def _build_new_array(instruction, program, source, destination):
    return [
        *_push_value(instruction, 0, source),
        BuildArrayCall(source=source, kind="array"),
        *_store(destination, source),
    ]


def _build_new_object(instruction, program, source, destination):
    return [BuildMap(source=source, count=0), *_store(destination, source)]


def _build_array_literal(instruction, program, source, destination):
    return [BuildArray(source=source, kind="list", count=0), *_store(destination, source)]


def _build_push_promise(instruction, program, source, destination):
    return [Push(source=source, value=Global(name="Promise", source=source)), *_store(destination, source)]


def _build_push_regenerator(instruction, program, source, destination):
    return [
        Push(source=source, value=Global(name="regenerator", source=source)),
        *_store(destination, source),
    ]


def _build_push_scope_global(instruction, program, source, destination):
    return [*_scope_field("c", source), *_store(destination, source)]


def _build_push_parent_scope(instruction, program, source, destination):
    key = instruction.value_operands()[0]
    if key.kind == "int":
        return [*_scope_field("H", source), StoreLocal(source=source, name=scope_slot_name(key.value))]
    return [
        _load(register_name(1), source),
        LoadAttr(source=source, attr="x"),
        *_push_value(instruction, 0, source),
        *_scope_field("H", source),
        StoreItemEffect(source=source, order="obj-key-value"),
    ]


def _build_get_global(instruction, program, source, destination):
    value = instruction.value_operands()[0]
    if value.kind == "string" and isinstance(value.value, str):
        return [Push(source=source, value=Global(name=value.value, source=source)), *_store(destination, source)]
    return [
        Push(source=source, value=Global(name=HOST_GLOBAL, source=source)),
        *_push_value(instruction, 0, source),
        LoadItem(source=source),
        *_store(destination, source),
    ]


def _build_scope_set(instruction, program, source, destination):
    return _slot_write_effects(instruction, 0, 1, source)


def _build_scope_assign(instruction, program, source, destination):
    return _slot_write_effects(instruction, 0, 1, source)


def _build_scope_lookup(instruction, program, source, destination):
    return [*_slot_read_effects(instruction, 0, source), *_store(destination, source)]


def _build_declare_var(instruction, program, source, destination):
    key = instruction.value_operands()[0]
    if key.kind == "int":
        return [
            Push(source=source, value=_const(None, source)),
            StoreLocal(source=source, name=scope_slot_name(key.value)),
        ]
    return [
        Push(source=source, value=Global(name=SCOPE_TABLE, source=source)),
        *_push_value(instruction, 0, source),
        Push(source=source, value=_const(None, source)),
        StoreItemEffect(source=source, order="obj-key-value"),
    ]


def _build_catch_bind(instruction, program, source, destination):
    key = instruction.value_operands()[0]
    effects: list[Effect] = [_load(PENDING_EXCEPTION, source)]
    if key.kind == "int":
        effects.append(StoreLocal(source=source, name=scope_slot_name(key.value)))
    else:
        effects.extend(
            [
                Push(source=source, value=Global(name=SCOPE_TABLE, source=source)),
                *_push_value(instruction, 0, source),
                StoreItemEffect(source=source, order="obj-key-value"),
            ]
        )
    effects.extend(_clear_local(PENDING_EXCEPTION, source))
    return effects


def _build_set_handler(instruction, program, source, destination):
    return [*_push_value(instruction, 0, source), StoreLocal(source=source, name=HANDLER_CELL)]


def _build_set_resume(instruction, program, source, destination):
    return [*_push_value(instruction, 0, source), StoreLocal(source=source, name=RESUME_CELL)]


def _build_clear_exception(instruction, program, source, destination):
    return _clear_local(PENDING_EXCEPTION, source)


def _build_get_exception(instruction, program, source, destination):
    return [
        _load(PENDING_EXCEPTION, source),
        LoadAttr(source=source, attr="f"),
        *_store(destination, source),
    ]


def _build_save_call_state(instruction, program, source, destination):
    key = instruction.value_operands()[0]
    record: list[Effect] = [
        _load(CALL_RECORD, source),
        _load(CALL_STATE, source),
        _load(CALL_DEPTH, source),
        _build_map_effect(source),
    ]
    if key.kind == "int":
        record.append(StoreLocal(source=source, name=scope_slot_name(key.value)))
    else:
        record.extend(
            [
                Push(source=source, value=Global(name=SCOPE_TABLE, source=source)),
                *_push_value(instruction, 0, source),
                StoreItemEffect(source=source, order="obj-key-value"),
            ]
        )
    record.extend(_clear_local(CALL_RECORD, source))
    record.extend(_clear_local(CALL_STATE, source))
    record.extend(_clear_local(CALL_DEPTH, source))
    return record


def _build_map_effect(source: SourceRef) -> Effect:
    from unidecompiler.core.effects import BuildMap

    return BuildMap(
        source=source,
        count=3,
        keys=(
            _const("a", source),
            _const("i", source),
            _const("v", source),
        ),
    )


def _build_restore_call_state(instruction, program, source, destination):
    key = instruction.value_operands()[0]
    slot = scope_slot_name(key.value) if key.kind == "int" else SAVED_CALL
    if key.kind != "int":
        return [
            Push(source=source, value=Global(name=SCOPE_TABLE, source=source)),
            *_push_value(instruction, 0, source),
            LoadItem(source=source),
            StoreLocal(source=source, name=SAVED_CALL),
        ]
    effects: list[Effect] = [
        _load(slot, source),
        StoreLocal(source=source, name=SAVED_CALL),
        *_clear_local(slot, source),
    ]
    for attr, local in (("a", CALL_RECORD), ("i", CALL_STATE), ("v", CALL_DEPTH)):
        effects.extend(
            [
                _load(SAVED_CALL, source),
                LoadAttr(source=source, attr=attr),
                StoreLocal(source=source, name=local),
            ]
        )
    return effects


def _build_make_closure(instruction, program, source, destination):
    values = instruction.value_operands()
    entry = values[0].value if values[0].kind == "int" else None
    name_value = values[1].value
    name = name_value if isinstance(name_value, str) else ""
    label = name or (f"sub_{entry:x}" if entry is not None else "<closure>")
    return [
        Push(source=source, value=_const(label, source)),
        MakeFunctionValue(source=source, fallback_name=label),
        *_store(destination, source),
    ]


def _build_load(instruction, program, source, destination):
    return [*_push_value(instruction, 0, source), *_store(destination, source)]


_SPECIAL: dict[int, Callable[..., list[Effect]]] = {
    0: _build_call_apply,
    3: _build_throw,
    4: _build_get_item,
    6: _build_halt,
    8: _build_new_regexp,
    10: _build_new_callee,
    12: _build_call1,
    15: _build_scope_set,
    17: _build_set_resume,
    18: _build_push_promise,
    19: _build_scope_lookup,
    20: _build_jump,
    26: _build_scope_assign,
    30: _build_clear_exception,
    41: _build_set_item,
    44: _build_frame_epilogue,
    45: _build_new_array,
    46: _build_new_object,
    49: _build_push_regenerator,
    50: _build_delete_item,
    52: _build_get_exception,
    53: _build_restore_call_state,
    55: _build_call0,
    57: _build_get_global,
    59: _build_make_closure,
    60: _build_array_literal,
    61: _build_catch_bind,
    62: _build_jump,
    63: _build_push_parent_scope,
    69: _build_call_frame,
    71: _build_save_call_state,
    74: _build_declare_var,
    75: _build_set_handler,
    77: _build_call_undefined_target,
    78: _build_push_scope_global,
    79: _build_jump,
    81: _build_load,
    82: _build_call3,
    83: _build_call2,
}


# --------------------------------------------------------------------------
# hints, steps, functions, module
# --------------------------------------------------------------------------


def _operand_role(instruction: KasadaInstruction, position: int) -> str:
    opcode = instruction.opcode
    if opcode == 79 and position == 0:
        return "target"
    if opcode in CONDITIONAL_BRANCHES and position == 1:
        return "target"
    if opcode == 59 and position == 0:
        return "target"
    return ""


def _vm_operands(instruction: KasadaInstruction) -> tuple[VMOperand, ...]:
    out: list[VMOperand] = []
    for position, operand in enumerate(instruction.operands):
        marked_target = _operand_role(instruction, position)
        if marked_target == "target":
            out.append(VMOperand(role="target", value=int(operand.as_value().value), text=operand.text))
            continue
        if operand.role == "store":
            out.append(VMOperand(role="register", value=operand.as_register(), text=operand.text))
            continue
        if operand.role == "register":
            out.append(VMOperand(role="register", value=operand.as_register(), text=operand.text))
            continue
        value = operand.as_value()
        if value.kind == "register":
            out.append(VMOperand(role="register", value=value.register, text=operand.text))
        elif value.kind == "int":
            out.append(VMOperand(role="immediate", value=value.value, text=operand.text))
        else:
            out.append(VMOperand(role="constant", value=value.value, text=operand.text))
    return tuple(out)


def _branch_flow(target: int, offset: int) -> str:
    return "loop-backedge" if target <= offset else "branch-target"


def _hints_for(instruction: KasadaInstruction, source: SourceRef) -> tuple[VMHint, ...]:
    opcode = instruction.opcode
    if opcode in BRANCHES:
        values = instruction.value_operands()
        candidate = values[0] if opcode == 79 else (values[1] if len(values) > 1 else None)
        if candidate is None or candidate.kind != "int":
            return ()
        target = int(candidate.value)
        return (
            VMHint(
                kind=_branch_flow(target, instruction.offset),
                source=source,
                target=target,
                label=instruction.mnemonic,
                flow="unconditional" if opcode == 79 else "conditional",
                detail=None if opcode == 79 else "target-if-true",
            ),
        )
    return ()


def make_step(instruction: KasadaInstruction, program: KasadaProgram) -> VMBytecodeStep:
    source = SourceRef(frontend=FRONTEND_ID, offset=instruction.offset)
    decoded = VMDecodedInstruction(
        opcode=instruction.mnemonic,
        source=source,
        operands=_vm_operands(instruction),
        raw=instruction.raw,
        artifact_range=_artifact_range(instruction),
    )
    return VMBytecodeStep(
        opcode=instruction.mnemonic,
        source=source,
        effects=_effects_for(instruction, program, source),
        raw=instruction.raw,
        decoded=decoded,
        hints=_hints_for(instruction, source),
    )


def _artifact_range(instruction: KasadaInstruction):
    from unidecompiler.provenance import ByteRange

    return ByteRange(instruction.artifact_offset, instruction.artifact_size)


def make_steps(function: KasadaFunction, program: KasadaProgram) -> tuple[VMBytecodeStep, ...]:
    return tuple(make_step(instruction, program) for instruction in function.instructions)


def local_names_of(steps: tuple[VMBytecodeStep, ...]) -> tuple[str, ...]:
    """Local names the step stream binds.

    Only names that are actually written count as declared.  A name that is
    read but never written keeps its ``undefined`` fallback instead of being
    advertised as a declared local, so the value-invariant pass never reports
    a read-before-bind for a cell the bytecode never assigns.
    """

    names: set[str] = set()
    for step in steps:
        for effect in step.effects or ():
            name = getattr(effect, "name", None)
            if isinstance(effect, StoreLocal) and isinstance(name, str):
                names.add(name)
    return tuple(sorted(names))


def _raw_window(function: KasadaFunction) -> Callable[[int], tuple[str, ...]]:
    def window(index: int) -> tuple[str, ...]:
        start = max(0, index - 2)
        end = min(len(function.instructions), index + 3)
        return tuple(
            f"{ins.offset}: {ins.raw}" for ins in function.instructions[start:end]
        )

    return window


def _branch_condition(instruction: KasadaInstruction, stack: tuple[Expr, ...]) -> Expr | None:
    """Target-taken condition for an immediate-target branch.

    ``JUMP_IF_TRUE``  takes the target when its operand value is truthy.
    ``JUMP_IF_FALSE`` takes the target when its operand value is falsy.
    Both consume no operand-stack values: the condition is an operand cell.
    """

    values = instruction.value_operands()
    if not values:
        return None
    condition_value = values[0]
    source = SourceRef(frontend=FRONTEND_ID, offset=instruction.offset)
    if condition_value.kind == "register":
        condition: Expr = Var(name=register_name(condition_value.register), source=source)
    else:
        condition = _const(condition_value.value, source)
    zero = _const(0, source)
    if instruction.opcode == 20:
        return BinaryOp(source=source, op="!=", left=condition, right=zero)
    return BinaryOp(source=source, op="==", left=condition, right=zero)


def _branch_stack_width(instruction: KasadaInstruction) -> int:
    return 0


def _make_stateful_callbacks(
    function: KasadaFunction, program: KasadaProgram
) -> VMStatefulCallbacks:
    instructions = function.instructions
    by_offset = {ins.offset: ins for ins in instructions}

    # Every name the function reads must be bound before the first read:
    # a frame cell or scope slot that has not been written yet holds
    # ``undefined`` in the VM, but it is still a declared variable.  Without
    # this seed, core's value-invariant pass reports the read as an unbound
    # local and the function degrades to unsupported.
    seeded: dict[str, Expr] = {}

    def initial_locals() -> dict[str, Expr]:
        return dict(seeded)

    def lift_linear(start: int, end: int, locals_, stack) -> VMLinearState:
        steps = tuple(make_step(ins, program) for ins in instructions[start:end])
        result = lift_steps(steps, initial_locals=locals_, initial_stack=stack)
        stopped_at = None
        if result.stopped_at is not None:
            try:
                stopped_at = start + steps.index(result.stopped_at)
            except ValueError:  # pragma: no cover - defensive
                stopped_at = None
        return VMLinearState(
            locals=result.state.locals,
            stack=tuple(result.state.stack),
            statements=tuple(result.state.statements),
            terminator=result.state.terminator,
            stopped_at=stopped_at,
        )

    def branch_condition(branch: VMBytecodeStep, stack: tuple[Expr, ...]) -> Expr | None:
        offset = branch.source.offset
        instruction = by_offset.get(offset)
        if instruction is None:
            return None
        return _branch_condition(instruction, stack)

    def branch_stack_width(branch: VMBytecodeStep) -> int:
        offset = branch.source.offset
        instruction = by_offset.get(offset)
        if instruction is None:
            return 0
        return _branch_stack_width(instruction)

    return VMStatefulCallbacks(
        initial_locals=initial_locals,
        lift_linear=lift_linear,
        branch_condition=branch_condition,
        branch_stack_width=branch_stack_width,
    )


def lift_function(function: KasadaFunction, program: KasadaProgram):
    steps = make_steps(function, program)
    profile = build_hint_region_profile(
        steps,
        frontend=FRONTEND_ID,
        opcode_classes=REGION_CLASSES,
        raw_window=_raw_window(function),
    )
    spec = VMFunctionSpec(
        name=function.name,
        params=(),
        frontend=FRONTEND_ID,
        instruction_count=len(steps),
        local_names=local_names_of(steps),
        metadata={
            "function_offset": function.entry,
            "function_origin": function.origin,
        },
    )
    return recover_vm_function(
        spec,
        lambda: lift_vm_step_function(
            spec,
            steps,
            profile=profile,
            stateful_callbacks=_make_stateful_callbacks(function, program),
            raw_window=_raw_window(function),
        ),
        raw=tuple(ins.raw for ins in function.instructions),
    )


def lift_module(program: KasadaProgram):
    return assemble_vm_module(
        name=program.filename or f"<{FRONTEND_ID}-program>",
        source_language=FRONTEND_ID,
        metadata={
            # Core renders this as provenance ("// input: <format> <version>"),
            # so it is a description mapping, not the frontend id.
            "frontend": {
                "id": FRONTEND_ID,
                "format": "kasadavm-wordstream",
                "version": "wordstream-86",
                "entry": program.entry,
                "word_count": len(program.words),
                "string_table_present": program.string_table is not None,
            },
            "bytecode_format": FRONTEND_ID,
        },
        functions=tuple(lift_function(function, program) for function in program.functions),
    )
