from __future__ import annotations

from unidecompiler.core.effects import (
    AssignValue, Binary, BuildArray, BuildCall, CallStackArgs, Emit, Invoke, LoadItem,
    LoadLocal, Pop, Push, RaiseTop, ReturnTop, ReturnVoid, StoreItemEffect, StoreLocal,
    Unary,
)
from unidecompiler.core.ir import (
    ArrayLiteral, BinaryOp, Call, Const, ExprStmt, GetAttr, GetItem, Global, IndirectCall, NewObject, ObjectLiteral, SourceRef, StoreItem, UndefinedLiteral,
    UnaryOp, Var,
)
from dataclasses import replace
from dataclasses import fields, is_dataclass
from collections import defaultdict, deque

from unidecompiler.core.vm_bytecode import VMBytecodeStep, run_vm_steps
from unidecompiler.core.vm_function import VMFunctionSpec, lift_vm_step_function
from unidecompiler.core.vm_hints import VMHint
from unidecompiler.core.vm_module import assemble_vm_module
from unidecompiler.core.vm_operands import VMDecodedInstruction, VMOperand
from unidecompiler.core.vm_region import (
    VMLinearState,
    VMRegionOpcodeClasses,
    VMRegionCallbacks,
    VMRegionProfile,
    VMStatefulCallbacks,
)
from unidecompiler.progress import ProgressReporter, report_progress

from .decoder import CHAR_OPERANDS, OPCODE_NAMES, _register_operand_indices
from .model import Function, Instruction, Program, TableMutation

FRONTEND_ID = "qimeivm"


def REG(index: int) -> str:
    return f"r{index}" if index >= 0 else f"r_neg_{abs(index)}"


def _register_index(name: str) -> int | None:
    if name.startswith("r_neg_") and name[6:].isdigit():
        return -int(name[6:])
    if name.startswith("r") and name[1:].isdigit():
        return int(name[1:])
    return None


def _load(index: int, source: SourceRef):
    value = Var(source=source, name=REG(index))
    return Push(source=source, value=value)


def _store(index: int, source: SourceRef):
    return StoreLocal(source=source, name=REG(index))


def _value(index: int, source: SourceRef):
    # VM registers are mutable frame slots.  They must be represented as
    # locals, not Global expressions: the latter cannot be assignment targets
    # in generic IR and makes the independent simulator stop at instruction 1.
    return Var(name=REG(index), source=source)


def _call_receiver(source: SourceRef):
    """The VM's private ``S`` receiver, distinct from slot 3 (``this``)."""
    return Global(name="call_receiver", source=source)


def _constant(value, source: SourceRef):
    return Const(source=source, value=value)


def _assign(index: int, value, source: SourceRef):
    return AssignValue(source=source, name=REG(index), target=_value(index, source), value=value)


def _item(obj: int, key: int, source: SourceRef):
    return GetItem(source=source, obj=_value(obj, source), key=_value(key, source))


def _item_imm(obj: int, key, source: SourceRef):
    return GetItem(source=source, obj=_value(obj, source), key=_constant(key, source))


def _call(callee, args, source: SourceRef):
    return Call(source=source, callee=callee, args=tuple(args))


def _method(obj: int, name: str, source: SourceRef):
    return GetAttr(source=source, obj=_value(obj, source), attr=name)


def _binary_expr(left, op: str, right, source: SourceRef):
    return BinaryOp(source=source, op=op, left=left, right=right, semantics="dynamic")


def _string_append(index: int, char, source: SourceRef):
    return _assign(index, _binary_expr(_value(index, source), "+", _constant(chr(char & 0xFFFF), source), source), source)


def _string_append_many(index: int, chars: tuple[int, ...], source: SourceRef):
    """Fold one VM multi-character append without changing its value flow."""
    text = "".join(chr(char & 0xFFFF) for char in chars)
    return _assign(index, _binary_expr(_value(index, source), "+", _constant(text, source), source), source)


def _store_item(obj: int, key, value: int, source: SourceRef):
    key_expr = _constant(key, source) if isinstance(key, int) else _value(key, source)
    return Emit(source=source, statement=StoreItem(
        source=source, obj=_value(obj, source), key=key_expr, value=_value(value, source)
    ))


def _store_item_dynamic(obj: int, key: int, value: int, source: SourceRef):
    return Emit(source=source, statement=StoreItem(
        source=source, obj=_value(obj, source), key=_value(key, source), value=_value(value, source)
    ))


def _store_static_property(obj: int, key: str, value, source: SourceRef):
    return Emit(source=source, statement=StoreItem(
        source=source,
        obj=_value(obj, source),
        key=_constant(key, source),
        value=_constant(value, source),
    ))


def _closure_value(ins: Instruction, source: SourceRef):
    o = ins.operands
    if ins.opcode == 43:
        count, base = max(o[0], 0), 1
        # The delta cell is consumed by ``F + a[++F]``.  The returned
        # function starts at the cell after that base, i.e. at the next
        # opcode cell.
        target = ins.offset + count + 4 + o[count + 2] - 1
    elif ins.opcode == 15:
        count, base = max(o[3], 0), 4
        target = ins.offset + len(o) + o[-2] - 1
    elif ins.opcode == 23:
        count, base = max(o[0], 0), 1
        target = ins.offset + len(o) + o[-2] - 1
    else:
        return None, None
    args = tuple(_value(index, source) for index in o[base:base + count])
    destination = o[base + count]
    if not isinstance(destination, int):
        return None, None
    # Keep the VM activation environment as data.  The generic simulator can
    # pass these five values back into the closure's r0/r1/r2/r5/r6 slots
    # when an IndirectCall resolves the target FunctionIR.
    captures = (
        _value(0, source), _value(1, source),
        # ``o(..., C, E, G, I)`` stores C as one frame slot (Y[2]).  C is
        # the argument array assembled by the closure opcode, not the
        # individual captured values.  Keeping it as an ArrayLiteral is
        # essential: every nested VM function expects ``r2`` to be indexed
        # (for example ``r2[0]``), and flattening it changes the program.
        ArrayLiteral(source=source, items=args),
        _value(5, source), _value(6, source),
    )
    return NewObject(
        source=source,
        type_name=f"closure_{target}",
        args=tuple(captures),
    ), destination


def _closure_capture(obj: int, index: int, source: SourceRef):
    captures = GetItem(source=source, obj=_value(obj, source), key=_constant("captures", source))
    return GetItem(source=source, obj=captures, key=_constant(index, source))


def _invoke_closure(obj: int, receiver, args, source: SourceRef):
    # Closure functions use the VM frame ABI:
    # r0,r1,r2,r3,r4,r5,r6,r7 = E,G,g,this,arguments,e,a,0.
    target = IndirectCall(
        source=source,
        selector=_value(obj, source),
        signature="closure",
    )
    return _call(target, (
        _closure_capture(obj, 0, source),
        _closure_capture(obj, 1, source),
        _closure_capture(obj, 2, source),
        receiver,
        ArrayLiteral(source=source, items=tuple(args)),
        _closure_capture(obj, 3, source),
        _closure_capture(obj, 4, source),
        _constant(0, source),
    ), source)


def _binary(ins: Instruction, source: SourceRef, op: str):
    d, left, right = ins.operands[:3]
    return (_load(left, source), _load(right, source), Binary(source=source, op=op), _store(d, source))


def _effects_impl(ins: Instruction, source: SourceRef):
    o = ins.operands
    op = ins.opcode
    if op == 47:
        return (Push(source=source, value=Const(source=source, value=None)), _store(o[0], source))
    if op == 70:
        return (Push(source=source, value=Const(source=source, value=o[1])), _store(o[0], source))
    if op == 72:
        return (_assign(o[0], ObjectLiteral(source=source), source),)
    if op == 76:
        return (_assign(o[0], _call_receiver(source), source),)
    if op == 79:
        return (Push(source=source, value=Const(source=source, value="")), _store(o[0], source))
    if op == 48:
        return (_load(o[1], source), _store(o[0], source))
    if op in {10, 16, 21, 37, 57, 62, 65, 67, 82, 83, 88}:
        return _binary(ins, source, {10: "^", 16: "|", 21: "-", 37: "%", 57: "/", 62: "*", 65: "+", 67: "<=", 82: "<", 83: ">", 88: ">="}[op])
    if op in {18, 59, 64, 90}:
        return (_load(o[1], source), Unary(source=source, op={18: "!", 59: "-", 64: "typeof", 90: "~"}[op]), _store(o[0], source))
    if op == 0:
        return (
            _assign(
                o[0],
                _call(
                    Global(name="delete_item", source=source),
                    (_value(o[1], source), _value(o[2], source)),
                    source,
                ),
                source,
            ),
        )
    if op == 1:
        return (_assign(o[0], _call(_method(o[1], "call", source), (_call_receiver(source), _value(o[2], source)), source), source),)
    if op == 2:
        return (
            _assign(o[0], _item_imm(o[1], o[2], source), source),
            _assign(o[3], _call(Global(name="Array", source=source), (_constant(o[4], source),), source), source),
            _store_item(o[5], o[6], o[7], source),
        )
    if op == 3:
        count = max(o[0], 0)
        args = tuple(_value(index, source) for index in o[1:1 + count])
        return (_assign(
            o[1 + count],
            _call(
                _method(o[2 + count], "apply", source),
                (_call_receiver(source), ArrayLiteral(source=source, items=args)),
                source,
            ),
            source,
        ),)
    if op == 4:
        return (
            _assign(o[0], _item(o[1], o[2], source), source),
            _assign(o[3], _constant("", source), source),
            _string_append(o[4], o[5], source),
        )
    if op in {5, 6}:
        count = 2 if op == 5 else 3
        if op == 6:
            args = tuple(_value(x, source) for x in o[2:5])
            return (_assign(o[0], _call(_method(o[1], "call", source), (_call_receiver(source), *args), source), source),)
        return (_assign(o[0], _call(_method(o[1], "call", source), tuple([_value(o[2], source), *(_value(x, source) for x in o[3:5])]), source), source),)
    if op == 7:
        return (_string_append(o[0], o[1], source), _assign(o[2], _item_imm(o[3], o[4], source), source))
    if op == 8:
        return (_assign(o[0], _binary_expr(_value(o[1], source), "&", _constant(o[2], source), source), source),)
    if op == 9:
        return (_assign(o[0], _item(o[1], o[2], source), source), _assign(o[3], _value(o[4], source), source))
    if op == 11:
        return (_assign(o[0], _binary_expr(_value(o[1], source), "instanceof", _value(o[2], source), source), source),)
    if op in {12, 44, 52}:
        effect = _assign(o[0], _binary_expr(_value(o[1], source), "==", _value(o[2], source), source), source)
        return (effect, Push(source=source, value=_value(o[3], source))) if op == 12 else (effect,)
    if op == 17:
        return (
            _assign(o[0], _binary_expr(_value(o[1], source), "==", _value(o[2], source), source), source),
            _assign(o[3], UnaryOp(source=source, op="!", value=_value(o[4], source)), source),
            Push(source=source, value=_value(o[5], source)),
        )
    if op == 20:
        return (
            _assign(o[0], _value(o[1], source), source),
            _assign(o[2], _binary_expr(_value(o[3], source), "==", _value(o[4], source), source), source),
            Push(source=source, value=_value(o[5], source)),
        )
    if op in {13, 25, 41, 56, 58, 68, 69, 71, 73, 74, 81, 95, 96}:
        names = {13: "<=", 25: ">=", 41: "===", 56: "+", 58: "<<", 68: "to_number", 69: ">", 71: ">>", 73: "-", 74: "+", 81: "|", 95: ">>>", 96: "<"}
        if op == 68:
            expr = _call(Global(name="to_number_preserving_bigint", source=source), (_value(o[1], source),), source)
        else:
            left = _constant(o[1], source) if op == 56 else _value(o[1], source)
            right = _value(o[2], source) if op in {56, 58, 68, 71, 73, 74, 81, 95} else _constant(o[2], source)
            if op in {13, 25, 41, 58, 69, 71, 73, 74, 81, 95, 96}:
                right = _constant(o[2], source)
            expr = _binary_expr(left, names[op], right, source)
        return (_assign(o[0], expr, source),)
    if op in {14, 54, 77, 85, 89}:
        if op == 14:
            return (_string_append(o[0], o[1], source),)
        if op == 77:
            if o[0] == o[2] == o[4]:
                return (_string_append_many(o[0], (o[1], o[3], o[5]), source),)
            return (_string_append(o[0], o[1], source), _string_append(o[2], o[3], source), _string_append(o[4], o[5], source))
        if op == 89:
            if o[0] == o[2]:
                return (_string_append_many(o[0], (o[1], o[3]), source),)
            return (_string_append(o[0], o[1], source), _string_append(o[2], o[3], source))
        if op == 85:
            return (_assign(o[0], _constant("", source), source), _string_append(o[1], o[2], source))
        return (_string_append(o[0], o[1], source), _assign(o[2], _item(o[3], o[4], source), source))
    if op == 19:
        return (_assign(o[0], _item(o[1], o[2], source), source),)
    if op == 45:
        # vm.js: Y[a[++F]] = Y[a[++F]][a[++F]]
        # The object is a register slot; the final a[++F] is the immediate
        # property key itself.
        return (_assign(o[0], _item_imm(o[1], o[2], source), source),)
    if op in {24, 34}:
        if op == 34:
            return (_assign(o[0], _call(Global(name="Array", source=source), (_constant(o[1], source),), source), source),)
        return (_assign(o[0], _call(Global(name="Array", source=source), (_constant(o[1], source),), source), source), _assign(o[2], _call(Global(name="Array", source=source), (_constant(o[3], source),), source), source))
    if op == 26:
        return (_assign(o[0], _value(o[1], source), source), _store_item_dynamic(o[2], o[3], o[4], source))
    if op == 27:
        return (_assign(o[0], _value(o[1], source), source), _store_item(o[2], o[3], o[4], source))
    if op == 28:
        return (_string_append(o[0], o[1], source), _store_item(o[2], o[3], o[4], source))
    if op == 30:
        return (_assign(o[0], _item_imm(o[1], o[2], source), source), _assign(o[3], _item_imm(o[4], o[5], source), source), _assign(o[6], _item_imm(o[7], o[8], source), source))
    if op == 31:
        return (_assign(o[0], _binary_expr(_value(o[1], source), "in", _value(o[2], source), source), source),)
    if op == 32:
        return (_store_item(o[0], o[1], o[2], source), _assign(o[3], _item_imm(o[4], o[5], source), source))
    if op == 33:
        return (_assign(o[0], _item_imm(o[1], o[2], source), source), _assign(o[3], _constant("", source), source))
    if op in {35, 61}:
        if op == 35:
            return (
                _assign(o[0], _item_imm(o[1], o[2], source), source),
                _assign(o[3], _value(o[4], source), source),
            )
        return (_assign(o[0], _value(o[1], source), source), _assign(o[2], _item_imm(o[3], o[4], source), source))
    if op == 60:
        return (
            _assign(o[0], _item_imm(o[1], o[2], source), source),
            _assign(o[3], _item_imm(o[4], o[5], source), source),
        )
    if op == 36:
        return (_assign(o[0], NewObject(source=source, constructor=_value(o[1], source)), source),)
    if op == 38:
        return (_store_item_dynamic(o[0], o[1], o[2], source),)
    if op == 39:
        return (_assign(o[0], NewObject(source=source, constructor=_value(o[1], source), args=(_value(o[2], source), _value(o[3], source))), source),)
    if op in {42, 91}:
        receiver = _value(o[2], source) if op == 42 and len(o) > 2 else _call_receiver(source)
        return (_assign(o[0], _call(_method(o[1], "call", source), (receiver,), source), source),)
    if op == 46:
        return (_assign(o[1], _call(Global(name="keys", source=source), (_value(o[0], source),), source), source),)
    if op == 49:
        count = max(o[0], 0)
        args = tuple(_value(index, source) for index in o[1:1 + count])
        return (_assign(
            o[1 + count],
            _call(
                _method(o[2 + count], "apply", source),
                (_value(o[3 + count], source), ArrayLiteral(source=source, items=args)),
                source,
            ),
            source,
        ),)
    if op in {50, 55, 86}:
        count = {50: 2, 55: 3, 86: 1}[op]
        receiver = _call_receiver(source) if op == 50 else _value(o[2], source)
        args = tuple(_value(x, source) for x in (o[2:2 + count] if op == 50 else o[3:3 + count]))
        return (_assign(o[0], _call(_method(o[1], "call", source), (receiver, *args), source), source),)
    if op == 66:
        return (_assign(o[0], _call_receiver(source), source), _load(o[1], source), ReturnTop(source=source, empty_is_void=True))
    if op == 75:
        length = _item_imm(o[0], "length", source)
        truthy_length = UnaryOp(source=source, op="!", value=UnaryOp(source=source, op="!", value=length))
        return (
            _assign(o[1], truthy_length, source),
            _assign(
                o[2],
                _call(
                    Global(name="shift_if_nonempty", source=source),
                    (_value(o[0], source), _value(o[2], source)),
                    source,
                ),
                source,
            ),
        )
    if op == 78:
        return (_assign(o[0], _value(o[1], source), source), _assign(o[2], _value(o[3], source), source))
    if op == 84:
        return (
            _assign(o[0], _item(o[1], o[2], source), source),
            _assign(o[3], _call(_method(o[4], "call", source), (_value(o[5], source), _value(o[6], source)), source), source),
        )
    if op == 87:
        increment = _binary_expr(_value(o[1], source), "+", _constant(1, source), source)
        if o[0] == o[1]:
            return (_assign(o[1], increment, source),)
        return (
            _assign(o[1], increment, source),
            _assign(o[0], _value(o[1], source), source),
        )
    if op == 92:
        return (_store_item(o[0], o[1], o[2], source),)
    if op == 94:
        return (_assign(o[0], NewObject(source=source, constructor=_value(o[1], source), args=(_value(o[2], source),)), source),)
    if op == 93:
        count = max(o[0], 0)
        args = tuple(_value(index, source) for index in o[1:1 + count])
        destination = o[1 + count]
        constructor = o[2 + count]
        return (_assign(destination, NewObject(source=source, constructor=_value(constructor, source), args=args), source),)
    if op == 97:
        return (_assign(o[0], Global(name="current_exception", source=source), source),)
    if op in {15, 23, 43}:
        value, destination = _closure_value(ins, source)
        if value is None:
            return ()
        effects = [_assign(destination, value, source)]
        # All three closure forms define the function's observable length
        # property immediately after construction.  The neutral IR cannot
        # encode the descriptor flags, but retaining the value is important
        # for later property reads and is preferable to silently dropping it.
        length = o[max(o[0], 0) + 3] if op == 43 else o[-1]
        effects.append(_store_static_property(destination, "length", length, source))
        if op == 15:
            effects.insert(0, _store_item(o[0], o[1], o[2], source))
        elif op == 43:
            effects.append(_store_item(o[-3], o[-2], o[-1], source))
        return tuple(effects)
    if op == 51:
        return (Push(source=source, value=_value(o[0], source)),)
    if op == 53:
        return (
            _assign(o[0], Global(name="current_exception", source=source), source),
            _assign(o[1], _value(o[2], source), source),
        )
    if op == 29:
        return (_load(o[0], source), RaiseTop(source=source))
    if op == 80:
        return (_load(o[0], source), ReturnTop(source=source, empty_is_void=True))
    if op == 22:
        # This pops the VM's private exception-handler stack `i`, not the
        # generic expression/value stack.
        return ()
    # Values outside 0..97 are branch/data cells, not executable opcodes.
    if op not in OPCODE_NAMES:
        return ()
    # Every executable opcode is handled above.  Keep malformed/data cells
    # inert; they are not VM instructions and must not be rendered as fake
    # expressions.
    return ()


def _effects(ins: Instruction, source: SourceRef):
    """Lift one opcode without allowing damaged data cells to abort a file.

    The decoder preserves truncated cells as diagnostics.  Such a cell is not
    executable bytecode, so it contributes no effect; the core can still
    recover the surrounding function and source reference.
    """
    try:
        return _effects_impl(ins, source)
    except (IndexError, ValueError):
        return ()


def _operands(ins: Instruction) -> tuple[VMOperand, ...]:
    target_positions = {
        12: (4, 5), 17: (6, 7), 20: (6, 7), 40: (0,),
        51: (1, 2), 53: (3,), 63: (0,),
    }.get(ins.opcode, ())
    if ins.opcode == 15 and len(ins.operands) >= 4:
        target_positions = (max(ins.operands[3], 0) + 5,)
    elif ins.opcode in {23, 43} and ins.operands:
        target_positions = (max(ins.operands[0], 0) + 2,)
    register_positions = frozenset(_register_operand_indices(ins.opcode, ins.operands))
    char_positions = frozenset(CHAR_OPERANDS.get(ins.opcode, ()))
    operands = []
    for index, value in enumerate(ins.operands):
        if index in target_positions:
            role = "target"
        elif index in register_positions:
            role = "register"
        elif index in char_positions:
            role = "constant"
        else:
            role = "immediate"
        operands.append(VMOperand(role=role, value=value, text=str(value)))
    return tuple(operands)


def _target(ins: Instruction, operand: int) -> int:
    # ``F += a[++F]`` captures the old F for the compound assignment before
    # evaluating the increment on its RHS.  The next switch increments once.
    return ins.offset + 1 + operand


def make_step(
    ins: Instruction,
    valid_offsets: frozenset[int] | None = None,
    loop_back_edges: frozenset[tuple[int, int]] = frozenset(),
) -> VMBytecodeStep:
    source = SourceRef(frontend=FRONTEND_ID, offset=ins.offset)
    hints: list[VMHint] = []
    if ins.opcode == 63 and ins.operands:
        target = _target(ins, ins.operands[0])
        kind = "loop-backedge" if (ins.offset, target) in loop_back_edges else "branch-target"
        hints.append(VMHint(kind=kind, source=source, target=target, label="JUMP", flow="unconditional"))
    elif ins.opcode in {12, 17, 20, 51} and ins.operands:
        indices = {
            12: (4, 5),
            17: (6, 7),
            20: (6, 7),
            51: (1, 2),
        }[ins.opcode]
        bases = {12: (4, 4), 17: (6, 6), 20: (6, 6), 51: (1, 1)}
        for position, (index, label) in enumerate(zip(indices, ("target-if-true", "target-if-false"), strict=True)):
            if len(ins.operands) > index:
                target = ins.offset + bases[ins.opcode][position] + ins.operands[index]
                if valid_offsets is not None and target not in valid_offsets:
                    continue
                kind = "loop-backedge" if (ins.offset, target) in loop_back_edges else "branch-target"
                hints.append(VMHint(kind=kind, source=source, target=target, label=label, detail=label, flow="conditional"))
        hints.append(VMHint(kind="materialized-condition", source=source, detail="register", flow="conditional"))
    if ins.opcode in {40, 53} and ins.operands:
        # The VM pushes a handler PC onto its private exception stack.  This
        # is a protected-region fact, not an ordinary branch: a throw can
        # transfer to the handler while normal execution continues through
        # the protected instructions.
        target = (
            ins.offset + 1 + ins.operands[0]
            if ins.opcode == 40
            else ins.offset + ins.size - 1 + ins.operands[-1]
        )
        if valid_offsets is None or target in valid_offsets:
            hints.append(
                VMHint(
                    kind="exception-handler",
                    source=source,
                    target=target,
                    value={"operation": "push"},
                    flow="conditional",
                )
            )
    elif ins.opcode == 22:
        hints.append(
            VMHint(
                kind="exception-handler-pop",
                source=source,
                value={"operation": "pop"},
            )
        )
    decoded = VMDecodedInstruction(opcode=OPCODE_NAMES.get(ins.opcode, "NOOP"), source=source, operands=_operands(ins), raw=ins.raw)
    return VMBytecodeStep(
        opcode=decoded.opcode,
        source=source,
        decoded=decoded,
        raw=ins.raw,
        effects=_effects(ins, source),
        hints=tuple(hints),
    )


def _profile(steps):
    jump_opcodes = frozenset(OPCODE_NAMES[i] for i in (12, 17, 20, 51, 63))
    conditional_opcodes = frozenset(OPCODE_NAMES[i] for i in (12, 17, 20, 51))

    def targets(step):
        return tuple(dict.fromkeys(
            hint.target
            for hint in step.hints
            if hint.kind in {"branch-target", "loop-backedge"}
            and hint.target is not None
        ))

    def has_kind(step, kind):
        return any(hint.kind == kind for hint in step.hints)

    return VMRegionProfile(
        frontend=FRONTEND_ID,
        is_noise=lambda _step: False,
        is_control=lambda step: step.opcode in jump_opcodes or bool(targets(step)),
        is_jump=lambda step: step.opcode in jump_opcodes,
        # Packed offsets have no source-order meaning.  These classes come
        # from the decoder's dominance proof encoded in neutral hint kinds.
        is_forward_jump=lambda step: has_kind(step, "branch-target"),
        is_backward_jump=lambda step: has_kind(step, "loop-backedge"),
        is_iter_start=lambda _step: False,
        is_async_iter_start=lambda _step: False,
        is_conditional_jump=lambda step: step.opcode in conditional_opcodes,
        is_cleanup=lambda _step: False,
        is_null_jump=lambda _step: False,
        is_not_null_jump=lambda _step: False,
        is_truthy_jump=lambda step: step.opcode == OPCODE_NAMES[51],
        target_offset=lambda step: next(iter(targets(step)), None),
        target_offsets=targets,
        offset=lambda step: step.source.offset,
        raw_window=lambda i: tuple(s.raw for s in steps[max(0, i - 2):i + 3]),
    )


def _stateful_callbacks(function: Function, prepared_steps: tuple[VMBytecodeStep, ...]):
    def lift_linear(start, end, locals_, stack):
        # Reuse the exact thin-IR stream submitted to core.  Re-decoding the
        # raw instructions here used to discard entry initializers and the
        # implicit fall-through return added by ``lift_program``.  Functions
        # containing a jump were consequently lifted with different effects
        # in the low-level CFG pass and then hidden by a linear fallback.
        slice_steps = prepared_steps[start:end]
        result = run_vm_steps(slice_steps, initial_locals=locals_, initial_stack=stack)
        stopped = None
        if result.stopped_at is not None:
            stopped = start + slice_steps.index(result.stopped_at)
        return VMLinearState(
            locals=result.state.locals,
            stack=tuple(result.state.stack),
            statements=tuple(result.state.statements),
            terminator=result.state.terminator,
            stopped_at=stopped,
        )

    def branch_condition(branch, _stack):
        operands = () if branch.decoded is None else tuple(item.value for item in branch.decoded.operands)
        if branch.opcode in {"EQ_BRANCH", "NE_BRANCH"} and len(operands) >= 4:
            index = 3 if branch.opcode == "EQ_BRANCH" else 5
        elif branch.opcode == "COPY_EQ_BRANCH" and len(operands) >= 6:
            index = 5
        elif branch.opcode == "JUMP_IF" and operands:
            index = 0
        else:
            return None
        return Global(name=REG(operands[index]), source=branch.source)

    return VMStatefulCallbacks(
        # VM registers are addressable slots, not lexical locals.  Seeding the
        # generic state with neutral register identities prevents the core from
        # mistaking a read of an input/closure register for an IR failure.
        initial_locals=lambda: {
            REG(index): Global(name=REG(index), source=SourceRef(frontend=FRONTEND_ID))
            for index in range(1024)
        },
        lift_linear=lift_linear,
        branch_condition=branch_condition,
        branch_stack_width=lambda instruction: 1 if instruction.opcode in {"EQ_BRANCH", "NE_BRANCH", "COPY_EQ_BRANCH", "JUMP_IF"} else 0,
    )


def _region_callbacks(function: Function):
    """Provide the core's structured walker with VM-neutral slice facts."""
    def lift_slice(start, end, stack):
        steps = tuple(make_step(ins) for ins in function.instructions[start:end])
        result = run_vm_steps(steps, initial_locals={}, initial_stack=stack)
        return tuple(result.state.statements)

    def lift_expr(start, end, _stack):
        if end >= len(function.instructions):
            return None
        ins = function.instructions[end]
        o = ins.operands
        if ins.opcode in {12, 17, 20}:
            index = {12: 3, 17: 5, 20: 5}[ins.opcode]
        elif ins.opcode == 51:
            index = 0
        else:
            return None
        return _value(o[index], SourceRef(frontend=FRONTEND_ID, offset=ins.offset)) if len(o) > index else None

    return VMRegionCallbacks(
        lift_slice=lift_slice,
        lift_expr=lift_expr,
        lift_iter_loop=lambda _condition, _value: None,
        lift_async_iter_loop=lambda _start, _end, _limit: None,
        lift_comprehension=lambda _start, _control, _end, _value: None,
    )


def _used_registers(value) -> set[str]:
    """Collect VM local references from neutral effects/IR values."""
    found: set[str] = set()
    if isinstance(value, Var):
        found.add(value.name)
        return found
    if is_dataclass(value):
        for field in fields(value):
            found.update(_used_registers(getattr(value, field.name)))
    elif isinstance(value, (tuple, list)):
        for item in value:
            found.update(_used_registers(item))
    elif isinstance(value, dict):
        for item in value.values():
            found.update(_used_registers(item))
    return found


def _contains_ir_type(value, expected_type: type) -> bool:
    """Return whether a neutral effect contains an IR node of one type."""
    if isinstance(value, expected_type):
        return True
    if is_dataclass(value):
        return any(
            _contains_ir_type(getattr(value, field.name), expected_type)
            for field in fields(value)
        )
    if isinstance(value, (tuple, list)):
        return any(_contains_ir_type(item, expected_type) for item in value)
    if isinstance(value, dict):
        return any(_contains_ir_type(item, expected_type) for item in value.values())
    return False


def _normal_successor_offsets(instruction: Instruction) -> tuple[int, ...]:
    """Return raw normal-flow targets without treating handlers as branches."""
    start = instruction.offset
    operands = instruction.operands
    opcode = instruction.opcode
    arity = instruction.size - 1
    if opcode == 63 and operands:
        return (start + 1 + operands[0],)
    if opcode == 51 and len(operands) >= 3:
        return (start + 1 + operands[1], start + 1 + operands[2])
    if opcode == 12 and len(operands) >= 6:
        return (start + 4 + operands[4], start + 4 + operands[5])
    if opcode in {17, 20} and len(operands) >= 8:
        return (start + 6 + operands[6], start + 6 + operands[7])
    if opcode == 75:
        return (start + arity + 1,)
    if opcode in {29, 66, 80}:
        return ()
    return (start + arity + 1,)


def _handler_target(instruction: Instruction) -> int | None:
    if not instruction.operands:
        return None
    if instruction.opcode == 40:
        return instruction.offset + 1 + instruction.operands[0]
    if instruction.opcode == 53:
        return instruction.offset + instruction.size - 1 + instruction.operands[-1]
    return None


def _incoming_handler_states(
    function: Function,
    steps: tuple[VMBytecodeStep, ...],
) -> dict[int, frozenset[tuple[int, ...] | None]]:
    """Compute neutral handler-stack facts at instruction boundaries.

    This is a bounded bytecode fact analysis used only to decide whether a
    handler pop or exceptional-edge state can be stated exactly.  It does not
    recover blocks or source structures; core still owns the resulting CFG.
    """
    instructions = {instruction.offset: instruction for instruction in function.instructions}
    steps_by_offset = {step.source.offset: step for step in steps}
    if function.offset not in instructions:
        return {}
    max_depth = max(
        1,
        sum(instruction.opcode in {40, 53} for instruction in function.instructions),
    )
    # ``None`` is an explicitly unknown/non-converging handler stack.  Never
    # saturate an over-deep state to the previous tuple: that would turn an
    # unproved loop state into a false exact exception fact.
    HandlerState = tuple[int, ...] | None
    incoming: dict[int, set[HandlerState]] = defaultdict(set)
    incoming[function.offset].add(())
    pending: deque[tuple[int, HandlerState]] = deque(((function.offset, ()),))
    visited: set[tuple[int, HandlerState]] = set()
    while pending:
        offset, state = pending.popleft()
        key = (offset, state)
        if key in visited:
            continue
        visited.add(key)
        instruction = instructions[offset]
        outgoing = state
        target = _handler_target(instruction)
        if state is None:
            outgoing = None
        elif target is not None:
            outgoing = (*state, target) if len(state) < max_depth else None
        elif instruction.opcode == 22 and state:
            outgoing = state[:-1]
        successors = list(_normal_successor_offsets(instruction))
        # Core's low-level VM pass preserves the decoder's packed instruction
        # order when a handler clone continues through a non-control step.
        # Mirror that neutral fall-through fact here so shared instructions
        # after an exception entry retain the active handler context even
        # when their public offsets are not numerically adjacent.
        if (
            state
            and instruction.opcode not in {12, 17, 20, 51, 63, 80}
            and offset in instructions
        ):
            position = next(
                index
                for index, item in enumerate(function.instructions)
                if item.offset == offset
            )
            if position + 1 < len(function.instructions):
                successors.append(function.instructions[position + 1].offset)
        for successor in tuple(dict.fromkeys(successors)):
            if successor not in instructions or outgoing in incoming[successor]:
                continue
            incoming[successor].add(outgoing)
            pending.append((successor, outgoing))
        # The interpreter transfers to the innermost handler after popping it
        # from its private stack.  Propagate only instructions whose submitted
        # generic IR can actually raise; core will own and validate the edge.
        step = steps_by_offset.get(offset)
        can_raise = (
            instruction.opcode == 29
            or (step is not None and _contains_ir_type(step.effects, Call))
        )
        if state is not None and state and can_raise:
            handler = state[-1]
            # Core models an exceptional transfer as replacing the innermost
            # protected frame with ACTIVE(handler).  Keep that handler target
            # in this static fact as well: subsequent shared instructions may
            # still execute in the active-handler clone until opcode 53's
            # conditional ACTIVE pop.  The target remains the innermost fact,
            # while outer protected/active contexts are preserved.
            handler_state = (*state[:-1], handler)
            if handler in instructions and handler_state not in incoming[handler]:
                incoming[handler].add(handler_state)
                pending.append((handler, handler_state))
    return {offset: frozenset(states) for offset, states in incoming.items()}


def _prepare_exception_facts(
    function: Function,
    steps: tuple[VMBytecodeStep, ...],
) -> tuple[VMBytecodeStep, ...]:
    """Adapt qimei handler facts to the precise generic exception contract."""
    incoming = _incoming_handler_states(function, steps)
    handler_targets = frozenset(
        target
        for instruction in function.instructions
        if (target := _handler_target(instruction)) is not None
    )
    prepared: list[VMBytecodeStep] = []
    for step in steps:
        states = incoming.get(step.source.offset, frozenset())
        known_handlers = tuple(sorted({
            handler
            for state in states
            if state is not None and state
            for handler in state
        }))
        top_handlers = tuple(sorted({
            state[-1]
            for state in states
            if state is not None and state
        }))
        generated: dict[int, tuple[VMHint, ...]] = {}
        if step.opcode == OPCODE_NAMES.get(22):
            # Opcode 22 pops qimei's private handler stack.  The same packed
            # instruction can be reached by clones with different handler
            # contexts, so submit one conditional fact per proven top frame.
            generated[22] = tuple(
                VMHint(
                    kind="exception-handler-pop",
                    source=step.source,
                    value={
                        "handler": handler,
                        "frame_kind": "protected",
                        "if_present": True,
                    },
                )
                for handler in top_handlers
            )

        # A handler entry starts with an ACTIVE frame on exceptional CFG
        # edges.  qimei opcode 53 replaces that frame with a new protected
        # handler, so pop the matching ACTIVE frame before the original push.
        if (
            step.source.offset in handler_targets
            and step.opcode == OPCODE_NAMES.get(53)
        ):
            entry_pop = VMHint(
                kind="exception-handler-pop",
                source=step.source,
                value={
                    "handler": step.source.offset,
                    "frame_kind": "active",
                    "if_present": True,
                },
            )
            hints_with_entry_pop: list[VMHint] = [entry_pop]
        else:
            hints_with_entry_pop = []

        # Replace the legacy opcode-22 pop in place, preserving hint order;
        # generated contextual facts are neutral VM facts consumed by core.
        hints: list[VMHint] = []
        for hint in step.hints:
            if hint.kind == "exception-handler-pop" and step.opcode == OPCODE_NAMES.get(22):
                hints.extend(generated[22])
            else:
                hints.append(hint)
        if step.source.offset in handler_targets and step.opcode == OPCODE_NAMES.get(53):
            # The handler-entry pop must precede opcode 53's existing
            # exception-handler push hint.
            insert_at = next(
                (
                    index
                    for index, hint in enumerate(hints)
                    if hint.kind == "exception-handler"
                ),
                len(hints),
            )
            hints[insert_at:insert_at] = hints_with_entry_pop

        if _contains_ir_type(step.effects, Call):
            # Submit contextual facts independently for each proven protected
            # clone.  Unprotected clones receive no fact, so core ignores the
            # protected fact rather than treating it as a global assertion.
            hints.extend(
                VMHint(
                    kind="exception-edge-state",
                    source=step.source,
                    value={
                        "stack_depth": 0,
                        "push_exception": False,
                        "handler": handler,
                    },
                )
                for handler in known_handlers
            )
        prepared.append(replace(step, hints=hints))
    return tuple(prepared)


def _lift_table_mutations(mutations: tuple[TableMutation, ...]):
    """Expose the pre-business self-modification as neutral table stores.

    The artifact is a final snapshot, so these writes are provenance facts,
    not instructions that should be replayed before ``main``.  A separate
    function keeps that distinction explicit while retaining each genuine
    writer offset in the generic instruction projection.
    """
    steps: list[VMBytecodeStep] = []
    for index, mutation in enumerate(mutations):
        source = SourceRef(frontend=FRONTEND_ID, offset=mutation.source_offset)
        effect = Emit(
            source=source,
            statement=StoreItem(
                source=source,
                obj=Var(source=source, name="a"),
                key=Const(source=source, value=mutation.target_offset),
                value=Const(source=source, value=mutation.after),
            ),
        )
        effects = (effect,)
        if index == len(mutations) - 1:
            effects += (ReturnVoid(source=source),)
        decoded = VMDecodedInstruction(
            opcode="TABLE_PATCH",
            source=source,
            operands=(
                VMOperand(role="target", value=mutation.target_offset, text=str(mutation.target_offset)),
                VMOperand(role="immediate", value=mutation.before, text=str(mutation.before)),
                VMOperand(role="immediate", value=mutation.after, text=str(mutation.after)),
            ),
            raw=(
                f"{mutation.source_offset:06d}: TABLE_PATCH "
                f"a[{mutation.target_offset}] {mutation.before} -> {mutation.after}"
            ),
        )
        steps.append(VMBytecodeStep(
            opcode="TABLE_PATCH",
            source=source,
            decoded=decoded,
            raw=decoded.raw,
            effects=effects,
        ))
    spec = VMFunctionSpec(
        name="bootstrap_table_writes",
        params=("a",),
        frontend=FRONTEND_ID,
        instruction_count=len(steps),
        local_names=("a",),
        metadata={
            "snapshot_semantics": "captured-before-main; informational-not-replayed",
            "table_mutations": tuple(
                {
                    "source_offset": mutation.source_offset,
                    "target_offset": mutation.target_offset,
                    "before": mutation.before,
                    "after": mutation.after,
                }
                for mutation in mutations
            ),
        },
    )
    return lift_vm_step_function(spec, tuple(steps))


def lift_program(
    program: Program,
    metadata,
    *,
    reporter: ProgressReporter | None = None,
):
    total = len(program.functions)
    report_progress(
        reporter,
        phase="lift",
        status="started",
        completed=0,
        total=total,
        unit="function",
        message="lifting qimeivm functions",
    )
    functions = []
    neutral_registers = {
        REG(index): UndefinedLiteral(source=SourceRef(frontend=FRONTEND_ID))
        for index in range(1024)
    }
    for function in program.functions:
        item_label = function.name or f"function@{function.offset}"
        report_progress(
            reporter,
            phase="lift",
            completed=len(functions),
            total=total,
            unit="function",
            item_label=item_label,
            message=f"lifting {item_label}",
        )
        valid_offsets = frozenset(ins.offset for ins in function.instructions)
        steps = tuple(
            make_step(ins, valid_offsets, function.loop_back_edges)
            for ins in function.instructions
        )
        if steps:
            used = set().union(*(_used_registers(effect) for step in steps for effect in step.effects))
            # r0..r7 are the VM's pre-existing frame values and are exposed as
            # function parameters.  Other referenced slots begin as JS
            # undefined.  This is observably distinct from opcode 47's null
            # under typeof, equality, coercion and property access.
            seed_names = {
                name for name in used
                if (index := _register_index(name)) is not None and index not in range(8)
            }
            seed = tuple(
                _assign(
                    index,
                    UndefinedLiteral(source=SourceRef(frontend=FRONTEND_ID)),
                    SourceRef(frontend=FRONTEND_ID),
                )
                for name in sorted(seed_names, key=lambda item: (_register_index(item) or 0))
                if (index := _register_index(name)) is not None
            )
            if seed:
                steps = (replace(steps[0], effects=seed + tuple(steps[0].effects)),) + steps[1:]
        if steps and function.instructions[-1].opcode not in {29, 66, 80}:
            # Recovered function regions may end at an implicit JavaScript
            # function boundary rather than an explicit VM return.  Model the
            # language-level fall-through return as a core effect.
            last = steps[-1]
            steps = steps[:-1] + (replace(last, effects=tuple(last.effects or ()) + (ReturnVoid(source=last.source),)),)
        steps = _prepare_exception_facts(function, steps)
        # The VM frame exposes these slots before bytecode starts. Keeping the
        # compact ABI explicit preserves the environment and arguments arrays
        # without treating registers as globals. Remaining slots are locals
        # created by assignments.
        spec = VMFunctionSpec(
            name=function.name,
            params=tuple(REG(i) for i in range(8)),
            frontend=FRONTEND_ID,
            instruction_count=len(steps),
            local_names=tuple(dict.fromkeys(
                (*tuple(REG(i) for i in range(1024)), *sorted(used))
            )),
            metadata={"offset": function.offset, "diagnostics": program.diagnostics},
        )
        lifted = lift_vm_step_function(
            spec,
            steps,
            profile=_profile(steps),
            callbacks=_region_callbacks(function),
            stateful_callbacks=_stateful_callbacks(function, steps),
            initial_locals=neutral_registers,
            raw_window=lambda i: tuple(s.raw for s in steps[max(0, i - 2):i + 3]),
        )
        functions.append(lifted)
        report_progress(
            reporter,
            phase="lift",
            completed=len(functions),
            total=total,
            unit="function",
            item_label=item_label,
            message=f"lifted {item_label}",
        )
    if program.table_mutations:
        functions.append(_lift_table_mutations(program.table_mutations))
    report_progress(
        reporter,
        phase="lift",
        status="completed",
        completed=total,
        total=total,
        unit="function",
        message=f"lifted {total} functions",
    )
    return assemble_vm_module(name=program.filename or "<qimeivm>", source_language=FRONTEND_ID, metadata={"frontend": metadata, "bytecode_format": FRONTEND_ID, "diagnostics": program.diagnostics}, functions=tuple(functions))


def lift_module(module, *, reporter: ProgressReporter | None = None):
    """Compatibility entry point used by the template plugin facade."""
    return lift_program(module.payload, module.metadata, reporter=reporter)
