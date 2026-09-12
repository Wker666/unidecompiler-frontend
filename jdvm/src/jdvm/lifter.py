"""Thin-IR lifter for the jdvm frontend.

Every decoded instruction becomes exactly one :class:`VMBytecodeStep` carrying
VM-neutral effects, operands, provenance and control-flow hints. The frontend
does not build blocks, branches, loops, or a control-flow graph: it submits the
step stream to :func:`lift_vm_step_function` and lets core recover structure.

Register model
--------------
jdvm has no named locals: the interpreter keeps its temporaries in JavaScript
closure slots (``_$Gr``, ``_$GP``, ...) and its operand stack in a plain array.
Slots that some function declares with ``var`` are treated as VM registers and
pre-declared per function so a read is never "unbound". Module-scope aliases
(``Gf``, ``_$k``, ``_$ue``, ``Date``, ...) are emitted as globals.
"""
from __future__ import annotations

from typing import Mapping

from unidecompiler.core.ir import (
    Const,
    Global,
    ObjectLiteral,
    SourceRef,
    UndefinedLiteral,
    Var,
)
from unidecompiler.core.effects import (
    Binary,
    Copy,
    DropBelowTop,
    DuplicateTop,
    DuplicateTopBelow,
    Effect,
    Invoke,
    InvokeMember,
    InvokeMethod,
    LoadAttr,
    LoadItem,
    LoadLocal,
    Pop,
    Push,
    ReturnTop,
    StoreAttr,
    StoreItemAtDepth,
    StoreLocal,
    Swap,
    Unary,
    UnknownOpcode,
)
from unidecompiler.core.vm_function import lift_steps
from unidecompiler.core.vm_bytecode import VMBytecodeStep
from unidecompiler.core.vm_function import VMFunctionSpec, lift_vm_step_function
from unidecompiler.core.vm_hints import VMHint
from unidecompiler.core.vm_module import assemble_vm_module
from unidecompiler.core.vm_operands import VMDecodedInstruction, VMOperand
from unidecompiler.core.vm_region import (
    VMRegionCallbacks,
    VMRegionOpcodeClasses,
    VMStatefulCallbacks,
    VMLinearState,
    VMRegionSlice,
    build_hint_region_profile,
)
from unidecompiler.plugins import FrontendModule

from .model import FRONTEND_ID, JdvmFunction, JdvmImage, JdvmInstruction
from .vmdef import vm_definition

#: Semantics that unconditionally transfer control.
_UNCONDITIONAL = frozenset({"jump"})
#: Semantics that transfer control when the popped condition is true.
_BRANCH_TRUE = frozenset({"branch-true-pop", "branch-true-keep"})
#: Semantics that transfer control when the popped condition is false.
_BRANCH_FALSE = frozenset({"branch-false-pop", "branch-false-keep"})
#: Branch semantics that leave the condition on the stack for the jump target.
_BRANCH_KEEP = frozenset({"branch-false-keep", "branch-true-keep"})
_BRANCH_ALL = _BRANCH_TRUE | _BRANCH_FALSE
_JUMPS_ALL = _BRANCH_ALL | _UNCONDITIONAL

#: `Function.prototype.call` sites. The VM always dispatches through
#: `Function.prototype.call.call(callee, thisArg, *args)`, which is exactly
#: `callee.call(thisArg, *args)`; the semantic name records the real argument
#: count after the receiver.
_CALL_SLOTS = {
    "invoke-call-1": 1,
    "invoke-call-2": 2,
    "invoke-call-3": 3,
    "invoke-call-4": 4,
    "invoke-call-5": 5,
    "invoke-call-6": 6,
    "invoke-call-7": 7,
}

#: Binary operator spellings for the VM's in-place arithmetic handlers.
_BINARY_OPS = {
    "add-assign": "+",
    "sub-assign": "-",
    "mul-assign": "*",
    "div-assign": "/",
    "mod-assign": "%",
    "or-assign": "|",
    "xor-assign": "^",
    "binary-eq": "==",
    "binary-ne": "!=",
    "binary-strict-eq": "===",
    "binary-strict-ne": "!==",
    "binary-lt": "<",
    "binary-gt": ">",
    "binary-ge": ">=",
    "binary-in": "in",
}

_CONST_SEMANTICS = {
    "push-null": None,
    "push-zero": 0,
    "push-one": 1,
}

#: Opcode classes handed to core's VM-neutral region profiler. ``jumps`` means
#: unconditional transfer only; conditional branches are classified separately
#: so core can structure them instead of treating them as region boundaries.
REGION_OPCODE_CLASSES = VMRegionOpcodeClasses(
    noise=frozenset(),
    control=_JUMPS_ALL | {"switch-multiway"},
    jumps=_UNCONDITIONAL,
    forward_jumps=_JUMPS_ALL,
    backward_jumps=_JUMPS_ALL,
    conditional_jumps=_BRANCH_ALL | {"switch-multiway"},
)

_REGISTERS: frozenset[str] | None = None


def _registers() -> frozenset[str]:
    """Every name the interpreter declares as a function-local slot."""
    global _REGISTERS
    if _REGISTERS is None:
        names: set[str] = set()
        for function in vm_definition().functions:
            names.update(function.locals)
        _REGISTERS = frozenset(names)
    return _REGISTERS


def lift_module(module: FrontendModule):
    """Lift a decoded jdvm module into generic core IR."""
    if module.frontend_id != FRONTEND_ID:
        raise TypeError(f"jdvm frontend cannot lift module from {module.frontend_id!r}")
    image: JdvmImage = module.payload
    functions = tuple(_lift_function(function) for function in image.functions)
    return assemble_vm_module(
        name=str(module.metadata.get("filename") or image.metadata.get("filename") or "<jdvm>"),
        source_language="jdvm",
        metadata={
            "frontend": module.metadata,
            "bytecode_format": "jdvm",
            "decoded": dict(image.metadata),
            "diagnostics": [
                {"code": d.code, "offset": d.offset, "message": d.message}
                for d in image.diagnostics
            ],
        },
        functions=functions,
    )


def _lift_function(function: JdvmFunction):
    steps = _steps(function)
    initial_locals = _initial_locals(function)
    raw_window = _raw_window(function)
    profile = build_hint_region_profile(
        steps,
        frontend=FRONTEND_ID,
        opcode_classes=REGION_OPCODE_CLASSES,
        raw_window=raw_window,
    )
    return lift_vm_step_function(
        _spec(function),
        steps,
        profile=profile,
        callbacks=_region_callbacks(steps),
        stateful_callbacks=_stateful_callbacks(steps, initial_locals),
        initial_locals=initial_locals,
        raw_window=raw_window,
    )


def _spec(function: JdvmFunction) -> VMFunctionSpec:
    return VMFunctionSpec(
        name=function.name,
        params=(),
        frontend=FRONTEND_ID,
        instruction_count=len(function.instructions),
        local_names=tuple(sorted(_function_registers(function))),
        metadata={
            "entry": function.entry,
            "end": function.end,
            "region": [function.entry, function.end],
        },
    )


def _function_registers(function: JdvmFunction) -> set[str]:
    known = _registers()
    used: set[str] = set()
    for instruction in function.instructions:
        for symbol in instruction.symbols:
            if symbol in known:
                used.add(symbol)
    return used


def _initial_locals(function: JdvmFunction) -> dict[str, Var]:
    return {
        name: Var(name=name, source=_source(function.entry, f"register {name}"))
        for name in sorted(_function_registers(function))
    }


def _source(offset: int, detail: str | None = None) -> SourceRef:
    return SourceRef(frontend=FRONTEND_ID, offset=offset, detail=detail)


def _raw_window(function: JdvmFunction):
    instructions = function.instructions

    def window(index: int, radius: int = 3) -> tuple[str, ...]:
        start = max(0, index - radius)
        end = min(len(instructions), index + radius + 1)
        return tuple(_raw_line(instruction) for instruction in instructions[start:end])

    return window


def _raw_line(instruction: JdvmInstruction) -> str:
    parts = [f"@{instruction.offset}", instruction.semantic]
    if instruction.pool_value is not None:
        parts.append(f"pool={instruction.pool_value!r}")
    if instruction.closure_entry is not None:
        parts.append(f"closure={instruction.closure_entry}")
    if instruction.target is not None:
        parts.append(f"->{instruction.target}")
    if instruction.cases:
        parts.append(
            "cases=" + ",".join(f"{c.pool_value!r}:{c.target}" for c in instruction.cases)
        )
    if instruction.default_target is not None:
        parts.append(f"default->{instruction.default_target}")
    return " ".join(parts)


def _operands(instruction: JdvmInstruction) -> tuple[VMOperand, ...]:
    operands: list[VMOperand] = []
    if instruction.pool_value is not None:
        operands.append(VMOperand(role="member", value=instruction.pool_value, text=instruction.pool_value))
    if instruction.pool_secondary is not None:
        operands.append(VMOperand(role="raw", value=instruction.pool_secondary, text=instruction.pool_secondary))
    for symbol in instruction.symbols:
        role = "register" if symbol in _registers() else "global"
        operands.append(VMOperand(role=role, value=symbol, text=symbol))
    if instruction.closure_entry is not None:
        operands.append(VMOperand(role="target", value=instruction.closure_entry, text=str(instruction.closure_entry)))
    if instruction.target is not None:
        operands.append(VMOperand(role="target", value=instruction.target, text=str(instruction.target)))
    if instruction.default_target is not None:
        operands.append(VMOperand(role="target", value=instruction.default_target, text=f"default:{instruction.default_target}"))
    for case in instruction.cases:
        operands.append(VMOperand(role="constant", value=case.pool_value, text=case.pool_value or ""))
        operands.append(VMOperand(role="target", value=case.target, text=f"case:{case.target}"))
    for value in instruction.operands:
        operands.append(VMOperand(role="immediate", value=value, text=str(value)))
    return tuple(operands)


def _hints(instruction: JdvmInstruction) -> tuple[VMHint, ...]:
    source = _source(instruction.offset, instruction.semantic)
    if instruction.semantic in _UNCONDITIONAL:
        return (VMHint(kind="branch-target", source=source, target=instruction.target, label=instruction.semantic, flow="unconditional"),)
    if instruction.semantic == "switch-multiway":
        hints: list[VMHint] = []
        for case in instruction.cases:
            hints.append(
                VMHint(
                    kind="case-target",
                    source=source,
                    target=case.target,
                    value=case.pool_value,
                    label=instruction.semantic,
                    flow="multiway",
                )
            )
        if instruction.default_target is not None:
            hints.append(
                VMHint(
                    kind="default-target",
                    source=source,
                    target=instruction.default_target,
                    label=instruction.semantic,
                    flow="multiway",
                )
            )
        return tuple(hints)
    if instruction.semantic in _BRANCH_ALL:
        detail = "target-if-true" if instruction.semantic in _BRANCH_TRUE else None
        return (
            VMHint(
                kind="branch-target",
                source=source,
                target=instruction.target,
                label=instruction.semantic,
                flow="conditional",
                detail=detail,
            ),
        )
    return ()


def _step(
    function: JdvmFunction,
    instruction: JdvmInstruction,
    *,
    fallthrough_pop: bool = False,
) -> VMBytecodeStep:
    source = _source(instruction.offset, f"{function.name}:{instruction.semantic}")
    decoded = VMDecodedInstruction(
        opcode=instruction.semantic,
        source=source,
        operands=_operands(instruction),
        raw=_raw_line(instruction),
    )
    return VMBytecodeStep(
        opcode=instruction.semantic,
        source=source,
        effects=_effects(instruction, source, fallthrough_pop=fallthrough_pop),
        raw=decoded.raw,
        decoded=decoded,
        hints=_hints(instruction),
    )


# ---------------------------------------------------------------------------
# effects
# ---------------------------------------------------------------------------
def _symbol(symbol: str, source: SourceRef) -> Effect:
    """Push a VM register, or a module-scope global when it is not a register."""
    if symbol in _registers():
        return LoadLocal(source=source, name=symbol, fallback=Var(name=symbol, source=source))
    return Push(source=source, value=Global(name=symbol, source=source))


def _attr(instruction: JdvmInstruction) -> str:
    return instruction.pool_value or ""


def _effects(
    instruction: JdvmInstruction,
    source: SourceRef,
    *,
    fallthrough_pop: bool = False,
) -> tuple[Effect, ...] | None:
    """Thin effects for one instruction, optionally with a fallthrough pop.

    ``fallthrough_pop`` carries the pop that a ``branch-*-keep`` handler
    performs on its fallthrough edge; see :func:`_fallthrough_pop_offsets`.
    """
    effects = _instruction_effects(instruction, source)
    if fallthrough_pop and effects is not None:
        return (Pop(source=source, count=1), *effects)
    return effects


def _instruction_effects(
    instruction: JdvmInstruction,
    source: SourceRef,
) -> tuple[Effect, ...] | None:
    semantic = instruction.semantic
    symbols = instruction.symbols

    if semantic == "unknown":
        return (
            UnknownOpcode(
                source=source,
                opcode=str(instruction.opcode),
                raw=_raw_line(instruction),
            ),
        )

    # --- control flow: conditions are owned by the stateful callbacks -------
    if semantic in _BRANCH_ALL or semantic == "jump":
        return ()
    if semantic == "switch-multiway":
        return (Pop(source=source, count=1),)

    # --- returns -----------------------------------------------------------
    if semantic == "return-value":
        return (ReturnTop(source=source),)
    if semantic == "return-void":
        return (ReturnTop(source=source, empty_is_void=True),)

    # --- constants and pushes ---------------------------------------------
    if semantic in _CONST_SEMANTICS:
        value = _CONST_SEMANTICS[semantic]
        return (Push(source=source, value=Const(value=value, source=source)),)
    if semantic == "push-undefined":
        return (Push(source=source, value=UndefinedLiteral(source=source)),)
    if semantic == "push-object":
        return (Push(source=source, value=ObjectLiteral(source=source)),)
    if semantic == "push-pool":
        return (Push(source=source, value=Const(value=instruction.pool_value, source=source)),)
    if semantic == "push-immediate":
        return (
            Push(
                source=source,
                value=Const(value=instruction.operands[0], source=source),
            ),
        )
    if semantic == "push-this":
        return (Push(source=source, value=Global(name="this", source=source)),)
    if semantic == "push-this-member":
        return (
            Push(source=source, value=Global(name="this", source=source)),
            LoadAttr(source=source, attr=_attr(instruction)),
        )
    if semantic == "push-closure":
        entry = instruction.closure_entry
        return (
            Push(source=source, value=Global(name=f"<function {entry}>", source=source)),
        )

    # --- registers ---------------------------------------------------------
    if semantic == "load-symbol":
        return (_symbol(symbols[0], source),)
    if semantic == "store-symbol-peek":
        # ``G = STK[STK.length - 1]`` peeks; an empty stack yields undefined.
        name = symbols[0]
        return (
            Copy(source=source, depth=1, allow_missing=True),
            StoreLocal(
                source=source,
                name=name,
                target=Var(name=name, source=source),
                missing_value=UndefinedLiteral(source=source),
            ),
        )
    if semantic == "load-symbol-postinc":
        name = symbols[0]
        return (
            _symbol(name, source),
            DuplicateTop(source=source),
            Push(source=source, value=Const(value=1, source=source)),
            Binary(source=source, op="+"),
            StoreLocal(source=source, name=name, target=Var(name=name, source=source)),
        )
    if semantic == "load-symbol-postdec":
        name = symbols[0]
        return (
            _symbol(name, source),
            DuplicateTop(source=source),
            Push(source=source, value=Const(value=1, source=source)),
            Binary(source=source, op="-"),
            StoreLocal(source=source, name=name, target=Var(name=name, source=source)),
        )
    if semantic == "load-symbol-predec":
        name = symbols[0]
        return (
            _symbol(name, source),
            Push(source=source, value=Const(value=1, source=source)),
            Binary(source=source, op="-"),
            DuplicateTop(source=source, materialized_name=name),
            StoreLocal(source=source, name=name, target=Var(name=name, source=source)),
        )
    if semantic == "typeof-symbol":
        return (
            _symbol(symbols[0], source),
            Unary(source=source, op="typeof "),
        )

    # --- members and items -------------------------------------------------
    if semantic == "load-length":
        return (LoadAttr(source=source, attr="length"),)
    if semantic in ("load-member", "load-member-raw"):
        return (LoadAttr(source=source, attr=_attr(instruction)),)
    if semantic in ("dup-load-member", "dup-load-member-raw"):
        # ``push(dup); STK[-2] = STK[-2][attr]`` leaves the attribute value
        # *below* the duplicated receiver.
        return (
            DuplicateTop(source=source),
            LoadAttr(source=source, attr=_attr(instruction)),
            Swap(source=source, depth=2),
        )
    if semantic == "store-member":
        # ``STK[-2][attr] = STK[-1]`` keeps the object slot on the stack.
        return (
            Copy(source=source, depth=2),
            StoreAttr(source=source, attr=_attr(instruction), order="value-obj"),
        )
    if semantic == "store-member-dup":
        return (
            DuplicateTopBelow(source=source, below_count=1),
            StoreAttr(source=source, attr=_attr(instruction), order="obj-value"),
        )
    if semantic == "load-item":
        return (LoadItem(source=source),)
    if semantic == "store-item":
        # ``STK[-3][STK[-2]] = STK[-1]`` keeps the object slot on the stack.
        return (StoreItemAtDepth(source=source, depth=1),)
    if semantic == "store-item-dup":
        # Same store, but the assigned value replaces the object slot.
        return (
            DuplicateTopBelow(source=source, below_count=2),
            StoreItemAtDepth(source=source, depth=1),
            Pop(source=source, count=1),
        )
    if semantic == "store-undefined":
        return (
            Pop(source=source, count=1),
            Push(source=source, value=UndefinedLiteral(source=source)),
        )

    # --- stack shuffling ---------------------------------------------------
    if semantic == "pop":
        return (Pop(source=source, count=1),)
    if semantic == "duplicate":
        return (DuplicateTop(source=source),)
    if semantic == "duplicate2":
        return (Copy(source=source, depth=2), Copy(source=source, depth=2))

    # --- operators ---------------------------------------------------------
    if semantic == "negate":
        return (Unary(source=source, op="-"),)
    if semantic == "logical-not":
        return (Unary(source=source, op="not "),)
    if semantic == "typeof":
        return (Unary(source=source, op="typeof "),)
    if semantic in _BINARY_OPS:
        return (Binary(source=source, op=_BINARY_OPS[semantic]),)

    # --- calls -------------------------------------------------------------
    if semantic in _CALL_SLOTS:
        # ``STK[-n] = Function.prototype.call.call(STK[-n], thisArg, *args)``.
        # ``InvokeMethod`` keeps the receiver on the stack, but the VM replaces
        # the receiver slot with the call result, so drop the stale receiver.
        return (
            InvokeMethod(
                source=source,
                attr="call",
                arg_count=_CALL_SLOTS[semantic],
                depth=1,
            ),
            DropBelowTop(source=source, count=1),
        )

    # --- construction ------------------------------------------------------
    if semantic == "construct-count":
        return (
            Push(source=source, value=Const(value=instruction.operands[0], source=source)),
            InvokeMember(
                source=source,
                static=True,
                constructor_type=symbols[0] if symbols else "Array",
                arg_count=1,
            ),
        )
    if semantic == "construct-pool1":
        return (
            Push(source=source, value=Const(value=instruction.pool_value, source=source)),
            InvokeMember(
                source=source,
                static=True,
                constructor_type=symbols[0] if symbols else "RegExp",
                arg_count=1,
            ),
        )
    if semantic == "construct-pool2":
        return (
            Push(source=source, value=Const(value=instruction.pool_value, source=source)),
            Push(source=source, value=Const(value=instruction.pool_secondary, source=source)),
            InvokeMember(
                source=source,
                static=True,
                constructor_type=symbols[0] if symbols else "RegExp",
                arg_count=2,
            ),
        )
    if semantic == "construct-stack0":
        return (
            Pop(source=source, count=1),
            Invoke(source=source, arg_count=0),
        )
    if semantic == "construct-stack1":
        return (
            DropBelowTop(source=source, count=1),
            Invoke(source=source, arg_count=1),
        )

    return (
        UnknownOpcode(
            source=source,
            opcode=semantic,
            raw=_raw_line(instruction),
        ),
    )


# ---------------------------------------------------------------------------
# region callbacks
# ---------------------------------------------------------------------------
def _fallthrough_pop_offsets(function: JdvmFunction) -> frozenset[int]:
    """Offsets that begin a keep-branch fallthrough and have no other entry.

    A ``branch-*-keep`` handler pops on its fallthrough edge and keeps the
    tested value on its taken edge. Core's conditional terminator consumes one
    condition value for *both* edges, so the pop has to be carried by the
    fallthrough itself. That is only equivalent to the VM when the fallthrough
    instruction has no other entry: if anything branches to it, the pop would
    be applied on a path that must not have it, so the offset is left out and
    the branch keeps core's uniform-condition model.
    """
    targets: set[int] = set()
    for instruction in function.instructions:
        if instruction.target is not None:
            targets.add(instruction.target)
        if instruction.default_target is not None:
            targets.add(instruction.default_target)
        for case in instruction.cases:
            targets.add(case.target)
    pops: set[int] = set()
    for instruction in function.instructions:
        if instruction.semantic not in _BRANCH_KEEP:
            continue
        fallthrough = instruction.offset + instruction.size
        if fallthrough < function.end and fallthrough not in targets:
            pops.add(fallthrough)
    return frozenset(pops)


def _steps(function: JdvmFunction) -> tuple[VMBytecodeStep, ...]:
    pops = _fallthrough_pop_offsets(function)
    return tuple(
        _step(function, instruction, fallthrough_pop=instruction.offset in pops)
        for instruction in function.instructions
    )


def _region_callbacks(steps: tuple[VMBytecodeStep, ...]) -> VMRegionCallbacks[VMBytecodeStep]:
    def lift_slice(start: int, end: int, stack: tuple) -> VMRegionSlice:
        result = lift_steps(steps[start:end], initial_stack=tuple(stack))
        if result.state.diagnostics:
            return VMRegionSlice(statements=(), stopped_at=start)
        return VMRegionSlice(
            statements=tuple(result.state.statements),
            stopped_at=None,
        )

    def lift_expr(start: int, end: int, stack: tuple):
        result = lift_steps(steps[start:end], initial_stack=tuple(stack))
        if result.state.diagnostics or result.stopped_at is not None:
            return None
        return result.state.stack[-1] if result.state.stack else None

    return VMRegionCallbacks(
        lift_slice=lift_slice,
        lift_expr=lift_expr,
        lift_iter_loop=lambda _index, _iterator: None,
        lift_async_iter_loop=lambda _start, _end, _index: None,
        lift_comprehension=lambda _start, _end, _index, _expr: None,
    )


def _stateful_callbacks(
    steps: tuple[VMBytecodeStep, ...],
    locals_: dict[str, Var],
) -> VMStatefulCallbacks[VMBytecodeStep]:
    def lift_linear(start: int, end: int, locals_in: Mapping[str, object], stack: tuple):
        result = lift_steps(
            steps[start:end],
            initial_locals=dict(locals_in),
            initial_stack=tuple(stack),
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

    def branch_condition(branch: VMBytecodeStep, stack: tuple):
        """The VM branches on the truthiness of the value it tests.

        A peeking branch on an empty operand stack tests ``undefined``; the VM
        reads ``STK[STK.length - 1]``, which is ``undefined`` rather than an
        error.
        """
        if not stack:
            return UndefinedLiteral(source=branch.source)
        return stack[-1]

    def branch_stack_width(branch: VMBytecodeStep) -> int:
        """Popping branches consume the condition; peeking branches keep it.

        A peeking branch keeps the tested value on its taken edge, so the
        condition must not be subtracted from the base stack. Its fallthrough
        edge pops instead, which is carried by the fallthrough instruction.
        """
        return 0 if branch.opcode in _BRANCH_KEEP else 1

    return VMStatefulCallbacks(
        initial_locals=lambda: dict(locals_),
        lift_linear=lift_linear,
        branch_condition=branch_condition,
        branch_stack_width=branch_stack_width,
    )
