"""Thin-IR lifting for the mtgsigvm bytecode family.

This module converts the frontend-private model into neutral
``VMBytecodeStep`` values and hands the complete stream to
``lift_vm_step_function``. Core owns stack recovery, CFG construction,
exception edges, loops, AST and pseudocode; nothing here builds any of those.
"""

from __future__ import annotations

from unidecompiler.core.effects import (
    Binary,
    BuildArray,
    BuildCall,
    BuildMap,
    BuildString,
    Compare,
    DuplicateTop,
    Effect,
    InvokeExpanded,
    LoadItem,
    MakeFunctionValue,
    Pop,
    Push,
    RaiseTop,
    ReturnTop,
    StoreItemEffect,
    Swap,
    Unary,
)
from unidecompiler.core.ir import Call, Const, Global, ModuleIR, SourceRef, Var
from unidecompiler.core.vm_bytecode import VMBytecodeStep
from unidecompiler.core.vm_function import VMFunctionSpec, lift_steps, lift_vm_step_function
from unidecompiler.core.vm_hints import VMHint
from unidecompiler.core.vm_module import assemble_vm_module
from unidecompiler.core.vm_operands import VMDecodedInstruction, VMOperand
from unidecompiler.core.vm_region import (
    VMLinearState,
    VMRegionCallbacks,
    VMRegionOpcodeClasses,
    VMRegionProfile,
    VMRegionSlice,
    VMStatefulCallbacks,
    build_hint_region_profile,
)
from unidecompiler.plugins import FrontendModule
from unidecompiler.provenance import ByteRange

from .model import (
    MtgsigDiagnostic,
    MtgsigFunction,
    MtgsigInstruction,
    MtgsigModule,
)
from .opcodes import (
    CONDITIONAL_JUMP_OPCODES,
    CONTROL_OPCODES,
    JUMP_OPCODES,
    NOISE_OPCODES,
    SWITCH_OPCODES,
)

FRONTEND_ID = "mtgsigvm"

#: Neutral name for the per-invocation scope object.
#:
#: The interpreter's ``run(name, arguments, scope)`` always receives a scope
#: object, and ``GETSCOPE``/``SETSCOPE``/``SETGLOBAL`` read and write it by a
#: key taken from the operand stack. Modelling it as a frame parameter keeps the
#: recovered IR exact without the frontend having to interpret the stack.
SCOPE_OBJECT = "__scope"

#: Neutral name for the argument object a VM invocation starts with.
#:
#: The interpreter seeds the operand stack with exactly one element,
#: ``[arguments]``, before the first instruction, and ``run`` receives the same
#: value as its second parameter.
ARGUMENTS_OBJECT = "__arguments"

#: Neutral runtime-helper names for VM primitives that the public thin IR has no
#: dedicated effect for. They are analysis markers, not host functions: the
#: helper call keeps every operand and the exact stack arity, so core can still
#: recover the surrounding data flow and control flow.
DELETE_HELPER = "vm_delete_item"
FORIN_HELPER = "vm_forin"
TRY_HELPER = "vm_try_call"

_COMPARISON_OPS = {
    5: "==",
    6: "!=",
    7: "===",
    8: "!==",
    9: "<",
    10: "<=",
    11: ">",
    12: ">=",
}
_BINARY_OPS = {
    13: "+",
    14: "-",
    15: "*",
    16: "**",
    17: "/",
    18: "%",
    21: "|",
    22: "^",
    23: "&",
    24: "<<",
    25: ">>",
    26: ">>>",
    27: "instanceof",
    31: "in",
}

#: The interpreter evaluates these with JavaScript's bitwise coercions, which
#: truncate both operands to a fixed 32-bit integer and wrap. ``>>>`` is the
#: unsigned variant. Recording the domain, width and overflow policy keeps core
#: transformations from silently changing them.
_INT32_BINARY_OPS = frozenset({21, 22, 23, 24, 25})
_UINT32_BINARY_OPS = frozenset({26})
_UNARY_OPS = {19: "!", 20: "~", 28: "typeof "}
_LITERAL_PUSHES = {
    39: None,       # undefined
    40: None,       # null
    43: True,
    44: False,
}

#: Opcodes submitted through a neutral runtime helper because the public thin IR
#: has no dedicated effect for them. Each still carries every operand and the
#: exact stack arity, and is reported as provenance.
_HELPER_OPCODES = {
    32: (FORIN_HELPER, "for-in over stack-provided callback and binding slot"),
    34: (TRY_HELPER, "try/catch/finally over stack-provided closures"),
}

REGION_OPCODE_CLASSES = VMRegionOpcodeClasses(
    noise=NOISE_OPCODES,
    control=CONTROL_OPCODES,
    # JMP is bidirectional, so it is listed in both direction classes; the
    # profile combines the class with the decoded target's direction.
    jumps=JUMP_OPCODES,
    forward_jumps=JUMP_OPCODES,
    backward_jumps=JUMP_OPCODES,
    conditional_jumps=CONDITIONAL_JUMP_OPCODES,
    truthy_jumps=frozenset({"JMPIF"}),
)


def _source(instruction: MtgsigInstruction) -> SourceRef:
    return SourceRef(frontend=FRONTEND_ID, offset=instruction.offset)


def _raw_window(instructions: tuple[MtgsigInstruction, ...], index: int) -> tuple[str, ...]:
    start = max(0, index - 2)
    end = min(len(instructions), index + 3)
    return tuple(instructions[position].raw for position in range(start, end))


def _operands(instruction: MtgsigInstruction) -> tuple[VMOperand, ...]:
    operands: list[VMOperand] = []
    for operand in instruction.operands:
        role = {
            "constant": "constant",
            "string": "constant",
            "count": "immediate",
            "flag": "immediate",
            "target": "branch-target",
            "case": "branch-target",
            "scope-key": "slot",
            "operand": "immediate",
        }.get(operand.role, "raw")
        operands.append(VMOperand(role=role, value=operand.value, text=operand.text))
    return tuple(operands)


def _artifact_range(instruction: MtgsigInstruction) -> ByteRange | None:
    """Absolute provenance, provable because the decoder does not relocate code."""
    if instruction.artifact_offset is None:
        return None
    return ByteRange(instruction.artifact_offset, instruction.size)


def _decoded(instruction: MtgsigInstruction) -> VMDecodedInstruction:
    return VMDecodedInstruction(
        opcode=instruction.mnemonic,
        source=_source(instruction),
        operands=_operands(instruction),
        raw=instruction.raw,
        artifact_range=_artifact_range(instruction),
    )


def _hints(instruction: MtgsigInstruction) -> tuple[VMHint, ...]:
    source = _source(instruction)
    hints: list[VMHint] = []
    if instruction.mnemonic in (JUMP_OPCODES | CONDITIONAL_JUMP_OPCODES) and instruction.targets:
        target = instruction.targets[0]
        conditional = instruction.mnemonic in CONDITIONAL_JUMP_OPCODES
        # The interpreter branches with a raw truthiness test on the popped
        # value: JMPIF jumps when it is truthy, JMPNOT when it is falsy.
        detail = (
            "target-if-false" if instruction.mnemonic == "JMPNOT"
            else "target-if-true" if conditional
            else None
        )
        hints.append(
            VMHint(
                kind="loop-backedge" if target <= instruction.offset else "branch-target",
                source=source,
                target=target,
                label=instruction.mnemonic,
                detail=detail,
                flow="conditional" if conditional else "unconditional",
            )
        )
    elif instruction.opcode == 56:
        cases = [operand for operand in instruction.operands if operand.role == "case"]
        for index, operand in enumerate(cases):
            hints.append(
                VMHint(
                    kind="case-target",
                    source=source,
                    target=operand.value,
                    label=instruction.mnemonic,
                    detail="case[%d]" % index,
                    flow="multiway",
                )
            )
        default = next(
            (operand for operand in instruction.operands
             if operand.role == "target" and str(operand.text).startswith("default=")),
            None,
        )
        if default is not None:
            hints.append(
                VMHint(
                    kind="default-target",
                    source=source,
                    target=default.value,
                    label=instruction.mnemonic,
                    detail="default",
                    flow="multiway",
                )
            )
    elif instruction.opcode == 34:
        # TRY dispatches over closures taken from the operand stack rather than
        # protecting a bytecode range, so it can only carry the value-less
        # operation marker. It deliberately supplies no protected interval:
        # claiming one would fabricate control flow the stream does not contain.
        hints.append(
            VMHint(
                kind="exception-region",
                source=source,
                label=instruction.mnemonic,
                detail="stack-provided try/catch/finally closures",
            )
        )
    return tuple(hints)


def _scope_effects(instruction: MtgsigInstruction, source: SourceRef) -> tuple[Effect, ...]:
    """Scoped read/write through the frame's scope object.

    The key sits on the operand stack, so the scope object is materialised and
    rotated into place for the neutral item effect. This reproduces the
    interpreter's ``scope[key]`` / ``scope[key] = value`` exactly.
    """
    scope = Var(name=SCOPE_OBJECT, source=source)
    if instruction.opcode == 51:  # GETSCOPE: value = scope[key]
        return (
            Push(source=source, value=scope),
            Swap(source=source, depth=2),
            LoadItem(source=source),
        )
    # SETSCOPE / SETGLOBAL: scope[key] = value
    return (
        Push(source=source, value=scope),
        Swap(source=source, depth=3),
        Swap(source=source, depth=2),
        StoreItemEffect(source=source, order="obj-key-value"),
    )


def _effects(instruction: MtgsigInstruction, module: MtgsigModule) -> tuple[Effect, ...]:
    source = _source(instruction)
    opcode = instruction.opcode

    if opcode in _COMPARISON_OPS:
        return (Compare(source=source, op=_COMPARISON_OPS[opcode]),)
    if opcode in _BINARY_OPS:
        if opcode in _INT32_BINARY_OPS:
            return (
                Binary(
                    source=source,
                    op=_BINARY_OPS[opcode],
                    numeric_domain="signed",
                    bit_width=32,
                    overflow_policy="wrap",
                ),
            )
        if opcode in _UINT32_BINARY_OPS:
            return (
                Binary(
                    source=source,
                    op=_BINARY_OPS[opcode],
                    numeric_domain="unsigned",
                    bit_width=32,
                    overflow_policy="wrap",
                ),
            )
        return (Binary(source=source, op=_BINARY_OPS[opcode]),)
    if opcode in _UNARY_OPS:
        return (Unary(source=source, op=_UNARY_OPS[opcode]),)
    if opcode in _LITERAL_PUSHES:
        return (Push(source=source, value=Const(value=_LITERAL_PUSHES[opcode], source=source)),)

    if opcode == 46:
        index = _first_value(instruction)
        text = module.strings[index] if isinstance(index, int) and 0 <= index < len(module.strings) else None
        return (Push(source=source, value=Const(value=text, source=source)),)
    if opcode == 45:
        index = _first_value(instruction)
        text = module.constants[index] if isinstance(index, int) and 0 <= index < len(module.constants) else None
        return (Push(source=source, value=Const(value=_number(text), source=source)),)
    if opcode == 30:
        return (BuildString(source=source, count=int(_first_value(instruction) or 0)),)
    if opcode == 41:
        return (BuildMap(source=source, count=int(_first_value(instruction) or 0)),)
    if opcode == 42:
        return (BuildArray(source=source, kind="list", count=int(_first_value(instruction) or 0)),)
    if opcode == 47:
        return (Pop(source=source, count=1),)
    if opcode == 48:
        # ``Swap(depth=N)`` exchanges the top of stack with the entry N below
        # it, so exchanging the top two requires depth 2.
        return (Swap(source=source, depth=2),)
    if opcode == 49:
        return (DuplicateTop(source=source),)
    if opcode == 38:
        return ()
    if opcode == 2:
        return (StoreItemEffect(source=source, order="obj-key-value"),)
    if opcode == 3:
        return (LoadItem(source=source),)
    if opcode == 4:
        # ``delete obj[key]`` also leaves a boolean result on the stack; the
        # public thin IR has no delete-and-return effect, so the deletion is
        # submitted exactly and the result is carried by a neutral helper call.
        return (
            BuildCall(
                source=source,
                callee=Global(name=DELETE_HELPER, source=source),
                arg_count=2,
                returns=0 if instruction.dead_result else 1,
            ),
        )
    if opcode == 29:
        return (
            BuildCall(
                source=source,
                callee=Global(name="RegExp", source=source),
                arg_count=2,
                returns=0 if instruction.dead_result else 1,
            ),
        )
    if opcode in (0, 37):
        # Both constructors consume an argument array from the operand stack.
        return (
            InvokeExpanded(
                source=source, has_keywords=False, returns=0 if instruction.dead_result else 1
            ),
        )
    if opcode == 58:
        # ``callee.apply(thisArg, argsArray)``. InvokeExpanded models the
        # expanded argument list; the receiver is not representable in the
        # public thin IR, so it is rotated off the stack.
        call: tuple[Effect, ...] = (
            Swap(source=source, depth=3),
            Pop(source=source, count=1),
            Swap(source=source, depth=2),
            InvokeExpanded(source=source, has_keywords=False),
        )
        if instruction.dead_result:
            # The decoder proved this result is never read, so the call is an
            # expression statement: ``emit_calls`` turns the pending call into a
            # statement and drops the value instead of leaving it on the stack
            # forever, which would make the operand-stack depth inconsistent at
            # every merge the discarded value reaches.
            return (*call, Pop(source=source, count=1, emit_calls=True))
        return call
    if opcode == 36 or opcode == 57:
        return (MakeFunctionValue(source=source, fallback_name=_closure_name(instruction, module)),)
    if opcode == 1 or opcode == 35:
        return (ReturnTop(source=source),)
    if opcode == 33:
        return (RaiseTop(source=source),)
    if opcode in (50, 51, 52):
        return _scope_effects(instruction, source)
    if opcode == 54 or opcode == 55:
        # The condition is consumed by core, not by this effect: core reads
        # ``branch_stack_width`` and removes that many values from the successor
        # stack itself. Emitting Pop here as well would drop one value twice.
        return ()
    if opcode == 53:
        return ()
    if opcode == 56:
        count = int(_first_value(instruction) or 0)
        return (Pop(source=source, count=count + 1),)
    if opcode == 34:
        # TRY consumes a try body, a catch body, an unused slot and an optional
        # finally body, all as closures on the operand stack. The neutral helper
        # keeps those operands in their stack order instead of discarding them.
        flag = int(_first_value(instruction) or 0)
        return (
            BuildCall(
                source=source,
                callee=Global(name=TRY_HELPER, source=source),
                arg_count=4 if flag else 3,
                returns=0,
            ),
        )
    if opcode == 32:
        return (
            BuildCall(
                source=source,
                callee=Global(name=FORIN_HELPER, source=source),
                arg_count=3,
                returns=0,
            ),
        )
    return ()


def _number(value):
    """Render an integral float64 constant as an integer.

    Every mtgsigvm number is an IEEE-754 double, so ``14.0`` and ``14`` are the
    same value; the integer spelling is used only for readability.
    """
    if isinstance(value, float) and value.is_integer() and abs(value) < 2 ** 53:
        return int(value)
    return value


def _first_value(instruction: MtgsigInstruction):
    return instruction.operands[0].value if instruction.operands else None


def _closure_name(instruction: MtgsigInstruction, module: MtgsigModule) -> str:
    if instruction.opcode == 57:
        return instruction.closure_name or "<closure>"
    return "<bound>"


def _steps(function: MtgsigFunction, module: MtgsigModule) -> tuple[VMBytecodeStep, ...]:
    """Submit one step per decoded instruction.

    The interpreter seeds the operand stack with the argument object before the
    first instruction. Core starts every function from an empty stack, so the
    first step also pushes that object explicitly; this keeps the stack shape
    identical on both the plain-effect and the stateful recovery paths.
    """
    steps: list[VMBytecodeStep] = []
    for index, instruction in enumerate(function.instructions):
        effects = _effects(instruction, module)
        if index == 0:
            effects = (
                Push(
                    source=_source(instruction),
                    value=Var(name=ARGUMENTS_OBJECT, source=_source(instruction)),
                ),
                *effects,
            )
        steps.append(
            VMBytecodeStep(
                opcode=instruction.mnemonic,
                source=_source(instruction),
                effects=effects,
                raw=instruction.raw,
                decoded=_decoded(instruction),
                hints=_hints(instruction),
            )
        )
    return tuple(steps)


def _function_spec(function: MtgsigFunction) -> VMFunctionSpec:
    return VMFunctionSpec(
        # The VM scope object is an implicit first parameter of every VM
        # invocation, so it is modelled as a bound parameter rather than an
        # unbound read.
        name=function.name,
        params=(SCOPE_OBJECT, ARGUMENTS_OBJECT),
        frontend=FRONTEND_ID,
        instruction_count=len(function.instructions),
        metadata={
            "vm_function": function.name,
            "vm_code_offset": function.code_offset,
            "vm_code_size": function.size,
            "vm_closure_targets": list(function.referenced_names),
        },
    )


def _region_profile(
    steps: tuple[VMBytecodeStep, ...], instructions: tuple[MtgsigInstruction, ...]
) -> VMRegionProfile[VMBytecodeStep]:
    return build_hint_region_profile(
        steps,
        frontend=FRONTEND_ID,
        opcode_classes=REGION_OPCODE_CLASSES,
        raw_window=lambda index: _raw_window(instructions, index),
    )


def _branch_stack_width(instruction: MtgsigInstruction) -> int:
    """How many operand-stack values the branch reads.

    ``JMPIF``/``JMPNOT`` test the value they pop. ``SWITCH`` reads its subject
    from below the case values, so the width is the whole switch stack.
    """
    if instruction.opcode == 56:
        count = int(instruction.operands[0].value) if instruction.operands else 0
        return count + 1
    if instruction.mnemonic in CONDITIONAL_JUMP_OPCODES:
        return 1
    return 0


def _branch_condition(instruction: MtgsigInstruction, stack: tuple[Expr, ...]):
    """The branch condition core renders.

    The interpreter branches on the raw truthiness of the popped value, so the
    value itself is the condition; the hint's ``target-if-true`` /
    ``target-if-false`` detail carries the polarity.
    """
    if not stack:
        return None
    if instruction.opcode == 56:
        return stack[0]
    return stack[-1]


def _linear_state(
    steps: tuple[VMBytecodeStep, ...],
    start: int,
    end: int,
    initial_locals: dict[str, Expr],
    initial_stack: tuple[Expr, ...],
) -> VMLinearState | None:
    """Lift one straight-line slice on top of core's incoming state.

    Core owns the walk; this only replays the effects this frontend declared for
    the same slice, which is exactly what the plain effect path does.
    """
    if start >= end:
        return VMLinearState(locals=dict(initial_locals), stack=tuple(initial_stack))
    result = lift_steps(
        steps[start:end], initial_locals=initial_locals, initial_stack=initial_stack
    )
    if result.state.diagnostics:
        return None
    if result.stopped_at is not None and result.state.terminator is None:
        return None
    # A terminator may sit before the end of the slice. This compiler emits a
    # trailing ``UNDEFINED; RET`` after a ``RETURN`` (the interpreter's normal
    # exit), and a ``RETURN`` already leaves the block, so the remainder is
    # unreachable and is not dropped.
    return VMLinearState(
        locals=result.state.locals,
        stack=tuple(result.state.stack),
        statements=tuple(result.state.statements),
        terminator=result.state.terminator,
    )


def _slice_state(
    steps: tuple[VMBytecodeStep, ...],
    start: int,
    end: int,
    locals: dict[str, Expr],
    stack: tuple[Expr, ...],
):
    if start >= end:
        return None
    return lift_steps(steps[start:end], initial_locals=locals, initial_stack=stack)


def _region_callbacks(
    steps: tuple[VMBytecodeStep, ...], instructions: tuple[MtgsigInstruction, ...]
) -> VMRegionCallbacks[VMBytecodeStep]:
    """Frontend facts core needs to structure control regions.

    Core owns the region walk, nesting and branch recovery. This only replays
    the declared effects over a slice core asks about.
    """

    def lift_slice(start: int, end: int, stack: tuple[Expr, ...]):
        if start >= end:
            # An empty slice is a successful no-op, not a failed lift.
            return VMRegionSlice()
        result = _slice_state(steps, start, end, {}, stack)
        if result is None or result.state.diagnostics:
            return VMRegionSlice(stopped_at=start)
        statements = list(result.state.statements)
        if result.stopped_at is not None and result.state.terminator is None:
            return VMRegionSlice(
                statements=tuple(statements),
                stopped_at=start + steps[start:end].index(result.stopped_at),
            )
        if result.state.terminator is not None:
            statements.append(result.state.terminator)
        return VMRegionSlice(statements=tuple(statements))

    def lift_expr(start: int, end: int, stack: tuple[Expr, ...]):
        if start >= end:
            return None
        result = _slice_state(steps, start, end, {}, stack)
        if result is None or result.state.diagnostics or not result.state.stack:
            return None
        return result.state.stack[-1]

    return VMRegionCallbacks(
        lift_slice=lift_slice,
        lift_expr=lift_expr,
        # The VM has no iterator, async-iterator or comprehension opcodes: its
        # only repetition construct is the plain conditional backedge.
        lift_iter_loop=lambda _get_iter_index, _iterable: None,
        lift_async_iter_loop=lambda _prefix_start, _get_aiter_index, _region_end: None,
        lift_comprehension=lambda _prefix_start, _get_iter_index, _region_end, _iterable: None,
    )


def lift_function(function: MtgsigFunction, module: MtgsigModule):
    """Lift one VM function into generic IR."""
    steps = _steps(function, module)
    instructions = function.instructions

    def make_callbacks(submitted: tuple[VMBytecodeStep, ...]) -> VMStatefulCallbacks:
        def linear(start: int, end: int, locals: dict[str, Expr], stack: tuple[Expr, ...]):
            return _linear_state(submitted, start, end, locals, stack)

        return VMStatefulCallbacks(
            initial_locals=lambda: {},
            lift_linear=linear,
            branch_condition=lambda branch, stack: _branch_condition(
                _instruction_for_step(instructions, branch), stack
            ),
            branch_stack_width=lambda branch: _branch_stack_width(
                _instruction_for_step(instructions, branch)
            ),
        )
    return lift_vm_step_function(
        _function_spec(function),
        steps,
        profile=_region_profile(steps, instructions),
        callbacks_factory=lambda submitted: _region_callbacks(submitted, instructions),
        stateful_callbacks_factory=make_callbacks,
        raw_window=lambda index: _raw_window(instructions, index),
    )


def _instruction_for_step(
    instructions: tuple[MtgsigInstruction, ...], step: VMBytecodeStep
) -> MtgsigInstruction:
    """Map a submitted step back to its decoded instruction."""
    for instruction in instructions:
        if instruction.offset == step.source.offset:
            return instruction
    return instructions[0]


def _function_metadata(function: MtgsigFunction, recovered) -> dict:
    return {
        "vm_function": function.name,
        "vm_code_offset": function.code_offset,
        "vm_code_size": function.size,
        "vm_instruction_count": len(function.instructions),
        "vm_closure_targets": list(function.referenced_names),
        "vm_decompile_status": recovered.metadata.get("decompile_status"),
    }


def _fallback_diagnostics(module: MtgsigModule) -> tuple[MtgsigDiagnostic, ...]:
    diagnostics: list[MtgsigDiagnostic] = []
    for function in module.functions:
        for instruction in function.instructions:
            entry = _HELPER_OPCODES.get(instruction.opcode)
            if entry is None:
                continue
            helper, reason = entry
            diagnostics.append(
                MtgsigDiagnostic(
                    code="vm.runtime-helper-modelling",
                    message=(
                        "%s (%s) is submitted as the neutral runtime helper %s; "
                        "every operand and the exact stack arity are preserved"
                        % (instruction.mnemonic, reason, helper)
                    ),
                    function=function.name,
                    offset=instruction.offset,
                )
            )
    return tuple(diagnostics)


def lift_module(module: FrontendModule) -> ModuleIR:
    """Submit complete VMBytecodeStep streams to core from this module's model."""
    if module.frontend_id != FRONTEND_ID:
        raise TypeError("mtgsigvm cannot lift module from %r" % (module.frontend_id,))
    decoded: MtgsigModule = module.payload

    functions = []
    for function in decoded.functions:
        recovered = lift_function(function, decoded)
        functions.append(recovered)

    if not functions:
        raise ValueError(
            "mtgsigvm: artifact %r contains no VM functions to lift" % decoded.name
        )

    diagnostics = [
        diagnostic.as_dict() for diagnostic in (*decoded.diagnostics, *_fallback_diagnostics(decoded))
    ]
    return assemble_vm_module(
        name=decoded.name,
        source_language=FRONTEND_ID,
        functions=tuple(functions),
        metadata={
            "frontend": module.metadata,
            "bytecode_format": "mtgsigvm-container",
            "vm": {
                "section_count": decoded.section_count,
                "debug_tables": decoded.debug_tables,
                "functions": len(decoded.functions),
                "constants": len(decoded.constants),
                "strings": len(decoded.strings),
                "string_xor_key": decoded.string_xor_key,
                "entry_function": decoded.entry_function.name if decoded.entry_function else None,
            },
            "diagnostics": diagnostics,
        },
    )
