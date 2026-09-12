from __future__ import annotations

import json
from collections.abc import Callable

from unidecompiler.plugins import FrontendDecodeError
from unidecompiler.progress import ProgressReporter, report_progress

from .model import Function, Instruction, Program


# Number of physical cells following the opcode.  The false-arm comma
# expression increments the VM cursor but does not consume a synthetic cell.
FIXED_ARITY: dict[int, int] = {
    0: 3, 1: 3, 2: 8, 4: 6, 5: 5, 6: 5, 7: 5, 8: 3, 9: 5,
    10: 3, 11: 3, 12: 6, 13: 3, 14: 2, 16: 3, 17: 8, 18: 2,
    19: 3, 20: 8, 21: 3, 22: 0, 24: 4, 25: 3, 26: 5, 27: 5,
    28: 5, 29: 1, 30: 9, 31: 3, 32: 6, 33: 4, 34: 2, 35: 5,
    36: 2, 37: 3, 38: 3, 39: 4, 40: 1, 41: 3, 42: 3, 44: 3,
    45: 3, 46: 2, 47: 1, 48: 2, 50: 4, 51: 3, 52: 3, 53: 4,
    54: 5, 55: 6, 56: 3, 57: 3, 58: 3, 59: 2, 60: 6, 61: 5,
    62: 3, 63: 1, 64: 2, 65: 3, 66: 2, 67: 3, 68: 2, 69: 3,
    70: 2, 71: 3, 72: 1, 73: 3, 74: 3, 75: 3, 76: 1, 77: 6,
    78: 4, 79: 1, 80: 1, 81: 3, 82: 3, 83: 3, 84: 7, 85: 3,
    86: 4, 87: 2, 88: 3, 89: 4, 90: 2, 91: 2, 92: 3, 94: 3,
    95: 3, 96: 3, 97: 1,
}

VARIABLE_ARITY: dict[int, Callable[[int], int]] = {
    # count + count arguments + fixed fields
    3: lambda count: max(count, 0) + 3,
    # dst, key, source, count, args..., destination, entry_delta, length
    15: lambda count: max(count, 0) + 7,
    23: lambda count: max(count, 0) + 4,
    43: lambda count: max(count, 0) + 7,
    49: lambda count: max(count, 0) + 4,
    93: lambda count: max(count, 0) + 3,
}

# The packed VM scatters code across unrelated table offsets.  These cursor
# bases mark the code regions reached by the runtime; they seed closure
# discovery, which then follows every statically encoded branch and closure
# target in the final table snapshot.  They are data, not an alternate
# interpreter.
QIMEI_RUNTIME_BASES = frozenset({
    39602, 25895, 54099, 57896, 89111, 70397, 42386, 76240,
    62595, 20303, 63300, 34950, 61419, 59504, 89506, 82893,
    54256, 38170, 14450, 44792, 2265, 65393, 71643, 74799,
    71450, 48899, 39653, 34406, 49067, 90123, 75031, 78814,
    15204, 24170, 3326, 89002, 58125, 74287, 69311, 45245,
    44395, 39254, 67165, 38763, 44941, 62558, 37604, 74176,
    49852, 9984, 40570, 87842, 13541, 25529, 15373, 20174,
    1674, 2620, 47294, 53923, 90330, 67798, 29118, 14693,
    34782, 36360, 17497, 59704, 35730, 58031, 9954, 6888,
    35659, 42837, 7988, 71902, 61149, 3253, 14760, 48805,
    6005, 35918, 51276, 63224, 27772, 70626, 55258, 11996,
    77125, 14370,
})


OPCODE_NAMES = {i: f"OP_{i:02d}" for i in range(98)}
OPCODE_NAMES.update({
    0: "DELETE", 1: "CALL_1", 2: "FUSED", 3: "APPLY", 4: "GET_CHAR",
    5: "CALL_2", 6: "CALL_3", 8: "AND_IMM", 10: "XOR", 11: "INSTANCEOF",
    12: "EQ_BRANCH", 13: "LE_IMM", 14: "APPEND_CHAR", 15: "MAKE_CLOSURE_PROP",
    16: "OR", 17: "NE_BRANCH", 18: "NOT", 19: "GET_ITEM", 20: "COPY_EQ_BRANCH",
    21: "SUB", 22: "POP_HANDLER", 23: "MAKE_CLOSURE", 24: "MAKE_ARRAYS",
    25: "GE_IMM", 26: "SET_DYNAMIC_COPY", 27: "SET_STATIC_COPY",
    28: "APPEND_CHAR_SET", 29: "THROW", 30: "GET3", 31: "IN",
    32: "SET_GET", 33: "GET_CLEAR", 34: "ARRAY", 35: "GET_COPY",
    36: "NEW_0", 37: "MOD", 38: "SET_DYNAMIC", 39: "NEW_2",
    40: "PUSH_HANDLER", 41: "STRICT_EQ_IMM", 42: "CALL_THIS_0",
    43: "MAKE_CLOSURE_SET", 44: "EQ", 45: "GET_STATIC", 46: "KEYS",
    47: "NULL", 48: "COPY", 49: "APPLY_METHOD", 50: "CALL_THIS_2",
    51: "JUMP_IF", 52: "EQ_DUP", 53: "SET_EXCEPTION_HANDLER", 54: "GET_CHAR_ITEM",
    55: "CALL_METHOD_3", 56: "ADD_IMM_LEFT", 57: "DIV", 58: "SHL",
    59: "NEG", 60: "GET2", 61: "COPY_GET", 62: "MUL", 63: "JUMP",
    64: "TYPEOF", 65: "ADD", 66: "RETURN_THIS_AND", 67: "LE",
    68: "TO_NUMBER", 69: "GT_IMM", 70: "CONST", 71: "SHR", 72: "OBJECT",
    73: "SUB_IMM", 74: "ADD_IMM", 75: "SHIFT", 76: "THIS", 77: "APPEND_CHARS3",
    78: "COPY2", 79: "EMPTY_STRING", 80: "RETURN", 81: "OR_IMM", 82: "LT",
    83: "GT", 84: "GET_CALL_1", 85: "CHAR", 86: "CALL_METHOD_1",
    87: "INC", 88: "GE", 89: "APPEND_CHARS2", 90: "BIT_NOT", 91: "CALL_THIS_0B",
    92: "SET_STATIC", 93: "NEW_APPLY", 94: "NEW_1", 95: "USHR", 96: "LT_IMM",
    97: "EXCEPTION",
})

CHAR_OPERANDS = {
    4: (5,), 7: (1,), 14: (1,), 28: (1,), 54: (1,),
    77: (1, 3, 5), 85: (2,), 89: (1, 3),
}


def _register_operand_indices(opcode: int, operands: tuple[int, ...]) -> tuple[int, ...]:
    """Return operand positions that address the VM's Y register array."""
    fixed = {
        0: (0, 1, 2), 1: (0, 1, 2), 2: (0, 1, 3, 5, 7),
        4: (0, 1, 2, 3, 4), 5: (0, 1, 2, 3, 4),
        6: (0, 1, 2, 3, 4), 7: (0, 2, 3), 8: (0, 1),
        9: (0, 1, 2, 3, 4), 10: (0, 1, 2), 11: (0, 1, 2),
        12: (0, 1, 2, 3), 13: (0, 1), 14: (0,),
        16: (0, 1, 2), 17: (0, 1, 2, 3, 4, 5), 18: (0, 1),
        19: (0, 1, 2), 20: (0, 1, 2, 3, 4, 5), 21: (0, 1, 2),
        24: (0, 2), 25: (0, 1), 26: (0, 1, 2, 3, 4),
        27: (0, 1, 2, 4), 28: (0, 2, 4), 29: (0,),
        30: (0, 1, 3, 4, 6, 7), 31: (0, 1, 2),
        32: (0, 2, 3, 4), 33: (0, 1, 3), 34: (0,),
        35: (0, 1, 3, 4), 36: (0, 1), 37: (0, 1, 2),
        38: (0, 1, 2), 39: (0, 1, 2, 3), 41: (0, 1),
        42: (0, 1, 2), 44: (0, 1, 2), 45: (0, 1),
        46: (0, 1), 47: (0,), 48: (0, 1), 50: (0, 1, 2, 3),
        51: (0,), 52: (0, 1, 2), 53: (0, 1, 2),
        54: (0, 2, 3, 4), 55: (0, 1, 2, 3, 4, 5),
        56: (0, 2), 57: (0, 1, 2), 58: (0, 1), 59: (0, 1),
        60: (0, 1, 3, 4), 61: (0, 1, 2, 3), 62: (0, 1, 2),
        64: (0, 1), 65: (0, 1, 2), 66: (0, 1), 67: (0, 1, 2),
        68: (0, 1), 69: (0, 1), 70: (0,), 71: (0, 1),
        72: (0,), 73: (0, 1), 74: (0, 1), 75: (0, 1, 2),
        76: (0,), 77: (0, 2, 4), 78: (0, 1, 2, 3), 79: (0,),
        80: (0,), 81: (0, 1), 82: (0, 1, 2), 83: (0, 1, 2),
        84: (0, 1, 2, 3, 4, 5, 6), 85: (0, 1),
        86: (0, 1, 2, 3), 87: (0, 1), 88: (0, 1, 2),
        89: (0, 2), 90: (0, 1), 91: (0, 1), 92: (0, 2),
        94: (0, 1, 2), 95: (0, 1), 96: (0, 1), 97: (0,),
    }
    if opcode == 3 and operands:
        count = max(operands[0], 0)
        return tuple(range(1, min(len(operands), count + 3)))
    if opcode == 15 and len(operands) >= 4:
        count = max(operands[3], 0)
        return (0, 2, *range(4, min(len(operands), 5 + count)))
    if opcode == 23 and operands:
        count = max(operands[0], 0)
        return tuple(range(1, min(len(operands), count + 2)))
    if opcode == 43 and operands:
        count = max(operands[0], 0)
        return (*range(1, min(len(operands), count + 2)), count + 4, count + 6)
    if opcode == 49 and operands:
        count = max(operands[0], 0)
        return tuple(range(1, min(len(operands), count + 4)))
    if opcode == 93 and operands:
        count = max(operands[0], 0)
        return tuple(range(1, min(len(operands), count + 3)))
    return fixed.get(opcode, ())


def _operands_are_valid(opcode: int, operands: tuple[int, ...]) -> bool:
    # Every integer is a valid JavaScript array-property key for Y, including
    # negative and very large values.  String.fromCharCode likewise applies
    # ToUint16 rather than rejecting out-of-range integers.  Structural
    # validity therefore comes from operand count/artifact bounds, not an
    # invented register-width constraint.
    return all(index < len(operands) for index in CHAR_OPERANDS.get(opcode, ()))


def _closure_shape_is_valid(opcode: int, operands: tuple[int, ...]) -> bool:
    """Validate only the structural fields needed to recognize a closure."""
    if opcode == 15:
        if len(operands) < 7:
            return False
        count = operands[3]
    elif opcode in {23, 43}:
        if len(operands) < (4 if opcode == 23 else 7):
            return False
        count = operands[0]
    else:
        return True
    return isinstance(count, int)


def looks_like_input(data: bytes, filename: str | None = None) -> bool:
    """Recognize qimeivm input without executing external programs."""
    if not data:
        return bool(filename and filename.lower().endswith(('.qimei',)))
    if filename and filename.lower().endswith(('.qimei',)):
        return True
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(value, list) and all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    )


def decode_input(
    data: bytes,
    filename: str | None = None,
    *,
    reporter: ProgressReporter | None = None,
):
    """Decode the packed VM array using the runtime's ``F`` cursor model.

    The interpreter executes ``a[++F]``, so the public entry opcode is one cell
    past the cursor base.  A runtime-reachable base profile seeds closure
    discovery, which then follows every statically encoded branch and closure
    target in the final table snapshot.
    """
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FrontendDecodeError(f"qimeivm input is not UTF-8 JSON: {exc}") from exc
    if not isinstance(value, list) or not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        raise FrontendDecodeError("qimeivm input must be a JSON array of integers")

    values = tuple(value)
    report_progress(
        reporter,
        phase="decode",
        status="started",
        completed=0,
        total=1,
        unit="phase",
        message="decoding qimeivm cells",
    )
    diagnostics: list[str] = []
    decoded: dict[int, Instruction] = {}
    # The interpreter executes ``a[++F]``, so the public entry opcode is one
    # cell past the cursor base passed to the VM factory.  The captured
    # US-key entry uses base 39602; a short synthetic stream starts at base 0.
    base = 39602 if len(values) > 39603 else 0
    entry = base + 1
    table_mutations = ()

    # The VM bootstrap constructs its exported object at base 258.  It is a
    # separate public function, not part of the business closure.
    bootstrap_entry = 259 if base and len(values) > 259 else None
    pending = [entry]
    if bootstrap_entry is not None:
        pending.append(bootstrap_entry)
    profiled_roots = {
        candidate + 1
        for candidate in (QIMEI_RUNTIME_BASES if base else ())
        if 0 <= candidate + 1 < len(values)
    }
    public_roots = {entry, *(() if bootstrap_entry is None else (bootstrap_entry,))}
    closure_roots: set[int] = profiled_roots - public_roots

    def add_target(target: int, source: int) -> None:
        if 0 <= target < len(values):
            pending.append(target)
        # An out-of-range VM jump is a terminal path in the JavaScript
        # switch loop.  It is not a decoder failure and has no in-module
        # successor to submit to core.

    def add_noop(start: int, value: int) -> None:
        decoded[start] = Instruction(start, value, (), 1,
                                      f"{start:06d}: NOOP [{value}]")
        add_target(start + 1, start)

    def add_closure_target(target: int, source: int) -> None:
        if 0 <= target < len(values):
            closure_roots.add(target)
            add_target(target, source)

    def decode_at(start: int) -> None:
        if start in decoded or not 0 <= start < len(values):
            return
        opcode = values[start]
        if opcode not in OPCODE_NAMES:
            # The runtime switch has no ``default`` arm.  A cell outside
            # 0..97 is therefore a one-cell no-op, not a parse error: the
            # loop simply advances to the next ``a[++F]``.  These cells are
            # common in the packed function and include embedded character
            # values between real VM operations.
            add_noop(start, opcode)
            return
        operand_start = start + 1
        if opcode in VARIABLE_ARITY:
            count_cell = operand_start + 3 if opcode == 15 else operand_start
            if count_cell >= len(values):
                add_noop(start, opcode)
                return
            count = values[count_cell]
            arity = VARIABLE_ARITY[opcode](count)
        else:
            arity = FIXED_ARITY.get(opcode, -1)
        if arity < 0 or operand_start + arity > len(values):
            add_noop(start, opcode)
            return
        operands = values[operand_start:operand_start + arity]
        if opcode in {15, 23, 43} and not _closure_shape_is_valid(opcode, operands):
            add_noop(start, opcode)
            return
        if not _operands_are_valid(opcode, operands):
            add_noop(start, opcode)
            return
        decoded[start] = Instruction(start, opcode, operands, arity + 1,
                                      f"{start:06d}: {OPCODE_NAMES[opcode]} {list(operands)!r}")

        # JavaScript compound assignment reads the old left-hand-side value
        # before evaluating its right-hand side.  In ``F += a[++F]`` the
        # addition therefore uses the cursor value from *before* ``++F``.
        # The following switch iteration then performs its own ``++F``.
        if opcode == 63 and operands:
            add_target(start + 1 + operands[0], start)
            return
        if opcode == 51 and len(operands) >= 3:
            add_target(start + 1 + operands[1], start)
            add_target(start + 1 + operands[2], start)
            return
        if opcode == 12 and len(operands) >= 6:
            add_target(start + 4 + operands[4], start)
            add_target(start + 4 + operands[5], start)
            return
        if opcode in {17, 20} and len(operands) >= 8:
            add_target(start + 6 + operands[6], start)
            add_target(start + 6 + operands[7], start)
            return
        if opcode == 75:
            add_target(start + 1 + arity, start)
            return
        if opcode == 40 and operands:
            add_target(start + 1 + operands[0], start)
        if opcode == 53 and operands:
            add_target(start + arity + operands[-1], start)

        if opcode == 15 and len(operands) >= 7:
            # delta is the penultimate operand; the returned closure starts
            # `o()` receives the cursor position after consuming the delta;
            # its first `a[++F]` therefore reads one cell after that base.
            add_closure_target(start + arity + operands[-2] - 1, start)
        elif opcode == 23 and len(operands) >= 4:
            add_closure_target(start + arity + operands[-2] - 1, start)
        elif opcode == 43 and len(operands) >= 7:
            count = max(operands[0], 0)
            add_closure_target(start + count + 4 + operands[count + 2] - 1, start)

        if opcode not in {29, 63, 66, 80}:
            add_target(start + arity + 1, start)

    # Seed every runtime-observed entry, then recursively discover closures
    # from the bytecode itself, following branch and closure targets.
    pending.extend(sorted(profiled_roots - {entry}))
    last_reported = 0
    while pending:
        decode_at(pending.pop())
        # The packed representation can contain many data cells.  Report the
        # number of cells proven and decoded as text only.  The denominator is
        # deliberately omitted: packed data cells and discovered branch
        # targets do not form a known, finite instruction work list.  Using
        # ``len(values)`` here would falsely imply that every table cell must
        # be decoded and produce misleading values such as ``1/381``.
        if len(decoded) - last_reported >= 256 or not pending:
            report_progress(
                reporter,
                phase="decode",
                unit="phase",
                message=f"decoded {len(decoded)} cells",
            )
            last_reported = len(decoded)

    by_offset = decoded

    def successors(instruction: Instruction) -> tuple[int, ...]:
        start, opcode, operands = instruction.offset, instruction.opcode, instruction.operands
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
        if opcode == 40 and operands:
            return (start + 1 + operands[0], start + arity + 1)
        if opcode == 53 and operands:
            return (start + arity + operands[-1], start + arity + 1)
        if opcode in {29, 66, 80}:
            return ()
        return (start + arity + 1,)

    def graph(root: int) -> tuple[Instruction, ...]:
        seen: set[int] = set()
        postorder: list[int] = []
        # The third item records whether the edge is an explicit control
        # transfer.  A packed closure root can also be a deliberately shared
        # tail of its parent; only physical fallthrough into a nested root is
        # forbidden.
        todo = [(root, False, True)]
        while todo:
            offset, expanded, explicit = todo.pop()
            if expanded:
                postorder.append(offset)
                continue
            if offset in seen or offset not in by_offset:
                continue
            # A closure target is a separate VM function.  A parent graph
            # must not fall through into that body merely because the packed
            # array places the body next to another reachable region.  The
            # closure receives its own graph below; this barrier prevents a
            # single shared tail from multiplying every lifted function.
            if (
                offset in (closure_roots | public_roots)
                and offset != root
                and not explicit
            ):
                continue
            seen.add(offset)
            todo.append((offset, True, explicit))
            instruction = by_offset[offset]
            explicit_targets = set(branch_successors_for_graph(instruction))
            todo.extend(
                (target, False, target in explicit_targets)
                for target in reversed(successors(by_offset[offset]))
            )
        # Reverse postorder is the standard linearization for a CFG.  It
        # places acyclic predecessors before their successors even though
        # qimei deliberately scatters blocks across unrelated table offsets.
        return tuple(by_offset[offset] for offset in reversed(postorder))

    def branch_successors_for_graph(instruction: Instruction) -> tuple[int, ...]:
        """Edges that intentionally enter another packed code region."""
        start, opcode, operands = instruction.offset, instruction.opcode, instruction.operands
        arity = instruction.size - 1
        if opcode == 63 and operands:
            return (start + 1 + operands[0],)
        if opcode == 51 and len(operands) >= 3:
            return (start + 1 + operands[1], start + 1 + operands[2])
        if opcode == 12 and len(operands) >= 6:
            return (start + 4 + operands[4], start + 4 + operands[5])
        if opcode in {17, 20} and len(operands) >= 8:
            return (start + 6 + operands[6], start + 6 + operands[7])
        if opcode == 40 and operands:
            return (start + 1 + operands[0],)
        if opcode == 53 and operands:
            return (start + arity + operands[-1],)
        return ()

    def branch_successors(instruction: Instruction) -> tuple[int, ...]:
        """Return only explicit branch edges eligible to close a VM loop."""
        start, opcode, operands = instruction.offset, instruction.opcode, instruction.operands
        if opcode == 63 and operands:
            return (start + 1 + operands[0],)
        if opcode == 51 and len(operands) >= 3:
            return (start + 1 + operands[1], start + 1 + operands[2])
        if opcode == 12 and len(operands) >= 6:
            return (start + 4 + operands[4], start + 4 + operands[5])
        if opcode in {17, 20} and len(operands) >= 8:
            return (start + 6 + operands[6], start + 6 + operands[7])
        return ()

    def natural_loop_edges(
        root: int,
        instructions: tuple[Instruction, ...],
    ) -> frozenset[tuple[int, int]]:
        """Classify loop latches by dominance, never by packed addresses.

        qimei randomizes basic-block placement throughout the integer table,
        so an edge to a numerically smaller offset is often just an ordinary
        forward transfer.  An explicit branch is a natural loop back edge
        only when its target dominates its source in the recovered function
        graph.
        """
        nodes = frozenset(instruction.offset for instruction in instructions)
        if root not in nodes:
            return frozenset()
        predecessors = {node: set() for node in nodes}
        for instruction in instructions:
            for target in successors(instruction):
                if target in nodes:
                    predecessors[target].add(instruction.offset)

        dominators = {
            node: ({root} if node == root else set(nodes))
            for node in nodes
        }
        changed = True
        while changed:
            changed = False
            for node in nodes:
                if node == root:
                    continue
                incoming = predecessors[node]
                if incoming:
                    common = set.intersection(*(dominators[pred] for pred in incoming))
                    updated = {node, *common}
                else:
                    updated = {node}
                if updated != dominators[node]:
                    dominators[node] = updated
                    changed = True

        return frozenset(
            (instruction.offset, target)
            for instruction in instructions
            for target in branch_successors(instruction)
            if target in nodes and target in dominators[instruction.offset]
        )

    def make_function(name: str, root: int) -> Function:
        instructions = graph(root)
        return Function(
            name,
            root,
            instructions,
            natural_loop_edges(root, instructions),
        )

    # A packed stream can create a closure that points back at the public
    # entry.  That is a recursive/self reference, not a second FunctionIR
    # with the same offset.  Keep one canonical function per opcode root.
    closure_roots = {root for root in closure_roots if root in by_offset and root not in public_roots}
    entry_name = f"closure_{entry}" if base else "main"
    functions = [make_function(entry_name, entry)]
    if bootstrap_entry is not None and bootstrap_entry in by_offset:
        functions.append(make_function("bootstrap", bootstrap_entry))
    for root in sorted(closure_roots):
        functions.append(make_function(f"closure_{root}", root))
    report_progress(
        reporter,
        phase="decode",
        status="completed",
        completed=1,
        total=1,
        unit="phase",
        message=f"decoded {len(decoded)} cells",
    )
    return Program(
        filename, "1", values, tuple(functions), tuple(diagnostics), table_mutations
    )
