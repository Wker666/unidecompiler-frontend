from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Instruction:
    offset: int
    opcode: int
    operands: tuple[int, ...]
    size: int
    raw: str


@dataclass(frozen=True)
class Function:
    name: str
    offset: int
    instructions: tuple[Instruction, ...]
    loop_back_edges: frozenset[tuple[int, int]] = frozenset()


@dataclass(frozen=True)
class TableMutation:
    """One captured bootstrap write to the packed bytecode table."""

    source_offset: int
    target_offset: int
    before: int
    after: int


@dataclass(frozen=True)
class Program:
    filename: str | None
    version: str
    values: tuple[int, ...]
    functions: tuple[Function, ...]
    diagnostics: tuple[str, ...] = ()
    table_mutations: tuple[TableMutation, ...] = ()
