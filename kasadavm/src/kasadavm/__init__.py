"""kasadavm — a thin unidecompiler VM frontend for the Kasada client VM.

The package decodes a little-endian ``int32`` word-stream artifact and submits
one neutral ``VMBytecodeStep`` per decoded instruction.  It never executes the
VM, never builds CFG or AST structures, and never contains source-language
special cases.
"""

from .decoder import decode_program, looks_like_input
from .lifter import lift_module
from .plugin import Frontend

__all__ = ["Frontend", "decode_program", "lift_module", "looks_like_input"]
