"""Frontend facade: validate, decode, delegate."""

from __future__ import annotations

from unidecompiler.plugins import FrontendModule

from .decoder import decode_program, looks_like_input
from .lifter import lift_module
from .support import VERSION_SUPPORT


class Frontend:
    """Thin adapter: recognize input, decode it, delegate lifting to core.

    ``string_table`` is optional companion data.  When it is absent the
    decoder still produces a complete decode: string operands stay explicit
    ``(offset, length)`` references instead of being guessed.
    """

    id = "kasadavm"
    display_name = "Kasada VM"
    supported_inputs = (".kasada",)
    version_support = VERSION_SUPPORT

    def __init__(self, *, string_table: str | None = None) -> None:
        self._string_table = string_table

    def can_load(self, data: bytes, filename: str | None = None) -> bool:
        return looks_like_input(data, filename)

    def decode(self, data: bytes, filename: str | None = None) -> FrontendModule:
        return FrontendModule(
            self.id,
            decode_program(data, filename, string_table=self._string_table),
            {"filename": filename, "frontend": self.id},
        )

    def lift(self, module: FrontendModule):
        if module.frontend_id != self.id:
            raise TypeError(f"cannot lift module from {module.frontend_id!r}")
        return lift_module(module.payload)
