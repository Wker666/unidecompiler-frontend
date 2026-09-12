"""Frontend registration for the mtgsigvm bytecode family."""

from __future__ import annotations

from unidecompiler.core.ir import ModuleIR
from unidecompiler.plugins import FrontendModule
from unidecompiler.progress import ProgressReporter, report_progress

from .decoder import decode_input, looks_like_input
from .lifter import FRONTEND_ID, lift_module
from .model import MtgsigModule
from .support import VERSION_SUPPORT


class Frontend:
    """Thin mtgsigvm frontend: decode the container, submit thin IR to core."""

    id = FRONTEND_ID
    display_name = "mtgsigvm"
    supported_inputs = (".mtgsig",)
    version_support = VERSION_SUPPORT

    def can_load(self, data: bytes, filename: str | None = None) -> bool:
        return looks_like_input(data, filename)

    def decode(self, data: bytes, filename: str | None = None) -> FrontendModule:
        module = decode_input(data, filename)
        return FrontendModule(
            frontend_id=self.id,
            payload=module,
            metadata=self._metadata(module, filename),
        )

    def lift(self, module: FrontendModule) -> ModuleIR:
        if module.frontend_id != self.id:
            raise TypeError(
                "mtgsigvm cannot lift module from %r" % (module.frontend_id,)
            )
        return lift_module(module)

    def decode_with_progress(
        self,
        data: bytes,
        filename: str | None,
        reporter: ProgressReporter,
    ) -> FrontendModule:
        report_progress(
            reporter, phase="decode", status="started", message="decoding mtgsigvm container"
        )
        module = self.decode(data, filename)
        report_progress(
            reporter,
            phase="decode",
            status="completed",
            completed=len(module.payload.functions),
            total=len(module.payload.functions),
            unit="function",
            message="decoded %d VM functions" % len(module.payload.functions),
        )
        return module

    def lift_with_progress(
        self,
        module: FrontendModule,
        reporter: ProgressReporter,
    ) -> ModuleIR:
        payload: MtgsigModule = module.payload
        total = len(payload.functions)
        report_progress(
            reporter, phase="lift", status="started", completed=0, total=total,
            unit="function", message="lifting mtgsigvm functions",
        )
        result = self.lift(module)
        report_progress(
            reporter, phase="lift", status="completed", completed=total, total=total,
            unit="function", message="lifted %d VM functions" % total,
        )
        return result

    @staticmethod
    def _metadata(module: MtgsigModule, filename: str | None) -> dict:
        return {
            "filename": filename,
            "format": "mtgsigvm-container",
            "section_count": module.section_count,
            "debug_tables": module.debug_tables,
            "functions": [function.name for function in module.functions],
            "entry_function": (module.entry_function.name if module.entry_function else None),
            "constants": len(module.constants),
            "strings": len(module.strings),
            "string_xor_key": module.string_xor_key,
            "diagnostics": [item.as_dict() for item in module.diagnostics],
        }
