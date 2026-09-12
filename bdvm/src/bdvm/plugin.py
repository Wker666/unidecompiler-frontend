from __future__ import annotations

from unidecompiler.core.ir import ModuleIR
from unidecompiler.plugins import FrontendModule
from unidecompiler.progress import ProgressReporter, report_progress

from .decoder import decode_input, looks_like_input
from .lifter import lift_module
from .support import VERSION_SUPPORT


class Frontend:
    """bdvm packed-bytecode frontend.

    Decoding and thin-IR submission only, with no VM execution or
    control-flow/source-structure recovery.
    """

    id = "bdvm"
    display_name = "bdvm packed bytecode"
    supported_inputs = (".bd",)
    version_support = VERSION_SUPPORT

    def can_load(self, data: bytes, filename: str | None = None) -> bool:
        return looks_like_input(data, filename)

    def decode(self, data: bytes, filename: str | None = None) -> FrontendModule:
        module = decode_input(data, filename)
        return FrontendModule(
            frontend_id=self.id,
            payload=module,
            metadata={
                "filename": filename,
                "format": "bdvm-packed",
                "endianness": "little",
                "debug_info_present": False,
                "diagnostics": [],
                "bdvm": {
                    "decoder": "bdvm-packed",
                    "string_pool_size": len(module.strings),
                    "function_count": len(module.functions),
                    "instruction_count": sum(
                        function.instruction_count for function in module.functions
                    ),
                    "entry_indices": list(module.entry_indices),
                    "packed_size": module.packed_size,
                    "inflated_size": module.inflated_size,
                    "xor_key": module.xor_key,
                },
            },
        )

    def lift(self, module: FrontendModule) -> ModuleIR:
        if module.frontend_id != self.id:
            raise TypeError(f"cannot lift module from {module.frontend_id!r}")
        return lift_module(module)

    def decode_with_progress(
        self,
        data: bytes,
        filename: str | None,
        reporter: ProgressReporter,
    ) -> FrontendModule:
        report_progress(
            reporter,
            phase="decode",
            status="started",
            unit="artifact",
            message="decoding bdvm packed payload",
        )
        module = self.decode(data, filename)
        payload = module.payload
        report_progress(
            reporter,
            phase="decode",
            status="completed",
            completed=len(payload.functions),
            total=len(payload.functions),
            unit="function",
            message=(
                f"decoded {len(payload.functions)} functions, "
                f"{len(payload.strings)} strings"
            ),
        )
        return module

    def lift_with_progress(
        self,
        module: FrontendModule,
        reporter: ProgressReporter,
    ) -> ModuleIR:
        if module.frontend_id != self.id:
            raise TypeError(f"cannot lift module from {module.frontend_id!r}")
        return lift_module(module, reporter=reporter)
