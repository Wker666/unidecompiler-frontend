from unidecompiler.plugins import FrontendModule
from unidecompiler.progress import ProgressReporter

from .decoder import decode_input, looks_like_input
from .lifter import lift_module
from .support import VERSION_SUPPORT


class Frontend:
    id = "qimeivm"
    display_name = "qimeivm"
    supported_inputs = ('.qimei',)
    version_support = VERSION_SUPPORT

    def can_load(self, data: bytes, filename: str | None = None) -> bool:
        return looks_like_input(data, filename)

    def decode(self, data: bytes, filename: str | None = None) -> FrontendModule:
        program = decode_input(data, filename)
        return FrontendModule(
            self.id,
            program,
            {
                "filename": filename,
                "format": self.id,
                "version": program.version,
                "instruction_count": sum(len(function.instructions) for function in program.functions),
                "diagnostics": program.diagnostics,
            },
        )

    def decode_with_progress(
        self,
        data: bytes,
        filename: str | None,
        reporter: ProgressReporter,
    ) -> FrontendModule:
        program = decode_input(data, filename, reporter=reporter)
        return FrontendModule(
            self.id,
            program,
            {
                "filename": filename,
                "format": self.id,
                "version": program.version,
                "instruction_count": sum(len(function.instructions) for function in program.functions),
                "diagnostics": program.diagnostics,
            },
        )

    def lift(self, module: FrontendModule):
        if module.frontend_id != self.id:
            raise TypeError(f"cannot lift module from {module.frontend_id!r}")
        return lift_module(module)

    def lift_with_progress(self, module: FrontendModule, reporter: ProgressReporter):
        if module.frontend_id != self.id:
            raise TypeError(f"cannot lift module from {module.frontend_id!r}")
        return lift_module(module, reporter=reporter)
