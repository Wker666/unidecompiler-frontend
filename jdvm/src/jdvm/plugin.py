"""jdvm frontend plugin registration."""
from __future__ import annotations

from unidecompiler.plugins import FrontendModule, ModuleIR

from .decoder import decode_input, looks_like_input
from .lifter import lift_module
from .support import VERSION_SUPPORT


class Frontend:
    """Thin frontend: validate, decode into a private model, and delegate lifting."""

    id = "jdvm"
    display_name = "jdvm"
    supported_inputs = (".jd",)
    version_support = VERSION_SUPPORT

    def can_load(self, data: bytes, filename: str | None = None) -> bool:
        return looks_like_input(data, filename)

    def decode(self, data: bytes, filename: str | None = None) -> FrontendModule:
        image = decode_input(data, filename)
        return FrontendModule(
            frontend_id=self.id,
            payload=image,
            metadata={
                "filename": filename,
                "format": "jdvm",
                "version": VERSION_SUPPORT.versions[0],
                "word_bytes": image.metadata.get("word_bytes"),
                "image_bytes": image.metadata.get("image_bytes"),
                "function_count": image.metadata.get("function_count"),
                "instruction_count": image.metadata.get("instruction_count"),
                "pool_size": image.metadata.get("pool_size"),
                "diagnostics": [
                    {"code": d.code, "offset": d.offset, "message": d.message}
                    for d in image.diagnostics
                ],
            },
        )

    def lift(self, module: FrontendModule) -> ModuleIR:
        if module.frontend_id != self.id:
            raise TypeError(f"jdvm frontend cannot lift module from {module.frontend_id!r}")
        return lift_module(module)
