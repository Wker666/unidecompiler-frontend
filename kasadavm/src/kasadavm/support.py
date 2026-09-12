"""Frontend-owned version-support declaration."""

from __future__ import annotations

from unidecompiler.plugins import FrontendVersionSupport

#: The artifact family is the Kasada client VM word stream.  Dispatch decodes
#: ``u = new Proxy([...86 handlers...])``; the console is fixed at 86 opcodes
#: for every sample inspected, so a single family version is declared.
VERSION_SUPPORT = FrontendVersionSupport(
    family="kasadavm",
    versions=("wordstream-86",),
    parser="kasadavm.decoder.decode_program",
    status="experimental",
    notes=(
        "little-endian int32 word stream; word 0 is the HALT sentinel; "
        "entry program counter is 1",
        "string operand cells reference an external string table that may be "
        "absent from the artifact",
    ),
)
