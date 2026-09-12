from unidecompiler.plugins import FrontendVersionSupport

VERSION_SUPPORT = FrontendVersionSupport(
    family="bdvm",
    versions=("1",),
    parser="internal packed-payload decoder (base64 + xor + raw-deflate varint stream)",
    status="experimental",
    notes=(
        "Payloads are the packed base64 form produced by the module loader "
        "J() in the reference interpreter; opcode semantics follow the "
        "dispatch tree of the interpreter entry X().",
    ),
)
