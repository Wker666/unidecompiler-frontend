"""Frontend-owned version support declaration."""

from __future__ import annotations

from unidecompiler.plugins import FrontendVersionSupport

#: String-pool obfuscation key of the reviewed bytecode family.
#:
#: The interpreter derives this from its own obfuscated key literal
#: ``b(673) == "3nl2kdn4f"`` by taking every odd character, parsing it in
#: radix 28 and concatenating the decimal spellings: ``n -> 23``, ``2 -> 2``,
#: ``d -> 13``, ``4 -> 4``. ``derive_string_xor_key`` reproduces that exactly
#: and is covered by tests, so the constant is evidence-backed rather than a
#: recorded guess.
DEFAULT_STRING_XOR_KEY = "232134"

#: The key literal the interpreter's decoder is seeded with.
KEY_LITERAL = "3nl2kdn4f"

#: Radix the interpreter uses when parsing key-literal characters.
KEY_RADIX = 28


def derive_string_xor_key(key_literal: str = KEY_LITERAL, radix: int = KEY_RADIX) -> str:
    """Reproduce the interpreter's string-pool key derivation."""
    out = []
    for index in range(1, len(key_literal), 2):
        char = key_literal[index]
        value = int(char, radix)
        out.append(str(value))
    return "".join(out)


VERSION_SUPPORT = FrontendVersionSupport(
    family="mtgsigvm",
    versions=("container-1",),
    parser="internal mtgsigvm container parser and thin-IR lifter",
    status="experimental",
    notes=(
        "The container carries no explicit version field; identity is decided "
        "by validating the header and section table.",
        "Scope-slot keys are supplied by the operand stack. They are resolved "
        "only when a purely local, branch-free window proves the literal; "
        "otherwise the slot stays unnamed and is reported as a diagnostic.",
        "Simulation is intentionally unsupported: this frontend only decodes "
        "and lifts, and never executes VM bytecode.",
    ),
)
