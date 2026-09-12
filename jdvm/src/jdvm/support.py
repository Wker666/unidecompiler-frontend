"""Version support declaration for the jdvm frontend."""
from __future__ import annotations

from unidecompiler.plugins import FrontendVersionSupport

VERSION_SUPPORT = FrontendVersionSupport(
    family="jdvm",
    versions=(
        "js_security_v3_main (m.jd.com) — 5126-word image, XOR 0x2e string pool",
    ),
    parser=(
        "flat signed 32-bit word image; per-function opcode tables and the "
        "decoded string pool are derived from the interpreter by "
        "tools/derive_vm_tables.py and frozen into src/jdvm/vm_definition.json"
    ),
    status="experimental",
    notes=(
        "jdvm permutes its opcode numbers per function, so an opcode word is "
        "only meaningful together with the function that owns it. The frozen "
        "definition records every (function entry, opcode) pair.",
        "A different VM build changes the image size, the string-pool XOR key, "
        "and the per-function opcode tables. Those builds need a re-derived "
        "definition, not a frontend change.",
        "The sibling js_security_v3_0.1.6.js build (XOR 0x05, 5182-word image, "
        "36 functions) shares the same canonical operation set and is not yet "
        "declared here.",
    ),
)
