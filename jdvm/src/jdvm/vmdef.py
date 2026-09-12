"""Static VM definition for the jdvm frontend.

The jdvm interpreter permutes its opcode numbers *per function*: the same
opcode word means different things in different functions, so a decoded opcode
is only meaningful together with the function that owns it. Core requires
VM-neutral facts, so this module resolves that indirection once:

    (function entry PC, opcode word)  ->  canonical semantic operation

The canonical operation set (68 entries) comes from
``analysis_inputs/interpreter/sec3.js`` and is frozen into
``vm_definition.json`` by ``tools/derive_vm_tables.py``. Nothing here executes
the interpreter; this is static decode data only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

_DEFINITION_RESOURCE = "vm_definition.json"

#: Canonical handler source text -> semantic operation name.
#: Keying by source text (not by array position) means a re-derived definition
#: can never silently renumber semantics.
CANONICAL_SEMANTICS: dict[str, str] = {
    'return STK.pop();': 'return-value',
    'if(STK.pop())++PC;else PC+=BC[PC];': 'branch-false-pop',
    'STK.push(STK[STK.length-1]);STK[STK.length-2]=STK[STK.length-2][POOL[BC[PC++]]];': 'dup-load-member-raw',
    'STK.push(G);': 'load-symbol',
    'STK[STK.length-1]=STK[STK.length-1].length;': 'load-length',
    'G=STK[STK.length-1];': 'store-symbol-peek',
    'PC+=BC[PC];': 'jump',
    'STK[STK.length-4]=G.call(STK[STK.length-4],STK[STK.length-3],STK[STK.length-2],STK[STK.length-1]);STK.length-=3;': 'invoke-call-3',
    'G=STK.pop();STK[STK.length-1]=STK[STK.length-1]>G;': 'binary-gt',
    'STK.push(null);': 'push-null',
    'STK.push(new G(BC[PC++]));': 'construct-count',
    'return;': 'return-void',
    'STK[STK.length-5]=G.call(STK[STK.length-5],STK[STK.length-4],STK[STK.length-3],STK[STK.length-2],STK[STK.length-1]);STK.length-=4;': 'invoke-call-4',
    'STK.pop();': 'pop',
    'STK[STK.length-1]=STK[STK.length-1][POOL[BC[PC++]]];': 'load-member-raw',
    'G=STK.pop();STK[STK.length-1]+=G;': 'add-assign',
    'STK.push(this);': 'push-this',
    'STK.push(BC[PC++]);': 'push-immediate',
    'if(STK[STK.length-2]!=null){STK[STK.length-3]=G.call(STK[STK.length-3],STK[STK.length-2],STK[STK.length-1]);STK.length-=2;}else{G=STK[STK.length-3];STK[STK.length-3]=G(STK[STK.length-1]);STK.length-=2;}': 'invoke-call-2',
    'STK[STK.length-2][POOL[@+BC[PC++]]]=STK[STK.length-1];STK[STK.length-2]=STK[STK.length-1];STK.length--;': 'store-member-dup',
    'STK.push(STK[STK.length-1]);STK[STK.length-2]=STK[STK.length-2][POOL[@+BC[PC++]]];': 'dup-load-member',
    'if(STK[STK.length-1]){++PC;--STK.length;}else PC+=BC[PC];': 'branch-false-keep',
    'STK[STK.length-1]=typeof STK[STK.length-1];': 'typeof',
    'STK[STK.length-1]=STK[STK.length-1][POOL[@+BC[PC++]]];': 'load-member',
    'STK.push(STK[STK.length-1]);': 'duplicate',
    'G=STK.pop();STK[STK.length-1]=STK[STK.length-1]==G;': 'binary-eq',
    'STK.push(this[POOL[@+BC[PC++]]]);': 'push-this-member',
    'STK.push(POOL[@+BC[PC++]]);': 'push-pool',
    'G=STK.pop();STK[STK.length-1]%=G;': 'mod-assign',
    'G=STK.pop();STK[STK.length-1]/=G;': 'div-assign',
    'STK.push(G++);': 'load-symbol-postinc',
    'if(STK.pop())PC+=BC[PC];else++PC;': 'branch-true-pop',
    'G=STK.pop();STK[STK.length-1]*=G;': 'mul-assign',
    'G=STK.pop();STK[STK.length-1]=STK[STK.length-1]===G;': 'binary-strict-eq',
    'G=STK.pop();STK[STK.length-1]=STK[STK.length-1]<G;': 'binary-lt',
    'G=STK.pop();STK[STK.length-1]-=G;': 'sub-assign',
    'STK[STK.length-2]=new STK[STK.length-2]();STK.length-=1;': 'construct-stack0',
    'G=STK.pop();STK[STK.length-1]^=G;': 'xor-assign',
    'STK[STK.length-2]=STK[STK.length-2][STK[STK.length-1]];STK.length--;': 'load-item',
    'STK[STK.length-3][STK[STK.length-2]]=STK[STK.length-1];STK[STK.length-3]=STK[STK.length-1];STK.length-=2;': 'store-item-dup',
    'if(STK[STK.length-1]!=null){STK[STK.length-2]=G.call(STK[STK.length-2],STK[STK.length-1]);}else{G=STK[STK.length-2];STK[STK.length-2]=G();}STK.length--;': 'invoke-call-1',
    'STK.push(STK[STK.length-2]);STK.push(STK[STK.length-2]);': 'duplicate2',
    'STK.push(undefined);': 'push-undefined',
    'STK.push({});': 'push-object',
    'STK.push(CLOSURE)': 'push-closure',
    'STK[STK.length-2][POOL[@+BC[PC++]]]=STK[STK.length-1];STK.length--;': 'store-member',
    'STK.push(0);': 'push-zero',
    'STK.push(--G);': 'load-symbol-predec',
    'STK.push(G--);': 'load-symbol-postdec',
    'G=STK.pop();STK[STK.length-1]|=G;': 'or-assign',
    'STK[STK.length-1]=-STK[STK.length-1];': 'negate',
    'STK[STK.length-3][STK[STK.length-2]]=STK[STK.length-1];STK.length-=2;': 'store-item',
    'STK.push(1);': 'push-one',
    'STK[STK.length-6]=G.call(STK[STK.length-6],STK[STK.length-5],STK[STK.length-4],STK[STK.length-3],STK[STK.length-2],STK[STK.length-1]);STK.length-=5;': 'invoke-call-5',
    'STK[STK.length-3]=new STK[STK.length-3](STK[STK.length-1]);STK.length-=2;': 'construct-stack1',
    'STK[STK.length-1]=!STK[STK.length-1];': 'logical-not',
    'G=STK.pop();for(G=0;G<BC[PC+1];++G)if(G===POOL[@+BC[PC+G*2+2]]){PC+=BC[PC+G*2+3];continue G;}PC+=BC[PC];': 'switch-multiway',
    'STK.push(typeof G);': 'typeof-symbol',
    'if(STK[STK.length-1])PC+=BC[PC];else{++PC;--STK.length;}': 'branch-true-keep',
    'G=STK.pop();STK[STK.length-1]=STK[STK.length-1]!==G;': 'binary-strict-ne',
    'G=BC[PC++];STK.push(new G(POOL[@+G],POOL[@+G+1]));': 'construct-pool2',
    'G=STK.pop();STK[STK.length-1]=STK[STK.length-1]!=G;': 'binary-ne',
    'STK[STK.length-1]=undefined;': 'store-undefined',
    'G=STK.pop();STK[STK.length-1]=STK[STK.length-1]in G;': 'binary-in',
    'STK.push(new G(POOL[@+BC[PC++]]));': 'construct-pool1',
    'G=STK.pop();STK[STK.length-1]=STK[STK.length-1]>=G;': 'binary-ge',
    'STK[STK.length-7]=G.call(STK[STK.length-7],STK[STK.length-6],STK[STK.length-5],STK[STK.length-4],STK[STK.length-3],STK[STK.length-2],STK[STK.length-1]);STK.length-=6;': 'invoke-call-6',
    'STK[STK.length-8]=G.call(STK[STK.length-8],STK[STK.length-7],STK[STK.length-6],STK[STK.length-5],STK[STK.length-4],STK[STK.length-3],STK[STK.length-2],STK[STK.length-1]);STK.length-=7;': 'invoke-call-7',
}


class VmDefinitionError(RuntimeError):
    """Raised when the frozen VM definition cannot be interpreted."""


@dataclass(frozen=True)
class OpcodeCell:
    """One opcode word as understood inside one function."""

    opcode: int
    semantic: str
    canonical_index: int
    operand_words: int
    pool_base: int
    closure_entry: int | None
    symbols: tuple[str, ...]

    @property
    def is_variable_length(self) -> bool:
        return self.operand_words < 0


@dataclass(frozen=True)
class FunctionDefinition:
    """One VM function root and its opcode table."""

    entry: int
    end: int
    name: str
    locals: tuple[str, ...]
    opcodes: dict[int, OpcodeCell]

    def cell(self, opcode: int) -> OpcodeCell | None:
        return self.opcodes.get(opcode)

    def __len__(self) -> int:
        return self.end - self.entry


@dataclass(frozen=True)
class VmDefinition:
    """The decoded, immutable description of the jdvm bytecode family."""

    function_id: str
    family: str
    word_bytes: int
    bytecode_words: int
    decoder_name: str
    decoder_xor: int
    decoder_escape: int
    decoder_threshold: int
    pool: tuple[str, ...]
    canonical: tuple[str, ...]
    functions: tuple[FunctionDefinition, ...]

    @property
    def image_bytes(self) -> int:
        return self.bytecode_words * self.word_bytes

    def function_for(self, offset: int) -> FunctionDefinition | None:
        """Return the function whose region contains ``offset``."""
        for function in self.functions:
            if function.entry <= offset < function.end:
                return function
        return None

    def pool_at(self, base: int, delta: int) -> str | None:
        index = base + delta
        if 0 <= index < len(self.pool):
            return self.pool[index]
        return None

    def semantics_for(self, canonical_index: int) -> str:
        return self._semantic_names[canonical_index]


def _load() -> VmDefinition:
    raw = json.loads(
        files(__package__).joinpath(_DEFINITION_RESOURCE).read_text(encoding="utf-8")
    )
    canonical = tuple(entry["source"] for entry in raw["canonical_ops"])
    names: list[str] = []
    for index, source in enumerate(canonical):
        try:
            names.append(CANONICAL_SEMANTICS[source])
        except KeyError as exc:  # pragma: no cover - guards interpreter drift
            raise VmDefinitionError(
                f"canonical operation {index} is not classified: {source!r}"
            ) from exc

    functions: list[FunctionDefinition] = []
    for entry in raw["functions"]:
        cells: dict[int, OpcodeCell] = {}
        for opcode_text, cell in entry["ops"].items():
            operand_words = cell["operand_words"]
            cells[int(opcode_text)] = OpcodeCell(
                opcode=int(opcode_text),
                semantic=names[cell["canon"]],
                canonical_index=cell["canon"],
                operand_words=-1 if operand_words is None else operand_words,
                pool_base=cell["pool_base"] or 0,
                closure_entry=cell["closure_entry"],
                symbols=tuple(cell["symbols"]),
            )
        functions.append(
            FunctionDefinition(
                entry=entry["entry"],
                end=entry["end"],
                name=entry["name"],
                locals=tuple(entry["locals"]),
                opcodes=cells,
            )
        )
    definition = VmDefinition(
        function_id="jdvm",
        family="jdvm",
        word_bytes=raw["word_bytes"],
        bytecode_words=raw["bytecode_len"],
        decoder_name=raw["decoder"]["name"],
        decoder_xor=raw["decoder"]["xor"],
        decoder_escape=raw["decoder"]["escape"],
        decoder_threshold=raw["decoder"]["threshold"],
        pool=tuple(raw["pool"]),
        canonical=canonical,
        functions=tuple(functions),
    )
    object.__setattr__(definition, "_semantic_names", tuple(names))
    return definition


@lru_cache(maxsize=1)
def vm_definition() -> VmDefinition:
    """Return the frozen jdvm VM definition (cached, immutable)."""
    return _load()


def decode_pool_string(encoded: str) -> str:
    """Decode one string-pool literal with the interpreter's own decoder.

    Exposed for tests and tooling that verify the frozen pool against the
    interpreter source without executing it.
    """
    definition = vm_definition()
    out: list[str] = []
    index = 0
    while index < len(encoded):
        code = ord(encoded[index])
        index += 1
        if code > definition.decoder_threshold:
            out.append(chr(code ^ definition.decoder_xor))
        elif code == definition.decoder_escape:
            out.append(encoded[index])
            index += 1
        else:
            out.append(chr(code))
    return "".join(out)
