from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
from typing import Any, Iterable

from .ir import OpCode, Program


# This module intentionally reuses the proven anti-unification idea from the
# preserved Lexigen v15 track, but changes the scientific boundary: proposals are
# mined from task-agnostic Ω IR fragments and are never admitted to long-term
# memory here. Causal admission remains solely GeneAdmissionPolicy's job.


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def tree_size(value: Any) -> int:
    if isinstance(value, dict):
        return 1 + sum(tree_size(item) for item in value.values())
    if isinstance(value, list):
        return 1 + sum(tree_size(item) for item in value)
    return 1


def variable_count(value: Any) -> int:
    if isinstance(value, dict) and set(value) == {"$var"}:
        return 1
    if isinstance(value, dict):
        return sum(variable_count(item) for item in value.values())
    if isinstance(value, list):
        return sum(variable_count(item) for item in value)
    return 0


def anti_unify(left: Any, right: Any) -> Any:
    counter = [0]
    pair_variables: dict[str, str] = {}

    def visit(a: Any, b: Any) -> Any:
        if a == b:
            return deepcopy(a)
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) == set(b):
                return {key: visit(a[key], b[key]) for key in sorted(a)}
        if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            return [visit(x, y) for x, y in zip(a, b)]
        key = canonical([a, b])
        if key not in pair_variables:
            pair_variables[key] = f"v{counter[0]}"
            counter[0] += 1
        return {"$var": pair_variables[key]}

    return visit(left, right)


def match_template(
    template: Any,
    value: Any,
    bindings: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    current = {} if bindings is None else dict(bindings)
    if isinstance(template, dict) and set(template) == {"$var"}:
        name = str(template["$var"])
        if name in current and current[name] != value:
            return None
        current[name] = value
        return current
    if isinstance(template, dict) and isinstance(value, dict) and set(template) == set(value):
        for key in sorted(template):
            current = match_template(template[key], value[key], current)
            if current is None:
                return None
        return current
    if isinstance(template, list) and isinstance(value, list) and len(template) == len(value):
        for expected, observed in zip(template, value):
            current = match_template(expected, observed, current)
            if current is None:
                return None
        return current
    return current if template == value else None


def instantiate_template(template: Any, bindings: dict[str, Any]) -> Any:
    if isinstance(template, dict) and set(template) == {"$var"}:
        return deepcopy(bindings[str(template["$var"])])
    if isinstance(template, dict):
        return {key: instantiate_template(value, bindings) for key, value in template.items()}
    if isinstance(template, list):
        return [instantiate_template(value, bindings) for value in template]
    return deepcopy(template)


@dataclass(frozen=True)
class FragmentRecord:
    source_family: str
    program_fingerprint: str
    start: int
    end: int
    fragment: dict[str, Any]


@dataclass(frozen=True)
class GeneProposal:
    proposal_id: str
    template: dict[str, Any]
    occurrences: int
    source_families: tuple[str, ...]
    fragment_size: int
    parameter_count: int
    mdl_gain: int
    stage: str = "proposal_only_not_causal_memory"

    @property
    def memory_eligible(self) -> bool:
        # Deliberately impossible at the genesis stage. A proposal must first be
        # materialized as an executable SemanticGene and earn prospective causal
        # evidence under GeneAdmissionPolicy.
        return False


def _source_family(program: Program) -> str:
    value = program.metadata.get("source_family")
    return str(value).strip() if value is not None else ""


def normalize_fragment(program: Program, start: int, end: int) -> dict[str, Any]:
    """Alpha-normalize a contiguous SSA fragment.

    Local SSA names become v0/v1/... and references entering from outside the
    fragment become arg0/arg1/... in first-use order. This removes accidental task
    naming while preserving executable dependency structure.
    """

    if start < 0 or end > len(program.instructions) or start >= end:
        raise ValueError("invalid fragment bounds")

    local_names: dict[str, str] = {}
    external_names: dict[str, str] = {}
    normalized: list[dict[str, Any]] = []

    for local_index, instruction in enumerate(program.instructions[start:end]):
        if instruction.op is OpCode.RETURN:
            raise ValueError("RETURN is not a mineable semantic fragment instruction")

        args: list[str] = []
        for name in instruction.args:
            if name in local_names:
                args.append(local_names[name])
            else:
                if name not in external_names:
                    external_names[name] = f"arg{len(external_names)}"
                args.append(external_names[name])

        normalized.append(
            {
                "op": instruction.op.value,
                "args": args,
                "literal": deepcopy(instruction.literal),
                "type": instruction.type_tag.value,
            }
        )
        local_names[instruction.name] = f"v{local_index}"

    return {
        "external_arity": len(external_names),
        "instructions": normalized,
    }


def enumerate_fragments(
    programs: Iterable[Program],
    *,
    min_window: int = 2,
    max_window: int = 4,
) -> tuple[FragmentRecord, ...]:
    if min_window < 1 or max_window < min_window:
        raise ValueError("invalid window limits")

    output: list[FragmentRecord] = []
    for program in programs:
        program.validate()
        family = _source_family(program)
        if not family:
            continue
        mineable = len(program.instructions) - 1  # final RETURN excluded
        for width in range(min_window, max_window + 1):
            if width > mineable:
                continue
            for start in range(0, mineable - width + 1):
                end = start + width
                fragment = normalize_fragment(program, start, end)
                if all(item["op"] == OpCode.CONST.value for item in fragment["instructions"]):
                    continue
                output.append(
                    FragmentRecord(
                        source_family=family,
                        program_fingerprint=program.fingerprint(),
                        start=start,
                        end=end,
                        fragment=fragment,
                    )
                )
    return tuple(output)


def _opcode_signature(fragment: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(item["op"]) for item in fragment["instructions"])


def _score(template: Any, occurrences: int) -> int:
    size = tree_size(template)
    variables = variable_count(template)
    reusable_body = max(0, size - variables - 2)
    return occurrences * reusable_body - (size + variables)


def mine_semantic_gene_proposals(
    programs: Iterable[Program],
    *,
    min_source_families: int = 2,
    min_window: int = 2,
    max_window: int = 4,
    limit: int = 16,
    require_parameterized: bool = True,
) -> tuple[GeneProposal, ...]:
    """Mine structural semantic-gene proposals without granting causal credit.

    Only fragments supported by multiple source families survive. Candidate
    generation is intentionally cheap; expensive prospective transfer is reserved
    for the much smaller set of materialized proposals.
    """

    records = enumerate_fragments(
        programs,
        min_window=min_window,
        max_window=max_window,
    )
    by_signature: dict[tuple[str, ...], list[FragmentRecord]] = {}
    for record in records:
        by_signature.setdefault(_opcode_signature(record.fragment), []).append(record)

    templates: dict[str, dict[str, Any]] = {}
    for group in by_signature.values():
        for left, right in combinations(group, 2):
            if left.source_family == right.source_family:
                continue
            if left.fragment["external_arity"] != right.fragment["external_arity"]:
                continue
            template = anti_unify(left.fragment, right.fragment)
            if require_parameterized and variable_count(template) == 0:
                continue
            templates[canonical(template)] = template

    proposals: list[GeneProposal] = []
    for template in templates.values():
        matched = [record for record in records if match_template(template, record.fragment) is not None]
        families = tuple(sorted({record.source_family for record in matched}))
        if len(families) < min_source_families:
            continue
        gain = _score(template, len(matched))
        if gain <= 0:
            continue
        digest = hashlib.sha256(canonical(template).encode("utf-8")).hexdigest()[:12]
        proposals.append(
            GeneProposal(
                proposal_id=f"omega_gene_{digest}",
                template=template,
                occurrences=len(matched),
                source_families=families,
                fragment_size=len(template["instructions"]),
                parameter_count=variable_count(template),
                mdl_gain=gain,
            )
        )

    proposals.sort(
        key=lambda item: (
            -len(item.source_families),
            -item.mdl_gain,
            -item.occurrences,
            item.proposal_id,
        )
    )
    return tuple(proposals[:limit])


def verify_exact_expansion(template: Any, fragment: Any) -> bool:
    bindings = match_template(template, fragment)
    return bindings is not None and instantiate_template(template, bindings) == fragment
