from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


@dataclass(frozen=True)
class Difference:
    path: tuple[str | int, ...]
    left: str
    right: str
    signature: str


def operator_signatures(precommit: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for signature, operators in precommit["typed_operator_catalog"].items():
        for operator in operators:
            if operator in result and result[operator] != signature:
                raise RuntimeError(f"operator has conflicting signatures: {operator}")
            result[operator] = signature
    return result


def count_fixed_operators(value: Any) -> int:
    if isinstance(value, list):
        return sum(count_fixed_operators(item) for item in value)
    if not isinstance(value, dict):
        return 0
    count = int(isinstance(value.get("op"), str))
    return count + sum(
        count_fixed_operators(item)
        for key, item in value.items()
        if key != "op"
    )


def antiunify(
    left: Any,
    right: Any,
    signatures: dict[str, str],
    path: tuple[str | int, ...] = (),
) -> tuple[Any, list[Difference]] | None:
    if type(left) is not type(right):
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return None
        output: list[Any] = []
        differences: list[Difference] = []
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            result = antiunify(left_item, right_item, signatures, path + (index,))
            if result is None:
                return None
            value, found = result
            output.append(value)
            differences.extend(found)
        return output, differences
    if isinstance(left, dict):
        if set(left) != set(right):
            return None
        output: dict[str, Any] = {}
        differences: list[Difference] = []
        for key in sorted(left):
            left_value = left[key]
            right_value = right[key]
            if key == "op" and isinstance(left_value, str) and isinstance(right_value, str):
                if left_value == right_value:
                    output[key] = left_value
                    continue
                left_signature = signatures.get(left_value)
                right_signature = signatures.get(right_value)
                if left_signature is None or left_signature != right_signature:
                    return None
                difference = Difference(path + (key,), left_value, right_value, left_signature)
                output[key] = {"$operator_hole": "h0"}
                differences.append(difference)
                continue
            result = antiunify(left_value, right_value, signatures, path + (key,))
            if result is None:
                return None
            value, found = result
            output[key] = value
            differences.extend(found)
        return output, differences
    if left != right:
        return None
    return left, []


def make_template(
    left: dict[str, Any],
    right: dict[str, Any],
    precommit: dict[str, Any],
    signatures: dict[str, str],
) -> dict[str, Any] | None:
    result = antiunify(left["structure"], right["structure"], signatures)
    if result is None:
        return None
    template_ast, differences = result
    rules = precommit["antiunification"]
    if not differences or len(differences) > int(rules["maximum_operator_holes"]):
        return None
    if len(differences) != 1:
        return None
    fixed_operator_nodes = count_fixed_operators(template_ast)
    if fixed_operator_nodes < int(rules["minimum_shared_fixed_operator_nodes"]):
        return None
    difference = differences[0]
    choices = list(precommit["typed_operator_catalog"][difference.signature])
    template = {
        "schema": "lexigen-v29-operator-hole-template-v1",
        "template_ast": template_ast,
        "operator_holes": [
            {
                "name": "h0",
                "signature": difference.signature,
                "path": list(difference.path),
                "allowed_choices": choices,
            }
        ],
        "fixed_operator_nodes": fixed_operator_nodes,
        "source_task_ids": sorted([left["task_id"], right["task_id"]]),
        "source_structure_sha256": sorted([
            left["structure_sha256"], right["structure_sha256"]
        ]),
        "source_instantiations": {
            left["task_id"]: {"h0": difference.left},
            right["task_id"]: {"h0": difference.right},
        },
        "origin": {
            "method": "all_pair_operator_value_antiunification",
            "differing_operator_count": 1,
            "whole_subtree_holes": 0,
        },
    }
    template["template_sha256"] = sha256_json(template)
    return template


def discover_templates(
    sources: dict[str, Any], precommit: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    signatures = operator_signatures(precommit)
    structures = list(sources["structures"])
    templates: dict[str, dict[str, Any]] = {}
    pair_reports: list[dict[str, Any]] = []
    for left, right in itertools.combinations(structures, 2):
        template = make_template(left, right, precommit, signatures)
        report = {
            "task_ids": sorted([left["task_id"], right["task_id"]]),
            "structure_sha256": sorted([
                left["structure_sha256"], right["structure_sha256"]
            ]),
            "eligible": template is not None,
            "template_sha256": None if template is None else template["template_sha256"],
        }
        pair_reports.append(report)
        if template is not None:
            templates[template["template_sha256"]] = template
    return [templates[key] for key in sorted(templates)], pair_reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "V29_TEMPLATE_LIBRARY.json")
    parser.add_argument("--templates-dir", type=Path, default=HERE / "templates")
    args = parser.parse_args()

    source_path = HERE / "V29_SOURCE_STRUCTURES.json"
    precommit_path = HERE / "V29_PRECOMMIT.json"
    sources = load(source_path)
    precommit = load(precommit_path)
    if sha256_file(source_path) != precommit["source_structures_sha256"]:
        raise RuntimeError("source structure registry changed")
    if len(sources["structures"]) != precommit["source_structure_count"]:
        raise RuntimeError("source structure denominator changed")

    templates, pair_reports = discover_templates(sources, precommit)
    expected_pair_count = len(sources["structures"]) * (len(sources["structures"]) - 1) // 2
    if len(pair_reports) != expected_pair_count:
        raise RuntimeError("source pair denominator changed")

    if args.templates_dir.exists():
        for path in args.templates_dir.glob("*.json"):
            path.unlink()
    args.templates_dir.mkdir(parents=True, exist_ok=True)
    template_entries: list[dict[str, Any]] = []
    for template in templates:
        path = args.templates_dir / f"template-{template['template_sha256']}.json"
        write(path, template)
        template_entries.append({
            "template_sha256": template["template_sha256"],
            "template_file_sha256": sha256_file(path),
            "source_task_ids": template["source_task_ids"],
            "fixed_operator_nodes": template["fixed_operator_nodes"],
            "operator_holes": template["operator_holes"],
        })

    library = {
        "schema": "lexigen-v29-template-library-v1",
        "precommit_sha256": sha256_file(precommit_path),
        "source_structures_sha256": sha256_file(source_path),
        "source_structure_count": len(sources["structures"]),
        "source_pair_count": len(pair_reports),
        "eligible_template_count": len(template_entries),
        "pair_reports": pair_reports,
        "templates": template_entries,
        "validation_generators_imported": 0,
        "validation_outputs_opened": False,
        "heldout_transfer_demonstrated": False,
        "world_level_breakthrough": False,
    }
    write(args.output, library)
    print(json.dumps({
        "source_pairs": len(pair_reports),
        "eligible_templates": len(template_entries),
        "template_sha256": [item["template_sha256"] for item in template_entries],
        "validation_outputs_opened": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()


def instantiate_template(template: dict[str, Any], arguments: dict[str, str]) -> dict[str, Any]:
    declarations = {item["name"]: item for item in template["operator_holes"]}
    if set(arguments) != set(declarations):
        raise RuntimeError("operator-hole argument mismatch")
    for name, value in arguments.items():
        if value not in declarations[name]["allowed_choices"]:
            raise RuntimeError(f"operator choice not allowed: {name}={value}")

    def visit(value: Any) -> Any:
        if isinstance(value, list):
            return [visit(item) for item in value]
        if isinstance(value, dict):
            if set(value) == {"$operator_hole"}:
                return arguments[str(value["$operator_hole"])]
            return {key: visit(item) for key, item in value.items()}
        return value

    return visit(template["template_ast"])
