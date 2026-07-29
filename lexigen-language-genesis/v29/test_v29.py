from __future__ import annotations

import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

from antiunify_v29 import (
    canonical,
    discover_templates,
    instantiate_template,
    load,
    sha256_json,
)


def fixtures():
    sources = load(HERE / "V29_SOURCE_STRUCTURES.json")
    precommit = load(HERE / "V29_PRECOMMIT.json")
    templates, pairs = discover_templates(sources, precommit)
    return sources, precommit, templates, pairs


def test_all_pairs_and_single_eligible_template():
    _, _, templates, pairs = fixtures()
    assert len(pairs) == 3
    assert sum(int(pair["eligible"]) for pair in pairs) == 1
    assert len(templates) == 1
    assert templates[0]["source_task_ids"] == ["67a423a3", "810b9b61"]


def test_template_hash_and_shape():
    _, _, templates, _ = fixtures()
    template = templates[0]
    stored_hash = template["template_sha256"]
    unhashed = copy.deepcopy(template)
    del unhashed["template_sha256"]
    assert sha256_json(unhashed) == stored_hash
    assert template["fixed_operator_nodes"] == 6
    assert len(template["operator_holes"]) == 1
    hole = template["operator_holes"][0]
    assert hole["signature"] == "PointSet->PointSet"
    assert hole["path"] == ["points", "points", "points", "op"]


def test_source_programs_reconstruct_exactly():
    sources, _, templates, _ = fixtures()
    template = templates[0]
    by_task = {item["task_id"]: item for item in sources["structures"]}
    for task_id, arguments in template["source_instantiations"].items():
        reconstructed = instantiate_template(template, arguments)
        assert canonical(reconstructed) == canonical(by_task[task_id]["structure"])


def test_unrelated_crop_structure_is_rejected():
    _, _, _, pairs = fixtures()
    rejected = [pair for pair in pairs if "b94a9452" in pair["task_ids"]]
    assert len(rejected) == 2
    assert all(not pair["eligible"] for pair in rejected)


def test_no_task_ids_or_subtree_holes_in_template_ast():
    sources, precommit, templates, _ = fixtures()
    template = templates[0]
    text = canonical(template["template_ast"])
    for item in sources["structures"]:
        assert item["task_id"] not in text
    assert "$operator_hole" in text
    assert "$subtree_hole" not in text
    hole = template["operator_holes"][0]
    assert hole["allowed_choices"] == precommit["typed_operator_catalog"][hole["signature"]]


def test_frozen_library_boundary():
    library = load(HERE / "V29_TEMPLATE_LIBRARY.json")
    assert library["source_pair_count"] == 3
    assert library["eligible_template_count"] == 1
    assert library["validation_generators_imported"] == 0
    assert not library["validation_outputs_opened"]
    assert not library["heldout_transfer_demonstrated"]
    assert not library["world_level_breakthrough"]


def main():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print(f"SUMMARY {len(tests)}/{len(tests)} tests passed")


if __name__ == "__main__":
    main()
