from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

from antiunify_v29 import load
from scan_one_v29 import candidate_instantiations, parameter_names


def template():
    library = load(HERE / "V29_TEMPLATE_LIBRARY.json")
    entry = library["templates"][0]
    return load(HERE / "templates" / f"template-{entry['template_sha256']}.json")


def test_candidate_denominator_and_order():
    values = list(candidate_instantiations(template()))
    assert len(values) == 160
    first_operator, first_colour, _ = values[0]
    last_operator, last_colour, _ = values[-1]
    assert first_operator == {"h0": "bbox_border"}
    assert first_colour == {"c0": 0}
    assert last_operator == {"h0": "row_span"}
    assert last_colour == {"c0": 9}


def test_every_candidate_is_fully_executable_ast():
    for operator_arguments, colour_arguments, program in candidate_instantiations(template()):
        text = json.dumps(program, sort_keys=True)
        assert "$operator_hole" not in text
        assert "param_color" not in text
        assert operator_arguments["h0"] in text
        assert f'"value": {colour_arguments["c0"]}' in text


def test_template_parameter_inventory():
    value = template()
    assert parameter_names(value["template_ast"]) == ["c0"]
    assert value["operator_holes"][0]["name"] == "h0"
    assert len(value["operator_holes"][0]["allowed_choices"]) == 16


def test_source_operator_instantiations_are_present():
    values = list(candidate_instantiations(template()))
    operators = {item[0]["h0"] for item in values}
    assert "erode4" in operators
    assert "holes" in operators


def test_scanner_does_not_import_validation_tasks_directly():
    text = (HERE / "scan_one_v29.py").read_text(encoding="utf-8")
    assert "tasks.task_" not in text
    assert "validation_task_ids" in text
    assert "replacement_used" in text


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
