from __future__ import annotations

import io
import json
import zipfile

from mini_origin import proof_carrying_reduction_synthesis_v91 as v91
from mini_origin import tsplib_external_holdout_v92 as v92
from mini_origin import tsplib_hash_lock_v92 as lock_v92


def _synthetic_archive() -> bytes:
    vertices = []
    for tail in range(9):
        edges = "".join(
            f'<edge cost="{1 + abs(tail-head)}.000000000000000e+00">{head}</edge>'
            for head in range(9) if head != tail
        )
        vertices.append(f"<vertex>{edges}</vertex>")
    xml = (
        "<?xml version=\"1.0\"?><travellingSalesmanProblemInstance>"
        "<name>synthetic</name><source>test</source><description>test</description>"
        "<doublePrecision>15</doublePrecision><ignoredDigits>5</ignoredDigits>"
        f"<graph>{''.join(vertices)}</graph>"
        "</travellingSalesmanProblemInstance>"
    ).encode()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("synthetic.xml", xml)
    return buf.getvalue()


def test_preregistration_and_frozen_v91_digest() -> None:
    payload = json.loads(lock_v92.PREREGISTRATION.read_text(encoding="utf-8"))
    assert payload["status"] == "preregistered_before_selected_archive_access"
    assert [row["name"] for row in lock_v92.selected_sources()] == ["gr21", "gr24", "p43"]
    assert v92.EXPECTED_SPEC.digest() == v92.EXPECTED_SPEC_DIGEST
    assert payload["frozen_v91_rule"]["v91_freeze_digest"] == v92.EXPECTED_V91_FREEZE_DIGEST


def test_xml_projection_and_independent_tour_certificate() -> None:
    matrix = v92.parse_projected_matrix("synthetic", _synthetic_archive())
    assert len(matrix) == 9
    assert all(len(row) == 9 for row in matrix)
    problem = v92.held_karp_problem("synthetic", matrix)
    brute = v92.brute_force_optimum(matrix)
    raw = v91.solve_with_spec(problem, None)
    candidate = v91.solve_with_spec(problem, v92.EXPECTED_SPEC)
    assert brute == raw.objective == candidate.objective
    assert candidate.relation_certificates == candidate.actions_pruned


def test_nonintegral_xml_cost_rejected() -> None:
    try:
        v92.parse_integral_cost("1.5")
    except RuntimeError as exc:
        assert "nonnegative integral" in str(exc)
    else:
        raise AssertionError("nonintegral TSPLIB XML cost was accepted")


def test_evaluation_locked_before_manifest(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(v92, "LOCK_MANIFEST", tmp_path / "missing.json")
    try:
        v92._load_lock_manifest()
    except RuntimeError as exc:
        assert "evaluation locked" in str(exc)
    else:
        raise AssertionError("v0.92 evaluation ran without lock manifest")
