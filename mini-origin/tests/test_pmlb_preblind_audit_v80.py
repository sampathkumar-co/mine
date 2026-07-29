from __future__ import annotations

import json
from pathlib import Path

from mini_origin import pmlb_preblind_audit_v80 as audit


def test_extracts_explicit_pmlb_dataset_names():
    text = """
    pmlb.fetch_data('alpha_case')
    fetch_data("beta-case")
    https://github.com/EpistasisLab/pmlb/tree/master/datasets/gamma_case
    datasets/delta-case/delta-case.tsv.gz
    """
    assert audit.extract_pmlb_names(text) == {
        "alpha-case",
        "beta-case",
        "gamma-case",
        "delta-case",
    }


def test_unrelated_strings_do_not_create_pmlb_names():
    assert audit.extract_pmlb_names("PMLB is a benchmark repository") == set()
    assert audit.extract_pmlb_names("datasets are useful") == set()


def test_preregistration_freezes_preblind_boundary():
    data = json.loads(audit.PREREGISTRATION.read_text(encoding="utf-8-sig"))
    assert data["status"] == "preregistered_before_pmlb_catalogue_access"
    assert data["parent_v79_commit"] == audit.FROZEN_V79_COMMIT
    assert data["parent_v79_evidence_digest"] == audit.V79_EVIDENCE_DIGEST


def test_parent_evidence_matches_frozen_pass():
    preregistration, parent = audit.load_inputs()
    assert preregistration["pmlb_catalogue_or_repository_tree_access_before_audit"] is False
    assert preregistration["pmlb_candidate_names_ids_or_urls_committed_before_audit"] is False
    assert preregistration["pmlb_dataset_bytes_access_before_audit"] is False
    assert preregistration["solver_execution_before_audit"] is False
    assert parent["status"] == "small_query_coverage_development_pass_v79"
    assert parent["development_gate"] is True
    assert parent["rust_mismatch_count"] == 0


def test_source_has_no_pmlb_client_or_network_dependency():
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert "import pmlb" not in source
    assert "requests" not in source
    assert "urlopen" not in source
    assert "urllib" not in source
