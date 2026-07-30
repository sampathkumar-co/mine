from mini_origin import pmlb_development_v85 as development


def test_development_install_preserves_frozen_gate_and_activates_only_v85_selector():
    preregistration = development._install()
    assert preregistration["opened_data_gate"]["contributing_dataset_count"] == 7
    assert (
        preregistration["opened_data_gate"]["threshold_source"]
        == "unchanged v0.82 preregistration"
    )
    assert development.parent.selector is development.repair
    protocol = development.repair.protocol()
    assert protocol["response_lattice_fallback"] == "zero-state-only"
    assert (
        protocol["response_lattice_closure"]
        == "joint generator partition refines each determined query partition"
    )
    assert protocol["development_data_status"] == "opened-but-not-accessed-by-v0.85-yet"


def test_selector_reexports_frozen_frontier_interface():
    development._install()
    assert development.repair.frontier is development.repair.parent.frontier
    assert (
        development.parent.selector.frontier.compiler_protocol()
        == development.repair.parent.frontier.compiler_protocol()
    )
