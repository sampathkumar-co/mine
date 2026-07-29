from mini_origin import conditioned_cell_frontier_v60 as conditioned
from mini_origin import label_free_selector_certificate_v71 as selector
from mini_origin import label_free_threshold_validation_v72 as v72


def test_configure_uses_label_free_oversized_cell_sampler():
    v72.configure()
    assert conditioned.sample_allowed is selector.label_free_sample_allowed


def test_preregistration_is_frozen():
    prereg = v72.verify_preregistration()
    assert prereg["locked_gate"]["contributing_datasets"] == 7
    assert prereg["locked_gate"]["minimum_base_states"] == 60
    assert prereg["locked_gate"]["minimum_profiled_states"] == 180
    assert prereg["locked_gate"]["minimum_median_plain_bounded_ratio"] == 10.0
    assert prereg["locked_gate"]["minimum_p90_plain_bounded_ratio"] == 30.0
