from mini_origin import label_free_selector_certificate_v71 as certificate
from mini_origin import numeric_threshold_repaired_v70 as numeric
from mini_origin import conditioned_cell_frontier_v60 as conditioned


def test_oversized_sampling_ignores_labels():
    certificate.configure_selector()
    first_records = certificate.synthetic_records(71_999, False)
    second_records = certificate.synthetic_records(71_999, True)
    first_task, _ = numeric.compile_task("same", first_records)
    second_task, _ = numeric.compile_task("same", second_records)
    assert first_task.rows == second_task.rows
    cell = first_task.full_mask
    first = certificate.label_free_sample_allowed(
        first_task, cell, 24, "salt"
    )
    second = certificate.label_free_sample_allowed(
        second_task, cell, 24, "salt"
    )
    assert first == second


def test_conditioned_cells_ignore_labels_after_patch():
    certificate.configure_selector()
    first_task, _ = numeric.compile_task(
        "same-cells", certificate.synthetic_records(72_000, False)
    )
    second_task, _ = numeric.compile_task(
        "same-cells", certificate.synthetic_records(72_000, True)
    )
    assert tuple(sorted(conditioned.conditioned_cells(first_task))) == tuple(
        sorted(conditioned.conditioned_cells(second_task))
    )


def test_selected_states_ignore_labels_after_patch():
    certificate.configure_selector()
    first_task, _ = numeric.compile_task(
        "same-selected", certificate.synthetic_records(72_001, False)
    )
    second_task, _ = numeric.compile_task(
        "same-selected", certificate.synthetic_records(72_001, True)
    )
    assert conditioned.select_states(first_task) == conditioned.select_states(
        second_task
    )
