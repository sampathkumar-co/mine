from mini_origin import prior_art_audit_v46 as v46


def test_subproblem_similarity_is_separate() -> None:
    text = (
        "Compute similarity lower bound for archived branch data and transfer "
        "cached assignments when the datasets are equivalent."
    )
    categories = v46.classify_window(text)
    assert "subproblem_similarity_or_cache_equivalence" in categories
    assert "strong_local_test_partition_candidate" not in categories


def test_instance_equivalence_is_separate() -> None:
    text = "Equivalent points with identical row support strengthen the bound."
    categories = v46.classify_window(text)
    assert "instance_or_support_equivalence" in categories
    assert "strong_local_test_partition_candidate" not in categories


def test_local_test_partition_candidate_is_flagged() -> None:
    text = (
        "If two feature tests create the same child partition mask, skip the "
        "duplicate feature and keep one canonical representative."
    )
    categories = v46.classify_window(text)
    assert "possible_local_test_partition_equivalence" in categories
    assert "strong_local_test_partition_candidate" in categories
