from mini_origin import openml_cross_source_lock_v77 as lock


def test_normalize_and_collect_ids():
    assert lock.normalize_name("AIDS / Study 175") == "aids-study-175"
    payload = {"benchmark_suite": {"data": [{"data_id": "11"}, {"dataset_id": 12}, {"did": 13}]}}
    assert lock.collect_dataset_ids(payload) == {11, 12, 13}


def test_registry_and_uci_exclusion_precede_selection():
    row = {"id": "123", "name": "Novel", "status": "active", "format": "ARFF", "file_id": "9"}
    qualities = {"NumberOfInstances": 1000, "NumberOfFeatures": 20, "NumberOfClasses": 2}
    assert lock.eligible(row, qualities, {123}, set()) == (False, "registry-excluded")
    row["id"] = "124"
    row["description"] = "Originally deposited at UCI"
    assert lock.eligible(row, qualities, set(), set()) == (False, "uci-origin")


def test_metadata_gate_and_ranking_are_deterministic():
    row = {"id": "124", "name": "Non U Source", "status": "active", "format": "ARFF", "file_id": "9", "version": "2"}
    qualities = {"NumberOfInstances": 1000, "NumberOfFeatures": 20, "NumberOfClasses": 2, "NumberOfInstancesWithMissingValues": 10}
    assert lock.eligible(row, qualities, set(), set()) == (True, "eligible")
    assert lock.rank_key("seed", row) == lock.rank_key("seed", dict(row))
    qualities["NumberOfInstancesWithMissingValues"] = 300
    assert lock.eligible(row, qualities, set(), set()) == (False, "missing-fraction")
