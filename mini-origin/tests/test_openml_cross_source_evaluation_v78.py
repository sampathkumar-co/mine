from __future__ import annotations

import hashlib

import pytest

from mini_origin import openml_cross_source_evaluation_v78 as v78


def dataset(**overrides):
    row = {
        "name": "toy",
        "dataset_id": 999,
        "default_target_attribute": "class label",
        "num_instances": 2,
        "num_features": 3,
    }
    row.update(overrides)
    return row


def test_dense_arff_preserves_order_quotes_and_missing_tokens():
    payload = b"""% comment
@relation toy
@attribute 'first feature' numeric
@attribute second {x,y}
@attribute 'class label' {no,yes}
@data
1,x,no
?, 'y', yes
"""
    records, summary = v78.parse_dense_arff(dataset(), payload)
    assert records == [(('1', 'x'), 'no'), (('?', 'y'), 'yes')]
    assert summary["feature_columns"] == ["first feature", "second"]
    assert summary["target_column"] == "class label"
    assert summary["missing_feature_tokens"] == 1


def test_sparse_rows_are_rejected():
    payload = b"""@relation toy
@attribute a numeric
@attribute b numeric
@attribute class {n,y}
@data
{0 1,2 y}
"""
    with pytest.raises(RuntimeError, match="sparse ARFF"):
        v78.parse_dense_arff(dataset(num_instances=1, default_target_attribute="class"), payload)


def test_shape_and_target_guards():
    wrong_width = b"""@relation toy
@attribute a numeric
@attribute b numeric
@attribute class {n,y}
@data
1,n
"""
    with pytest.raises(RuntimeError, match="width mismatch"):
        v78.parse_dense_arff(dataset(num_instances=1, default_target_attribute="class"), wrong_width)
    missing_target = b"""@relation toy
@attribute a numeric
@attribute b numeric
@attribute class {n,y}
@data
1,2,?
"""
    with pytest.raises(RuntimeError, match="missing target"):
        v78.parse_dense_arff(dataset(num_instances=1, default_target_attribute="class"), missing_target)


def test_frozen_lock_and_selected_ids_are_bound():
    prereg, evidence = v78.load_frozen_inputs()
    assert prereg["parent_v77_lock_digest"] == v78.LOCK_DIGEST
    assert evidence["lock_digest"] == v78.LOCK_DIGEST
    assert [row["dataset_id"] for row in evidence["datasets"]] == [1068, 188, 40499, 1067, 1063, 469, 1049]
    assert all(len(row["raw_sha256"]) == 64 for row in evidence["datasets"])
