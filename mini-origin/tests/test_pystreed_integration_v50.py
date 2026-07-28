from pathlib import Path

from mini_origin import pystreed_integration_v50 as v50


def test_augmentation_creates_descendant_only_equivalence() -> None:
    labels = []
    features = []
    for repeat in range(12):
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    labels.append(a ^ b ^ c)
                    features.append([
                        a, b, c,
                        a ^ c,
                        b ^ c,
                        a ^ b,
                        (a & b) ^ c,
                        (a | b) ^ c,
                        repeat % 2,
                        (repeat // 2) % 2,
                    ])
    columns, metadata = v50.make_augmented(labels, features)
    assert metadata["derived_feature_count"] > 0
    assert len(columns) > len(metadata["base_feature_indices"])
    signatures = {tuple(column) for column in columns}
    assert len(signatures) == len(columns)

    descendant_equivalence = False
    for anchor_position in (0, 1):
        anchor = columns[anchor_position]
        for outcome in (0, 1):
            selected = [
                index for index, value in enumerate(anchor)
                if value == outcome
            ]
            restricted = {}
            for column_index, column in enumerate(columns):
                signature = tuple(column[index] for index in selected)
                restricted.setdefault(signature, []).append(column_index)
            if any(len(group) > 1 for group in restricted.values()):
                descendant_equivalence = True
    assert descendant_equivalence


def test_parse_table_accepts_binary_accuracy_data(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    rows = []
    for index in range(64):
        label = index % 3
        features = [(index >> bit) & 1 for bit in range(10)]
        rows.append(" ".join(map(str, [label] + features)))
    path.write_text("\n".join(rows), encoding="utf-8")
    parsed = v50.parse_table(path)
    assert parsed is not None
    labels, features = parsed
    assert len(labels) == 64
    assert len(features[0]) == 10
    assert set(labels) == {0, 1, 2}


def test_external_scope_is_pinned_and_restricted() -> None:
    assert v50.PINNED_COMMIT == (
        "9ad41626a1f26c4b7481e8360c5c8b1871e10d96"
    )
    assert v50.MAX_DEPTH == 3
    assert v50.MAX_NODES == 5
    assert v50.DATASET_COUNT == 4
