from __future__ import annotations

import hashlib
import json

from . import external_response_cost_v58 as external
from . import numeric_threshold_frontier_v70 as original


def label_free_sample(name: str, records):
    """Select at most MAX_RECORDS without consulting labels.

    Feature vectors determine the SHA-256 rank. The original archive row index is
    used only to make duplicate feature vectors deterministic. Labels remain
    attached to selected rows but never influence selection.
    """
    ranked = list(enumerate(records))
    ranked.sort(key=lambda item: (
        hashlib.sha256(json.dumps(
            [name, item[1][0]],
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest(),
        item[0],
    ))
    return [row for _, row in ranked[:external.MAX_RECORDS]]


def compiler_protocol() -> dict[str, object]:
    protocol = dict(original.compiler_protocol())
    protocol.pop("maximum_sampled_distinct_records", None)
    protocol["maximum_sampled_records"] = external.MAX_RECORDS
    protocol["sampling"] = (
        "rank original rows by SHA-256(dataset name, feature vector) with "
        "original row index as the duplicate-feature tie-break; labels excluded"
    )
    protocol["labels_or_costs_used"] = False
    return protocol


# Patch only the contradictory inherited sampling dependency and its protocol
# description. The compiler, state selector, exact planners and gate remain in
# the frozen original module.
original.external.deterministic_sample = label_free_sample
original.compiler_protocol = compiler_protocol

is_missing = original.is_missing
finite_decimal = original.finite_decimal
decimal_name = original.decimal_name
quantile_thresholds = original.quantile_thresholds
compile_task = original.compile_task
protocol = original.protocol
compact_state = original.compact_state
configure_parent = original.configure_parent
run_reference = original.run_reference
validate = original.validate
main = original.main


if __name__ == "__main__":
    main()
