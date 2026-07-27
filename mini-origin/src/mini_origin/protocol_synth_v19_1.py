from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from . import protocol_synth_v19 as v19


def safe_accepts(
    rules: Iterable[v19.ProtocolRule],
    case: v19.ProtocolCase,
) -> bool:
    for rule in rules:
        try:
            if not rule.predicate(case.contract, case.manifest, case.bundle):
                return False
        except (ArithmeticError, KeyError, TypeError, ValueError, IndexError):
            return False
    return True


# All synthesis, minimisation, training evaluation and hidden evaluation use
# total predicates. This assignment changes the module-global lookup used by
# the imported implementation without duplicating the synthesis algorithm.
v19.accepts = safe_accepts


def run(seed: int = 401) -> dict[str, object]:
    report = v19.run(seed)
    report["claim_scope"] = (
        "counterexample-guided synthesis selects and minimises total executable scientific-validity "
        "predicates from labelled bundles, freezes the protocol digest, and only then constructs "
        "withheld mutation families; malformed contracts reject rather than crash; external novelty "
        "still requires an independently authored grammar and outside reproduction"
    )
    report["total_predicate_semantics"] = True
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected_rule_count": report["selected_rule_count"],
                "library_size": report["library_size"],
                "hidden_detection": report["hidden"]["detection_rate"],
                "hidden_false_reject": report["hidden"]["false_reject_rate"],
                "random_detection_median": report["random_equal_size"]["detection_median"],
                "selected_rules": report["selected_rules"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
