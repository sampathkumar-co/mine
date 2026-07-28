from __future__ import annotations

from dataclasses import dataclass
import argparse, hashlib, json, math
from pathlib import Path

import numpy as np

from . import tree_compiler_v26 as v26


@dataclass(frozen=True)
class DecisionExample:
    minimum_response: float
    should_move: bool


@dataclass(frozen=True)
class ThresholdFit:
    threshold: float
    development_accuracy: float
    development_move_accuracy: float
    development_accept_accuracy: float
    training_accuracy: float
    candidates_checked: int


@dataclass(frozen=True)
class Evaluation:
    accuracy: float
    mean_queries: float
    maximum_queries: int
    invalid_rate: float
    low_signal_accuracy: float
    high_signal_accuracy: float


def collect_examples(
    seed: int,
    tasks: list[tuple[list[set[int]], int]],
    query: v26.QueryProgram,
    replicates: int,
    rho_low: float,
    rho_high: float,
) -> list[DecisionExample]:
    rng = np.random.default_rng(seed)
    examples: list[DecisionExample] = []
    for index, (tree, root) in enumerate(tasks):
        allowed = set(range(len(tree)))
        step = 0
        rho = float(rng.uniform(rho_low, rho_high))
        while len(allowed) > 1:
            node = v26.select_query(tree, allowed, query)
            responses = v26.local_responses(
                seed + index * 104_729 + step * 7_919,
                tree,
                allowed,
                node,
                root,
                rho,
                replicates,
            )
            examples.append(
                DecisionExample(
                    minimum_response=min(value for _, value in responses),
                    should_move=node != root,
                )
            )
            parent = v26.next_on_path(tree, allowed, node, root)
            if parent is None:
                break
            allowed = v26.component(tree, allowed, parent, node)
            step += 1
    return examples


def class_metrics(examples: list[DecisionExample], threshold: float) -> tuple[float, float, float]:
    correct = []
    move = []
    accept = []
    for example in examples:
        prediction = example.minimum_response <= threshold
        value = prediction == example.should_move
        correct.append(value)
        (move if example.should_move else accept).append(value)
    return float(np.mean(correct)), float(np.mean(move)), float(np.mean(accept))


def threshold_candidates(examples: list[DecisionExample]) -> list[float]:
    values = sorted({example.minimum_response for example in examples})
    candidates = [values[0] - 1e-9]
    candidates.extend((left + right) / 2.0 for left, right in zip(values, values[1:]))
    candidates.append(values[-1] + 1e-9)
    return candidates


def fit_threshold(training: list[DecisionExample], development: list[DecisionExample]) -> ThresholdFit:
    best = None
    candidates = threshold_candidates(training)
    for threshold in candidates:
        train_accuracy, _, _ = class_metrics(training, threshold)
        dev_accuracy, move_accuracy, accept_accuracy = class_metrics(development, threshold)
        score = (
            min(move_accuracy, accept_accuracy),
            dev_accuracy,
            train_accuracy,
            -abs(threshold),
        )
        row = (score, threshold, dev_accuracy, move_accuracy, accept_accuracy, train_accuracy)
        if best is None or row[0] > best[0]:
            best = row
    assert best is not None
    _, threshold, dev_accuracy, move_accuracy, accept_accuracy, train_accuracy = best
    return ThresholdFit(
        threshold=float(threshold),
        development_accuracy=float(dev_accuracy),
        development_move_accuracy=float(move_accuracy),
        development_accept_accuracy=float(accept_accuracy),
        training_accuracy=float(train_accuracy),
        candidates_checked=len(candidates),
    )


def evaluate(
    seed: int,
    tasks: list[tuple[list[set[int]], int]],
    policy: v26.Policy,
    replicates: int,
    rho_low: float,
    rho_high: float,
) -> Evaluation:
    rng = np.random.default_rng(seed)
    rows = []
    low_signal = []
    high_signal = []
    for index, (tree, root) in enumerate(tasks):
        rho = float(rng.uniform(rho_low, rho_high))
        row = v26.run_trial(
            seed + index * 104_729,
            tree,
            root,
            policy,
            rho,
            replicates,
        )
        rows.append(row)
        (low_signal if rho < 0.30 else high_signal).append(row[0])
    return Evaluation(
        accuracy=float(np.mean([row[0] for row in rows])),
        mean_queries=float(np.mean([row[1] for row in rows])),
        maximum_queries=max(row[1] for row in rows),
        invalid_rate=float(np.mean([row[2] for row in rows])),
        low_signal_accuracy=float(np.mean(low_signal)),
        high_signal_accuracy=float(np.mean(high_signal)),
    )


def digest(query: v26.QueryProgram, fit: ThresholdFit) -> str:
    return hashlib.sha256(
        f"{query.text()}:{fit.threshold:.12f}:{fit.candidates_checked}".encode()
    ).hexdigest()


def run(seed: int = 1201) -> dict[str, object]:
    query = v26.QueryProgram("sum_distance", 3)
    training_tasks = v26.make_tasks(seed * 10_000 + 89, (7, 11, 15), 5, 2)
    development_tasks = v26.make_tasks(seed * 10_000 + 1_000_093, (9, 13, 17), 6, 2)
    training = collect_examples(
        seed * 10_000 + 2_000_003,
        training_tasks,
        query,
        replicates=512,
        rho_low=0.18,
        rho_high=0.92,
    )
    development = collect_examples(
        seed * 10_000 + 3_000_007,
        development_tasks,
        query,
        replicates=512,
        rho_low=0.16,
        rho_high=0.94,
    )
    fit = fit_threshold(training, development)
    candidate = v26.Policy(
        query,
        v26.DecoderProgram("minimum_below", fit.threshold, 2),
    )
    fixed = v26.Policy(
        query,
        v26.DecoderProgram("minimum_below", 0.20, 2),
    )
    frozen_digest = digest(query, fit)

    # Hidden weak-signal tasks are created only after the fitted threshold,
    # policy, fixed control and digest are frozen.
    hidden_tasks = v26.make_tasks(
        seed * 10_000 + 12_000_001,
        (17, 31, 63, 127),
        12,
        3,
    )
    candidate_result = evaluate(
        seed * 10_000 + 13_000_003,
        hidden_tasks,
        candidate,
        replicates=768,
        rho_low=0.15,
        rho_high=0.95,
    )
    fixed_result = evaluate(
        seed * 10_000 + 13_000_005,
        hidden_tasks,
        fixed,
        replicates=768,
        rho_low=0.15,
        rho_high=0.95,
    )
    random_decoder = v26.random_decoder_control(
        seed * 10_000 + 13_000_009,
        hidden_tasks,
        query,
    )
    fixed_gap = candidate_result.accuracy - fixed_result.accuracy
    low_signal_gap = (
        candidate_result.low_signal_accuracy - fixed_result.low_signal_accuracy
    )
    random_gap = candidate_result.accuracy - random_decoder["median_accuracy"]
    gate = (
        fit.candidates_checked >= 100
        and fit.development_accuracy >= 0.985
        and fit.development_move_accuracy >= 0.97
        and fit.development_accept_accuracy >= 0.97
        and candidate_result.accuracy >= 0.985
        and candidate_result.low_signal_accuracy >= 0.95
        and candidate_result.invalid_rate <= 0.01
        and fixed_gap >= 0.01
        and low_signal_gap >= 0.03
        and random_gap >= 0.50
        and abs(fit.threshold - 0.20) >= 0.01
    )
    return {
        "status": "data_calibrated_decoder_candidate" if gate else "not_yet",
        "claim_scope": "the decoder boundary is selected from every empirical decision interval rather than a hand-listed threshold family, frozen, and transferred to unseen weak-signal tree interventions; this is calibrated decision-stump induction, not a world breakthrough",
        "seed": seed,
        "candidate_gate": gate,
        "threshold_fit": fit.__dict__,
        "training_example_count": len(training),
        "development_example_count": len(development),
        "frozen_decoder_digest": frozen_digest,
        "candidate": candidate_result.__dict__,
        "fixed_020_control": fixed_result.__dict__,
        "fixed_accuracy_gap": fixed_gap,
        "low_signal_accuracy_gap": low_signal_gap,
        "random_decoder_control": random_decoder,
        "random_accuracy_gap": random_gap,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1201)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "threshold": report["threshold_fit"]["threshold"],
        "accuracy": report["candidate"]["accuracy"],
        "low_signal_accuracy": report["candidate"]["low_signal_accuracy"],
        "fixed_gap": report["fixed_accuracy_gap"],
    }, indent=2))


if __name__ == "__main__":
    main()
