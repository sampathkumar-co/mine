from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

import ipwm_eval_r1 as r1

NULL_SEEDS = [101, 211, 307]


def no_cross_features(rec: dict) -> dict[str, float]:
    out = {"bias": 1.0}
    out.update({f"p:{k}": float(v) for k, v in rec["program_features"].items()})
    out.update({f"i:{k}": float(v) for k, v in rec["intervention_features"].items()})
    out.update(r1.categorical("intervention_family", rec["intervention_family"]))
    out.update(r1.categorical("language", rec["language"]))
    return out


def _joint_permutation(records: list[dict], seed: int, strata: tuple[str, ...] | None) -> list[int]:
    rng = random.Random(seed)
    groups: dict[tuple, list[int]] = defaultdict(list)
    if strata is None:
        groups[("ALL",)] = list(range(len(records)))
    else:
        for i, rec in enumerate(records):
            groups[tuple(rec[k] for k in strata)].append(i)
    perm = list(range(len(records)))
    for key in sorted(groups, key=lambda x: tuple(map(str, x))):
        dst = list(groups[key])
        src = list(dst)
        rng.shuffle(src)
        for d, s in zip(dst, src):
            perm[d] = s
    return perm


def permuted_targets(records: list[dict], seed: int, strata: tuple[str, ...] | None):
    perm = _joint_permutation(records, seed, strata)
    return (
        [float(records[i]["positive_speedup"]) for i in perm],
        [float(records[i]["log_speedup"]) for i in perm],
        [float(records[i]["valid_label"]) for i in perm],
    )


def fit_three(train: list[dict], test: list[dict], extractor, *, targets=None):
    if targets is None:
        y_pos = [float(x["positive_speedup"]) for x in train]
        y_log = [float(x["log_speedup"]) for x in train]
        y_valid = [float(x["valid_label"]) for x in train]
    else:
        y_pos, y_log, y_valid = targets
    mp = r1.SparseLinear(binary=True)
    ml = r1.SparseLinear(binary=False, lr=0.12)
    mv = r1.SparseLinear(binary=True)
    mp.fit(train, list(y_pos), extractor)
    ml.fit(train, list(y_log), extractor)
    mv.fit(train, list(y_valid), extractor)
    return (
        [mp.predict(x, extractor) for x in test],
        [ml.predict(x, extractor) for x in test],
        [mv.predict(x, extractor) for x in test],
    )


def fit_two(train: list[dict], test: list[dict], extractor):
    yp = [float(x["positive_speedup"]) for x in train]
    yl = [float(x["log_speedup"]) for x in train]
    mp = r1.SparseLinear(binary=True)
    ml = r1.SparseLinear(binary=False, lr=0.12)
    mp.fit(train, yp, extractor)
    ml.fit(train, yl, extractor)
    return [mp.predict(x, extractor) for x in test], [ml.predict(x, extractor) for x in test]


def metric_pack(test: list[dict], pred_pos: list[float], pred_log: list[float], pred_valid: list[float] | None = None) -> dict:
    yp = [x["positive_speedup"] for x in test]
    yl = [x["log_speedup"] for x in test]
    yv = [x["valid_label"] for x in test]
    out = {
        "positive_speedup_auroc": r1.auc(yp, pred_pos),
        "spearman_log_speedup": r1.spearman(yl, pred_log),
        "brier_positive_speedup": r1.brier(yp, pred_pos),
        "top_k_mean_observed_speedup": r1.top_k_mean_speedup(test, pred_log),
        "confidence_gap": r1.confidence_gap(yp, pred_pos),
    }
    if pred_valid is not None:
        out["validity_auroc"] = r1.auc(yv, pred_valid)
    return out


def average_metric_dict(rows: list[dict]) -> dict:
    keys = sorted({k for x in rows for k in x})
    return {k: mean(float(x[k]) for x in rows if k in x) for k in keys}


def evaluate_group_holdout(records: list[dict], group_key: str) -> dict:
    groups = sorted({x[group_key] for x in records})
    folds = []
    global_null_by_seed: dict[int, list[dict]] = {s: [] for s in NULL_SEEDS}
    strat_null_by_seed: dict[int, list[dict]] = {s: [] for s in NULL_SEEDS}

    for fold_no, group in enumerate(groups):
        train = [x for x in records if x[group_key] != group]
        test = [x for x in records if x[group_key] == group]
        if not train or not test:
            continue

        full_pos, full_log, full_valid = fit_three(train, test, r1.full_features)
        nc_pos, nc_log = fit_two(train, test, no_cross_features)
        st_pos, st_log = fit_two(train, test, r1.static_features)
        int_pos, int_log = fit_two(train, test, r1.intervention_features)
        freq_pos, freq_log = r1.frequency_predictions(train, test)

        fold = {
            "group": group,
            "train_n": len(train),
            "test_n": len(test),
            "full": metric_pack(test, full_pos, full_log, full_valid),
            "no_cross": metric_pack(test, nc_pos, nc_log),
            "static_only": metric_pack(test, st_pos, st_log),
            "intervention_only": metric_pack(test, int_pos, int_log),
            "frequency": metric_pack(test, freq_pos, freq_log),
        }

        for seed in NULL_SEEDS:
            gtargets = permuted_targets(train, seed + fold_no * 10000, None)
            p, l, v = fit_three(train, test, r1.full_features, targets=gtargets)
            global_null_by_seed[seed].append(metric_pack(test, p, l, v))

            stargets = permuted_targets(
                train,
                seed + fold_no * 10000 + 5000,
                ("intervention_family", "repository_family"),
            )
            p, l, v = fit_three(train, test, r1.full_features, targets=stargets)
            strat_null_by_seed[seed].append(metric_pack(test, p, l, v))

        folds.append(fold)

    macro = {
        name: average_metric_dict([f[name] for f in folds])
        for name in ["full", "no_cross", "static_only", "intervention_only", "frequency"]
    }

    global_null = [
        {"seed": seed, "macro": average_metric_dict(global_null_by_seed[seed])}
        for seed in NULL_SEEDS
    ]
    strat_null = [
        {"seed": seed, "macro": average_metric_dict(strat_null_by_seed[seed])}
        for seed in NULL_SEEDS
    ]

    full = macro["full"]
    no_cross = macro["no_cross"]
    static = macro["static_only"]
    freq = macro["frequency"]
    global_aurocs = [x["macro"]["positive_speedup_auroc"] for x in global_null]
    global_spears = [x["macro"]["spearman_log_speedup"] for x in global_null]
    global_valid = [x["macro"].get("validity_auroc", 0.5) for x in global_null]
    strat_aurocs = [x["macro"]["positive_speedup_auroc"] for x in strat_null]
    strat_spears = [x["macro"]["spearman_log_speedup"] for x in strat_null]
    strat_valid = [x["macro"].get("validity_auroc", 0.5) for x in strat_null]

    out = {
        "group_key": group_key,
        "group_count": len(folds),
        "record_count": sum(f["test_n"] for f in folds),
        "folds": folds,
        "macro": macro,
        "global_null": global_null,
        "stratified_null": strat_null,
        "null_summary": {
            "global_mean_positive_auroc": mean(global_aurocs),
            "global_max_positive_auroc": max(global_aurocs),
            "global_mean_spearman": mean(global_spears),
            "global_max_abs_spearman": max(abs(x) for x in global_spears),
            "global_mean_validity_auroc": mean(global_valid),
            "global_max_validity_auroc": max(global_valid),
            "stratified_mean_positive_auroc": mean(strat_aurocs),
            "stratified_max_positive_auroc": max(strat_aurocs),
            "stratified_mean_spearman": mean(strat_spears),
            "stratified_max_abs_spearman": max(abs(x) for x in strat_spears),
            "stratified_mean_validity_auroc": mean(strat_valid),
            "stratified_max_validity_auroc": max(strat_valid),
        },
        "deltas": {
            "full_minus_no_cross_positive_auroc": full["positive_speedup_auroc"] - no_cross["positive_speedup_auroc"],
            "full_minus_no_cross_spearman": full["spearman_log_speedup"] - no_cross["spearman_log_speedup"],
            "full_minus_static_positive_auroc": full["positive_speedup_auroc"] - static["positive_speedup_auroc"],
            "full_minus_static_spearman": full["spearman_log_speedup"] - static["spearman_log_speedup"],
            "full_minus_stratified_null_positive_auroc": full["positive_speedup_auroc"] - max(strat_aurocs),
            "full_minus_stratified_null_spearman": full["spearman_log_speedup"] - max(strat_spears),
            "relative_top_k_gain_over_frequency": (
                full["top_k_mean_observed_speedup"] / freq["top_k_mean_observed_speedup"] - 1.0
                if freq["top_k_mean_observed_speedup"] > 0 else 0.0
            ),
        },
    }
    return out


def evaluate(records: list[dict]) -> dict:
    languages = sorted({x["language"] for x in records})
    interventions = sorted({x["intervention_family"] for x in records})
    return {
        "schema": "lexigen-v8-ipwm-evaluation-r2",
        "repository_holdout": evaluate_group_holdout(records, "repository_id"),
        "repository_family_holdout": evaluate_group_holdout(records, "repository_family"),
        "language_holdout": evaluate_group_holdout(records, "language") if len(languages) >= 2 else None,
        "intervention_family_holdout": evaluate_group_holdout(records, "intervention_family") if len(interventions) >= 3 else None,
        "null_seeds": NULL_SEEDS,
        "scientific_transfer_evidence": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    records = r1.load_jsonl(args.input)
    result = evaluate(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
