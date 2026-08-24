from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

import ipwm_eval_r1 as r1

NULL_SEEDS = [101, 211, 307, 401, 503]


def _context_features(rec: dict) -> dict[str, float]:
    out = {"bias": 1.0}
    out.update(r1.categorical("language", rec["language"]))
    out.update(r1.categorical("environment", rec["environment_id"]))
    return out


def _intervention_context_products(rec: dict) -> dict[str, float]:
    out = {}
    iv = {k: float(v) for k, v in rec["intervention_features"].items()}
    for k, v in iv.items():
        out[f"li:{rec['language']}|{k}"] = v
        out[f"ei:{rec['environment_id']}|{k}"] = v
    return out


def no_alignment_features(rec: dict) -> dict[str, float]:
    out = _context_features(rec)
    out.update({f"p:{k}": float(v) for k, v in rec["program_features"].items()})
    out.update({f"i:{k}": float(v) for k, v in rec["intervention_features"].items()})
    out.update(_intervention_context_products(rec))
    return out


def aligned_features(rec: dict) -> dict[str, float]:
    out = no_alignment_features(rec)
    p = {k: float(v) for k, v in rec["program_features"].items()}
    i = {k: float(v) for k, v in rec["intervention_features"].items()}
    for key in sorted(set(p) & set(i)):
        out[f"a:{key}"] = p[key] * i[key]
    return out


def static_only_features(rec: dict) -> dict[str, float]:
    out = _context_features(rec)
    out.update({f"p:{k}": float(v) for k, v in rec["program_features"].items()})
    return out


def intervention_only_features(rec: dict) -> dict[str, float]:
    out = _context_features(rec)
    out.update({f"i:{k}": float(v) for k, v in rec["intervention_features"].items()})
    out.update(_intervention_context_products(rec))
    return out


def _joint_permutation(records: list[dict], seed: int, strata: tuple[str, ...] | None) -> list[int]:
    rng = random.Random(seed)
    groups: dict[tuple, list[int]] = defaultdict(list)
    if strata is None:
        groups[("ALL",)] = list(range(len(records)))
    else:
        for idx, rec in enumerate(records):
            groups[tuple(rec[k] for k in strata)].append(idx)
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


def fit_three(train: list[dict], test: list[dict], extractor, targets=None):
    if targets is None:
        yp = [float(x["positive_speedup"]) for x in train]
        yl = [float(x["log_speedup"]) for x in train]
        yv = [float(x["valid_label"]) for x in train]
    else:
        yp, yl, yv = targets
    mp = r1.SparseLinear(binary=True)
    ml = r1.SparseLinear(binary=False, lr=0.12)
    mv = r1.SparseLinear(binary=True)
    mp.fit(train, list(yp), extractor)
    ml.fit(train, list(yl), extractor)
    mv.fit(train, list(yv), extractor)
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


def metric_pack(test: list[dict], pred_pos: list[float], pred_log: list[float], pred_valid=None) -> dict:
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
    keys = sorted({k for row in rows for k in row})
    return {k: mean(float(row[k]) for row in rows if k in row) for k in keys}


def evaluate_group_holdout(records: list[dict], group_key: str) -> dict:
    groups = sorted({x[group_key] for x in records})
    folds = []
    global_null: dict[int, list[dict]] = {s: [] for s in NULL_SEEDS}
    stratified_null: dict[int, list[dict]] = {s: [] for s in NULL_SEEDS}

    for fold_no, group in enumerate(groups):
        train = [x for x in records if x[group_key] != group]
        test = [x for x in records if x[group_key] == group]
        if not train or not test:
            continue

        fp, fl, fv = fit_three(train, test, aligned_features)
        np, nl = fit_two(train, test, no_alignment_features)
        sp, sl = fit_two(train, test, static_only_features)
        ip, il = fit_two(train, test, intervention_only_features)
        qp, ql = r1.frequency_predictions(train, test)

        fold = {
            "group": group,
            "train_n": len(train),
            "test_n": len(test),
            "full": metric_pack(test, fp, fl, fv),
            "no_alignment": metric_pack(test, np, nl),
            "static_only": metric_pack(test, sp, sl),
            "intervention_only": metric_pack(test, ip, il),
            "frequency": metric_pack(test, qp, ql),
        }

        for seed in NULL_SEEDS:
            targets = permuted_targets(train, seed + fold_no * 10000, None)
            pp, pl, pv = fit_three(train, test, aligned_features, targets)
            global_null[seed].append(metric_pack(test, pp, pl, pv))

            targets = permuted_targets(
                train,
                seed + fold_no * 10000 + 5000,
                ("intervention_family", "repository_family"),
            )
            pp, pl, pv = fit_three(train, test, aligned_features, targets)
            stratified_null[seed].append(metric_pack(test, pp, pl, pv))

        folds.append(fold)

    macro = {
        name: average_metric_dict([f[name] for f in folds])
        for name in ["full", "no_alignment", "static_only", "intervention_only", "frequency"]
    }
    gnull = [{"seed": s, "macro": average_metric_dict(global_null[s])} for s in NULL_SEEDS]
    snull = [{"seed": s, "macro": average_metric_dict(stratified_null[s])} for s in NULL_SEEDS]

    ga = [x["macro"]["positive_speedup_auroc"] for x in gnull]
    gs = [x["macro"]["spearman_log_speedup"] for x in gnull]
    gv = [x["macro"]["validity_auroc"] for x in gnull]
    sa = [x["macro"]["positive_speedup_auroc"] for x in snull]
    ss = [x["macro"]["spearman_log_speedup"] for x in snull]
    sv = [x["macro"]["validity_auroc"] for x in snull]
    full = macro["full"]
    no_alignment = macro["no_alignment"]
    freq = macro["frequency"]

    return {
        "group_key": group_key,
        "group_count": len(folds),
        "record_count": sum(f["test_n"] for f in folds),
        "folds": folds,
        "macro": macro,
        "global_null": gnull,
        "stratified_null": snull,
        "null_summary": {
            "global_mean_positive_auroc": mean(ga),
            "global_max_positive_auroc": max(ga),
            "global_mean_spearman": mean(gs),
            "global_max_abs_spearman": max(abs(x) for x in gs),
            "global_mean_validity_auroc": mean(gv),
            "global_max_validity_auroc": max(gv),
            "stratified_mean_positive_auroc": mean(sa),
            "stratified_max_positive_auroc": max(sa),
            "stratified_mean_spearman": mean(ss),
            "stratified_max_abs_spearman": max(abs(x) for x in ss),
            "stratified_mean_validity_auroc": mean(sv),
            "stratified_max_validity_auroc": max(sv),
        },
        "deltas": {
            "full_minus_no_alignment_positive_auroc": full["positive_speedup_auroc"] - no_alignment["positive_speedup_auroc"],
            "full_minus_no_alignment_spearman": full["spearman_log_speedup"] - no_alignment["spearman_log_speedup"],
            "full_minus_stratified_null_positive_auroc": full["positive_speedup_auroc"] - max(sa),
            "full_minus_stratified_null_spearman": full["spearman_log_speedup"] - max(ss),
            "relative_top_k_gain_over_frequency": (
                full["top_k_mean_observed_speedup"] / freq["top_k_mean_observed_speedup"] - 1.0
                if freq["top_k_mean_observed_speedup"] > 0 else 0.0
            ),
        },
    }


def evaluate(records: list[dict]) -> dict:
    return {
        "schema": "lexigen-v8-ipwm-evaluation-r3",
        "representation": "aligned_primitive_susceptibility_x_action",
        "intervention_family_identity_used_as_feature": False,
        "null_seeds": NULL_SEEDS,
        "repository_holdout": evaluate_group_holdout(records, "repository_id"),
        "repository_family_holdout": evaluate_group_holdout(records, "repository_family"),
        "language_holdout": evaluate_group_holdout(records, "language"),
        "intervention_family_holdout": evaluate_group_holdout(records, "intervention_family"),
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
