from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable

REQUIRED = {
    "record_id", "repository_id", "repository_family", "language", "environment_id",
    "program_features", "intervention_id", "intervention_family", "intervention_features",
    "valid", "runtime_before", "runtime_after", "measurement_repetitions", "provenance",
}


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-min(x, 60.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(x, -60.0))
    return z / (1.0 + z)


def validate_record(r: dict) -> None:
    missing = REQUIRED - set(r)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if not isinstance(r["program_features"], dict) or not isinstance(r["intervention_features"], dict):
        raise ValueError("program_features and intervention_features must be objects")
    if float(r["runtime_before"]) <= 0 or float(r["runtime_after"]) <= 0:
        raise ValueError("runtimes must be positive")
    if int(r["measurement_repetitions"]) < 1:
        raise ValueError("measurement_repetitions must be >=1")
    for group in (r["program_features"], r["intervention_features"]):
        for key, value in group.items():
            if not isinstance(key, str) or not isinstance(value, (int, float)):
                raise ValueError("feature maps must contain numeric values keyed by strings")


def enrich(r: dict) -> dict:
    x = dict(r)
    speedup = float(r["runtime_before"]) / float(r["runtime_after"])
    x["speedup"] = speedup
    x["log_speedup"] = math.log(speedup)
    x["positive_speedup"] = 1 if speedup > 1.0 else 0
    x["valid_label"] = 1 if bool(r["valid"]) else 0
    return x


def load_jsonl(path: Path) -> list[dict]:
    records = []
    seen = set()
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        validate_record(r)
        if r["record_id"] in seen:
            raise ValueError(f"duplicate record_id at line {i}: {r['record_id']}")
        seen.add(r["record_id"])
        records.append(enrich(r))
    if not records:
        raise ValueError("empty corpus")
    return records


def categorical(name: str, value: str) -> dict[str, float]:
    return {f"{name}={value}": 1.0}


def full_features(r: dict) -> dict[str, float]:
    out = {"bias": 1.0}
    p = {f"p:{k}": float(v) for k, v in r["program_features"].items()}
    q = {f"i:{k}": float(v) for k, v in r["intervention_features"].items()}
    out.update(p)
    out.update(q)
    out.update(categorical("intervention_family", r["intervention_family"]))
    out.update(categorical("language", r["language"]))
    for pk, pv in p.items():
        for ik, iv in q.items():
            out[f"x:{pk}|{ik}"] = pv * iv
    return out


def static_features(r: dict) -> dict[str, float]:
    out = {"bias": 1.0}
    out.update({f"p:{k}": float(v) for k, v in r["program_features"].items()})
    out.update(categorical("language", r["language"]))
    return out


def intervention_features(r: dict) -> dict[str, float]:
    out = {"bias": 1.0}
    out.update({f"i:{k}": float(v) for k, v in r["intervention_features"].items()})
    out.update(categorical("intervention_family", r["intervention_family"]))
    return out


class SparseLinear:
    def __init__(self, *, binary: bool, l2: float = 0.01, epochs: int = 350, lr: float = 0.25):
        self.binary = binary
        self.l2 = l2
        self.epochs = epochs
        self.lr = lr
        self.vocab: dict[str, int] = {}
        self.scale: list[float] = []
        self.w: list[float] = []

    def _prepare(self, rows: list[dict[str, float]]) -> list[list[tuple[int, float]]]:
        names = sorted({k for r in rows for k in r})
        self.vocab = {k: i for i, k in enumerate(names)}
        sumsq = [0.0] * len(names)
        sparse = []
        for r in rows:
            sr = []
            for k, v in r.items():
                j = self.vocab[k]
                fv = float(v)
                sumsq[j] += fv * fv
                sr.append((j, fv))
            sparse.append(sr)
        n = max(1, len(rows))
        self.scale = [max(1.0, math.sqrt(s / n)) for s in sumsq]
        return [[(j, v / self.scale[j]) for j, v in row] for row in sparse]

    def fit(self, records: list[dict], target: list[float], extractor: Callable[[dict], dict[str, float]]) -> None:
        raw = [extractor(r) for r in records]
        rows = self._prepare(raw)
        self.w = [0.0] * len(self.vocab)
        n = max(1, len(rows))
        for epoch in range(self.epochs):
            grad = [0.0] * len(self.w)
            for row, y in zip(rows, target):
                z = sum(self.w[j] * v for j, v in row)
                pred = sigmoid(z) if self.binary else z
                err = pred - y
                for j, v in row:
                    grad[j] += err * v
            step = self.lr / math.sqrt(1.0 + epoch * 0.05)
            for j in range(len(self.w)):
                reg = 0.0 if list(self.vocab.keys())[j] == "bias" else self.l2 * self.w[j]
                self.w[j] -= step * ((grad[j] / n) + reg)

    def predict(self, r: dict, extractor: Callable[[dict], dict[str, float]]) -> float:
        z = 0.0
        for k, v in extractor(r).items():
            j = self.vocab.get(k)
            if j is not None:
                z += self.w[j] * (float(v) / self.scale[j])
        return sigmoid(z) if self.binary else z


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(order):
        end = pos + 1
        while end < len(order) and values[order[end]] == values[order[pos]]:
            end += 1
        rank = (pos + 1 + end) / 2.0
        for k in range(pos, end):
            ranks[order[k]] = rank
        pos = end
    return ranks


def auc(y: list[int], score: list[float]) -> float:
    npos = sum(y)
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return 0.5
    ranks = average_ranks(score)
    sum_pos = sum(r for r, yy in zip(ranks, y) if yy)
    return (sum_pos - npos * (npos + 1) / 2.0) / (npos * nneg)


def pearson(a: list[float], b: list[float]) -> float:
    if len(a) < 2:
        return 0.0
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va <= 1e-15 or vb <= 1e-15:
        return 0.0
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)


def spearman(a: list[float], b: list[float]) -> float:
    return pearson(average_ranks(a), average_ranks(b))


def brier(y: list[int], p: list[float]) -> float:
    return sum((yy - pp) ** 2 for yy, pp in zip(y, p)) / max(1, len(y))


def top_k_mean_speedup(records: list[dict], pred_log: list[float], frac: float = 0.2) -> float:
    k = max(1, int(math.ceil(len(records) * frac)))
    idx = sorted(range(len(records)), key=lambda i: pred_log[i], reverse=True)[:k]
    return sum(records[i]["speedup"] for i in idx) / k


def confidence_gap(y: list[int], p: list[float]) -> float:
    if len(y) < 4:
        return 0.0
    conf = [abs(x - 0.5) for x in p]
    order = sorted(range(len(y)), key=lambda i: conf[i])
    q = max(1, len(y) // 4)
    lo = order[:q]
    hi = order[-q:]
    def acc(ids):
        return sum((p[i] >= 0.5) == bool(y[i]) for i in ids) / len(ids)
    return acc(hi) - acc(lo)


def frequency_predictions(train: list[dict], test: list[dict]) -> tuple[list[float], list[float]]:
    pos = defaultdict(list)
    logs = defaultdict(list)
    for r in train:
        pos[r["intervention_family"]].append(r["positive_speedup"])
        logs[r["intervention_family"]].append(r["log_speedup"])
    gp = sum(r["positive_speedup"] for r in train) / len(train)
    gl = sum(r["log_speedup"] for r in train) / len(train)
    pp, pl = [], []
    for r in test:
        fam = r["intervention_family"]
        pp.append(sum(pos[fam]) / len(pos[fam]) if pos[fam] else gp)
        pl.append(sum(logs[fam]) / len(logs[fam]) if logs[fam] else gl)
    return pp, pl


def fit_predict(train: list[dict], test: list[dict], extractor, *, shuffled: bool = False, seed: int = 17):
    y_pos = [float(r["positive_speedup"]) for r in train]
    y_log = [float(r["log_speedup"]) for r in train]
    y_valid = [float(r["valid_label"]) for r in train]
    if shuffled:
        rng = random.Random(seed)
        rng.shuffle(y_pos)
        rng.shuffle(y_log)
        rng.shuffle(y_valid)
    m_pos = SparseLinear(binary=True)
    m_log = SparseLinear(binary=False, lr=0.12)
    m_valid = SparseLinear(binary=True)
    m_pos.fit(train, y_pos, extractor)
    m_log.fit(train, y_log, extractor)
    m_valid.fit(train, y_valid, extractor)
    return (
        [m_pos.predict(r, extractor) for r in test],
        [m_log.predict(r, extractor) for r in test],
        [m_valid.predict(r, extractor) for r in test],
    )


def evaluate_group_holdout(records: list[dict], group_key: str) -> dict:
    pred = defaultdict(list)
    obs = []
    fold_summaries = []
    groups = sorted({r[group_key] for r in records})
    for fold_no, group in enumerate(groups):
        train = [r for r in records if r[group_key] != group]
        test = [r for r in records if r[group_key] == group]
        if not train or not test:
            continue
        f_pos, f_log = frequency_predictions(train, test)
        full_pos, full_log, full_valid = fit_predict(train, test, full_features)
        static_pos, static_log, _ = fit_predict(train, test, static_features)
        int_pos, int_log, _ = fit_predict(train, test, intervention_features)
        shuf_pos, shuf_log, shuf_valid = fit_predict(train, test, full_features, shuffled=True, seed=1000 + fold_no)
        for r, fp, fl, xp, xl, sp, sl, ip, il, hp, hl, hv, xv in zip(
            test, f_pos, f_log, full_pos, full_log, static_pos, static_log,
            int_pos, int_log, shuf_pos, shuf_log, shuf_valid, full_valid,
        ):
            obs.append(r)
            pred["frequency_pos"].append(fp); pred["frequency_log"].append(fl)
            pred["full_pos"].append(xp); pred["full_log"].append(xl); pred["full_valid"].append(xv)
            pred["static_pos"].append(sp); pred["static_log"].append(sl)
            pred["intervention_pos"].append(ip); pred["intervention_log"].append(il)
            pred["shuffled_pos"].append(hp); pred["shuffled_log"].append(hl); pred["shuffled_valid"].append(hv)
        fold_summaries.append({"group": group, "train_n": len(train), "test_n": len(test)})

    y_pos = [r["positive_speedup"] for r in obs]
    y_log = [r["log_speedup"] for r in obs]
    y_valid = [r["valid_label"] for r in obs]
    out = {
        "group_key": group_key,
        "group_count": len(groups),
        "record_count": len(obs),
        "folds": fold_summaries,
        "full": {
            "validity_auroc": auc(y_valid, pred["full_valid"]),
            "positive_speedup_auroc": auc(y_pos, pred["full_pos"]),
            "spearman_predicted_vs_observed_log_speedup": spearman(y_log, pred["full_log"]),
            "brier_positive_speedup": brier(y_pos, pred["full_pos"]),
            "top_k_mean_observed_speedup": top_k_mean_speedup(obs, pred["full_log"]),
            "high_confidence_vs_low_confidence_gap": confidence_gap(y_pos, pred["full_pos"]),
        },
        "frequency": {
            "positive_speedup_auroc": auc(y_pos, pred["frequency_pos"]),
            "spearman_predicted_vs_observed_log_speedup": spearman(y_log, pred["frequency_log"]),
            "top_k_mean_observed_speedup": top_k_mean_speedup(obs, pred["frequency_log"]),
        },
        "static_only": {
            "positive_speedup_auroc": auc(y_pos, pred["static_pos"]),
            "spearman_predicted_vs_observed_log_speedup": spearman(y_log, pred["static_log"]),
        },
        "intervention_only": {
            "positive_speedup_auroc": auc(y_pos, pred["intervention_pos"]),
            "spearman_predicted_vs_observed_log_speedup": spearman(y_log, pred["intervention_log"]),
        },
        "shuffled": {
            "validity_auroc": auc(y_valid, pred["shuffled_valid"]),
            "positive_speedup_auroc": auc(y_pos, pred["shuffled_pos"]),
            "spearman_predicted_vs_observed_log_speedup": spearman(y_log, pred["shuffled_log"]),
        },
    }
    freq_top = out["frequency"]["top_k_mean_observed_speedup"]
    out["relative_top_k_gain_over_frequency_baseline"] = (
        out["full"]["top_k_mean_observed_speedup"] / freq_top - 1.0 if freq_top > 0 else 0.0
    )
    return out


def evaluate(records: list[dict]) -> dict:
    repo = evaluate_group_holdout(records, "repository_id")
    fam = evaluate_group_holdout(records, "repository_family")
    languages = sorted({r["language"] for r in records})
    lang = evaluate_group_holdout(records, "language") if len(languages) >= 2 else None
    return {
        "schema": "lexigen-v8-ipwm-evaluation-r1",
        "repository_holdout": repo,
        "repository_family_holdout": fam,
        "language_holdout": lang,
        "language_count": len(languages),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    records = load_jsonl(args.input)
    result = evaluate(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
