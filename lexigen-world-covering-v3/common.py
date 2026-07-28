from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import time
import urllib.request
from array import array
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SNAPSHOT_URL = "https://zenodo.org/records/19735294/files/coverdata.json?download=1"
SNAPSHOT_MD5 = "b2c626b07f216aac830d344eff5ad523"
REFERENCE_DATE = datetime(2026, 4, 24, tzinfo=timezone.utc)
KEY_RE = re.compile(r"^C\((\d+),(\d+),(\d+)\)$")

V1_SEED = (
    "32c897005c91865319f1b7da264b6162fc1ff4de|"
    "b2c626b07f216aac830d344eff5ad523|LEXIGEN_WORLD_COVERING_V1"
)
V2_SEED = (
    "32c897005c91865319f1b7da264b6162fc1ff4de|"
    "b2c626b07f216aac830d344eff5ad523|LEXIGEN_WORLD_COVERING_V2"
)
V3_SEED = (
    "32c897005c91865319f1b7da264b6162fc1ff4de|"
    "b2c626b07f216aac830d344eff5ad523|LEXIGEN_WORLD_COVERING_V3"
)

TARGET_COUNT = 3
GREEDY_DETERMINISTIC = 10
GREEDY_RANDOMIZED = 30
REPAIR_RESTARTS = 8
REPAIR_SECONDS = 180.0
RESTRICTED_CP_SECONDS = 180.0
FULL_CP_SECONDS = 1250.0
CP_WORKERS = 4


@dataclass(frozen=True)
class Target:
    name: str
    v: int
    k: int
    t: int
    upper: int
    lower: int
    last_update: str
    gap: int
    candidate_blocks: int
    t_subsets: int
    incidence_edges: int
    opportunity_score: float
    tie_break: str


@dataclass
class Incidence:
    blocks: list[tuple[int, ...]]
    tsets: list[tuple[int, ...]]
    cover_by_block: list[array]
    blocks_by_t: list[array]


def download_snapshot() -> bytes:
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            request = urllib.request.Request(
                SNAPSHOT_URL, headers={"User-Agent": "LEXIGEN-world-covering-v3"}
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                data = response.read()
            digest = hashlib.md5(data).hexdigest()
            if digest != SNAPSHOT_MD5:
                raise RuntimeError(f"snapshot MD5 mismatch: {digest} != {SNAPSHOT_MD5}")
            return data
        except Exception as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError("could not download frozen covering snapshot") from last_error


def load_snapshot_json() -> dict[str, object]:
    raw = json.loads(download_snapshot())
    if not isinstance(raw, dict):
        raise TypeError("coverdata root must be a dictionary")
    return raw


def parse_date(text: str) -> datetime:
    value = (text or "").strip()
    if not value:
        return datetime(1996, 1, 1, tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return datetime(1996, 1, 1, tzinfo=timezone.utc)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def last_update(entry: dict[str, object]) -> str:
    improvements = entry.get("imps") or []
    if isinstance(improvements, list) and improvements:
        row = improvements[0]
        if isinstance(row, list) and len(row) >= 4:
            return str(row[3])
    return ""


def make_target(
    name: str,
    v: int,
    k: int,
    t: int,
    upper: int,
    lower: int,
    updated: str,
    score: float,
    tie: str,
) -> Target:
    candidates = math.comb(v, k)
    tsets = math.comb(v, t)
    edges = candidates * math.comb(k, t)
    return Target(
        name=name,
        v=v,
        k=k,
        t=t,
        upper=upper,
        lower=lower,
        last_update=updated,
        gap=upper - lower,
        candidate_blocks=candidates,
        t_subsets=tsets,
        incidence_edges=edges,
        opportunity_score=score,
        tie_break=tie,
    )


def basic_parameters(name: str, entry: dict[str, object]) -> tuple[int, int, int, int, int] | None:
    match = KEY_RE.match(name)
    if not match:
        return None
    v, k, t = map(int, match.groups())
    upper = int(entry["size"])
    lower = int(entry["low_bd"])
    if not (
        10 <= v <= 22
        and 4 <= k <= min(10, v - 2)
        and 3 <= t <= min(5, k - 1)
        and upper - lower >= 2
        and upper <= 120
        and upper - 1 >= lower
    ):
        return None
    return v, k, t, upper, lower


def v1_target(name: str, entry: dict[str, object]) -> Target | None:
    params = basic_parameters(name, entry)
    if params is None:
        return None
    v, k, t, upper, lower = params
    if upper > 100:
        return None
    candidates = math.comb(v, k)
    tsets = math.comb(v, t)
    edges = candidates * math.comb(k, t)
    if candidates > 50_000 or tsets > 5_000 or edges > 3_000_000:
        return None
    updated = last_update(entry)
    age = max(0.0, (REFERENCE_DATE - parse_date(updated)).days / 365.25)
    score = (
        (float(upper - lower) ** 1.35)
        * (1.0 + min(age, 25.0) / 18.0)
        * ((100.0 / float(upper)) ** 0.15)
        / (float(edges) ** 0.35)
    )
    tie = hashlib.sha256(f"{V1_SEED}|{name}".encode()).hexdigest()
    return make_target(name, v, k, t, upper, lower, updated, score, tie)


def v2_target(name: str, entry: dict[str, object]) -> Target | None:
    params = basic_parameters(name, entry)
    if params is None:
        return None
    v, k, t, upper, lower = params
    if upper > 100:
        return None
    candidates = math.comb(v, k)
    tsets = math.comb(v, t)
    edges = candidates * math.comb(k, t)
    if candidates > 60_000 or tsets > 5_000 or edges > 3_500_000:
        return None
    updated = last_update(entry)
    age = max(0.0, (REFERENCE_DATE - parse_date(updated)).days / 365.25)
    score = (
        (float(upper - lower) ** 1.45)
        * (1.0 + min(age, 25.0) / 16.0)
        * ((100.0 / float(upper)) ** 0.12)
        / (float(edges) ** 0.32)
    )
    tie = hashlib.sha256(f"{V2_SEED}|{name}".encode()).hexdigest()
    return make_target(name, v, k, t, upper, lower, updated, score, tie)


def v3_target(name: str, entry: dict[str, object]) -> Target | None:
    params = basic_parameters(name, entry)
    if params is None:
        return None
    v, k, t, upper, lower = params
    candidates = math.comb(v, k)
    tsets = math.comb(v, t)
    edges = candidates * math.comb(k, t)
    if candidates > 75_000 or tsets > 6_500 or edges > 4_500_000:
        return None
    updated = last_update(entry)
    age = max(0.0, (REFERENCE_DATE - parse_date(updated)).days / 365.25)
    score = (
        (float(upper - lower) ** 1.55)
        * (1.0 + min(age, 25.0) / 15.0)
        * ((120.0 / float(upper)) ** 0.10)
        / (float(edges) ** 0.30)
    )
    tie = hashlib.sha256(f"{V3_SEED}|{name}".encode()).hexdigest()
    return make_target(name, v, k, t, upper, lower, updated, score, tie)


def select_with_cap(items: list[Target], count: int) -> list[Target]:
    items.sort(key=lambda x: (-x.opportunity_score, x.tie_break, x.name))
    selected: list[Target] = []
    pair_counts: dict[tuple[int, int], int] = {}
    for target in items:
        pair = (target.k, target.t)
        if pair_counts.get(pair, 0) >= 2:
            continue
        selected.append(target)
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        if len(selected) == count:
            return selected
    raise RuntimeError(f"found only {len(selected)} of {count} targets")


def select_target_lineage(
    coverdata: dict[str, object],
) -> tuple[list[Target], list[Target], list[Target]]:
    v1_items = [
        target
        for name, raw in coverdata.items()
        if isinstance(raw, dict) and (target := v1_target(name, raw)) is not None
    ]
    v1 = select_with_cap(v1_items, TARGET_COUNT)
    v1_names = {x.name for x in v1}

    v2_items = [
        target
        for name, raw in coverdata.items()
        if name not in v1_names
        and isinstance(raw, dict)
        and (target := v2_target(name, raw)) is not None
    ]
    v2 = select_with_cap(v2_items, TARGET_COUNT)
    reserved = v1_names | {x.name for x in v2}

    v3_items = [
        target
        for name, raw in coverdata.items()
        if name not in reserved
        and isinstance(raw, dict)
        and (target := v3_target(name, raw)) is not None
    ]
    v3 = select_with_cap(v3_items, TARGET_COUNT)
    if reserved.intersection(x.name for x in v3):
        raise RuntimeError("v3 overlaps reserved v1/v2 target lineage")
    return v1, v2, v3


def build_incidence(target: Target) -> Incidence:
    tsets = list(itertools.combinations(range(target.v), target.t))
    index = {subset: i for i, subset in enumerate(tsets)}
    blocks: list[tuple[int, ...]] = []
    cover_by_block: list[array] = []
    blocks_by_t = [array("I") for _ in tsets]
    for block_index, block in enumerate(itertools.combinations(range(target.v), target.k)):
        covered = array("I", (index[s] for s in itertools.combinations(block, target.t)))
        blocks.append(block)
        cover_by_block.append(covered)
        for subset_index in covered:
            blocks_by_t[subset_index].append(block_index)
    if len(blocks) != target.candidate_blocks or len(tsets) != target.t_subsets:
        raise RuntimeError("incidence dimensions mismatch")
    return Incidence(blocks, tsets, cover_by_block, blocks_by_t)


def coverage_counts(incidence: Incidence, selected: list[int]) -> list[int]:
    counts = [0] * len(incidence.tsets)
    for block in selected:
        for subset in incidence.cover_by_block[block]:
            counts[subset] += 1
    return counts


def verify_design(target: Target, blocks: list[tuple[int, ...]]) -> tuple[bool, str]:
    if len(blocks) != len(set(blocks)):
        return False, "duplicate blocks"
    universe = set(range(target.v))
    for block in blocks:
        if len(block) != target.k or tuple(sorted(block)) != block:
            return False, "malformed block"
        if not set(block).issubset(universe):
            return False, "point outside universe"
    covered: set[tuple[int, ...]] = set()
    for block in blocks:
        covered.update(itertools.combinations(block, target.t))
    expected = math.comb(target.v, target.t)
    if len(covered) != expected:
        return False, f"covered {len(covered)} of {expected} t-subsets"
    if len(blocks) >= target.upper:
        return False, "not below frozen upper bound"
    return True, "verified"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
