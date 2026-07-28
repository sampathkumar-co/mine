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

# v1 reservation must reproduce the exact frozen v1 selector.
V1_SEED_MATERIAL = (
    "32c897005c91865319f1b7da264b6162fc1ff4de|"
    "b2c626b07f216aac830d344eff5ad523|LEXIGEN_WORLD_COVERING_V1"
)
V1_TARGET_COUNT = 3

# v2 ranks only after removing the exact v1-selected names.
SEED_MATERIAL = (
    "32c897005c91865319f1b7da264b6162fc1ff4de|"
    "b2c626b07f216aac830d344eff5ad523|LEXIGEN_WORLD_COVERING_V2"
)
REFERENCE_DATE = datetime(2026, 4, 24, tzinfo=timezone.utc)
TARGET_COUNT = 3
GREEDY_DETERMINISTIC = 8
GREEDY_RANDOMIZED = 24
LOCAL_RESTARTS = 6
LOCAL_SECONDS = 120.0
CP_SAT_SECONDS = 1400.0
CP_SAT_WORKERS = 4
KEY_RE = re.compile(r"^C\((\d+),(\d+),(\d+)\)$")


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
                SNAPSHOT_URL, headers={"User-Agent": "LEXIGEN-world-covering-v2"}
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


def parse_date(text: str) -> datetime:
    value = (text or "").strip()
    if not value:
        return datetime(1996, 1, 1, tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return datetime(1996, 1, 1, tzinfo=timezone.utc)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _last_update(entry: dict[str, object]) -> str:
    improvements = entry.get("imps") or []
    if isinstance(improvements, list) and improvements:
        row = improvements[0]
        if isinstance(row, list) and len(row) >= 4:
            return str(row[3])
    return ""


def _make_target(
    name: str,
    v: int,
    k: int,
    t: int,
    upper: int,
    lower: int,
    last_update: str,
    opportunity_score: float,
    tie_break: str,
) -> Target:
    candidate_blocks = math.comb(v, k)
    t_subsets = math.comb(v, t)
    incidence_edges = candidate_blocks * math.comb(k, t)
    return Target(
        name=name,
        v=v,
        k=k,
        t=t,
        upper=upper,
        lower=lower,
        last_update=last_update,
        gap=upper - lower,
        candidate_blocks=candidate_blocks,
        t_subsets=t_subsets,
        incidence_edges=incidence_edges,
        opportunity_score=opportunity_score,
        tie_break=tie_break,
    )


def v1_target_from_entry(name: str, entry: dict[str, object]) -> Target | None:
    """Reproduce the exact frozen v1 eligibility, score, and tie-break."""
    match = KEY_RE.match(name)
    if not match:
        return None
    v, k, t = map(int, match.groups())
    upper = int(entry["size"])
    lower = int(entry["low_bd"])
    gap = upper - lower
    if not (
        10 <= v <= 22
        and 4 <= k <= min(10, v - 2)
        and 3 <= t <= min(5, k - 1)
        and gap >= 2
        and upper <= 100
        and upper - 1 >= lower
    ):
        return None
    candidate_blocks = math.comb(v, k)
    t_subsets = math.comb(v, t)
    incidence_edges = candidate_blocks * math.comb(k, t)
    if candidate_blocks > 50_000 or t_subsets > 5_000 or incidence_edges > 3_000_000:
        return None

    last_update = _last_update(entry)
    age_years = max(0.0, (REFERENCE_DATE - parse_date(last_update)).days / 365.25)
    age_factor = 1.0 + min(age_years, 25.0) / 18.0
    gap_factor = float(gap) ** 1.35
    complexity_factor = float(incidence_edges) ** 0.35
    size_factor = (100.0 / float(upper)) ** 0.15
    opportunity_score = gap_factor * age_factor * size_factor / complexity_factor
    tie_break = hashlib.sha256(f"{V1_SEED_MATERIAL}|{name}".encode()).hexdigest()
    return _make_target(
        name, v, k, t, upper, lower, last_update, opportunity_score, tie_break
    )


def target_from_entry(name: str, entry: dict[str, object]) -> Target | None:
    """Apply the frozen v2 eligibility and ranking to non-v1 targets."""
    match = KEY_RE.match(name)
    if not match:
        return None
    v, k, t = map(int, match.groups())
    upper = int(entry["size"])
    lower = int(entry["low_bd"])
    gap = upper - lower
    if not (
        10 <= v <= 22
        and 4 <= k <= min(10, v - 2)
        and 3 <= t <= min(5, k - 1)
        and gap >= 2
        and upper <= 100
        and upper - 1 >= lower
    ):
        return None
    candidate_blocks = math.comb(v, k)
    t_subsets = math.comb(v, t)
    incidence_edges = candidate_blocks * math.comb(k, t)
    if candidate_blocks > 60_000 or t_subsets > 5_000 or incidence_edges > 3_500_000:
        return None

    last_update = _last_update(entry)
    age_years = max(0.0, (REFERENCE_DATE - parse_date(last_update)).days / 365.25)
    age_factor = 1.0 + min(age_years, 25.0) / 16.0
    gap_factor = float(gap) ** 1.45
    complexity_factor = float(incidence_edges) ** 0.32
    upper_factor = (100.0 / float(upper)) ** 0.12
    opportunity_score = gap_factor * age_factor * upper_factor / complexity_factor
    tie_break = hashlib.sha256(f"{SEED_MATERIAL}|{name}".encode()).hexdigest()
    return _make_target(
        name, v, k, t, upper, lower, last_update, opportunity_score, tie_break
    )


def _select_with_pair_cap(eligible: list[Target], count: int) -> list[Target]:
    eligible.sort(key=lambda x: (-x.opportunity_score, x.tie_break, x.name))
    selected: list[Target] = []
    pair_counts: dict[tuple[int, int], int] = {}
    for target in eligible:
        pair = (target.k, target.t)
        if pair_counts.get(pair, 0) >= 2:
            continue
        selected.append(target)
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"selector found only {len(selected)} of {count} required targets")
    return selected


def select_targets(coverdata: dict[str, object]) -> tuple[list[Target], list[Target]]:
    v1_eligible: list[Target] = []
    for name, raw in coverdata.items():
        if isinstance(raw, dict):
            target = v1_target_from_entry(name, raw)
            if target is not None:
                v1_eligible.append(target)
    reserved = _select_with_pair_cap(v1_eligible, V1_TARGET_COUNT)
    reserved_names = {target.name for target in reserved}

    v2_eligible: list[Target] = []
    for name, raw in coverdata.items():
        if name in reserved_names or not isinstance(raw, dict):
            continue
        target = target_from_entry(name, raw)
        if target is not None:
            v2_eligible.append(target)
    selected = _select_with_pair_cap(v2_eligible, TARGET_COUNT)

    if reserved_names.intersection(target.name for target in selected):
        raise RuntimeError("v2 target overlaps exact frozen v1 target set")
    return reserved, selected


def build_incidence(target: Target) -> Incidence:
    tsets = list(itertools.combinations(range(target.v), target.t))
    t_index = {subset: index for index, subset in enumerate(tsets)}
    blocks: list[tuple[int, ...]] = []
    cover_by_block: list[array] = []
    blocks_by_t = [array("I") for _ in tsets]
    for block_index, block in enumerate(itertools.combinations(range(target.v), target.k)):
        covered = array("I", (t_index[s] for s in itertools.combinations(block, target.t)))
        blocks.append(block)
        cover_by_block.append(covered)
        for subset_index in covered:
            blocks_by_t[subset_index].append(block_index)
    if len(blocks) != target.candidate_blocks or len(tsets) != target.t_subsets:
        raise RuntimeError("incidence dimensions differ from frozen selector metadata")
    return Incidence(blocks, tsets, cover_by_block, blocks_by_t)


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
        return False, "not strictly smaller than frozen upper bound"
    return True, "verified"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_snapshot_json() -> dict[str, object]:
    raw = json.loads(download_snapshot())
    if not isinstance(raw, dict):
        raise TypeError("coverdata root must be a dictionary")
    return raw
