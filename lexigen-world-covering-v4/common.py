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
SEED_MATERIAL = (
    "32c897005c91865319f1b7da264b6162fc1ff4de|"
    "b2c626b07f216aac830d344eff5ad523|LEXIGEN_WORLD_COVERING_V4"
)
REFERENCE_DATE = datetime(2026, 4, 24, tzinfo=timezone.utc)
RESERVED_COUNT = 9
TARGET_COUNT = 3
GREEDY_DETERMINISTIC = 12
GREEDY_RANDOMIZED = 36
LNS_SECONDS = 360.0
LNS_ROUNDS = 42
CP_SAT_SECONDS = 1050.0
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
                SNAPSHOT_URL, headers={"User-Agent": "LEXIGEN-world-covering-v4"}
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


def target_from_entry(name: str, entry: dict[str, object]) -> Target | None:
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
        and upper <= 120
        and upper - 1 >= lower
    ):
        return None
    candidate_blocks = math.comb(v, k)
    t_subsets = math.comb(v, t)
    incidence_edges = candidate_blocks * math.comb(k, t)
    if candidate_blocks > 75_000 or t_subsets > 6_500 or incidence_edges > 4_500_000:
        return None

    improvements = entry.get("imps") or []
    last_update = ""
    if isinstance(improvements, list) and improvements:
        row = improvements[0]
        if isinstance(row, list) and len(row) >= 4:
            last_update = str(row[3])
    age_years = max(0.0, (REFERENCE_DATE - parse_date(last_update)).days / 365.25)
    age_factor = 1.0 + min(age_years, 25.0) / 15.0
    gap_factor = float(gap) ** 1.55
    complexity_factor = float(incidence_edges) ** 0.30
    upper_factor = (120.0 / float(upper)) ** 0.10
    opportunity_score = gap_factor * age_factor * upper_factor / complexity_factor
    tie_break = hashlib.sha256(f"{SEED_MATERIAL}|{name}".encode()).hexdigest()
    return Target(
        name=name,
        v=v,
        k=k,
        t=t,
        upper=upper,
        lower=lower,
        last_update=last_update,
        gap=gap,
        candidate_blocks=candidate_blocks,
        t_subsets=t_subsets,
        incidence_edges=incidence_edges,
        opportunity_score=opportunity_score,
        tie_break=tie_break,
    )


def select_targets(coverdata: dict[str, object]) -> tuple[dict[str, list[Target]], list[Target]]:
    eligible: list[Target] = []
    for name, raw in coverdata.items():
        if isinstance(raw, dict):
            target = target_from_entry(name, raw)
            if target is not None:
                eligible.append(target)
    eligible.sort(key=lambda x: (-x.opportunity_score, x.tie_break, x.name))

    selected: list[Target] = []
    pair_counts: dict[tuple[int, int], int] = {}
    needed = RESERVED_COUNT + TARGET_COUNT
    for target in eligible:
        pair = (target.k, target.t)
        if pair_counts.get(pair, 0) >= 2:
            continue
        selected.append(target)
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        if len(selected) == needed:
            break
    if len(selected) != needed:
        raise RuntimeError(f"selector found only {len(selected)} of {needed} targets")

    expected_v1 = ["C(15,8,5)", "C(11,6,5)", "C(14,5,3)"]
    expected_v2 = ["C(12,7,5)", "C(14,8,5)", "C(16,7,4)"]
    expected_v3 = ["C(15,6,4)", "C(17,9,5)", "C(16,8,5)"]
    actual = [target.name for target in selected[:9]]
    expected = expected_v1 + expected_v2 + expected_v3
    if actual != expected:
        raise RuntimeError(
            "deterministic lineage mismatch before fresh-target selection: "
            f"{actual} != {expected}"
        )
    lineage = {
        "v1": selected[:3],
        "corrected_v2": selected[3:6],
        "v3": selected[6:9],
    }
    return lineage, selected[9:12]


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
