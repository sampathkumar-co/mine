from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
V19R4 = HERE.parent / "v19r4"
V19R2 = HERE.parent / "v19r2"
for folder in (HERE, V19R4, V19R2):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from portable_runtime_v19r2 import execute_portable
from run_cegis_v19r4 import expand_production
from runtime_v19r2 import as_grid, execute, sha256_json


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path: Path, value: Any) -> None:
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def colours(examples) -> list[int]:
    return sorted({cell for source, target in examples for grid in (source, target) for row in grid for cell in row})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "V20_REPRODUCTION.json")
    args = parser.parse_args()

    precommit = load(HERE / "V20_PRECOMMIT.json")
    committed = load(HERE / "V20_TRANSFER_SCAN.json")
    production = load(V19R4 / "production" / "v19r4-production.json")
    if sha256_json(production) != precommit["production_sha256"]:
        raise RuntimeError("production identity changed")
    if precommit["hidden_outputs_opened"] or committed["hidden_outputs_opened"]:
        raise RuntimeError("hidden-output flag changed")

    reports = []
    package_hashes_verified = 0
    for item in precommit["items"]:
        gate = int(item["gate"])
        package_path = args.package_root / f"v13-campaign-{gate:02d}" / "redacted-task.json"
        actual_package_hash = file_sha256(package_path)
        if actual_package_hash != item["package_sha256"]:
            raise RuntimeError(f"package hash changed for gate {gate}")
        package_hashes_verified += 1
        package = load(package_path)
        examples = [(as_grid(entry["input"]), as_grid(entry["output"])) for entry in package["train"]]
        palette = colours(examples)
        survivors = []
        runtime_invalid = 0
        checked = 0
        for marker, background in itertools.product(palette, repeat=2):
            if checked >= 100:
                break
            checked += 1
            arguments = {"marker_colour": marker, "output_background": background}
            concrete = expand_production(production, arguments)
            try:
                primary = [execute(concrete, source) for source, _ in examples]
                portable = [execute_portable(concrete, source) for source, _ in examples]
            except Exception:
                runtime_invalid += 1
                continue
            exact = sum(output == target for output, (_, target) in zip(primary, examples))
            portable_exact = sum(output == target for output, (_, target) in zip(portable, examples))
            if exact == portable_exact == len(examples) and primary == portable:
                survivors.append({
                    "arguments": arguments,
                    "arguments_sha256": sha256_json(arguments),
                    "concrete_program_sha256": sha256_json(concrete),
                })
        reports.append({
            "gate": gate,
            "demonstrations": len(examples),
            "palette": palette,
            "argument_pairs_checked": checked,
            "runtime_invalid_pairs": runtime_invalid,
            "exact_survivors": len(survivors),
            "survivors": survivors,
        })

    reproduced = {
        "schema": "lexigen-v20-fixed-production-transfer-scan-v1",
        "precommit_sha256": sha256_json(precommit),
        "production_sha256": sha256_json(production),
        "fixed_gates": precommit["fixed_gates"],
        "gates_checked": len(reports),
        "gates_with_exact_survivor": sum(report["exact_survivors"] > 0 for report in reports),
        "total_exact_survivors": sum(report["exact_survivors"] for report in reports),
        "hidden_outputs_opened": False,
        "sealed_external_success": False,
        "transfer_demonstrated": False,
        "world_level_breakthrough": False,
        "reports": reports,
    }
    if reproduced != committed:
        raise RuntimeError("reproduced scan differs from committed scan")
    if reproduced["total_exact_survivors"] != 0:
        raise RuntimeError("negative transfer result changed")

    audit = {
        "schema": "lexigen-v20-transfer-reproduction-v1",
        "precommit_commit": "17b96af07f726c202cede3a54fba70ecd6da3cbc",
        "negative_result_commit": "6b24fd0f74c02448eefb22456d3b8fc5c7549f24",
        "production_sha256": reproduced["production_sha256"],
        "fixed_gates": reproduced["fixed_gates"],
        "package_hashes_verified": package_hashes_verified,
        "gates_checked": reproduced["gates_checked"],
        "argument_pairs_checked": sum(report["argument_pairs_checked"] for report in reports),
        "runtime_invalid_pairs": sum(report["runtime_invalid_pairs"] for report in reports),
        "total_exact_survivors": 0,
        "committed_scan_sha256": file_sha256(HERE / "V20_TRANSFER_SCAN.json"),
        "byte_equivalent_reproduction": True,
        "hidden_outputs_opened": False,
        "transfer_demonstrated": False,
        "v20_pass": False,
        "world_level_breakthrough": False,
    }
    write(args.output, audit)
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
