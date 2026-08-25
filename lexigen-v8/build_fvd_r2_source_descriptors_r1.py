from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path}")
    return value


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_blob_sha1(path: Path) -> str:
    return subprocess.check_output(["git", "hash-object", str(path)], text=True).strip()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def first_v4_operators(analysis: dict[str, Any], aux: dict[str, Any] | None) -> list[str]:
    direct = analysis.get("top_v4_operators")
    if isinstance(direct, list):
        return [str(x) for x in direct]
    top = analysis.get("v4_full_top")
    if isinstance(top, dict) and isinstance(top.get("operators"), list):
        return [str(x) for x in top["operators"]]
    if aux:
        seqs = aux.get("v4_full_operator_sequences")
        if isinstance(seqs, list) and seqs and isinstance(seqs[0], list):
            return [str(x) for x in seqs[0]]
        mappings = aux.get("mappings")
        if isinstance(mappings, dict):
            rows = mappings.get("v4_full")
            if isinstance(rows, list) and rows:
                row = rows[0]
                if isinstance(row, list) and len(row) >= 3 and isinstance(row[2], list):
                    return [str(x) for x in row[2]]
    return []


def solve_body(source: str) -> str:
    marker = "def solve("
    pos = source.find(marker)
    if pos < 0:
        return source
    tail = source[pos:]
    end = tail.find("\n    def ", len(marker))
    return tail if end < 0 else tail[:end]


def normalize_traits(
    analysis: dict[str, Any],
    aux: dict[str, Any] | None,
    source: str,
    operators: list[str],
) -> list[str]:
    fp = {str(x) for x in analysis.get("fingerprint_features", []) if isinstance(x, (str, int, float))}
    traits = set(fp)
    lower = source.lower()
    body = solve_body(source).lower()
    ops = set(operators)
    semantic = ""
    if aux:
        semantic = str(aux.get("source_semantic_observation", "")).lower()

    discrete_markers = {"boolean", "discrete", "graph", "set"}
    if fp & discrete_markers or "newboolvar" in lower or "adjacency" in lower:
        traits.update({"discrete", "set_or_boolean_structure", "specialized_representation_possible"})

    if fp & {"array", "matrix", "numeric", "linear", "convex"} or "numpy" in lower or "np." in lower:
        traits.update({"numeric_array", "precision_change_possible", "specialized_representation_possible"})

    if "approximate_verifier" in fp or "tolerance" in fp or "math.isclose" in lower or "np.allclose" in lower or "rtol" in lower or "atol" in lower:
        traits.update({"approximate_verifier", "tolerance_verifier", "exact_or_tolerance_certificate"})

    if "certificate" in fp or "verifier" in fp or "is_solution" in lower:
        traits.add("local_predicate_available")
        if "discrete" in traits:
            traits.add("exact_or_structural_certificate")

    if fp & {"constraints", "threshold", "projection", "convex", "order_statistic"} or "model.add(" in lower:
        traits.add("constraints_or_threshold")

    if "order_statistic" in fp or "projection" in fp or "active_set_decomposition" in ops:
        traits.add("small_active_core_possible")

    if fp & {"block_structure", "grouped_generator"} or "structure_aware_initialization" in ops or "separable" in semantic or "only x1" in semantic:
        traits.update({"high_dimensional_numeric_structure", "redundant_representation_or_modes"})

    if "sparse_frontier_search" in ops:
        traits.update({"explicit_frontier_work", "custom_search_hotpath"})

    if ops & {"sort_partition_reduction", "closed_form_reduction", "active_set_decomposition"} or "separable" in semantic or "only x1" in semantic:
        traits.add("reducible_domain")

    if "early_certificate_exit" in ops or "separable" in semantic or "only x1" in semantic:
        traits.add("provably_irrelevant_work_possible")

    if "risk_aware_staging" in ops:
        traits.add("fallback_available")

    if any(token in lower for token in ("numpy", "scipy", "networkx", "cvxpy", "ortools", "torch")):
        traits.add("native_backend_available")

    opaque_markers = ("cp_model.cpsolver", "solve_ivp", "odeint", "cvxpy", "cp.problem(")
    if any(token in lower for token in opaque_markers):
        traits.add("opaque_external_solver_dominant")

    if any(token in lower for token in ("solve_ivp", "odeint", "integrate.solve_ivp", "integrate.ode")):
        traits.add("error_amplification_risk")

    if any(token in lower for token in ("fftpack", "scipy.fft", "np.fft", " dst(", " dct(")):
        traits.add("bounded_error_propagation")

    if "def solve(" in lower:
        traits.add("hot_execution_path")

    # Only classify Python-loop cost when the solve path itself visibly contains
    # repeated Python iteration and is not delegated to one of the opaque solver APIs.
    if body.count("for ") >= 2 and "opaque_external_solver_dominant" not in traits:
        traits.add("python_loop_bottleneck")

    if "numeric_array" in traits or "discrete" in traits:
        traits.add("source_visible_type_shape_or_value_regime")

    return sorted(traits)


def build(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_json(args.manifest)
    split = load_json(args.split)
    if manifest.get("outcome_paths_present") is not False:
        raise RuntimeError("source manifest unexpectedly contains outcome paths")
    if manifest.get("official_fvd_holdouts_accessed") is not False:
        raise RuntimeError("official holdout boundary crossed")

    split_map = {str(x["task"]): str(x["partition"]) for x in split["ordered"]}
    records = []
    for row in manifest["tasks"]:
        task = str(row["task"])
        if split_map.get(task) != row.get("partition"):
            raise RuntimeError(f"split mismatch: {task}")

        analysis_path = Path(str(row["source_analysis_path"]))
        observed_blob = git_blob_sha1(analysis_path)
        if observed_blob != row["source_analysis_blob_sha1"]:
            raise RuntimeError(f"source-analysis blob mismatch for {task}: {observed_blob}")
        analysis = load_json(analysis_path)

        aux = None
        aux_path_value = row.get("source_aux_path")
        aux_blob = None
        if aux_path_value:
            aux_path = Path(str(aux_path_value))
            aux_blob = git_blob_sha1(aux_path)
            if aux_blob != row["source_aux_blob_sha1"]:
                raise RuntimeError(f"source-aux blob mismatch for {task}: {aux_blob}")
            aux = load_json(aux_path)

        raw_path = args.algotune_root / str(row["raw_source_path"])
        raw = raw_path.read_bytes()
        raw_sha = sha256_bytes(raw)
        raw_blob = git_blob_sha1(raw_path)
        if raw_sha != row["raw_source_sha256"]:
            raise RuntimeError(f"raw source sha256 mismatch for {task}: {raw_sha}")

        explicit_raw_blob = row.get("raw_source_git_blob_sha1")
        if explicit_raw_blob is not None and raw_blob != str(explicit_raw_blob):
            raise RuntimeError(f"raw source git blob mismatch for {task}: {raw_blob}")

        historical_blob = analysis.get("source_git_blob_sha1")
        if historical_blob is not None and raw_blob != str(historical_blob):
            raise RuntimeError(f"historical source git blob disagrees for {task}: {raw_blob} != {historical_blob}")

        analysis_sha = analysis.get("source_sha256")
        historical_sha_mismatch_documented = False
        if analysis_sha is not None and str(analysis_sha) != raw_sha:
            if row.get("historical_recorded_source_sha256") != str(analysis_sha):
                raise RuntimeError(f"historical source sha256 disagrees for {task} without documented provenance anomaly")
            if explicit_raw_blob is None or str(explicit_raw_blob) != raw_blob:
                raise RuntimeError(f"historical source sha256 anomaly lacks authoritative Git blob lock for {task}")
            historical_sha_mismatch_documented = True

        source_text = raw.decode("utf-8")
        operators = first_v4_operators(analysis, aux)
        traits = normalize_traits(analysis, aux, source_text, operators)
        records.append({
            "task": task,
            "partition": row["partition"],
            "traits": traits,
            "source_only_v4_top_operators": operators,
            "source_analysis_blob_sha1": observed_blob,
            "source_aux_blob_sha1": aux_blob,
            "raw_source_git_blob_sha1": raw_blob,
            "raw_source_sha256": raw_sha,
            "historical_sha256_mismatch_documented": historical_sha_mismatch_documented,
        })

    payload: dict[str, Any] = {
        "schema": "lexigen-v8-fvd-r2-source-descriptors-r1",
        "source_only": True,
        "calibration_outcomes_loaded": False,
        "official_fvd_holdouts_accessed": False,
        "blocked_ipwm_real_data_accessed": False,
        "all_source_hashes_verified": True,
        "all_split_memberships_verified": True,
        "records": sorted(records, key=lambda x: x["task"]),
    }
    payload["artifact_sha256"] = canonical_sha256(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("lexigen-v8/FVD_R2_V4_SOURCE_MANIFEST_R1.json"))
    parser.add_argument("--split", type=Path, default=Path("lexigen-v8/FVD_R2_V4_SPLIT_R1.json"))
    parser.add_argument("--algotune-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "artifact_sha256": payload["artifact_sha256"],
        "record_count": len(payload["records"]),
        "all_source_hashes_verified": payload["all_source_hashes_verified"],
        "calibration_outcomes_loaded": payload["calibration_outcomes_loaded"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
