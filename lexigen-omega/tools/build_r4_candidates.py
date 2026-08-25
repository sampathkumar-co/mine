from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import subprocess
from pathlib import Path

TARGET = Path("src/datasets/load.py")
CANDIDATES = ("F1", "F2", "F4", "N1", "N3", "N6", "R1", "R3", "R6")
PROPOSAL_BLOB = "95012e5d3643987c40ed30fd6cd9c125cd767257"
EXPECTED_BASE_BLOB = "13562ec82b01898334b2eaa455f4ce38bb7176da"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one replacement target, got {count}")
    return text.replace(old, new, 1)


def write_patch(path: Path, before: str, after: str, patch_path: Path) -> None:
    diff = difflib.unified_diff(
        before.splitlines(True),
        after.splitlines(True),
        fromfile=f"a/{path.as_posix()}",
        tofile=f"b/{path.as_posix()}",
    )
    patch_path.write_text("".join(diff), encoding="utf-8")


def verify_protocol_inputs() -> None:
    got = subprocess.check_output(
        ["git", "hash-object", "lexigen-omega/evidence/PROSPECTIVE_R4_SOURCE_PROPOSALS.json"],
        text=True,
    ).strip()
    if got != PROPOSAL_BLOB:
        raise RuntimeError(f"R4 proposal blob drift: {got}")


def transform(before: str, cid: str) -> str:
    after = before

    if cid == "F1":
        old = '''                if _require_default_config_name:
                    with fs.open(f"datasets/{path}/{filename}", "r", revision=revision, encoding="utf-8") as f:
                        can_load_config_from_parquet_export = "DEFAULT_CONFIG_NAME" not in f.read()
                else:
                    can_load_config_from_parquet_export = True
'''
        new = '''                if _require_default_config_name:
                    with fs.open(f"datasets/{path}/{filename}", "r", revision=revision, encoding="utf-8") as f:
                        can_load_config_from_parquet_export = True
                        for line in f:
                            if "DEFAULT_CONFIG_NAME" in line:
                                can_load_config_from_parquet_export = False
                                break
                else:
                    can_load_config_from_parquet_export = True
'''
        after = replace_once(after, old, new, cid)

    elif cid == "F2":
        old = '''    dataset_module = dataset_module_factory(
        path,
        revision=revision,
        download_config=download_config,
        download_mode=download_mode,
        data_dir=data_dir,
        data_files=data_files,
        trust_remote_code=trust_remote_code,
        _require_default_config_name=_require_default_config_name,
    )
'''
        new = '''    if path in _PACKAGED_DATASETS_MODULES:
        packaged_download_config = download_config if download_config is not None else DownloadConfig()
        packaged_download_config.extract_compressed_file = True
        packaged_download_config.force_extract = True
        packaged_download_config.force_download = download_mode == DownloadMode.FORCE_REDOWNLOAD
        dataset_module = PackagedDatasetModuleFactory(
            path,
            data_dir=data_dir,
            data_files=data_files,
            download_config=packaged_download_config,
            download_mode=download_mode,
        ).get_module()
    else:
        dataset_module = dataset_module_factory(
            path,
            revision=revision,
            download_config=download_config,
            download_mode=download_mode,
            data_dir=data_dir,
            data_files=data_files,
            trust_remote_code=trust_remote_code,
            _require_default_config_name=_require_default_config_name,
        )
'''
        after = replace_once(after, old, new, cid)

    elif cid == "F4":
        old = '''    elif is_relative_path(path) and path.count("/") <= 1:
        try:
            _raise_if_offline_mode_is_enabled()
            hf_api = HfApi(config.HF_ENDPOINT)
'''
        new = '''    elif is_relative_path(path) and path.count("/") <= 1:
        try:
            if config.HF_DATASETS_OFFLINE:
                try:
                    return CachedDatasetModuleFactory(path, dynamic_modules_path=dynamic_modules_path).get_module()
                except Exception:
                    pass
            _raise_if_offline_mode_is_enabled()
            hf_api = HfApi(config.HF_ENDPOINT)
'''
        after = replace_once(after, old, new, cid)

    elif cid == "N1":
        old = '''    filename = list(filter(lambda x: x, path.replace(os.sep, "/").split("/")))[-1]
    if not filename.endswith(".py"):
        filename = filename + ".py"
    combined_path = os.path.join(path, filename)
'''
        new = '''    normalized_path = path.replace(os.sep, "/")
    stripped_path = normalized_path.rstrip("/")
    if stripped_path:
        filename = stripped_path.rsplit("/", 1)[-1]
    else:
        filename = list(filter(lambda x: x, normalized_path.split("/")))[-1]
    if not filename.endswith(".py"):
        filename = filename + ".py"
    combined_path = os.path.join(path, filename)
'''
        after = replace_once(after, old, new, cid)

    elif cid == "N3":
        old1 = '''    builder_kwargs = dataset_module.builder_kwargs
    data_dir = builder_kwargs.pop("data_dir", data_dir)
'''
        new1 = '''    builder_kwargs = dataset_module.builder_kwargs
    builder_configs_parameters = dataset_module.builder_configs_parameters
    metadata_configs = builder_configs_parameters.metadata_configs
    data_dir = builder_kwargs.pop("data_dir", data_dir)
'''
        after = replace_once(after, old1, new1, cid + "-aliases")
        after = replace_once(
            after,
            '''        "config_name", name or dataset_module.builder_configs_parameters.default_config_name
''',
            '''        "config_name", name or builder_configs_parameters.default_config_name
''',
            cid + "-default",
        )
        old2 = '''    if (
        dataset_module.builder_configs_parameters.metadata_configs
        and config_name in dataset_module.builder_configs_parameters.metadata_configs
    ):
        hash = update_hash_with_config_parameters(
            hash, dataset_module.builder_configs_parameters.metadata_configs[config_name]
        )
'''
        new2 = '''    if metadata_configs and config_name in metadata_configs:
        hash = update_hash_with_config_parameters(hash, metadata_configs[config_name])
'''
        after = replace_once(after, old2, new2, cid + "-metadata")

    elif cid == "N6":
        old = '''    if (
        dataset_module.builder_configs_parameters.metadata_configs
        and config_name in dataset_module.builder_configs_parameters.metadata_configs
    ):
        hash = update_hash_with_config_parameters(
            hash, dataset_module.builder_configs_parameters.metadata_configs[config_name]
        )
'''
        new = '''    metadata_configs = dataset_module.builder_configs_parameters.metadata_configs
    metadata_config = metadata_configs.get(config_name) if metadata_configs else None
    if metadata_config is not None:
        hash = update_hash_with_config_parameters(hash, metadata_config)
'''
        after = replace_once(after, old, new, cid)

    elif cid == "R1":
        old = '''            if filename in [sibling.rfilename for sibling in dataset_info.siblings]:  # contains a dataset script
'''
        new = '''            if any(sibling.rfilename == filename for sibling in dataset_info.siblings):  # contains a dataset script
'''
        after = replace_once(after, old, new, cid)

    elif cid == "R3":
        old = '''        def _get_modification_time(module_hash):
            return (Path(importable_directory_path) / module_hash / (self.name.split("/")[-1] + ".py")).stat().st_mtime

        hash = sorted(hashes, key=_get_modification_time)[-1]
'''
        new = '''        def _get_modification_time(module_hash):
            return (Path(importable_directory_path) / module_hash / (self.name.split("/")[-1] + ".py")).stat().st_mtime

        hash = max(
            enumerate(hashes),
            key=lambda item: (_get_modification_time(item[1]), item[0]),
        )[1]
'''
        after = replace_once(after, old, new, cid)

    elif cid == "R6":
        old1 = '''    builder_kwargs = dataset_module.builder_kwargs
    data_dir = builder_kwargs.pop("data_dir", data_dir)
'''
        new1 = '''    builder_kwargs = dataset_module.builder_kwargs
    builder_configs_parameters = dataset_module.builder_configs_parameters
    metadata_configs = builder_configs_parameters.metadata_configs
    data_dir = builder_kwargs.pop("data_dir", data_dir)
'''
        after = replace_once(after, old1, new1, cid + "-aliases")
        after = replace_once(
            after,
            '''        "config_name", name or dataset_module.builder_configs_parameters.default_config_name
''',
            '''        "config_name", name or builder_configs_parameters.default_config_name
''',
            cid + "-default",
        )
        old2 = '''    if (
        dataset_module.builder_configs_parameters.metadata_configs
        and config_name in dataset_module.builder_configs_parameters.metadata_configs
    ):
        hash = update_hash_with_config_parameters(
            hash, dataset_module.builder_configs_parameters.metadata_configs[config_name]
        )
'''
        new2 = '''    metadata_config = metadata_configs.get(config_name) if metadata_configs else None
    if metadata_config is not None:
        hash = update_hash_with_config_parameters(hash, metadata_config)
'''
        after = replace_once(after, old2, new2, cid + "-metadata")
    else:
        raise RuntimeError(f"unknown candidate {cid}")

    if after == before:
        raise RuntimeError(f"{cid}: transform made no change")
    return after


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", choices=CANDIDATES, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    verify_protocol_inputs()
    path = args.root / TARGET
    before = path.read_text(encoding="utf-8")
    base_blob = subprocess.check_output(["git", "hash-object", str(path)], text=True).strip()
    if base_blob != EXPECTED_BASE_BLOB:
        raise RuntimeError(f"R4 base source drift: {base_blob} != {EXPECTED_BASE_BLOB}")

    after = transform(before, args.candidate)
    path.write_text(after, encoding="utf-8")

    args.output.mkdir(parents=True, exist_ok=True)
    patch_path = args.output / f"R4-{args.candidate}.patch"
    write_patch(TARGET, before, after, patch_path)

    report = {
        "project": "LEXIGEN OMEGA",
        "stage": "R4_candidate_materialization_before_execution",
        "instance_id": "huggingface__datasets-ef3b5dd",
        "candidate": args.candidate,
        "target": TARGET.as_posix(),
        "base_git_blob_sha1": base_blob,
        "proposal_git_blob_sha1": PROPOSAL_BLOB,
        "before_sha256": sha256_bytes(before.encode()),
        "after_sha256": sha256_bytes(after.encode()),
        "patch_sha256": sha256_bytes(patch_path.read_bytes()),
        "candidate_execution_count": 0,
        "candidate_timing_observed": False,
        "expert_opt_commit_accessed": False,
        "expert_diff_accessed": False,
        "hints_accessed": False,
        "R5_source_accessed": False,
    }
    (args.output / f"R4-{args.candidate}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
