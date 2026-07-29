from __future__ import annotations

import hashlib
import json
import os

import arcgen_gate as base
import arcgen_gate_v13 as delegate

ALLOWED_GATES = {f"v13-campaign-{index:02d}" for index in range(1, 41)}
EXCLUDED_TASK_IDS = frozenset({
    "00dbd492", "0a938d79", "0becf7df", "12997ef3", "1d61978c", "212895b5",
    "22806e14", "228f6490", "22a4bbc2", "234bbc79", "2b9ef948", "33067df9",
    "470c91de", "668eec9a", "695367ec", "6bcdb01e", "7c9b52a0", "7e576d6e",
    "8ee62060", "98cf29f8", "9b2a60aa", "9dfd6313", "a64e4611", "a78176bb",
    "af24b4cc", "b1948b0a", "b7256dcd", "c444b776", "c64f1187", "c8cbb738",
    "cbded52d", "cce03e0d", "e48d4e1a", "e57337a4", "f823c43c", "f9012d9b",
})


def excluded_task_ids_sha256() -> str:
    payload = "\n".join(sorted(EXCLUDED_TASK_IDS)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def command_select(args) -> None:
    lexigen_root = args.lexigen_root.resolve()
    arcgen_root = args.arcgen_root.resolve()
    lexigen_commit = base.git_commit(lexigen_root)
    arcgen_commit = base.git_commit(arcgen_root)
    all_task_ids = base.eligible_task_ids(arcgen_root)
    eligible = sorted(task_id for task_id in all_task_ids if task_id not in EXCLUDED_TASK_IDS)
    task_id, index, digest = base.select_task(lexigen_commit, arcgen_commit, eligible)
    record = {
        "protocol": base.PROTOCOL,
        "lexigen_commit": lexigen_commit,
        "arcgen_commit": arcgen_commit,
        "eligible_task_count_before_exclusion": len(all_task_ids),
        "excluded_task_count": len(EXCLUDED_TASK_IDS),
        "excluded_task_ids_sha256": excluded_task_ids_sha256(),
        "eligible_task_count": len(eligible),
        "selection_digest": digest,
        "selection_index": index,
        "selected_task_id": task_id,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


def main() -> None:
    gate_id = os.environ.get("LEXIGEN_V13_CAMPAIGN_GATE")
    if gate_id not in ALLOWED_GATES:
        raise SystemExit("LEXIGEN_V13_CAMPAIGN_GATE must be one of the forty precommitted v13 campaign gates")
    protocol = f"arcgen-{gate_id}"
    base.PROTOCOL = protocol
    delegate.PROTOCOL = protocol
    parser = base.build_parser()
    args = parser.parse_args()
    if args.command == "select":
        command_select(args)
    elif args.command == "solve":
        delegate.command_solve(args)
    else:
        args.function(args)


if __name__ == "__main__":
    main()
