from __future__ import annotations

import json

from . import openml_hash_lock_v77 as lock


def install_suite_id_compatibility() -> int:
    import openml

    preregistration = json.loads(
        lock.PREREGISTRATION.read_text(encoding="utf-8-sig")
    )
    expected_suite_id = int(preregistration["benchmark_suite_id"])
    original_get_suite = openml.study.get_suite

    def get_suite_with_committed_id(*args, **kwargs):
        suite = original_get_suite(*args, **kwargs)
        existing = getattr(suite, "suite_id", None)
        if existing is None:
            setattr(suite, "suite_id", expected_suite_id)
        elif int(existing) != expected_suite_id:
            raise RuntimeError(
                f"OpenML suite ID mismatch: expected {expected_suite_id}, got {existing}"
            )
        return suite

    openml.study.get_suite = get_suite_with_committed_id
    return expected_suite_id


def main() -> None:
    install_suite_id_compatibility()
    lock.main()


if __name__ == "__main__":
    main()
