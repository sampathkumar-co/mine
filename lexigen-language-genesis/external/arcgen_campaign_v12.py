from __future__ import annotations

import os

import arcgen_gate as base
import arcgen_gate_v12 as delegate

ALLOWED_GATES = {f"v12-campaign-{index:02d}" for index in range(1, 21)}


def main() -> None:
    gate_id = os.environ.get("LEXIGEN_V12_CAMPAIGN_GATE")
    if gate_id not in ALLOWED_GATES:
        raise SystemExit(
            "LEXIGEN_V12_CAMPAIGN_GATE must be one of the twenty precommitted v12 campaign gates"
        )
    protocol = f"arcgen-{gate_id}"
    base.PROTOCOL = protocol
    delegate.PROTOCOL = protocol
    delegate.main()


if __name__ == "__main__":
    main()
