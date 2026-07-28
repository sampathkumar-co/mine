from __future__ import annotations

import os
import sys

import arcgen_gate as base
import arcgen_gate_v11 as delegate

ALLOWED_GATES = {f"v11-campaign-{index:02d}" for index in range(1, 11)}


def main() -> None:
    gate_id = os.environ.get("LEXIGEN_V11_CAMPAIGN_GATE")
    if gate_id not in ALLOWED_GATES:
        raise SystemExit(
            "LEXIGEN_V11_CAMPAIGN_GATE must be one of the ten precommitted v11 campaign gates"
        )
    protocol = f"arcgen-{gate_id}"
    base.PROTOCOL = protocol
    delegate.PROTOCOL = protocol
    delegate.main()


if __name__ == "__main__":
    main()
