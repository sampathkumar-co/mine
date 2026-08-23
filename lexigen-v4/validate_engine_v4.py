from __future__ import annotations

from pathlib import Path

import engine_v4
import validate_engine_v3 as validation

validation.ENGINE_VERSION = engine_v4.ENGINE_VERSION
validation.fingerprint = engine_v4.fingerprint
validation.generate_proposals = engine_v4.generate_proposals
validation.OUTPUT = Path("validation-evidence-r4")

if __name__ == "__main__":
    validation.main()
