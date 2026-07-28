from __future__ import annotations

import arcgen_gate as base
from arc_language_v6 import execute_program, synthesize

base.PROTOCOL = "arcgen-gate-v6"
base.synthesize = synthesize
base.execute_program = execute_program


if __name__ == "__main__":
    base.main()
