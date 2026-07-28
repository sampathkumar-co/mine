from __future__ import annotations

# Reuse the audited gate protocol implementation while freezing a new protocol
# identity and the v2 execution/synthesis semantics. Gate 1 remains immutable.
import arcgen_gate as base
from arc_language_v2 import execute_program, synthesize

base.PROTOCOL = "arcgen-gate-v2"
base.synthesize = synthesize
base.execute_program = execute_program


if __name__ == "__main__":
    base.main()
