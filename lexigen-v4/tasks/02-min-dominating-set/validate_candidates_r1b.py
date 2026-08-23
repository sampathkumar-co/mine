from __future__ import annotations

import runpy

# Import first so the frozen candidate module receives only the preregistered generic
# empty-input correction before the unchanged validator imports it.
import candidates_r1b  # noqa: F401

runpy.run_path("validate_candidates.py", run_name="__main__")
