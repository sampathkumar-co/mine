from __future__ import annotations

import sys

import parallel_task46_overlay as overlay
import parallel_train_v2 as runner


def main() -> None:
    task=None
    for i,arg in enumerate(sys.argv[:-1]):
        if arg=="--task":
            task=sys.argv[i+1]
            break
    if task not in {"ode_lorenz96_nonchaotic","edge_expansion"}:
        raise SystemExit(f"Task4/6 repair harness cannot run task {task!r}")
    overlay.activate()
    runner.main()


if __name__=="__main__":
    main()
