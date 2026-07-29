from __future__ import annotations

from . import fresh_external_evaluation_v75 as evaluation

_ORIGINAL_CONFIGURE = evaluation.configure


def configure_frozen_v72():
    prereg, manifest = _ORIGINAL_CONFIGURE()
    evaluation.opened.external.task_from_records = evaluation.core.compile_task
    evaluation.opened.compact_state = evaluation.v72.compact_state
    evaluation.opened.protocol = evaluation.v72.protocol
    evaluation.core.compiler_protocol = evaluation.v72.compiler_protocol
    return prereg, manifest


evaluation.configure = configure_frozen_v72


if __name__ == "__main__":
    evaluation.main()
