from __future__ import annotations

import inspect

from mini_origin import budget_compiler_v29 as v29


def test_program_library_contains_progressive_schedules() -> None:
    programs = v29.programs()
    assert len(programs) == 12
    assert all(program.batches[-1] == 512 for program in programs)
    assert any(program.batches[0] == 16 for program in programs)
    assert {program.z for program in programs} == {1.96, 2.58, 3.29}


def test_hidden_tasks_follow_budget_freeze() -> None:
    source = inspect.getsource(v29.run)
    freeze = source.index("frozen_digest = digest")
    hidden = source.index("hidden_instances = v28.make_instances")
    evaluation = source.index("candidate = evaluate")
    assert freeze < hidden < evaluation


def test_gate_requires_real_cost_reduction() -> None:
    source = inspect.getsource(v29.run)
    assert "observation_ratio <= 0.60" in source
    assert "p95_ratio <= 0.80" in source
    assert "accuracy_gap >= -0.005" in source
    assert "candidate.accuracy >= 0.985" in source


def test_budget_text_is_deterministic() -> None:
    program = v29.BudgetProgram((16, 32, 64), 2.58)
    assert program.text() == "batches=16,32,64;z=2.58"
