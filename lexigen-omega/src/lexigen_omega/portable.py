from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .ir import BudgetExceeded, ExecutionBudget, ExecutionResult, OpCode, Program


def _items_created(value: Any) -> int:
    if isinstance(value, Mapping):
        return len(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return len(value)
    return 0


def execute_portable_metered(
    program: Program,
    inputs: Mapping[str, Any],
    budget: ExecutionBudget | None = None,
) -> ExecutionResult:
    """Independent Ω1 interpreter used only for semantic parity checks.

    This implementation intentionally does not call the primary interpreter or its
    helper functions. Sharing the IR schema is allowed; sharing execution logic is
    not, so differential tests can catch semantic drift.
    """

    program.validate()
    limits = budget or ExecutionBudget()
    required = set(program.inputs)
    supplied = set(inputs)
    if required != supplied:
        raise ValueError(
            f"input mismatch missing={sorted(required - supplied)} "
            f"extra={sorted(supplied - required)}"
        )

    values: dict[str, Any] = dict(inputs)
    executed = 0
    created = 0

    for instruction in program.instructions:
        executed += 1
        if executed > limits.max_instructions:
            raise BudgetExceeded(
                f"instruction budget exceeded: {executed} > {limits.max_instructions}"
            )

        operands = [values[name] for name in instruction.args]
        opcode = instruction.op

        if opcode is OpCode.RETURN:
            return ExecutionResult(operands[0], executed, created)
        if opcode is OpCode.CONST:
            result = instruction.literal
        elif opcode is OpCode.ADD:
            result = operands[0] + operands[1]
        elif opcode is OpCode.SUB:
            result = operands[0] - operands[1]
        elif opcode is OpCode.MUL:
            result = operands[0] * operands[1]
        elif opcode is OpCode.DIV:
            result = operands[0] / operands[1]
        elif opcode is OpCode.MOD:
            result = operands[0] % operands[1]
        elif opcode is OpCode.NEG:
            result = -operands[0]
        elif opcode is OpCode.ABS:
            result = abs(operands[0])
        elif opcode is OpCode.EQ:
            result = operands[0] == operands[1]
        elif opcode is OpCode.LT:
            result = operands[0] < operands[1]
        elif opcode is OpCode.LE:
            result = operands[0] <= operands[1]
        elif opcode is OpCode.AND:
            result = bool(operands[0]) and bool(operands[1])
        elif opcode is OpCode.OR:
            result = bool(operands[0]) or bool(operands[1])
        elif opcode is OpCode.NOT:
            result = not bool(operands[0])
        elif opcode is OpCode.SELECT:
            if len(operands) != 3:
                raise ValueError("SELECT requires condition, true value, false value")
            result = operands[1] if bool(operands[0]) else operands[2]
        elif opcode is OpCode.TUPLE:
            result = tuple(operands)
        elif opcode is OpCode.GETITEM:
            result = operands[0][operands[1]]
        elif opcode is OpCode.LEN:
            result = len(operands[0])
        elif opcode is OpCode.CONCAT:
            result = operands[0] + operands[1]
        elif opcode is OpCode.RANGE:
            if len(operands) == 1:
                result = list(range(operands[0]))
            elif len(operands) == 2:
                result = list(range(operands[0], operands[1]))
            elif len(operands) == 3:
                result = list(range(operands[0], operands[1], operands[2]))
            else:
                raise ValueError("RANGE requires 1..3 arguments")
        elif opcode is OpCode.SORT:
            result = sorted(operands[0])
        elif opcode is OpCode.UNIQUE:
            result = list(dict.fromkeys(operands[0]))
        elif opcode is OpCode.REVERSE:
            result = list(reversed(operands[0]))
        elif opcode is OpCode.DIFF:
            sequence = list(operands[0])
            result = [right - left for left, right in zip(sequence, sequence[1:])]
        elif opcode is OpCode.CUMSUM:
            running = 0
            result = []
            for item in operands[0]:
                running += item
                result.append(running)
        elif opcode is OpCode.REDUCE_SUM:
            result = sum(operands[0])
        else:
            raise NotImplementedError(opcode.value)

        created += _items_created(result)
        if created > limits.max_collection_items_created:
            raise BudgetExceeded(
                f"collection-item budget exceeded: {created} > "
                f"{limits.max_collection_items_created}"
            )
        values[instruction.name] = result

    raise RuntimeError("validated program terminated without RETURN")


def execute_portable(program: Program, inputs: Mapping[str, Any]) -> Any:
    return execute_portable_metered(program, inputs).value
