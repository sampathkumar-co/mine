from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from typing import Any, Iterable, Mapping


class TypeTag(str, Enum):
    ANY = "any"
    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    SEQ = "seq"
    MAP = "map"
    GRAPH = "graph"
    TENSOR = "tensor"


class OpCode(str, Enum):
    INPUT = "input"
    CONST = "const"
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    MOD = "mod"
    NEG = "neg"
    ABS = "abs"
    EQ = "eq"
    LT = "lt"
    LE = "le"
    AND = "and"
    OR = "or"
    NOT = "not"
    SELECT = "select"
    TUPLE = "tuple"
    GETITEM = "getitem"
    LEN = "len"
    CONCAT = "concat"
    RANGE = "range"
    SORT = "sort"
    UNIQUE = "unique"
    REVERSE = "reverse"
    DIFF = "diff"
    CUMSUM = "cumsum"
    REDUCE_SUM = "reduce_sum"
    RETURN = "return"


@dataclass(frozen=True)
class Instruction:
    name: str
    op: OpCode
    args: tuple[str, ...] = ()
    literal: Any = None
    type_tag: TypeTag = TypeTag.ANY

    def canonical(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "op": self.op.value,
            "args": list(self.args),
            "literal": self.literal,
            "type": self.type_tag.value,
        }


@dataclass(frozen=True)
class Program:
    inputs: tuple[str, ...]
    instructions: tuple[Instruction, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        seen = set(self.inputs)
        if len(seen) != len(self.inputs):
            raise ValueError("duplicate input name")

        return_count = 0
        for index, ins in enumerate(self.instructions):
            if not ins.name:
                raise ValueError(f"instruction {index} has empty name")
            if ins.name in seen:
                raise ValueError(f"duplicate SSA name: {ins.name}")
            for arg in ins.args:
                if arg not in seen:
                    raise ValueError(
                        f"instruction {ins.name} references unknown value {arg}"
                    )
            if ins.op is OpCode.INPUT:
                raise ValueError("INPUT is represented by Program.inputs, not instructions")
            if ins.op is OpCode.RETURN:
                return_count += 1
                if len(ins.args) != 1:
                    raise ValueError("RETURN requires exactly one argument")
                if index != len(self.instructions) - 1:
                    raise ValueError("RETURN must be the final instruction")
            seen.add(ins.name)

        if return_count != 1:
            raise ValueError("program must contain exactly one RETURN")

    def canonical_payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "inputs": list(self.inputs),
            "instructions": [i.canonical() for i in self.instructions],
            "metadata": _canonicalize(dict(self.metadata)),
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExecutionBudget:
    max_instructions: int = 10000
    max_collection_items_created: int = 100000

    def __post_init__(self) -> None:
        if self.max_instructions <= 0:
            raise ValueError("max_instructions must be positive")
        if self.max_collection_items_created < 0:
            raise ValueError("max_collection_items_created must be non-negative")


@dataclass(frozen=True)
class ExecutionResult:
    value: Any
    instructions_executed: int
    collection_items_created: int


class BudgetExceeded(RuntimeError):
    pass


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _canonicalize(value[k]) for k in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v) for v in value]
    if isinstance(value, set):
        return sorted(_canonicalize(v) for v in value)
    return value


def _resolve(env: Mapping[str, Any], names: Iterable[str]) -> list[Any]:
    return [env[name] for name in names]


def _created_collection_items(value: Any) -> int:
    if isinstance(value, Mapping):
        return len(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return len(value)
    return 0


def execute_program_metered(
    program: Program,
    inputs: Mapping[str, Any],
    budget: ExecutionBudget | None = None,
) -> ExecutionResult:
    """Execute Ω1 IR under deterministic resource accounting.

    The counters intentionally measure abstract IR work rather than wall time so
    candidate comparisons remain reproducible across machines. Benchmark adapters
    may add wall-clock/CPU/GPU accounting separately.
    """

    program.validate()
    budget = budget or ExecutionBudget()
    expected = set(program.inputs)
    supplied = set(inputs)
    if supplied != expected:
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        raise ValueError(f"input mismatch missing={missing} extra={extra}")

    env: dict[str, Any] = dict(inputs)
    instructions_executed = 0
    collection_items_created = 0

    for ins in program.instructions:
        instructions_executed += 1
        if instructions_executed > budget.max_instructions:
            raise BudgetExceeded(
                f"instruction budget exceeded: {instructions_executed} > "
                f"{budget.max_instructions}"
            )

        args = _resolve(env, ins.args)
        op = ins.op

        if op is OpCode.CONST:
            value = ins.literal
        elif op is OpCode.ADD:
            value = args[0] + args[1]
        elif op is OpCode.SUB:
            value = args[0] - args[1]
        elif op is OpCode.MUL:
            value = args[0] * args[1]
        elif op is OpCode.DIV:
            value = args[0] / args[1]
        elif op is OpCode.MOD:
            value = args[0] % args[1]
        elif op is OpCode.NEG:
            value = -args[0]
        elif op is OpCode.ABS:
            value = abs(args[0])
        elif op is OpCode.EQ:
            value = args[0] == args[1]
        elif op is OpCode.LT:
            value = args[0] < args[1]
        elif op is OpCode.LE:
            value = args[0] <= args[1]
        elif op is OpCode.AND:
            value = bool(args[0]) and bool(args[1])
        elif op is OpCode.OR:
            value = bool(args[0]) or bool(args[1])
        elif op is OpCode.NOT:
            value = not bool(args[0])
        elif op is OpCode.SELECT:
            if len(args) != 3:
                raise ValueError("SELECT requires condition, true value, false value")
            value = args[1] if bool(args[0]) else args[2]
        elif op is OpCode.TUPLE:
            value = tuple(args)
        elif op is OpCode.GETITEM:
            value = args[0][args[1]]
        elif op is OpCode.LEN:
            value = len(args[0])
        elif op is OpCode.CONCAT:
            value = args[0] + args[1]
        elif op is OpCode.RANGE:
            if len(args) == 1:
                value = list(range(args[0]))
            elif len(args) == 2:
                value = list(range(args[0], args[1]))
            elif len(args) == 3:
                value = list(range(args[0], args[1], args[2]))
            else:
                raise ValueError("RANGE requires 1..3 arguments")
        elif op is OpCode.SORT:
            value = sorted(args[0])
        elif op is OpCode.UNIQUE:
            value = list(dict.fromkeys(args[0]))
        elif op is OpCode.REVERSE:
            value = list(reversed(args[0]))
        elif op is OpCode.DIFF:
            seq = list(args[0])
            value = [b - a for a, b in zip(seq, seq[1:])]
        elif op is OpCode.CUMSUM:
            total = 0
            out = []
            for item in args[0]:
                total += item
                out.append(total)
            value = out
        elif op is OpCode.REDUCE_SUM:
            value = sum(args[0])
        elif op is OpCode.RETURN:
            return ExecutionResult(
                value=args[0],
                instructions_executed=instructions_executed,
                collection_items_created=collection_items_created,
            )
        else:  # pragma: no cover - enum exhaustiveness guard
            raise NotImplementedError(op.value)

        collection_items_created += _created_collection_items(value)
        if collection_items_created > budget.max_collection_items_created:
            raise BudgetExceeded(
                "collection-item budget exceeded: "
                f"{collection_items_created} > "
                f"{budget.max_collection_items_created}"
            )
        env[ins.name] = value

    raise RuntimeError("validated program terminated without RETURN")


def execute_program(program: Program, inputs: Mapping[str, Any]) -> Any:
    """Compatibility wrapper returning only the program value."""

    return execute_program_metered(program, inputs).value
