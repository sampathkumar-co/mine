from __future__ import annotations

import hashlib
import json
import pickle
import sqlite3
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
RECOVERY = HERE.parent / "v25-recovery"
V25 = HERE.parent / "v25"
import sys
for folder in (RECOVERY, V25):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from enumerator_v25_recovery import (
    TYPE_ORDER,
    Expression,
    Grid,
    RuntimeV25Error,
    abstract_literal_colours,
    build_streams,
    canonical,
    initial_expressions,
    make_expression,
    merge_streams,
    semantic_signature,
    sha256_json,
)


class ControlledCheckpointStop(RuntimeError):
    pass


class CheckpointDatabase:
    def __init__(self, path: Path, *, create: bool) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if create and self.path.exists():
            self.path.unlink()
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=DELETE")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA temp_store=MEMORY")
        self.connection.execute("PRAGMA foreign_keys=ON")
        if create:
            self.connection.executescript(
                """
                CREATE TABLE metadata (
                    key TEXT PRIMARY KEY,
                    value BLOB NOT NULL
                ) WITHOUT ROWID;
                CREATE TABLE seen (
                    namespace TEXT NOT NULL,
                    digest BLOB NOT NULL,
                    payload BLOB NOT NULL,
                    PRIMARY KEY(namespace, digest, payload)
                ) WITHOUT ROWID;
                CREATE TABLE expressions (
                    type_name TEXT NOT NULL,
                    depth INTEGER NOT NULL,
                    sequence INTEGER NOT NULL,
                    nodes INTEGER NOT NULL,
                    ast_text TEXT NOT NULL,
                    values_blob BLOB NOT NULL,
                    PRIMARY KEY(type_name, depth, sequence)
                ) WITHOUT ROWID;
                CREATE TABLE state (
                    id INTEGER PRIMARY KEY CHECK(id = 1),
                    payload BLOB NOT NULL
                );
                """
            )
            self.connection.commit()
        self.connection.execute("BEGIN IMMEDIATE")
    def set_metadata(self, key: str, value: Any) -> None:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES(?, ?)",
            (key, payload),
        )

    def metadata(self, key: str) -> Any:
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"missing checkpoint metadata: {key}")
        return json.loads(bytes(row[0]).decode("utf-8"))

    def add_payload(self, namespace: str, payload: bytes) -> bool:
        digest = hashlib.sha256(payload).digest()
        cursor = self.connection.execute(
            "INSERT OR IGNORE INTO seen(namespace, digest, payload) VALUES(?, ?, ?)",
            (namespace, digest, payload),
        )
        return cursor.rowcount == 1

    def add_signature(self, type_name: str, signature: tuple[Any, ...]) -> bool:
        return self.add_payload(
            type_name,
            pickle.dumps(signature, protocol=5),
        )
    def add_expression(
        self,
        expression: Expression,
        sequence: int,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO expressions(
                type_name, depth, sequence, nodes, ast_text, values_blob
            ) VALUES(?, ?, ?, ?, ?, ?)
            """,
            (
                expression.type_name,
                expression.depth,
                sequence,
                expression.nodes,
                expression.ast_text,
                pickle.dumps(expression.values, protocol=5),
            ),
        )

    def save_state(self, state: dict[str, Any]) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO state(id, payload) VALUES(1, ?)",
            (pickle.dumps(state, protocol=5),),
        )
        self.connection.commit()
        self.connection.execute("BEGIN IMMEDIATE")

    def load_state(self) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT payload FROM state WHERE id = 1"
        ).fetchone()
        if row is None:
            raise RuntimeError("checkpoint has no saved state")
        return pickle.loads(bytes(row[0]))
    def load_store(
        self,
        maximum_depth: int,
    ) -> dict[str, list[list[Expression]]]:
        store = {
            type_name: [[] for _ in range(maximum_depth + 1)]
            for type_name in TYPE_ORDER
        }
        rows = self.connection.execute(
            """
            SELECT type_name, depth, nodes, ast_text, values_blob
            FROM expressions
            ORDER BY type_name, depth, sequence
            """
        )
        for type_name, depth, nodes, ast_text, values_blob in rows:
            expression = Expression(
                type_name=str(type_name),
                depth=int(depth),
                nodes=int(nodes),
                ast_text=str(ast_text),
                values=pickle.loads(bytes(values_blob)),
            )
            store[str(type_name)][int(depth)].append(expression)
        return store

    def close(self) -> None:
        try:
            self.connection.rollback()
        except sqlite3.Error:
            pass
        self.connection.close()


def input_binding(
    examples: Sequence[tuple[Grid, Grid]],
    budgets: dict[str, int],
) -> dict[str, Any]:
    document = [
        {"input": source, "output": target}
        for source, target in examples
    ]
    return {
        "examples_sha256": sha256_json(document),
        "budgets": budgets,
    }


def initialize_checkpoint(
    database: CheckpointDatabase,
    examples: Sequence[tuple[Grid, Grid]],
    maximum_depth: int,
    binding: dict[str, Any],
) -> tuple[dict[str, list[list[Expression]]], dict[str, Any]]:
    sources = tuple(source for source, _ in examples)
    targets = tuple(target for _, target in examples)
    nontrivial = any(source != target for source, target in examples)
    store = {
        type_name: [[] for _ in range(maximum_depth + 1)]
        for type_name in TYPE_ORDER
    }
    state: dict[str, Any] = {
        "schema": "lexigen-v25-checkpoint-state-v1",
        "phase": "enumerating",
        "current_depth": 1,
        "current_type_index": 0,
        "current_layer_accepted": 0,
        "current_layer_baseline": None,
        "last_completed_order_key": None,
        "processed_candidate_count": 0,
        "total_unique": 0,
        "raw_candidates": 0,
        "runtime_invalid": 0,
        "semantic_duplicates": 0,
        "ast_duplicates": 0,
        "exact_expressions": [],
        "statistics": [],
        "exhausted_reason": None,
    }
    database.set_metadata("binding", binding)
    leaves = initial_expressions(sources)
    for type_name in TYPE_ORDER:
        accepted = 0
        for expression in leaves[type_name]:
            signature = semantic_signature(type_name, expression.values)
            if not database.add_signature(type_name, signature):
                state["semantic_duplicates"] += 1
                continue
            sequence = len(store[type_name][0])
            store[type_name][0].append(expression)
            database.add_expression(expression, sequence)
            state["total_unique"] += 1
            accepted += 1
            if (
                type_name == "Grid"
                and nontrivial
                and expression.values == targets
            ):
                state["exact_expressions"].append(expression)
        state["statistics"].append(
            {
                "depth": 0,
                "type": type_name,
                "unique_expressions": accepted,
                "raw_candidates": 0,
                "runtime_invalid": 0,
                "semantic_duplicates": 0,
            }
        )
    database.save_state(state)
    return store, state


def verify_resume_binding(
    database: CheckpointDatabase,
    binding: dict[str, Any],
) -> None:
    if database.metadata("binding") != binding:
        raise RuntimeError("checkpoint input or budget binding mismatch")


def advance_layer(state: dict[str, Any], maximum_depth: int) -> None:
    type_index = int(state["current_type_index"]) + 1
    depth = int(state["current_depth"])
    if type_index >= len(TYPE_ORDER):
        type_index = 0
        depth += 1
    state["current_type_index"] = type_index
    state["current_depth"] = depth
    state["current_layer_accepted"] = 0
    state["current_layer_baseline"] = None
    state["last_completed_order_key"] = None
    if depth > maximum_depth:
        state["phase"] = "complete_pending_result"


def finalize_result(
    store: dict[str, list[list[Expression]]],
    state: dict[str, Any],
    *,
    nontrivial: bool,
    maximum_depth: int,
) -> dict[str, Any]:
    exact_structures: dict[str, dict[str, Any]] = {}
    exact_expressions = sorted(
        state["exact_expressions"],
        key=lambda expression: expression.order_key,
    )
    for expression in exact_expressions:
        abstract_ast, arguments = abstract_literal_colours(expression.ast)
        structure_hash = sha256_json(abstract_ast)
        entry = exact_structures.setdefault(
            structure_hash,
            {
                "structure_sha256": structure_hash,
                "structure": abstract_ast,
                "minimum_depth": expression.depth,
                "minimum_nodes": expression.nodes,
                "concrete_programs": [],
            },
        )
        entry["minimum_depth"] = min(
            entry["minimum_depth"],
            expression.depth,
        )
        entry["minimum_nodes"] = min(
            entry["minimum_nodes"],
            expression.nodes,
        )
        concrete = {
            "arguments": arguments,
            "concrete_ast_sha256": sha256_json(expression.ast),
            "depth": expression.depth,
            "nodes": expression.nodes,
        }
        if concrete not in entry["concrete_programs"]:
            entry["concrete_programs"].append(concrete)

    for entry in exact_structures.values():
        entry["concrete_programs"].sort(key=canonical)

    return {
        "schema": "lexigen-v25-semantic-enumeration-result-v1",
        "nontrivial_task": nontrivial,
        "maximum_depth": maximum_depth,
        "enumeration_complete": state["exhausted_reason"] is None,
        "exhausted_reason": state["exhausted_reason"],
        "raw_candidate_evaluations": state["raw_candidates"],
        "runtime_invalid_candidates": state["runtime_invalid"],
        "semantic_duplicates": state["semantic_duplicates"],
        "ast_duplicates": state["ast_duplicates"],
        "total_unique_expressions": state["total_unique"],
        "unique_by_type": {
            type_name: sum(len(layer) for layer in store[type_name])
            for type_name in TYPE_ORDER
        },
        "exact_concrete_programs": len(exact_expressions),
        "exact_abstract_structures": len(exact_structures),
        "exact_structures": [
            exact_structures[key]
            for key in sorted(exact_structures)
        ],
        "statistics": state["statistics"],
    }


def enumerate_programs_checkpointed(
    examples: Sequence[tuple[Grid, Grid]],
    *,
    maximum_depth: int,
    maximum_unique_per_type_per_depth: int,
    maximum_total_unique: int,
    maximum_raw_candidates: int,
    checkpoint_path: Path,
    resume: bool,
    checkpoint_interval_processed_candidates: int = 25000,
    stop_after_processed_candidates: int | None = None,
) -> dict[str, Any]:
    if not examples:
        raise ValueError("at least one example is required")
    if checkpoint_interval_processed_candidates <= 0:
        raise ValueError("checkpoint interval must be positive")
    budgets = {
        "maximum_depth": int(maximum_depth),
        "maximum_unique_per_type_per_depth": int(
            maximum_unique_per_type_per_depth
        ),
        "maximum_total_unique": int(maximum_total_unique),
        "maximum_raw_candidates": int(maximum_raw_candidates),
    }
    binding = input_binding(examples, budgets)
    checkpoint_path = Path(checkpoint_path)
    if resume and not checkpoint_path.exists():
        raise RuntimeError("resume requested but checkpoint does not exist")
    database = CheckpointDatabase(
        checkpoint_path,
        create=not resume,
    )
    try:
        if resume:
            verify_resume_binding(database, binding)
            state = database.load_state()
            store = database.load_store(maximum_depth)
        else:
            store, state = initialize_checkpoint(
                database,
                examples,
                maximum_depth,
                binding,
            )

        if state["phase"] == "complete":
            return state["final_result"]

        sources = tuple(source for source, _ in examples)
        targets = tuple(target for _, target in examples)
        nontrivial = any(
            source != target
            for source, target in examples
        )
        processed_since_checkpoint = 0

        while (
            int(state["current_depth"]) <= maximum_depth
            and state["exhausted_reason"] is None
        ):
            depth = int(state["current_depth"])
            type_index = int(state["current_type_index"])
            type_name = TYPE_ORDER[type_index]
            if type_name == "Color":
                state["statistics"].append(
                    {
                        "depth": depth,
                        "type": type_name,
                        "unique_expressions": 0,
                        "raw_candidates": 0,
                        "runtime_invalid": 0,
                        "semantic_duplicates": 0,
                    }
                )
                advance_layer(state, maximum_depth)
                database.save_state(state)
                if (
                    stop_after_processed_candidates is not None
                    and state["processed_candidate_count"]
                    >= stop_after_processed_candidates
                    and state["phase"] != "complete_pending_result"
                ):
                    raise ControlledCheckpointStop(
                        "forced interruption after checkpoint"
                    )
                continue

            if state["current_layer_baseline"] is None:
                state["current_layer_baseline"] = {
                    "raw": state["raw_candidates"],
                    "invalid": state["runtime_invalid"],
                    "duplicates": state["semantic_duplicates"],
                }
                state["current_layer_accepted"] = len(
                    store[type_name][depth]
                )

            accepted = int(state["current_layer_accepted"])
            baseline = state["current_layer_baseline"]
            marker_value = state["last_completed_order_key"]
            marker = (
                (int(marker_value[0]), str(marker_value[1]))
                if marker_value is not None
                else None
            )
            streams = build_streams(
                store,
                type_name,
                depth,
                sources,
            )
            current_key: tuple[int, str] | None = None
            for candidate in merge_streams(streams):
                if accepted >= maximum_unique_per_type_per_depth:
                    break
                if state["total_unique"] >= maximum_total_unique:
                    state["exhausted_reason"] = (
                        "maximum_total_unique_expressions"
                    )
                    break
                if state["raw_candidates"] >= maximum_raw_candidates:
                    state["exhausted_reason"] = (
                        "maximum_raw_candidate_evaluations"
                    )
                    break

                candidate_key = candidate.order_key
                if marker is not None and candidate_key <= marker:
                    continue
                if (
                    current_key is not None
                    and candidate_key != current_key
                ):
                    state["last_completed_order_key"] = [
                        current_key[0],
                        current_key[1],
                    ]
                    state["current_layer_accepted"] = accepted
                    if (
                        processed_since_checkpoint
                        >= checkpoint_interval_processed_candidates
                    ):
                        database.save_state(state)
                        processed_since_checkpoint = 0
                        if (
                            stop_after_processed_candidates is not None
                            and state["processed_candidate_count"]
                            >= stop_after_processed_candidates
                        ):
                            raise ControlledCheckpointStop(
                                "forced interruption after checkpoint"
                            )
                current_key = candidate_key
                state["processed_candidate_count"] += 1
                processed_since_checkpoint += 1
                if not database.add_payload(
                    "__ast__",
                    candidate.ast_text.encode("utf-8"),
                ):
                    state["ast_duplicates"] += 1
                    continue

                state["raw_candidates"] += 1
                try:
                    values = candidate.evaluator(
                        candidate.children,
                        candidate.constants,
                    )
                    signature = semantic_signature(type_name, values)
                except (
                    RuntimeV25Error,
                    ValueError,
                    IndexError,
                    KeyError,
                    TypeError,
                    OverflowError,
                ):
                    state["runtime_invalid"] += 1
                    continue

                if not database.add_signature(type_name, signature):
                    state["semantic_duplicates"] += 1
                    continue

                expression = make_expression(
                    type_name,
                    depth,
                    candidate.ast,
                    values,
                    nodes=candidate.nodes,
                    ast_text=candidate.ast_text,
                )
                sequence = len(store[type_name][depth])
                store[type_name][depth].append(expression)
                database.add_expression(expression, sequence)
                state["total_unique"] += 1
                accepted += 1
                state["current_layer_accepted"] = accepted
                if (
                    type_name == "Grid"
                    and nontrivial
                    and values == targets
                ):
                    state["exact_expressions"].append(expression)

            if current_key is not None:
                state["last_completed_order_key"] = [
                    current_key[0],
                    current_key[1],
                ]

            store[type_name][depth].sort(
                key=lambda expression: expression.order_key
            )
            state["statistics"].append(
                {
                    "depth": depth,
                    "type": type_name,
                    "unique_expressions": accepted,
                    "raw_candidates": (
                        state["raw_candidates"] - baseline["raw"]
                    ),
                    "runtime_invalid": (
                        state["runtime_invalid"] - baseline["invalid"]
                    ),
                    "semantic_duplicates": (
                        state["semantic_duplicates"]
                        - baseline["duplicates"]
                    ),
                }
            )
            advance_layer(state, maximum_depth)
            database.save_state(state)
            processed_since_checkpoint = 0
            if (
                stop_after_processed_candidates is not None
                and state["processed_candidate_count"]
                >= stop_after_processed_candidates
                and state["phase"] != "complete_pending_result"
            ):
                raise ControlledCheckpointStop(
                    "forced interruption after completed layer"
                )

        result = finalize_result(
            store,
            state,
            nontrivial=nontrivial,
            maximum_depth=maximum_depth,
        )
        state["phase"] = "complete"
        state["final_result"] = result
        database.save_state(state)
        return result
    finally:
        database.close()
