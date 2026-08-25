import unittest

from lexigen_omega.gene import CausalEvidence, GeneAdmissionPolicy, SemanticGene
from lexigen_omega.ir import (
    BudgetExceeded,
    ExecutionBudget,
    Instruction,
    OpCode,
    Program,
    TypeTag,
    execute_program,
    execute_program_metered,
)


class IRTests(unittest.TestCase):
    def make_program(self) -> Program:
        return Program(
            inputs=("x",),
            instructions=(
                Instruction("r", OpCode.REVERSE, ("x",), type_tag=TypeTag.SEQ),
                Instruction("c", OpCode.CUMSUM, ("r",), type_tag=TypeTag.SEQ),
                Instruction("out", OpCode.RETURN, ("c",), type_tag=TypeTag.SEQ),
            ),
            metadata={"z": 1, "a": [2, 3]},
        )

    def test_execute(self) -> None:
        self.assertEqual(execute_program(self.make_program(), {"x": [1, 2, 3]}), [3, 5, 6])

    def test_metered_execution_is_deterministic(self) -> None:
        result = execute_program_metered(self.make_program(), {"x": [1, 2, 3]})
        self.assertEqual(result.value, [3, 5, 6])
        self.assertEqual(result.instructions_executed, 3)
        self.assertEqual(result.collection_items_created, 6)

    def test_instruction_budget_is_hard(self) -> None:
        with self.assertRaisesRegex(BudgetExceeded, "instruction budget exceeded"):
            execute_program_metered(
                self.make_program(),
                {"x": [1, 2, 3]},
                ExecutionBudget(max_instructions=2),
            )

    def test_collection_budget_is_hard(self) -> None:
        with self.assertRaisesRegex(BudgetExceeded, "collection-item budget exceeded"):
            execute_program_metered(
                self.make_program(),
                {"x": [1, 2, 3]},
                ExecutionBudget(max_collection_items_created=5),
            )

    def test_fingerprint_is_deterministic(self) -> None:
        p1 = self.make_program()
        p2 = Program(
            inputs=p1.inputs,
            instructions=p1.instructions,
            metadata={"a": [2, 3], "z": 1},
        )
        self.assertEqual(p1.fingerprint(), p2.fingerprint())

    def test_unknown_reference_rejected(self) -> None:
        p = Program(
            inputs=("x",),
            instructions=(
                Instruction("bad", OpCode.ADD, ("x", "missing")),
                Instruction("out", OpCode.RETURN, ("bad",)),
            ),
        )
        with self.assertRaisesRegex(ValueError, "unknown value"):
            p.validate()


class GeneTests(unittest.TestCase):
    def gene(self, evidence):
        program = Program(
            inputs=("x",),
            instructions=(
                Instruction("r", OpCode.REVERSE, ("x",)),
                Instruction("out", OpCode.RETURN, ("r",)),
            ),
        )
        return SemanticGene(
            gene_id="G-001",
            expansion=program,
            source_families=("development-a",),
            evidence=tuple(evidence),
        )

    def win(self, task, ecosystem, family):
        return CausalEvidence(
            task_id=task,
            ecosystem=ecosystem,
            repository_or_family=family,
            full_clean_win=True,
            full_used_gene=True,
            no_memory_qualifying_win=False,
            random_memory_qualifying_win=False,
            removal_preserved_advantage=False,
            equal_budget=True,
        )

    def test_two_diverse_causal_wins_admit(self) -> None:
        gene = self.gene(
            [
                self.win("t1", "gso", "repo-one"),
                self.win("t2", "math", "family-two"),
            ]
        )
        self.assertTrue(GeneAdmissionPolicy().admissible(gene))

    def test_random_control_reproduction_blocks_credit(self) -> None:
        bad = CausalEvidence(
            task_id="t1",
            ecosystem="gso",
            repository_or_family="repo-one",
            full_clean_win=True,
            full_used_gene=True,
            no_memory_qualifying_win=False,
            random_memory_qualifying_win=True,
            removal_preserved_advantage=False,
            equal_budget=True,
        )
        gene = self.gene([bad, self.win("t2", "math", "family-two")])
        self.assertFalse(GeneAdmissionPolicy().admissible(gene))

    def test_removal_must_destroy_advantage(self) -> None:
        bad = CausalEvidence(
            task_id="t1",
            ecosystem="gso",
            repository_or_family="repo-one",
            full_clean_win=True,
            full_used_gene=True,
            no_memory_qualifying_win=False,
            random_memory_qualifying_win=False,
            removal_preserved_advantage=True,
            equal_budget=True,
        )
        gene = self.gene([bad, self.win("t2", "math", "family-two")])
        self.assertFalse(GeneAdmissionPolicy().admissible(gene))


if __name__ == "__main__":
    unittest.main()
