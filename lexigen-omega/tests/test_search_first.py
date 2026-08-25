import random
import unittest

from lexigen_omega.genesis import (
    enumerate_fragments,
    match_template,
    mine_semantic_gene_proposals,
    verify_exact_expansion,
)
from lexigen_omega.ir import (
    ExecutionBudget,
    Instruction,
    OpCode,
    Program,
    TypeTag,
    execute_program_metered,
)
from lexigen_omega.portable import execute_portable_metered


class PortableParityTests(unittest.TestCase):
    def _chain(self, ops):
        instructions = []
        current = "x"
        for index, op in enumerate(ops):
            name = f"v{index}"
            instructions.append(Instruction(name, op, (current,), type_tag=TypeTag.SEQ))
            current = name
        instructions.append(Instruction("out", OpCode.RETURN, (current,), type_tag=TypeTag.SEQ))
        return Program(inputs=("x",), instructions=tuple(instructions))

    def test_generated_sequence_programs_match_independent_runtime(self):
        rng = random.Random(20260824)
        unary = [OpCode.REVERSE, OpCode.SORT, OpCode.UNIQUE, OpCode.DIFF, OpCode.CUMSUM]
        for _ in range(80):
            width = rng.randint(1, 5)
            program = self._chain([rng.choice(unary) for _ in range(width)])
            values = [rng.randint(-9, 9) for _ in range(rng.randint(0, 12))]
            budget = ExecutionBudget(max_instructions=20, max_collection_items_created=200)
            primary = execute_program_metered(program, {"x": values}, budget)
            portable = execute_portable_metered(program, {"x": values}, budget)
            self.assertEqual(primary, portable)

    def test_scalar_and_branch_semantics_match(self):
        program = Program(
            inputs=("x", "y"),
            instructions=(
                Instruction("zero", OpCode.CONST, literal=0, type_tag=TypeTag.INT),
                Instruction("lt", OpCode.LT, ("x", "y"), type_tag=TypeTag.BOOL),
                Instruction("neg", OpCode.NEG, ("x",), type_tag=TypeTag.INT),
                Instruction("pick", OpCode.SELECT, ("lt", "neg", "y"), type_tag=TypeTag.INT),
                Instruction("out", OpCode.RETURN, ("pick",), type_tag=TypeTag.INT),
            ),
        )
        for x in range(-4, 5):
            for y in range(-4, 5):
                self.assertEqual(
                    execute_program_metered(program, {"x": x, "y": y}),
                    execute_portable_metered(program, {"x": x, "y": y}),
                )


class SemanticGenesisTests(unittest.TestCase):
    def _program(self, family: str, offset: int) -> Program:
        return Program(
            inputs=("x",),
            instructions=(
                Instruction("k", OpCode.CONST, literal=offset, type_tag=TypeTag.INT),
                Instruction("shift", OpCode.ADD, ("x", "k"), type_tag=TypeTag.INT),
                Instruction("magnitude", OpCode.ABS, ("shift",), type_tag=TypeTag.INT),
                Instruction("out", OpCode.RETURN, ("magnitude",), type_tag=TypeTag.INT),
            ),
            metadata={"source_family": family},
        )

    def test_cross_family_parameterized_gene_is_mined_but_not_admitted(self):
        programs = [
            self._program("numerical", 1),
            self._program("scheduling", 3),
            self._program("graph", 7),
        ]
        proposals = mine_semantic_gene_proposals(programs, min_window=2, max_window=3)
        self.assertTrue(proposals)
        best = proposals[0]
        self.assertGreaterEqual(len(best.source_families), 2)
        self.assertGreater(best.parameter_count, 0)
        self.assertGreater(best.mdl_gain, 0)
        self.assertFalse(best.memory_eligible)

        fragments = enumerate_fragments(programs, min_window=best.fragment_size, max_window=best.fragment_size)
        matched = [record for record in fragments if match_template(best.template, record.fragment) is not None]
        self.assertGreaterEqual(len({record.source_family for record in matched}), 2)
        self.assertTrue(all(verify_exact_expansion(best.template, record.fragment) for record in matched))

    def test_same_family_repetition_does_not_create_cross_family_proposal(self):
        programs = [self._program("same", 1), self._program("same", 2), self._program("same", 3)]
        proposals = mine_semantic_gene_proposals(programs, min_window=2, max_window=3)
        self.assertEqual(proposals, ())


if __name__ == "__main__":
    unittest.main()
