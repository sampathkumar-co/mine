"""LEXIGEN Ω core research package.

Development-only. Importing this package makes no AGI or breakthrough claim.
"""

from .gene import CausalEvidence, GeneAdmissionPolicy, SemanticGene
from .ir import (
    BudgetExceeded,
    ExecutionBudget,
    ExecutionResult,
    Instruction,
    OpCode,
    Program,
    TypeTag,
    execute_program,
    execute_program_metered,
)

__all__ = [
    "BudgetExceeded",
    "CausalEvidence",
    "ExecutionBudget",
    "ExecutionResult",
    "GeneAdmissionPolicy",
    "Instruction",
    "OpCode",
    "Program",
    "SemanticGene",
    "TypeTag",
    "execute_program",
    "execute_program_metered",
]
