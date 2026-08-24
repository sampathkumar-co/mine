"""LEXIGEN Ω core research package.

Development-only. Importing this package makes no AGI or breakthrough claim.
"""

from .gene import CausalEvidence, GeneAdmissionPolicy, SemanticGene
from .genesis import GeneProposal, mine_semantic_gene_proposals
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
from .portable import execute_portable, execute_portable_metered

__all__ = [
    "BudgetExceeded",
    "CausalEvidence",
    "ExecutionBudget",
    "ExecutionResult",
    "GeneAdmissionPolicy",
    "GeneProposal",
    "Instruction",
    "OpCode",
    "Program",
    "SemanticGene",
    "TypeTag",
    "execute_portable",
    "execute_portable_metered",
    "execute_program",
    "execute_program_metered",
    "mine_semantic_gene_proposals",
]
