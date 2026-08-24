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
from .trajectory import (
    EvaluationObservation,
    HindsightPreference,
    MechanismObservation,
    MechanismPreference,
    ProposalProvenance,
    attach_proposal_provenance,
    build_hindsight_preferences,
    build_mechanism_preferences,
    ingest_v7_gso_preflight_result,
    parse_v7_gso_proposals,
)

__all__ = [
    "BudgetExceeded",
    "CausalEvidence",
    "EvaluationObservation",
    "ExecutionBudget",
    "ExecutionResult",
    "GeneAdmissionPolicy",
    "GeneProposal",
    "HindsightPreference",
    "Instruction",
    "MechanismObservation",
    "MechanismPreference",
    "OpCode",
    "Program",
    "ProposalProvenance",
    "SemanticGene",
    "TypeTag",
    "attach_proposal_provenance",
    "build_hindsight_preferences",
    "build_mechanism_preferences",
    "execute_portable",
    "execute_portable_metered",
    "execute_program",
    "execute_program_metered",
    "ingest_v7_gso_preflight_result",
    "mine_semantic_gene_proposals",
    "parse_v7_gso_proposals",
]
