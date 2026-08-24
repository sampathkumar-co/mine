"""LEXIGEN Ω core research package.

Development-only. Importing this package makes no AGI or breakthrough claim.
"""

from .development import (
    DevelopmentArchive,
    DevelopmentObservation,
    MechanismPosterior,
    mechanism_key,
    preferences_to_development_observations,
)
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
    "DevelopmentArchive",
    "DevelopmentObservation",
    "EvaluationObservation",
    "ExecutionBudget",
    "ExecutionResult",
    "GeneAdmissionPolicy",
    "GeneProposal",
    "HindsightPreference",
    "Instruction",
    "MechanismObservation",
    "MechanismPosterior",
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
    "mechanism_key",
    "mine_semantic_gene_proposals",
    "parse_v7_gso_proposals",
    "preferences_to_development_observations",
]
