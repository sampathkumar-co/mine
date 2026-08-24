"""LEXIGEN Ω core research package.

Development-only. Importing this package makes no AGI or breakthrough claim.
"""

from .gene import CausalEvidence, GeneAdmissionPolicy, SemanticGene
from .ir import Instruction, OpCode, Program, TypeTag, execute_program

__all__ = [
    "CausalEvidence",
    "GeneAdmissionPolicy",
    "Instruction",
    "OpCode",
    "Program",
    "SemanticGene",
    "TypeTag",
    "execute_program",
]
