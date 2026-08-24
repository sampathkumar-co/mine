from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .ir import Program


@dataclass(frozen=True)
class CausalEvidence:
    """One prospective transfer experiment for a semantic gene.

    A positive record is intentionally strict: the full arm must win, both controls
    must fail to reproduce the qualifying advantage, and removing the gene must
    remove that advantage under equal budgets.
    """

    task_id: str
    ecosystem: str
    repository_or_family: str
    full_clean_win: bool
    full_used_gene: bool
    no_memory_qualifying_win: bool
    random_memory_qualifying_win: bool
    removal_preserved_advantage: bool
    equal_budget: bool
    post_result_revision: bool = False

    @property
    def causal_win(self) -> bool:
        return (
            self.full_clean_win
            and self.full_used_gene
            and not self.no_memory_qualifying_win
            and not self.random_memory_qualifying_win
            and not self.removal_preserved_advantage
            and self.equal_budget
            and not self.post_result_revision
        )


@dataclass(frozen=True)
class SemanticGene:
    gene_id: str
    expansion: Program
    source_families: tuple[str, ...]
    description: str = ""
    evidence: tuple[CausalEvidence, ...] = field(default_factory=tuple)

    @property
    def semantic_fingerprint(self) -> str:
        return self.expansion.fingerprint()

    def causal_wins(self) -> tuple[CausalEvidence, ...]:
        return tuple(e for e in self.evidence if e.causal_win)


@dataclass(frozen=True)
class GeneAdmissionPolicy:
    """Conservative long-term-memory admission policy.

    Development may create speculative genes freely. `admissible()` controls only
    admission to causal long-term memory, where later solvers may rely on a gene as
    learned transferable knowledge.
    """

    min_causal_wins: int = 2
    min_ecosystems: int = 2
    min_distinct_target_families: int = 2
    require_source_target_separation: bool = True

    def admissible(self, gene: SemanticGene) -> bool:
        wins = gene.causal_wins()
        if len(wins) < self.min_causal_wins:
            return False

        ecosystems = {e.ecosystem for e in wins}
        if len(ecosystems) < self.min_ecosystems:
            return False

        targets = {e.repository_or_family for e in wins}
        if len(targets) < self.min_distinct_target_families:
            return False

        if self.require_source_target_separation:
            sources = set(gene.source_families)
            if any(e.repository_or_family in sources for e in wins):
                return False

        return True

    def explain_rejection(self, gene: SemanticGene) -> tuple[str, ...]:
        reasons: list[str] = []
        wins = gene.causal_wins()
        if len(wins) < self.min_causal_wins:
            reasons.append(
                f"causal wins {len(wins)} < required {self.min_causal_wins}"
            )
        if len({e.ecosystem for e in wins}) < self.min_ecosystems:
            reasons.append("insufficient ecosystem diversity")
        if len({e.repository_or_family for e in wins}) < self.min_distinct_target_families:
            reasons.append("insufficient target-family diversity")
        if self.require_source_target_separation:
            sources = set(gene.source_families)
            overlap = sorted(
                {e.repository_or_family for e in wins}.intersection(sources)
            )
            if overlap:
                reasons.append(f"source/target family overlap: {overlap}")
        return tuple(reasons)


def unique_semantic_genes(genes: Iterable[SemanticGene]) -> tuple[SemanticGene, ...]:
    """Deduplicate genes by executable semantics, not by human-visible ID."""

    chosen: dict[str, SemanticGene] = {}
    for gene in genes:
        chosen.setdefault(gene.semantic_fingerprint, gene)
    return tuple(chosen[key] for key in sorted(chosen))
