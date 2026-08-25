from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from .trajectory import MechanismPreference, ProposalProvenance


@dataclass(frozen=True)
class DevelopmentObservation:
    """One deliberately exposed development-only learning event.

    These observations may influence Ω development/search policy, but they can never
    be counted as prospective causal evidence for the final breakthrough claim.
    Infrastructure incidents are represented with ``scientific=False`` and receive
    zero learning weight.
    """

    task_id: str
    family: str
    mechanism_key: str
    reward: float
    source_campaign: str
    scientific: bool = True
    exposed: bool = True
    final_claim_eligible: bool = False

    def validate(self) -> None:
        if not self.task_id.strip():
            raise ValueError("development observation requires task_id")
        if not self.family.strip():
            raise ValueError("development observation requires family")
        if not self.mechanism_key.strip():
            raise ValueError("development observation requires mechanism_key")
        if not 0.0 <= self.reward <= 1.0:
            raise ValueError("development reward must be in [0, 1]")
        if not self.exposed:
            raise ValueError("development archive accepts exposed tasks only")
        if self.final_claim_eligible:
            raise ValueError("development evidence can never be final-claim eligible")


@dataclass(frozen=True)
class MechanismPosterior:
    mechanism_key: str
    task_count: int
    family_count: int
    reward_sum: float
    mean_reward: float
    exploration_bonus: float
    priority: float


def mechanism_key(provenance: ProposalProvenance) -> str:
    sequence = ">".join(provenance.primitive_sequence) or "EMPTY"
    return f"seq:{sequence}"


class DevelopmentArchive:
    """Contamination-safe memory for aggressive Ω iteration.

    The archive intentionally separates *development utility* from *scientific
    causal memory*. Nothing stored here can be promoted automatically to
    ``SemanticGene`` causal evidence. A future clean prospective experiment must
    independently establish that.
    """

    def __init__(self) -> None:
        self._observations: list[DevelopmentObservation] = []

    @property
    def observations(self) -> tuple[DevelopmentObservation, ...]:
        return tuple(self._observations)

    def add(self, observation: DevelopmentObservation) -> None:
        observation.validate()
        self._observations.append(observation)

    def extend(self, observations: Iterable[DevelopmentObservation]) -> None:
        for observation in observations:
            self.add(observation)

    def rank_mechanisms(self, *, exploration: float = 0.35) -> tuple[MechanismPosterior, ...]:
        if exploration < 0:
            raise ValueError("exploration must be non-negative")
        usable = [item for item in self._observations if item.scientific]
        total_tasks = max(1, len({item.task_id for item in usable}))
        by_key: dict[str, list[DevelopmentObservation]] = {}
        for item in usable:
            by_key.setdefault(item.mechanism_key, []).append(item)

        ranked: list[MechanismPosterior] = []
        for key, rows in by_key.items():
            # Aggregate within task first so a task with many pairwise comparisons
            # cannot dominate the archive merely by generating more rows.
            by_task: dict[str, list[float]] = {}
            families: set[str] = set()
            for row in rows:
                by_task.setdefault(row.task_id, []).append(row.reward)
                families.add(row.family)
            task_rewards = [sum(values) / len(values) for values in by_task.values()]
            reward_sum = sum(task_rewards)
            mean_reward = reward_sum / len(task_rewards)
            bonus = exploration * math.sqrt(math.log(total_tasks + 1.0) / len(task_rewards))
            ranked.append(
                MechanismPosterior(
                    mechanism_key=key,
                    task_count=len(task_rewards),
                    family_count=len(families),
                    reward_sum=reward_sum,
                    mean_reward=mean_reward,
                    exploration_bonus=bonus,
                    priority=mean_reward + bonus,
                )
            )

        ranked.sort(
            key=lambda item: (
                -item.priority,
                -item.family_count,
                -item.task_count,
                item.mechanism_key,
            )
        )
        return tuple(ranked)


def preferences_to_development_observations(
    preferences: Iterable[MechanismPreference],
    *,
    family: str,
    source_campaign: str,
) -> tuple[DevelopmentObservation, ...]:
    """Convert frozen pairwise evaluator preferences into development rewards.

    Each comparison contributes a winner=1 / loser=0 event. ``DevelopmentArchive``
    later aggregates within each task before ranking, preventing tasks with more
    comparisons from receiving extra weight.
    """

    output: list[DevelopmentObservation] = []
    for preference in preferences:
        output.append(
            DevelopmentObservation(
                task_id=preference.task_id,
                family=family,
                mechanism_key=mechanism_key(preference.preferred),
                reward=1.0,
                source_campaign=source_campaign,
            )
        )
        output.append(
            DevelopmentObservation(
                task_id=preference.task_id,
                family=family,
                mechanism_key=mechanism_key(preference.rejected),
                reward=0.0,
                source_campaign=source_campaign,
            )
        )
    return tuple(output)
