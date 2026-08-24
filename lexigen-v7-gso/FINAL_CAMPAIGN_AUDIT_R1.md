# LEXIGEN V7 GSO Real Causal-Transfer Pilot R1 — Final Audit

## Verdict

**FAIL — preregistered pilot gate is mathematically unreachable.**

The frozen protocol requires at least two causal-transfer wins. At the kill point, Tasks 1, 3, 4, 5, and 6 could no longer contribute causal credit, leaving only Task 2 capable of adding at most one causal win. Therefore the campaign cannot satisfy the frozen causal gate regardless of the eventual Task-2 diagnostic result.

## What V7 did demonstrate

- Task 1 produced a clean official GSO performance win: full F1 reached about 2.129x harmonic speedup over base and about 1.012x of the expert target, and passed the frozen anti-gaming audit.
- Task 6 produced a very large representative-subset performance result: full F3 reached about 69.573x harmonic speedup, and the frozen anti-gaming audit ultimately passed after a serialization-only infrastructure repair.
- These results show useful repository-scale optimization ability.

## Why this is not causal-transfer success

The central V7 claim is stronger than raw optimization. The learned abstraction library must causally enable useful mechanisms that equally budgeted no-library/random controls do not independently reconstruct.

That requirement failed:

- Task 1 is a performance win but causal-negative under the frozen equal-budget causal rule.
- Task 3's direct native fast path was independently reconstructed byte-for-byte by no-library candidates. Revision 1's execute-certify-recover repair was also reconstructed byte-for-byte by no-library. The remaining frozen learned mechanism families are explicitly mirrored by no-library/random proposals, so revision 2 cannot restore causal exclusivity.
- Task 4 became confirmatory-ineligible after expert-boundary contamination was detected during compile-failure diagnosis. No revision was generated from the exposed expert patch.
- Task 5 was already frozen as diagnostic-only/ineligible.
- Task 6's high-performing mechanism class is independently available to the no-library arm, so the speedup cannot receive learned-library causal credit.
- Task 2 was still running when the kill rule became mathematically triggered. It can contribute at most one additional causal win and therefore cannot rescue the >=2 requirement.

## Integrity decisions

- No selected task was swapped or dropped.
- No threshold was relaxed after observing results.
- Task 3 revision 2 was not run after causal impossibility was established.
- Task 4's exposed expert patch was not used to generate a revision.
- Task 2 may finish and be preserved as post-kill diagnostic evidence, but it cannot reopen the frozen campaign verdict.

## Scientific conclusion

V7 is **not** established as a general causal-transfer breakthrough by this pilot. The strongest lesson is not that V7 cannot optimize; it can. The problem is that the no-library search frequently reconstructs the same useful mechanisms, so the frozen learned library has not yet demonstrated a unique causal capability.

This negative result should be used to redesign the next architecture around mechanism classes that are genuinely learned, compressive, and unavailable to the control search at equal budget—not by adding more task-specific recipes or relaxing gates.
