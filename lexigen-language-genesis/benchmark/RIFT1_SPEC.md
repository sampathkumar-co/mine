# RIFT-1 — hidden control-law induction

Status: public synthetic research stage; no breakthrough claim.

## Purpose

RIFT-0 could be overfit to one missing idea: iterate a monotone transition until it stabilizes. RIFT-1 removes that single-mechanism advantage.

The engine is shown input/output examples and black-box transition functions. It is not told which control law generated the target. It must synthesize a persistent executable artifact and transfer it across differently encoded worlds.

## Mechanism families

1. **Stabilizing closure**
   - repeatedly apply the transition;
   - return the first stable state.

2. **Trajectory union**
   - follow a deterministic transition until a state repeats;
   - return the union of every state observed before repetition.

3. **Two-cycle canonicalization**
   - follow a deterministic transition through a transient into a period-two cycle;
   - return the lexicographically canonical state from the two-cycle.

Each mechanism appears through several unrelated surface encodings and longer unseen transfer depths.

## Starting language

The frozen baseline may apply the transition only three times and return the current state. It has no history, accumulator, branching, or looping.

## Synthesis meta-machine

The public synthetic stage supplies low-level machine operations:

- apply the black-box transition;
- copy/advance state;
- compare current and next states;
- record and query previously seen states;
- union a state into an accumulator;
- branch and jump;
- return one of the machine registers or a canonical choice.

The synthesizer must discover program order, branch targets, return mode, and the subset of operations required for each mechanism. Candidate ordering is deterministic and hash-based, not hand-arranged to favour the target.

This still supplies the opcode meanings. Therefore success is fixed-meta-language synthesis, not full L4 semantic invention.

## Required public-stage results

For every mechanism:

- three-step baseline succeeds on shallow calibration cases;
- baseline fails materially on longer transfer cases;
- synthesized artifact reaches exact transfer accuracy across all surfaces;
- artifact executes in a separately implemented runtime;
- deterministic reruns produce matching artifact and report hashes;
- a single artifact is reused without modification across all surfaces of its mechanism.

## Stronger RIFT-1 research gate

The next stage removes mechanism-labelled training partitions. A selector must infer which synthesized artifact applies to a new task from examples alone, without mechanism names.

## Why this is not yet a breakthrough

Program synthesis inside a human-supplied universal or domain-specific machine is established research. RIFT-1 is valuable only as an anti-cheating benchmark and infrastructure step toward a later stage where the system must propose new operational semantics not present in its initial opcode inventory.
