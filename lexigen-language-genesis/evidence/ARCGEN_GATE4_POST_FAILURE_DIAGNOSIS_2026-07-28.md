# ARC-GEN gate 4 post-failure diagnosis — 2026-07-28

Status: **post-failure language extension; no rescore and no breakthrough claim**.

## Authentic blind result

- frozen engine commit: `9f24d4e696ecea1e799d0c8b508e0ca3b88acb3f`
- pinned ARC-GEN commit: `a15cbdb44c776610aeeb9f487a06af875d3d0878`
- protocol: `arcgen-gate-v5`
- selected task: `228f6490`
- redacted-task SHA-256: `3f642ccb79b55f5430b3675f7e6fbfbe354f5888a23a1880a8407d1ed02a1a2e`
- sealed-output SHA-256: `2e2734865ab532e861195d38f70846137bfbb2e52f8884c918ba71728c74f8b0`
- candidates tested: `75,000`
- execution signatures: `30,469`
- program found: `false`
- hidden scoring: not performed

Gate 4 remains a permanent negative result. Later language versions are ineligible for rescoring it.

## Training-only diagnosis

After the failure was committed, a separate reveal-only branch regenerated the six training pairs and verified both committed hashes before publishing them. Hidden test outputs were not published or used.

The missing semantic family is object-centric shape transplantation:

1. identify gray (`5`) enclosure objects;
2. identify enclosed background components that form holes;
3. identify external single-colour connected components;
4. match source and hole shapes, optionally under rotation/reflection;
5. transfer the source colour into the matching hole;
6. erase the external source component.

This is represented in `external/arc_language_v6.py` as `transplant_matching_components_into_gray_holes`.

## Claim boundary

This extension is informed by an external failure and is therefore meaningful vocabulary growth, but it is still human-diagnosed and human-implemented. It is not autonomous semantic invention and is not a world breakthrough. A valid next gate must freeze v6 before selecting a new untouched external task and must preserve candidate predictions before hidden scoring.

Public human-solution descriptions for ARC task `228f6490` independently describe the same object-transfer rule, which supports the diagnosis but also confirms that the semantic is known rather than novel.
