# Lexigen v26 — guided semantic search

v26 is a controlled improvement over the frozen v25 implementation. It uses the same task identities, generated examples, typed language semantics, depth, and five-million raw-candidate ceiling.

The only architectural change is search control: support types are enumerated before Grid, and non-exact Grid expressions are retained through a deterministic target-distance beam. Exact target matches are never pruned.

This branch must not modify v25 evidence. Held-out generators remain unopened until a v26 discovery library is frozen. Any result remains public ARC-GEN development evidence, not a world-level breakthrough.
