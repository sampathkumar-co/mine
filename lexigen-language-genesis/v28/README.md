# Lexigen v28 — liveness-pruned typed search

v28 keeps the exact v27 candidate semantics, support beams, identities, demonstrations, depth, and raw-candidate ceiling.

It removes only frontiers that cannot reach a Grid output within depth 5. PointSet generation stops after depth 4, ObjectSet generation stops after depth 3, and non-exact Grid expressions at depth 5 are evaluated but not retained. Exact depth-5 Grid outputs are always preserved.

This pruning follows from the frozen type graph and should preserve every expressible depth-5 Grid solution. Held-out generators remain unopened until discovery evidence is frozen.
