# Lexigen v30 exact memoized recovery

This branch recovers the nine v30 identities that reached the original 20-minute infrastructure limit.

It preserves the frozen candidate sequence, parameter order, examples, limits, runtime semantics and report format. The only change is exact shared-subtree memoization.

The optimized scanner is forbidden from running recovery identities until it reproduces the completed `469497ad` report byte-for-byte.
