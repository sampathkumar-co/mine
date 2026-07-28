# Lexigen World Covering v2 — Synthetic Validation

This branch validates the already-frozen v2 engine without downloading the covering repository snapshot or revealing any real target identity.

It checks:

- exact Git blob hashes from the frozen v2 lock;
- deterministic reserved/v2 target slicing using synthetic metadata;
- independent acceptance of a known seven-block Fano-plane covering;
- rejection of an incomplete covering;
- solver discovery of a valid seven-block covering when the synthetic prior upper bound is eight;
- rejection of an impossible six-block target.

This validation branch is not a research result and must not be merged. It does not modify the frozen v2 branch.
