# v25 exact checkpoint recovery

This branch fixes the unresolved v25 runner-loss problem without changing the frozen search.

The enumerator writes atomic checkpoints only after every candidate sharing a global order key has been processed. A resumed run restores the retained expressions, exact SQLite deduplication database, counters and current layer, then rebuilds the deterministic streams and skips the completed prefix.

The first gate forcibly interrupts `c074846d` after at least 50,000 processed candidates, resumes it, and requires the final task report to match the frozen SHA-256 `4d6b326e3f8334aa5d5542cae59344a76a1baa55b1734ddacd3a70c783440091` byte-for-byte.

Task `0dfd9992` is not authorized until that gate passes. Checkpoints are operational state, not scientific evidence.
