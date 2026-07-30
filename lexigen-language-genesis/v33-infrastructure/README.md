# Lexigen v33 infrastructure hardening

This branch fixes recurring research-infrastructure failures without changing any frozen scientific result.

The generator path keeps one Python worker alive per task scan, but reseeds and re-imports the task module for every request. A timed-out request restarts the worker and retries only the identical seed once; replacement seeds are forbidden.

The first gate must reproduce the complete v32 `9caf5b84` task report byte-for-byte. No new task generator may be opened before that gate passes.

Long enumerations will use atomic, hash-bound checkpoints. A resumed run must produce the same final report bytes as an uninterrupted reference run. Partial checkpoints are operational state, never scientific evidence.
