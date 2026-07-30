# Lexigen v25 long recovery for 0dfd9992

This isolated branch runs the already validated exact disk-backed v25 recovery engine on the final identity `0dfd9992`.

No implementation, seed, search-budget, candidate-order, report-format or aggregation rule changes are allowed. The only operational change is raising the job ceiling from 40 to 180 minutes.

The final aggregate must contain exactly 37 immutable original reports, two immutable recovered reports and this one report. Held-out validation remains unopened unless the resulting frozen library is nonempty.
