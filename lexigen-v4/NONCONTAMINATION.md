# Noncontamination boundary

This campaign may read only its own `lexigen-v4/` directory and the frozen external AlgoTune snapshots specified in `PROTOCOL.md`.

It must not read from, write to, import from, trigger, cancel, or compare against unsealed evidence in:

- `lexigen-world-covering-*`
- `lexigen/language-genesis-*`
- Mini-ORIGIN directories
- v3 task branches

Previous v3 results may enter v4 only through the abstract, precommitted transfer-memory records already embedded in `engine.py`. No task payload, target identity, candidate source, hidden metric or post-seal frontier value may be copied into the v4 campaign.
