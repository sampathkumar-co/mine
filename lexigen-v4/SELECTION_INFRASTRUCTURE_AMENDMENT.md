# Selection infrastructure amendment

The first locked selection run (`30372778744`) verified `ENGINE_LOCK.json` successfully and then failed before selection with `source tree contains no task names`.

Redacted diagnostic run `30373004021` confirmed that no task content, manifest, payload, report or public solver was opened. Path-only runs `30373226269` and `30373399055` established the frozen source layout:

- 154 task source files use `AlgoTuneTasks/<task>/<task>.py`;
- each is paired with `AlgoTuneTasks/<task>/description.txt`;
- the original enumerator incorrectly expected `AlgoTuneTasks/<task>/task.py`.

`select_holdouts_layout_v2.py` changes only source-name extraction to the observed frozen path convention. It imports the locked seed, exclusions, family classifier, task scoring, deterministic selector, task count and diversity constraints without alteration.

This is classified as an infrastructure-only frozen-snapshot layout amendment. The engine and every scientific campaign decision remain unchanged.
