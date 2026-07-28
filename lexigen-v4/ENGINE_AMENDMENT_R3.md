# Second preselection amendment

PR #84 and workflow run `30371542030` failed snapshot-free semantic validation before producing any holdout selection or task access.

The correction remains generic:

- lexical atoms now include function names, class names, argument names and import aliases;
- this allows structural semantics expressed in identifiers such as `project_constraints` to be recognised without using task names;
- the grouped warm-start synthetic card now includes an independent approximate verifier, so bounded exact refinement is tested only in a scientifically valid context;
- validation writes and uploads a redacted failure artifact even when an assertion fails.

No benchmark inventory, holdout task identity, task source, manifest, payload, public solver or report was accessed before this amendment. PRs #83 and #84 remain preserved as rejected prelock designs and cannot be used as the campaign engine.
