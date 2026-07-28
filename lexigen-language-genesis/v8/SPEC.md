# Lexigen v8 — Autonomous Meta-Grammar Growth

Status: initialized after the reproducible v7 checkpoint; no breakthrough claim.

## Objective

Move beyond choosing an AST from a human-enumerated semantic grammar. v8 must construct new typed predicates, selectors, relations, or edit combinators from a smaller fixed substrate, serialize the resulting grammar extension, and execute it in an interpreter that did not contain the extension as a named operation.

## Frozen substrate

The starting substrate may expose only generic operations such as:

- finite sets, tuples, numbers and booleans;
- equality and ordering;
- map, filter, fold and composition;
- grid coordinates and cell lookup;
- four-neighbour adjacency;
- bounded recursion or fixed-point evaluation;
- deterministic rendering of explicitly selected cells.

It must not expose finished notions such as `matching hole`, `sprite`, `single-component colour`, `frame transplant`, `reflected trajectory`, or any other previously diagnosed ARC task rule.

## Required loop

1. Freeze the substrate, synthesizer, budgets and external-task selector.
2. Observe demonstrations from a task that the frozen v7 grammar cannot solve.
3. Diagnose a missing type, predicate, relation or combinator.
4. synthesize an executable grammar extension from the substrate;
5. serialize both the extension and the task program;
6. execute them in a separately implemented interpreter;
7. commit the extension and predictions before hidden scoring;
8. score once;
9. reuse the extension on another untouched task without modification.

## v8 gate

A v8 pass requires:

- no human-authored post-failure operator;
- a new executable grammar production absent from v7;
- a frozen fixed-grammar baseline failure under equal budget;
- exact hidden predictions committed before scoring;
- independent interpreter agreement;
- ablation failure when the extension is removed;
- no unresolved observational ambiguity in scored cases;
- full provenance and immutable negative results.

## Claim boundary

One blind v8 win would be the first credible autonomous language-growth candidate, not yet a world-level breakthrough. Transfer to a second untouched family and independent reproduction remain necessary.
