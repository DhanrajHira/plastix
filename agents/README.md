# Agent skills

Procedural recipes for common tasks in this repository. Each skill is self-contained and targets a single workflow an AI agent is likely to be asked to perform. Keep them short, concrete, and checkable — recipes, not essays.

Before using any of these, read [../AGENTS.md](../AGENTS.md) for the architectural context that makes the recipes make sense.

## Index

- [build-and-test.md](build-and-test.md) — configure, build, run tests/benches/examples; common flag combinations and where CMake caches live.
- [add-example.md](add-example.md) — add a new entry under `examples/`, wire it into CMake, and match the prevailing style.
- [define-traits.md](define-traits.md) — assemble a `NetworkTraits` struct by overriding individual policies on top of `DefaultNetworkTraits<>`.
- [add-soa-field.md](add-soa-field.md) — extend a unit or connection allocator with a new field via `ExtraUnitFields` / `ExtraConnFields`.
- [implement-pass-policy.md](implement-pass-policy.md) — write a custom `ForwardPass` or `BackwardPass` policy (Map / Combine / Apply), including the source/destination orientation rules.
- [dynamic-topology.md](dynamic-topology.md) — implement `AddUnit` / `AddConn` / `PruneUnit` / `PruneConn` policies and understand how the proposal + neighborhood window drives them.
- [paper-sync.md](paper-sync.md) — when a framework change lands, the cross-references between code and the `paper/` LaTeX sources that must stay consistent.

## Adding a new skill

Create a new Markdown file in this directory. Give it a concrete task in the title (imperative verb). Structure:

1. *When to use* — one or two sentences identifying the trigger.
2. *Steps* — an ordered list of concrete edits / commands, with file paths and line-level pointers where useful.
3. *Pitfalls* — known gotchas from prior attempts, especially cases where the C++ template error messages are misleading.
4. *Verify* — the check that tells you the change worked (test that should pass, example that should run, build output to inspect).

Then add it to the index above.
