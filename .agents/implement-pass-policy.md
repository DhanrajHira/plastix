# Implement a forward or backward pass policy

## When to use

Plugging in a new activation, loss-gradient propagator, attention-style aggregation, or anything else expressible as a local per-edge contribution reduced into a per-unit update. This is the most frequently overridden policy category.

## Signature

Both `ForwardPass` and `BackwardPass` satisfy the same `PassPolicy` concept:

```cpp
struct MyPass {
  using Accumulator = /* default-initializable type */;

  static Accumulator Map(auto &U, size_t Self, size_t Other,
                         auto &C, size_t ConnId, auto &G);
  static Accumulator Combine(Accumulator A, Accumulator B);
  static void Apply(auto &U, size_t Id, auto &G, Accumulator Acc);
};
```

`Combine` must be associative (the framework makes no ordering guarantee across connections destined for the same unit). `Apply` receives the reduced accumulator and updates the unit's state; the framework resets the accumulator slot to `Accumulator{}` afterward.

## Orientation rules

This is the single most error-prone detail. The framework treats each connection as a pair `(FromId, ToId)` — from a source unit to a destination unit — but the meaning of `Self` and `Other` in `Map` depends on the direction.

| Pass | `Self` | `Other` | Where `Apply` runs | Accumulator lives on |
|---|---|---|---|---|
| Forward | `ToId` (destination) | `FromId` (source) | each non-input unit, after its level sweep | destination unit (via `ForwardAccTag`) |
| Backward | `FromId` (source) | `ToId` (destination) | each source unit, after its level's gradient sweep | source unit (via `BackwardAccTag`) |

Concretely: in a `ForwardPass::Map`, `U` indexed at `Self` is the unit whose activation you're *computing*. In a `BackwardPass::Map`, `U` indexed at `Self` is the unit whose upstream gradient you're *accumulating* into — the destination's gradient (already in `BackwardAccTag` for output units, put there by `Loss::CalculateLoss`) is read through `Other`.

## Examples to copy from

- `DefaultForwardPass` in `traits.hpp` — the minimal linear combiner (`Weight * Activation`).
- `SigmoidForwardPass` / `SigmoidBackwardPass` in `examples/mlp-xor/mlp_xor.cpp` — a complete matching pair, with the backward pass routing `dL/dz` through a custom `GradPreActTag` unit field because `BackwardAccTag` gets cleared after each level.
- `iPCForwardPass` / `iPCBackwardPass` in `examples/ipc-multilayer/ipc_multilayer.cpp` — uses a non-trivial custom `Accumulator` struct and propagates through persistent value nodes.
- `SwiftTDForward` in `examples/swifttd/swifttd.cpp` — shows the pattern of leaking the forward result into `GlobalState.V` via `Apply`.

## Pitfalls

- Confusing `Self` and `Other`. Write a one-line comment in each policy body naming which id is source and which is destination, especially in backward passes. Silent sign errors in gradients are the number-one cause of "my loss diverges" in Plastix examples.
- Accumulators are cleared after `Apply` — you cannot rely on `BackwardAcc` still holding its value at the next level. If you need the pre-activation gradient to survive across levels, stash it in a dedicated unit field (see `mlp-xor`'s `GradPreActTag`).
- `Combine` is called with arbitrary pairs of accumulators, potentially including partially-reduced intermediate values. It must form a commutative-enough monoid that any grouping yields the same result. Non-associative operations (like max with a tie-breaker) break in the topological loop as soon as the sort order changes.
- `Accumulator` must be default-constructible (concept requirement). Non-POD accumulators work, but the per-unit buffer is not destroyed between steps — the framework just reassigns `Accumulator{}`. Make sure `Accumulator{}` is a valid "zero" for your reduction.
- In `Propagation::Pipeline` mode, the forward pass accumulates all connections in one sweep before any `Apply`, so every destination's accumulator sees the snapshot of source activations at the start of the step. This is why pipeline mode is preferred for predictive-coding-style algorithms that need simultaneous-update semantics.

## Verify

- Numerical spot check: for a tiny network with known weights (e.g. single connection, identity activation), `DoForwardPass` produces the expected scalar at the output.
- Gradient check: compare `DoBackwardPass`'s weight updates against a finite-difference loss derivative on a 2-unit network. `linear-regression` converging to the true weights `[2.0, -1.0, 0.5]` is the canonical regression check for a correct MSE + gradient-descent pair.
- `static_assert(plastix::PassPolicy<MyPass, UnitAllocFor<MyTraits>, ConnAllocFor<MyTraits>, MyGlobals>);` fires early and narrowly if a signature is wrong.
