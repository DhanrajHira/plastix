# Define a NetworkTraits struct

## When to use

You need to plug policies into `plastix::Network`. This is the integration point — every algorithm in Plastix starts by declaring its traits.

## Steps

1. Pick (or define) a `GlobalState`. If you have no per-step scalars to share between policies, use `plastix::EmptyGlobalState` and skip this step. Otherwise, declare a plain struct with default member initializers:

   ```cpp
   struct MyGlobals {
     float LearningRate = 0.01f;
     float V = 0.0f;
     float VOld = 0.0f;
   };
   ```

   Hyperparameters live here as default values. The `Network` owns this struct by value and exposes no setter, so defaults are the only way to configure them.

2. Inherit from `plastix::DefaultNetworkTraits<GlobalState>`:

   ```cpp
   struct MyTraits : plastix::DefaultNetworkTraits<MyGlobals> {
     using ForwardPass = MyForwardPass;
     using BackwardPass = MyBackwardPass;
     using Loss        = plastix::MSELoss;
     using UpdateConn  = MyUpdateConn;
     // ExtraUnitFields, ExtraConnFields, Neighbourhood, Model can also be overridden.
   };
   ```

   Leave any policy you don't need untouched; it inherits the `NoX` sentinel from `DefaultNetworkTraits` and the matching `Do...()` loop compiles out entirely.

3. Validate the traits at the point of definition:

   ```cpp
   static_assert(plastix::NetworkTraits<MyTraits>);
   ```

   This check fires the individual policy concepts, which produce far more targeted diagnostics than instantiating `Network<MyTraits>` and waiting for the template expansion to fail.

4. Alias the network type for readability:

   ```cpp
   using MyNet = plastix::Network<MyTraits>;
   ```

## Policy overrides — quick reference

| Member | Concept | Default | Notes |
|---|---|---|---|
| `GlobalState` | — | `EmptyGlobalState` | Plain struct, owned by value. |
| `ForwardPass` | `PassPolicy` | `DefaultForwardPass` (weighted sum) | Map/Combine/Apply, `Accumulator` type. |
| `BackwardPass` | `PassPolicy` | `NoBackwardPass` | `Self` is the source; `Other` is the destination. |
| `Loss` | `LossPolicy` | `NoLoss` | Also available: `MSELoss`, `RMSLoss`, `SoftmaxCrossEntropyLoss`. Stages `dL/dA` into `BackwardAccTag`. |
| `UpdateUnit` | `UpdateUnitPolicy` | `NoUpdateUnit` | Per-unit post-step hook. |
| `UpdateConn` | `UpdateConnPolicy` | `NoUpdateConn` | Two phases: Incoming then Outgoing. Use the split for two-phase reductions (SwiftTD reference). |
| `PruneUnit` | `PruneUnitPolicy` | `NoPruneUnit` | Sets `PrunedTag`; connections touching pruned units die automatically. |
| `PruneConn` | `PruneConnPolicy` | `NoPruneConn` | Sets `DeadTag`. |
| `AddUnit` | `AddUnitPolicy` | `NoAddUnit` | Return `optional<int16_t>` offset from parent's level, clamped to `[1, MaxLevels-1]`. |
| `AddConn` | `AddConnPolicy` | `NoAddConn` | Queried over unordered level pairs within `Neighbourhood`. |
| `ResetGlobal` | `ResetGlobalStatePolicy` | `NoResetGlobalState` | Fires at the end of each `DoStep`. |
| `ExtraUnitFields` | — | `UnitFieldList<>` | See `agents/add-soa-field.md`. |
| `ExtraConnFields` | — | `ConnFieldList<SOAField<WeightTag, float>>` | Overriding replaces, not extends. |
| `Neighbourhood` | — | `1` | Max inter-level distance for `AddConn` queries. |
| `Model` | — | `Propagation::Topological` | Set to `Propagation::Pipeline` for one-layer-per-step semantics. |

## Pitfalls

- Replacing `ExtraConnFields` drops `WeightTag`. If you still want a weight, include `plastix::alloc::SOAField<plastix::WeightTag, float>` as the first entry in your `ConnFieldList`.
- `static_assert(plastix::NetworkTraits<T>)` only checks signatures, not semantics. A `BackwardPass` whose `Map` flips the source and destination is still a valid `PassPolicy` but will compute the wrong gradients.
- `GlobalState` is copied, not referenced, into the `Network`. Modifying fields from outside after construction has no effect. Anything that needs to be configurable per-step should flow through `DoStep`'s `Inputs` or `Targets` spans.
- Concepts in diagnostics can be verbose. When `NetworkTraits<T>` fails, check the individual sub-concepts (`PassPolicy<T::ForwardPass, ...>`, etc.) first — those errors are much narrower.

## Verify

- `static_assert(plastix::NetworkTraits<MyTraits>);` compiles.
- `plastix::Network<MyTraits> Net(InputDim, FullyConnected<>{OutputDim});` compiles and the default constructor does not trigger any runtime error.
- A single `Net.DoStep(Inputs);` runs without touching any policy you did not intend to activate (confirm by temporarily putting a `static_assert(false)` inside any `NoX`-replaced policy and seeing that it does not fire).
