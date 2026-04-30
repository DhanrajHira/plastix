# Add an SOA field to a unit or connection

## When to use

Your algorithm needs state that persists across steps — per-unit scalars, per-connection tensors, per-edge traces, per-unit enums, bitmasks, etc. This is the standard way to extend network state without touching the framework.

## Steps

1. Declare one empty tag struct per field you're adding. Convention: PascalCase, suffix `Tag` for scalars that could collide with a natural name; bare domain nouns (`Pulse`, `Delay`) are also fine when context makes it unambiguous.

   ```cpp
   struct GradPreActTag {};
   struct MomentumTag {};
   struct TenureTag {};
   ```

2. List the new fields in `ExtraUnitFields` or `ExtraConnFields` inside your traits, pairing each tag with the storage type:

   ```cpp
   struct MyTraits : plastix::DefaultNetworkTraits<> {
     using ExtraUnitFields = plastix::UnitFieldList<
       plastix::alloc::SOAField<GradPreActTag, float>,
       plastix::alloc::SOAField<TenureTag,     uint8_t>>;

     using ExtraConnFields = plastix::ConnFieldList<
       plastix::alloc::SOAField<plastix::WeightTag, float>,   // keep the default weight
       plastix::alloc::SOAField<MomentumTag, float>>;
   };
   ```

   `UnitFieldList<...>` appends to the built-in unit fields (`Activation`, `ForwardAcc`, `BackwardAcc`, `Pruned`, `Level`). `ConnFieldList<...>` **replaces** the built-in extras, so include `WeightTag` yourself if you still want a weight. Core connection fields (`FromId`, `ToId`, `Dead`, `SrcLevel`) are always present regardless.

3. Read and write the field inside policies via the free-function accessor:

   ```cpp
   float &g = plastix::GetField<GradPreActTag>(U, UnitId);
   plastix::GetField<MomentumTag>(C, ConnId) = 0.9f * plastix::GetField<MomentumTag>(C, ConnId) + grad;
   ```

   Prefer `GetField<Tag>(alloc, id)` over `alloc.template Get<Tag>(id)` — it drops the `.template` disambiguator and const-propagates through `auto&` policy parameters.

4. Initialize non-default values as needed:

   - Connections: pass a `ConnInit` functor into `FullyConnected<ConnInit>{}`. The functor runs on each newly allocated connection id (see `RandomUniformWeight` and `SwiftTDConnInit` for reference shapes).
   - Units: pass a `UnitInit` functor as the second template parameter of `FullyConnected<ConnInit, UnitInit>` for hidden-layer units, or an `InputInit` lambda as the second constructor argument to `Network` for input units (see `imprinting-learner`).

## Pitfalls

- Field capacity is fixed at `Network` construction. The unit allocator hardcodes `4096` capacity and the connection allocator `4096 * 4 = 16384` (see `plastix.hpp`). Algorithms that grow far past those limits need that constructor reworked — until then, `Allocate()` silently returns `-1` on exhaustion.
- Storage is mmap'd `MAP_ANONYMOUS | MAP_NORESERVE`, so adding a wide new field (e.g. a 1 KB per-unit struct) does not spend memory unless you touch every page. Large field types are cheap if sparsely written, but the arena size is `Capacity * max(sizeof(Field))` — a single giant field inflates every field's arena.
- Overriding `ExtraConnFields` without re-including `SOAField<WeightTag, float>` silently drops the default weight. Either inherit from `DefaultNetworkTraits` and don't touch the list if you want the default, or redeclare the weight field explicitly.
- Tag types must be empty structs (`static_assert` inside `SOAField` enforces this). You cannot use a non-empty struct or a scalar type as a tag — this is a compile-time error with a helpful message.
- The tag index is computed by position in the field list, so reordering fields does not break code (accessors are by tag, not by index) but the mmap layout changes. Don't try to persist arenas to disk and expect them to be layout-compatible across builds that reorder fields.

## Verify

- `static_assert(plastix::NetworkTraits<MyTraits>);` still compiles.
- `plastix::GetField<NewTag>(UnitAlloc, 0)` returns the zero-initialized value before any policy touches it (placement-new zero-inits POD types).
- Round-trip test: construct a `Network<MyTraits>`, run a few `DoStep` calls, assert that `GetField<NewTag>` returns the values your policies wrote.
