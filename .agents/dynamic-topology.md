# Implement dynamic-topology policies (Add/Prune)

## When to use

Your algorithm grows or shrinks the network online — generating hidden units, proposing new edges, pruning dead weights. This is the feature that differentiates Plastix from static-architecture frameworks, and the imprinting-learner example is the canonical reference.

## Ordering inside DoStep

Every `DoStep` runs the structure-changing stages in this fixed order:

1. `DoPruneUnits` — sets `PrunedTag` on units your `PruneUnit::ShouldPrune` says to kill.
2. `DoPruneConnections` — sets `DeadTag` on connections whose `PruneConn::ShouldPrune` fires, and also on every connection that touches a unit marked with `PrunedTag`.
3. `DoAddUnits` — allocates new units with a level offset returned by `AddUnit::AddUnit`.
4. `DoAddConnections` — queries `AddConn::ShouldAddIncomingConnection` / `ShouldAddOutgoingConnection` over every unordered level pair within `Traits::Neighbourhood` of each other, deduplicates proposals, and commits the unique ones with `InitConnection`.
5. If `Propagation::Topological` and the structure changed, the connection list is re-sorted by level (`NeedsResort`).
6. `DoResetGlobalState`.

## AddUnit

```cpp
struct MyAddUnit {
  static std::optional<int16_t> AddUnit(auto &U, size_t ParentId, auto &G) {
    // Return nullopt to skip. Return an offset to spawn a child.
    // Offset is relative to Level(ParentId), clamped to [1, MaxLevels-1].
    return should_grow(U, ParentId, G) ? std::optional<int16_t>(0) : std::nullopt;
  }

  static void InitUnit(auto &U, size_t NewId, size_t ParentId, auto &G) {
    // Assign tags / activation / kind to the newly allocated unit.
  }
};
```

The framework iterates over the *current* set of units; new units allocated during `DoAddUnits` are not iterated again in the same step. Parent-unit state is read through `U` at `ParentId`; `NewId` is the freshly allocated unit. `LevelTag` is pre-populated for you before `InitUnit` runs.

## AddConn

```cpp
struct MyAddConn {
  static bool ShouldAddIncomingConnection(auto &U, size_t Self, size_t Candidate, auto &G) {
    return /* Should a connection from Candidate to Self be proposed? */;
  }
  static bool ShouldAddOutgoingConnection(auto &U, size_t Self, size_t Candidate, auto &G) {
    return /* Should a connection from Self to Candidate be proposed? */;
  }
  static void InitConnection(auto &U, size_t From, size_t To,
                             auto &C, size_t ConnId, auto &G) {
    // Fill in weight and any other per-connection fields.
    plastix::GetWeight(C, ConnId) = 0.0f;
  }
};
```

`Neighbourhood` controls the sliding level window for proposals. With `Neighbourhood = 1` every pair within adjacent levels (including same-level pairs) is queried once. `Self` and `Candidate` are both unit ids within the window; the Incoming hook is asked "should a `Candidate -> Self` edge be added?" and the Outgoing hook is asked "should a `Self -> Candidate` edge be added?". For cross-level pairs, both perspectives are queried, so a single commit decision can come from either side — duplicates are resolved by the phase-3 dedupe sort.

## Prune policies

```cpp
struct MyPruneUnit { static bool ShouldPrune(auto &U, size_t Id, auto &G); };
struct MyPruneConn { static bool ShouldPrune(auto &U, size_t Dst, size_t Src,
                                             auto &C, size_t ConnId, auto &G); };
```

A pruned unit does not free its slot — the allocator still counts it toward capacity. Connections touching a pruned unit are tombstoned via `DeadTag`. Dead connections are skipped in every loop but not compacted; capacity pressure accumulates across steps.

## Pitfalls

- Capacity exhaustion is silent. `UnitAlloc.Allocate()` returns `static_cast<size_t>(-1)` past capacity (currently hardcoded to `4096` units and `16384` connections in `plastix.hpp`). Check for this in `AddUnit::InitUnit` or raise capacity when the algorithm expects to grow large.
- `DoAddConnections` uses `ProposalAlloc`'s fixed capacity (equal to `ConnAlloc.GetCapacity()`). Oversubscribing the proposal buffer silently drops excess proposals. If your neighborhood × unit count is large, raise capacity in the `Network` constructor.
- The proposal phase queries every unordered pair in the window — cost scales with `Neighbourhood * (units_per_level)^2`. Keep `Neighbourhood = 1` unless you genuinely need long-range proposals, and gate expensive per-pair predicates on cheap rejections first.
- `Propagation::Topological` requires a resort after every structural change. The framework sets `NeedsResort = true` automatically, but a policy that mutates `SrcLevelTag` directly can desynchronize the sort. Don't write to `SrcLevelTag` from user code — let `SortConnectionsByLevel` compute it.
- The proposal dedup is by packed `(From, To)` bits only. If two policies disagree about whether to commit the same edge, whichever got sorted first wins; `InitConnection` runs exactly once per unique edge. Don't rely on `InitConnection` side-effects to trigger multiple times per pair.

## Verify

- For static-topology algorithms: `NoAddUnit` / `NoAddConn` / `NoPruneUnit` / `NoPruneConn` at defaults, confirm unit and connection counts stay constant across `DoStep` by printing `Net.GetUnitAlloc().Size()` and `Net.GetConnAlloc().Size()`.
- For growing algorithms: assert that the counts increase at the expected cadence, then level off if your algorithm has a natural cap. The imprinting-learner `std::cout` traces ("Init Units", "Adding connection to output from ...") are an example of lightweight structural logging.
- Post-prune sanity: iterate `ConnAlloc` and assert that every connection with `GetField<DeadTag>` = true also has both endpoints unreachable — no live forward-pass contribution from it.
