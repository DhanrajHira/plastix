# Plastix

Plastix is a C++20 library for dynamic neural networks with structural sparsity, targeting continual learning algorithms where network topology changes after every iteration (adding/removing units and connections).

## Project goals

- **GPU acceleration**: CPU-first but all data structures and interfaces are designed to be GPU-portable (mmap arenas, SOA layout, no STL containers in hot paths).
- **Dynamic structure**: Networks can add/remove units and connections every iteration. Tombstone-based deletion for connections, policy-driven unit addition.
- **Extensibility**: Users define their own unit state and connection state (via SOA field macros) and customize all network phases through a traits-based policy system.
- **Constrained interfaces for performance**: Patterns like MapReduce for state updates are intentional — they constrain the user interface to enable efficient parallel execution (reductions, scans) on both CPU and GPU.

## Architecture

### SOA allocator (`include/plastix/alloc.hpp`)
- `SOAAllocator<EntityType, SOAField<Tag, Type>...>` — mmap-backed arena with separate arrays per field.
- **Double-buffered fields**: Each field has a primary and a back (secondary) mmap region, both allocated at construction. `Get<Tag>()` reads from primary. Enables zero-copy reorder via `Gather`.
- Placement-new default construction on `Allocate()` (initializes both primary and back).
- All allocations return `AllocId<T>` (a `size_t`). Access via `Alloc.Get<Tag>(Id)`.
- `GetArrayFor<Tag>()` — returns a raw pointer to the primary array for a given field tag. Useful for scratch allocators where algorithms need direct pointer access (e.g., `memset`, `memcpy`).
- `PermutationScratch()` — returns a reusable `size_t*` array (mmap'd at construction, capacity elements). Callers write gather indices into it before calling `Gather`.
- `Gather(N)` — for each field, gathers from primary into back using `PermScratch` (`Back[i] = Primary[PermScratch[i]]`), then swaps primary/back pointers. No per-call mmap/munmap.

### SOA macros (`include/plastix/soa.hpp`)
- X-macro system using `.inc` files with `SOA_TYPE(Name)`, `FIELD(Name, Type)`, `SOA_END()`.
- Three passes via mode flags: `PLASTIX_SOA_MODE_TAGS`, `PLASTIX_SOA_MODE_ALLOC`, `PLASTIX_SOA_MODE_HANDLE`.
- `soa.hpp` has no include guards — designed for multiple inclusion.
- Used for `conn_state.inc` (fixed field types). Unit state uses direct tag/template definitions in `unit_state.hpp` because accumulator field types are parameterized by policy types.

### Traits-based Network (`include/plastix/plastix.hpp`)
- `Network<Traits>` takes a single `NetworkTraits` concept-constrained type.
- Traits bundles: `UnitAllocator`, `ConnAllocator`, `GlobalState`, and 7 policy types.
- Users inherit from `DefaultNetworkTraits<UnitAlloc, ConnAlloc, Global>` and override individual policies.
- Policy concepts: `PassPolicy`, `UpdateUnitPolicy`, `UpdateConnPolicy`, `PruneUnitPolicy`, `PruneConnPolicy`, `AddUnitPolicy`.
- Policy methods receive `(Allocator &, AllocId, GlobalState &, ...)` — explicit, no hidden state.
- `UpdateConnPolicy` and `PruneConnPolicy` receive `(UnitAlloc, DstId, SrcId, ConnAlloc, ConnId, Global)` — a single `ConnId` identifies each connection.
- Default policies use `auto` parameters to work with any allocator type.
- **Noop sentinel types**: `NoBackwardPass`, `NoUpdateUnit`, `NoUpdateConn`, `NoPruneUnit`, `NoPruneConn`, `NoAddUnit`. These satisfy their concepts but `DoX()` methods compile out the entire body via `if constexpr` when detected. `DefaultNetworkTraits` uses these for all policies except `ForwardPass`.
- **`DefaultNetworkTraits<ConnAlloc, Global>`** — only takes `ConnAllocator` and optional `GlobalState`. `UnitAllocator` is derived automatically by `Network` from policy accumulator types via `MakeUnitAllocator`.

### Generalized pass policies
- **PassPolicy** — defines `Accumulator` type (must be default-initializable). `Map(UnitAlloc, UnitIdA, UnitIdB, ConnAlloc, ConnId, Global)` returns `Accumulator`. `Combine(Acc, Acc)` reduces. `Apply(UnitAlloc, Id, Global, Acc)` writes results. Forward pass: `UnitIdA=ToId`, `UnitIdB=FromId`, accumulates into `ForwardAcc`. Backward pass: `UnitIdA=FromId`, `UnitIdB=ToId`, accumulates into `BackwardAcc`. Policies read weight and activation from allocators directly — no explicit float parameters.
- **DefaultForwardPass** — `Accumulator = float`. `Map` reads `WeightTag * ActivationTag(SrcId)`, `Combine` sums, `Apply` writes to `ActivationTag`.

### Update policies
- **UpdateUnitPolicy** — MapReduce with user-defined `Partial` type. `Map(UnitAlloc, DstId, SrcId, Global, Weight)` returns `Partial`, `Combine(A, B)` reduces, `Apply(UnitAlloc, Id, Global, Acc)` writes results. Accumulator stored in `UpdateAcc` SOA field (type must match `Partial`). Two-pass: accumulate across all connections, then apply+reset per unit.
- **UpdateConnPolicy** — per-connection update in two separate loops: all incoming first, then all outgoing.

### Prune policies
- **DoPruneUnits** — iterates units, writes `ShouldPrune` result to `Pruned` SOA field (marking only).
- **DoPruneConnections** — iterates connections: if source or destination unit is pruned, marks connection `Dead = true` (tombstone). Otherwise checks `PruneConnPolicy::ShouldPrune`. Dead connections are skipped in all subsequent iteration.
- Both checks guarded by `if constexpr` — compile out when using noop sentinels.

### Add policies
- **AddUnitPolicy** — `AddUnit(UnitAlloc, UnitId, Global)` is called for each existing unit. Returns a `UnitPosition`. If any coordinate is non-zero, a new unit is allocated at that position. The iteration captures unit count before the loop, so newly added units are not visited in the same call.

### Unit state and position
- `UnitState` has `Activation` (float), `ForwardAcc`, `BackwardAcc`, `UpdateAcc`, `Pruned`, `Position`, and `Level` fields.
- `ForwardAcc`, `BackwardAcc`, `UpdateAcc` are parameterized by policy accumulator types via `MakeUnitAllocator<FwdAcc, BwdAcc, UpdAcc>`. Network derives the unit allocator automatically.
- Single `Activation` field (no double-buffering). Level-based ordering ensures lower levels finalize before upper levels read them.
- `Position` is a `UnitPosition` struct: `_Float16 X, Y, Z` + `uint16_t Pad` (8 bytes). Has `IsZero()` and `explicit operator bool()` methods.
- `Level` (`uint16_t`) — topological level of the unit. Input units are level 0. For all other units, `Level = max(Level of all predecessor units) + 1`. Max 1024 levels (`MaxLevels` constant).
- `FullyConnected` layer builder positions units in a standard neural network layout: X = layer depth (0, 1, 2, ...), Y = centered within layer with unit spacing, Z = 0. Sets `Level` based on predecessor layer.

### Iteration order (`DoStep`)
`DoStep(Inputs)` runs the full pipeline in this fixed order:
1. ForwardPass — 2. BackwardPass — 3. UpdateUnitState — 4. UpdateConnectionState — 5. PruneUnits — 6. PruneConnections — 7. AddUnits — 8. AddConnections — 9. SortConnectionsByLevel (if connections were added)

### Forward and backward passes
- **Forward pass** evaluates level by level. For each target level L (1 to NumLevels): iterates connections in `LevelRanges[L-1]`, accumulating into `ForwardAcc(ToId)` via `Combine(ForwardAcc, Map(...))`, then calls `Apply` on all units at level L and resets `ForwardAcc` to `Acc{}`. Increments `Step`. All layers propagate in a single `DoForwardPass` call.
- **Backward pass** iterates levels in reverse (NumLevels down to 1). For each level L: iterates connections in `LevelRanges[L]` (connections FROM level L carry gradients back), accumulating into `BackwardAcc(FromId)` via `Combine`, then calls `Apply` on all units at level L and resets `BackwardAcc`. Does NOT increment `Step`. The default backward pass is a noop — custom policies implement gradient computation.
- Intermediate scratch data should live in SOA fields, not local containers (GPU portability).

### Connection structure
- Connections are stored as individual SOA entities with `FromId` (uint32_t), `ToId` (uint32_t), `Weight` (float), `Dead` (bool), and `SrcLevel` (uint16_t) fields, defined in `conn_state.inc`.
- `SrcLevel` stores the `Level` of the source unit (`FromId`). Used as the sort key.
- `ConnStateAllocator` is a plain `SOAAllocator` — users add per-connection metadata by adding fields to `conn_state.inc`.
- **Tombstone deletion**: pruned connections have `Dead = true` and are skipped during iteration. No physical removal.
- **Sorted by source level**: Connections are physically sorted in `ConnAlloc` by `SrcLevel` via counting sort (1024-bin histogram). This ensures evaluation-order correctness in the forward pass.
- `LevelRanges` — a `std::array<LevelRange, 1024>` on `Network`, where `LevelRange{Begin, End}` marks the connection index range for each source level. Built as a byproduct of the counting sort prefix sum. Connections within a level range can be evaluated in parallel.
- **Level recomputation** (`RecomputeLevels`): Level-parallel Kahn's algorithm. Builds an ephemeral CSR (outgoing adjacency) from the live connections using `ConnAlloc.PermutationScratch()` for the edge list and a `KahnScratchAllocator` (SOA allocator with 5 `uint32_t` fields: `InDegree`, `OutOffset`, `KahnWritePos`, `Frontier`, `NextFrontier`) for per-unit arrays. The initial frontier is seeded directly from the input range `[0, NumInput)` rather than scanning all units. BFS by level assigns `Level = max(predecessor levels) + 1` in O(V + E). Runs before every sort.
- **Sort trigger**: `SortConnectionsByLevel()` runs after construction and whenever `DoAddConnections` adds new connections (guarded by `NeedsResort` flag, also checked at start of `DoForwardPass`).

## Code conventions

- LLVM naming style: `PascalCase` for types/functions/variables, `ALL_CAPS` for macros.
- No STL containers in allocator hot paths (mmap + placement new directly).
- Macros prefixed with `PLASTIX_` to avoid collisions.
- `.inc` files have no include guards (intentional — multiple inclusion).
- `-Wall -Wextra -Wpedantic` enforced.

## Build

```bash
cmake -B build && cmake --build build
ctest --test-dir build
```
