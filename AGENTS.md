# AGENTS.md

Guidance for AI coding agents working in this repository. Keep this file current when the architecture, build commands, or conventions change.

## What this project is

Plastix is a C++20 neural-network library that treats a network as a policy-driven simulation over pluggable Structure-of-Arrays (SOA) allocators. Instead of hard-coding a forward pass, loss, or learning rule, the library exposes a `Network<Traits>` template parameterized by a `NetworkTraits` type. The traits are a bundle of policy structs (forward pass, backward pass, loss, unit update, connection update, prune, add-unit, add-connection, reset-global) plus lists of extra per-unit and per-connection fields. The framework's `DoStep` orchestrates these policies in a fixed order, and `if constexpr` compiles out any policy that is left at its `NoX` sentinel default.

The result: algorithms as different as vanilla MLP backprop (`mlp-xor`), linear regression (`linear-regression`), TD(λ) with meta-learned step sizes (`swifttd`), streaming incremental Predictive Coding (`ipc-linear`, `ipc-multilayer`), and a dynamically growing imprinting learner all compile to tight loops over the same core `DoStep` scaffolding.

## Repository layout

```
paper/                     LaTeX sources for the Plastix paper (see "Paper" below)
include/plastix/           Public headers — the entire framework lives here
  alloc.hpp                SOAAllocator (mmap-backed, tag-indexed fields, Gather/permute)
  conn.hpp                 Connection allocator + core tags (FromId/ToId/Dead/SrcLevel/Weight)
  unit_state.hpp           Unit allocator + core tags (Activation/ForwardAcc/BackwardAcc/Pruned/Level)
  traits.hpp               Policy concepts, default/noop policies, built-in losses, DefaultNetworkTraits
  layers.hpp               LayerBuilder concept, FullyConnected, RandomUniformWeight, UnitRange
  plastix.hpp              Network<Traits> class (DoForwardPass/DoBackwardPass/DoStep, topology sort)
  soa.hpp                  X-macro helpers for declaring custom SOA types in user code
src/plastix.cpp            Sole .cpp file — just the version() symbol
tests/                     GoogleTest (fetched via FetchContent). Split by area:
  test_plastix.cpp         Topological-mode network behavior
  test_plastix_pipeline.cpp Pipeline-mode network behavior
  test_alloc.cpp           SOAAllocator unit tests
  test_soa.cpp             soa.hpp X-macro tests
benchmarks/                Google Benchmark stubs (FetchContent)
examples/                  Runnable algorithm demos, one subdir each (see below)
agents/                    Skills/recipes for AI agents (see agents/README.md)
.clang-format              LLVM-based style with customizations — obey it
```

## Build and test

CMake ≥ 3.20, C++20. The tree uses `FetchContent` for GoogleTest 1.15.2 and Google Benchmark 1.9.1, so the first configure downloads them.

```bash
cmake -S . -B build                    # configure (all options ON by default)
cmake --build build -j                 # build library, tests, benchmarks, examples
ctest --test-dir build --output-on-failure   # run tests
build/tests/plastix_tests              # or run the test binary directly
build/benchmarks/bench_plastix         # run benchmarks
build/examples/mlp-xor/mlp_xor         # run an example
```

CMake options (all default ON except CUDA):

- `PLASTIX_BUILD_TESTS`
- `PLASTIX_BUILD_EXAMPLES`
- `PLASTIX_BUILD_BENCHMARKS`
- `PLASTIX_ENABLE_CUDA` — wires in `src/kernels/*.cu` if present and defines `PLASTIX_HAS_CUDA`. No kernels currently ship in the tree.

The library target is named `plastix`. Link examples and tests with `target_link_libraries(<target> PRIVATE plastix)`.

## Core architecture

### SOA allocator (`alloc.hpp`)

`alloc::SOAAllocator<Entity, SOAField<Tag1, T1>, SOAField<Tag2, T2>, ...>` is the spine of the library. Each field is a separately-mmapped parallel array. `Allocate()` is atomic (bumps `Count`), and `GetField<Tag>(alloc, id)` / `GetField<Tag>(alloc, id)` / the free function `GetField<Tag>(alloc, id)` read/write a specific field. The allocator also holds a mirrored "back" set of arrays and a `PermScratch` array: `Gather(N)` applies a permutation in `PermScratch` from the primary arrays into the back arrays and swaps, which the topology sort uses to reorder connections by source level in-place without extra passes of allocation.

Key properties worth remembering:

- Capacity is fixed at construction. Over-allocation returns `static_cast<size_t>(-1)`.
- Field storage is mmap'd `MAP_ANONYMOUS | MAP_NORESERVE`; physical pages are only touched on first write. Huge capacities are cheap if sparsely used.
- Fields are placement-new'd (zero-initialized for POD) on `Allocate()`.
- Tags are empty structs used only as type-level keys. Every tag must be declared somewhere in user or library code.

### Unit and connection state

A unit has these core fields (declared in `unit_state.hpp`):

- `ActivationTag` → `float` — the current output of the unit
- `ForwardAccTag` → policy-defined `ForwardPass::Accumulator` — reduction buffer for the forward Map/Combine
- `BackwardAccTag` → policy-defined `BackwardPass::Accumulator` — reduction buffer for the backward pass; also the handoff from `Loss::CalculateLoss` (which stages dL/dActivation here for output units)
- `PrunedTag` → `bool` — set by `PruneUnit`, read by `PruneConnections`
- `LevelTag` → `uint16_t` — topological depth from input, used by Topological propagation

A connection has these core fields (declared in `conn.hpp`):

- `FromIdTag` / `ToIdTag` → `uint32_t` — source and destination unit ids
- `DeadTag` → `bool` — tombstone; dead connections are skipped in every loop but not compacted
- `SrcLevelTag` → `uint16_t` — maintained by the topology sort; enables bucket-sort of connections by source level so the topological forward pass walks one level at a time
- `WeightTag` → `float` — the default `ExtraConnFields` list in `DefaultNetworkTraits` adds this, but it is a user-level convention, not a framework invariant. The noop `NoConnInit` leaves it zero-initialized.

Extra fields are added via `ExtraUnitFields = UnitFieldList<SOAField<Tag, T>, ...>` / `ExtraConnFields = ConnFieldList<...>` in the traits. See the SwiftTD example for an 11-field connection and the imprinting-learner example for heterogeneous per-unit fields (enums, flags, bitmask pulses, etc.).

### Policies and concepts (`traits.hpp`)

Policies are stateless structs with static methods. Each policy has a matching concept and a `NoX` sentinel type. The framework checks `if constexpr (std::is_same_v<typename Traits::X, NoX>)` and compiles out the entire `DoX()` loop body when the sentinel is active, so there is no cost to leaving policies at their defaults.

The pass policies (`ForwardPass`, `BackwardPass`) expose:

```cpp
using Accumulator = ...;                         // must be default_initializable
static Accumulator Map(U&, size_t Self, size_t Other, C&, size_t ConnId, G&);
static Accumulator Combine(Accumulator, Accumulator);
static void        Apply(U&, size_t Id, G&, Accumulator);
```

For the forward pass, `Self` is the destination (ToId) and `Other` is the source (FromId). For the backward pass, they are flipped: `Self` is the source and `Other` is the destination — gradients flow backward, so `Apply` runs on source units and reads `Map` results that came from destination-side state. The framework clears the accumulator to `Accumulator{}` after `Apply` in both directions.

Other policy categories:

- `Loss::CalculateLoss(U&, UnitRange Outputs, span<const float> Targets, G&)` — runs after the forward pass and before the backward pass; the built-in `MSELoss`, `RMSLoss`, and `SoftmaxCrossEntropyLoss` stage `dL/dActivation` into `BackwardAccTag` on each output unit.
- `UpdateUnit::Update(U&, size_t Id, G&)` — per-unit post-step hook.
- `UpdateConn::UpdateIncomingConnection(...)` and `UpdateOutgoingConnection(...)` — the framework calls both for every live connection, in that order. The two-phase split lets algorithms like SwiftTD compute reductions in phase 1 and consume them in phase 2 of the same step.
- `PruneUnit::ShouldPrune` → `bool`, sets `PrunedTag`. `PruneConn::ShouldPrune` → `bool`, sets `DeadTag`. Connections to/from a pruned unit are automatically killed.
- `AddUnit::AddUnit` → `optional<int16_t>`; return value is a level offset relative to the parent unit, clamped to `[1, MaxLevels-1]`. `InitUnit` runs on the newly allocated id. Capacity is the unit allocator's capacity (currently hardcoded to 4096 in `plastix.hpp`); exhaustion is silent.
- `AddConn::ShouldAddIncomingConnection(U, Self, Candidate, G)` / `ShouldAddOutgoingConnection` are queried over every unordered level pair within `Neighbourhood` levels of each other (see `Traits::Neighbourhood`). Approved proposals are deduplicated, then `InitConnection` runs on each committed connection. The window is rolled level-by-level so cost scales with neighborhood × units-per-level, not units².
- `ResetGlobal::Reset(G&)` — fires at the end of every `DoStep`.
- `GlobalState` — a plain user struct (default: `EmptyGlobalState`). The network owns it by value; policies receive it by reference. There is no public setter — hyperparameters must be `constexpr` in a namespace or default member initializers on the global state struct.

### Propagation models (`Traits::Model`)

Both modes run through the same `DoStep` ordering:
`DoForwardPass → DoCalculateLoss → DoBackwardPass → DoUpdateUnitState → DoUpdateConnectionState → DoPruneUnits → DoPruneConnections → DoAddUnits → DoAddConnections → (resort if structural change) → DoResetGlobalState`.

They differ only in how `DoForwardPass` and `DoBackwardPass` iterate connections:

- `Propagation::Topological` (default) — connections are bucket-sorted by `SrcLevelTag`. The forward pass walks one level at a time: accumulate across all connections in level L, then `Apply` on all units whose `LevelTag == L`, then move to L+1. A signal crosses the entire network in a single `DoStep`. The level computation (Kahn's algorithm, level-parallel) runs in `RecomputeLevels` and is re-triggered automatically whenever `DoAddConnections` commits new edges (`NeedsResort` flag). Inputs are always at level 0.
- `Propagation::Pipeline` — no sort; all live connections are accumulated in one sweep into each destination's `ForwardAcc`, then every non-input unit's `Apply` runs once. A signal advances exactly one layer per `DoStep`. Used by `pipeline-fcc`, `imprinting-learner`, and `ipc-multilayer`.

The topology sort (`SortConnectionsByLevel` in `plastix.hpp`) uses the SOA allocator's `PermScratch` + `Gather` machinery for an in-place reorder of every connection field. `MaxLevels = 1024` caps the depth.

### Layers and construction

`Network(InputDim, Layers...)` allocates input units, then folds over `Layers...`, each of which is a `LayerBuilder`: a callable of signature `UnitRange(UnitAlloc&, ConnAlloc&, UnitRange PrevLayer)`. The built-in `FullyConnected<ConnInit, UnitInit>` allocates its units at level `prev_level + 1`, connects every source-destination pair in source-id order, and returns the new range. For algorithms that need more control, supply your own builder (see `examples/manual-fcc`).

Constructors:

- `Network(InputDim, Builders...)` — uses `NoUnitInit` for input units.
- `Network(InputDim, InputInit, Builders...)` — `InputInit` is `void(UnitAlloc&, size_t)` and runs on each input unit id after allocation.
- `Network(InputDim, OutputDim = 1)` — shorthand for `FullyConnected<>{OutputDim}` with default-initialized connections.

### Running a step

```cpp
Net.DoStep(Inputs, Targets);   // orchestrates the full lifecycle
Net.DoForwardPass(Inputs);     // manual granular control also supported
auto output = Net.GetOutput(); // span<const float> over ActivationTag at OutputRange
```

## Examples — what each one is for

| Dir | What it demonstrates |
|---|---|
| `fcc-nn` | Minimal topological forward-only network with custom sigmoid activation. Smoke test for the default traits. |
| `manual-fcc` | Hand-written `LayerBuilder` instead of `FullyConnected`, showing the raw allocator API. |
| `pipeline-fcc` | `Propagation::Pipeline` mode; one layer of propagation per `DoStep`. |
| `linear-regression` | `MSELoss` + a single-layer gradient-descent `UpdateConn` policy. The canonical supervised-learning shape. |
| `mlp-xor` | Two-layer sigmoid MLP with manual backprop via a custom `SigmoidBackwardPass` that stages `dL/dz` in an extra unit field `GradPreActTag`. |
| `swifttd` | TD(λ) with per-feature meta-learned step sizes (IDBD/Auto-step). The `UpdateConn` policy is deliberately split across `UpdateIncomingConnection` (phase 1: weight/beta/h + reductions) and `UpdateOutgoingConnection` (phase 2: consume reductions, refresh traces). Reference use case for the two-phase hook. |
| `imprinting-learner` | Dynamic unit and connection generation via `AddUnit` + `AddConn`, unit kinds (pattern / memory / output) with per-kind forward logic, and `Propagation::Pipeline`. Non-trivial example of growing a topology online. |
| `ipc-linear` | Streaming incremental Predictive Coding collapsed to one layer (equivalent to online SGD on linear regression). Minimal iPC reference. |
| `ipc-multilayer` | Full Streaming iPC with persistent hidden value nodes, bottom-up + top-down corrections, Hebbian weight updates. |

## Conventions

### Code style

- `.clang-format` is authoritative. Run it before committing. The style is LLVM-derived.
- Naming: `PascalCase` for types *and* local variables; `UPPER_CASE` only for `#define`/macros; types and concepts both `PascalCase`. Field tags are empty structs suffixed with `Tag` (or a bare domain name when the file already makes the intent clear, e.g. `Pulse`, `Delay` in the imprinting example). Match existing style rather than standard C++ `snake_case`.
- Include paths use the `plastix/` prefix from outside the library (`#include <plastix/plastix.hpp>`). Headers inside `include/plastix/` include siblings with quotes and the same prefix (`#include "plastix/conn.hpp"`).
- Prefer the free-function `GetField<Tag>(alloc, id)` over `alloc.template Get<Tag>(id)` in new code — it avoids the `.template` disambiguator and const-propagates through `auto&`.
- Built-in accessors: `GetActivation`, `GetForwardAcc`, `GetBackwardAcc`, `GetLevel`, `GetWeight` for the most common fields.
- Comments: keep them focused on *why*, especially in policy implementations. The algorithms are subtle — a sentence about what invariants a phase maintains is worth more than a paragraph restating the code.

### Adding policies and fields

- Pick empty-struct tag types, then list the fields in `ExtraUnitFields` / `ExtraConnFields` inside your traits. The allocator's capacity is fixed and memory is lazy — adding fields is cheap.
- Inherit your traits from `plastix::DefaultNetworkTraits<GlobalState>` and override only the policies you care about. Leaving a policy at its `NoX` default compiles the corresponding `DoX()` body out entirely.
- `static_assert(plastix::NetworkTraits<MyTraits>);` at the point of definition catches policy-signature mistakes early, while the concept's error messages are still readable.
- Every policy receives the unit allocator, connection allocator, and global state with `auto&` so the exact allocator types (which depend on accumulator types and extra-field lists) are inferred at instantiation. Don't hardcode allocator typedefs in policy bodies.

### Testing

- Tests live next to what they test; each subfile (`test_alloc.cpp`, `test_soa.cpp`, `test_plastix.cpp`, `test_plastix_pipeline.cpp`) is compiled into the single `plastix_tests` binary declared in `tests/CMakeLists.txt`. Add new `test_*.cpp` files to that list.
- Prefer narrow traits in tests — define a one-off `struct X : DefaultNetworkTraits<> { ... };` inside a TEST's anonymous namespace rather than importing example traits. Example traits change for algorithmic reasons, and tests shouldn't churn with them.
- Pipeline-mode and Topological-mode tests are kept in separate files because they exercise substantially different code paths in `DoForwardPass`/`DoBackwardPass`. Follow that split when adding new traits-level tests.

### When in doubt

Read the closest example first. Every non-trivial pattern in Plastix is exercised by at least one example, and most have explanatory block comments at the top of the file. `examples/swifttd/swifttd.cpp` and `examples/imprinting-learner/imprinting_learner.cpp` are especially dense with cross-cutting uses of the policy system.

## Paper

The `paper/` directory contains the LaTeX source for *Plastix: GPU Acceleration for Sparse Networks With Dynamic Structure* (Hira, Meyer, Chilibeck — NeurIPS 2026 submission template). The paper is the authoritative prose description of the framework's motivation and design; when the code and the paper disagree, prefer the code but flag the discrepancy. Files worth knowing:

- `main.tex` — top-level paper; abstract, introduction, related work, high-level framework description. Start here to understand *why* Plastix exists and how the authors frame it (neuron-first, GPU-friendly, structural adaptation at step frequency).
- `interface_overview.tex` — user-facing transition-function interface. Defines the stage vocabulary (Forward, Backward, UpdateUnit, UpdateConn, Add/Prune for both) and their inputs/outputs using the $s(u)$, $s(u,v)$ notation. Maps directly onto the policy concepts in `traits.hpp`.
- `implementation_details.tex` — the narrow, polished implementation story: organizing principle (no global interactions in a pass), Map/Combine/Apply primitives, the GPU execution model for connection sweeps, why the interface collapses to efficient kernels.
- `implementation_elaboration.tex` — extended implementation discussion, including memory layout motivation, pipelined vs. layered computation, duplicate weight buffers, and other design-choice rationale outlined in `main.tex`'s outline.
- `examples.tex` — prose walkthroughs of the headline examples (imprinting learner, Streaming iPC). Keep these in sync with the code in `examples/` when materially changing an algorithm.
- `future_work.tex`, `preamble.tex`, `neurips_2026.sty`, `sample.bib`, `tikz/` — supporting boilerplate and figures.

Useful framing to lift from the paper when explaining the code:

- *Neuron-first / locality constraint.* Every per-step stage is either per-unit or per-connection; no stage peeks at global state except through the explicit `GlobalState` handle. The restriction is what makes the framework GPU-friendly and keeps the interface narrow.
- *Map / Combine / Apply* corresponds one-to-one with the `PassPolicy` concept — when writing a new forward or backward pass, think of it as declaring these three primitives, with `Combine` required to be associative.
- *Stages compile out.* The paper repeatedly emphasizes that stages left unspecified are skipped with no overhead. The code implements this via `if constexpr` checks against `NoX` sentinels in `plastix.hpp`.
- *Pipelined vs. layered.* `Propagation::Pipeline` corresponds to the paper's pipelined model; `Propagation::Topological` is the layered model. The paper's implementation section motivates when each is preferable.

If you change the framework's stage ordering, add a new policy category, rename a concept, or change the propagation semantics, update both `implementation_details.tex` / `interface_overview.tex` and this `AGENTS.md`. A single source of truth is less important than keeping the paper's definitions matching what the code does.

## Agent skills

See `agents/` for reusable recipes covering common tasks in this repo (adding an example, extending SOA fields, defining custom traits, running the tooling). Those files are the place to add more procedural knowledge as patterns stabilize.
