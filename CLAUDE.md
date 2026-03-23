# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Commands

Uses `just` as a task runner and CMake/Ninja as the build system.

```bash
# Configure and build (tests + benchmarks + examples)
just build

# Full rebuild from scratch
just rebuild

# Run all tests (after building)
cd build && ctest

# Run a single test binary directly
./build/tests/plastix_tests

# Run a specific test by name
./build/tests/plastix_tests --gtest_filter=NetworkTest.SingleLayerPerceptronConnections

# Run benchmarks
./build/benchmarks/plastix_benchmarks

# Run an example
./build/examples/ipc-linear/ipc_linear
```

The build copies `compile_commands.json` to the repo root for clangd.

CUDA support is off by default; enable with `-DPLASTIX_ENABLE_CUDA=ON`.

## Architecture

Plastix is a C++20 header-only neural network framework (only `src/plastix.cpp` contains non-header code — it just returns the version string). Everything lives in `include/plastix/`.

### Core design: policy-based traits

`Network<Traits>` is the central class. `Traits` is a struct satisfying the `NetworkTraits` concept, which bundles six pluggable policies:

| Policy type | What it does |
|---|---|
| `ForwardPass` | `Map(weight, activation)` + `CalculateAndApply(accumulated)` per unit |
| `BackwardPass` | Same interface, runs in reverse topology order |
| `UpdateUnit` | Accumulates per-incoming-connection partial values, then applies |
| `UpdateConn` | Called per connection for incoming and outgoing updates |
| `PruneUnit` | Returns bool whether a unit should be removed |
| `PruneConn` | Returns bool whether a connection should be removed |

`DefaultNetworkTraits<UnitAlloc, ConnAlloc, Global>` provides sensible no-ops for all policies except `ForwardPass` (which defaults to `weight * activation` with identity activation). Extend by inheriting and overriding specific `using` aliases.

### SOA allocators

`alloc::SOAAllocator<Entity, Fields...>` is the memory backbone. Each field is declared as `SOAField<Tag, Type>` where `Tag` is an empty marker struct. Fields are backed by separate `mmap`'d arenas. Access: `alloc.Get<Tag>(id)`.

`alloc::PageAllocator` extends `SOAAllocator` with `CompactPage(pageId, liveMask)` for in-place stream compaction (used by pruning).

### Unit state

`UnitState` is defined via X-macro pattern in `include/plastix/unit_state.inc` + `include/plastix/soa.hpp`. Including `soa.hpp` three times with different `#define PLASTIX_SOA_MODE_*` guards generates: tag structs, a `UnitStateAllocator` typedef, and a `UnitStateHandle` helper. The built-in fields are: `ActivationA`, `ActivationB` (ping-pong buffers), `BackwardAcc`, `UpdateAcc`, `Pruned`.

To extend unit state with custom fields (e.g., for iPC), construct a new `SOAAllocator<UnitState, ...all base fields..., SOAField<CustomTag, T>>` directly — no framework modification needed.

### Connection storage

`ConnPage` holds up to 7 `(src_unit_idx, weight)` pairs pointing to a single destination unit. Pages are allocated via `ConnStateAllocator` (a `PageAllocator`). The forward pass iterates pages sequentially, detecting destination changes to accumulate before applying activation.

### Execution model (pipelined)

`Network::DoStep(inputs)` runs: ForwardPass → BackwardPass → UpdateUnit → UpdateConn → PruneUnits → PruneConnections. Activations double-buffer between `ActivationA` and `ActivationB` each step (ping-pong). This means output from a multi-layer network requires N steps to propagate N layers — see `examples/fcc-nn/fcc_nn.cpp`.

Policies that are `No*` sentinels (e.g., `NoBackwardPass`) compile away their entire loop via `if constexpr`.

### Adding a new learning algorithm

See `examples/ipc-linear/ipc_linear.cpp` as a reference. The pattern is:
1. Define custom SOA field tags and extend `iPCUnitAllocator`
2. Implement static policy structs (`ForwardPass`, `UpdateConn`, etc.)
3. Define a traits struct inheriting `DefaultNetworkTraits` with overridden `using` aliases
4. Instantiate `Network<YourTraits>`
