# ADR-006: Device-resident SOA on the CUDA backend

Status: accepted (staged; lazy backing deferred)
Supersedes/extends: ADR-005 (SoaView split)

## Context

The CUDA SOA backend allocates every arena with `cudaMallocManaged` +
`cudaMemset(0, whole arena)` (`alloc.hpp` `detail::AllocStorage`). Two problems:

1. **Eager backing.** The full-arena `cudaMemset` touches every page, forcing the
   UVM driver to physically back the entire `Capacity` arena at construction --
   so a large-but-sparse net (e.g. DEEP R's dense candidate pool, mostly dead
   slots) costs full memory. The CPU path (`mmap(MAP_NORESERVE)`) is lazy, so
   this is a GPU-only regression.
2. **Migration thrashing.** Managed memory gives one pointer coherent across
   host and device, but a workload that alternates host and device access to the
   same arena (DEEP R: device kernels + host loss/metrics per step) page-faults
   and migrates pages back and forth every step -- non-deterministic and slow.

Dropping only the `cudaMemset` (ADR-candidate, "Phase 0") fixes (1) cheaply but
leaves (2). For a real GPU DEEP R run we want the arena **device-resident for the
duration of GPU execution**: no per-step host access, no migration.

## Decision

Move the CUDA SOA backend to **explicit `cudaMalloc` device buffers that are
device-resident during the GPU-execution phase**, with host access confined to
two boundaries: **construction** (host builds a staging mirror, one H2D upload)
and **readback** (D2H of the small slices host code needs -- outputs, metrics,
the live count). During the DoStep loop the arena lives only on the device.

Scope for this ADR (agreed with Ayrton):
- **In:** allocator split (device buffer + host staging), atomic-count readback,
  construction upload + output readback, a `device_ptr` compile-time host-deref
  guard, RAII/`Defer` leak safety.
- **Deferred:** lazy backing (accept eager `cudaMalloc` over-allocation for now;
  VMM/`cudaMallocAsync` pool is a later ADR), multi-GPU per-device arenas, and
  revisiting the CPU (`mmap`) backend.

## Design

### 1. `device_ptr<T>` -- compile-time host-deref guard
`include/plastix/device_ptr.hpp`. Wraps a device pointer. Under the CUDA build
(`__CUDACC__`) `operator*`/`operator[]`/`operator->` are `__device__`-only, so
any **host** dereference is a compile error (calling a `__device__` function from
host code). On the CPU build it is a transparent pointer. `get()` is `PLASTIX_HD`
and returns the raw `T*` for allocation/free/`cudaMemcpy` bookkeeping -- an
explicit escape hatch, not an accidental deref. This makes "a host pass forgot it
is now device memory" a build failure instead of a segfault.

### 2. Allocator split (device buffer + host staging)
The device-resident allocator holds, per field arena: a `device_ptr<T>` (the
`cudaMalloc` buffer the `SoaView` hands to kernels) and a host staging `T*`
(`malloc`). `PushToDevice()` = H2D of the live prefix; `PullToHost()` = D2H.
Builders and other host construction write the staging buffer; one `PushToDevice`
after construction makes the device authoritative. `SoaView::view()` returns the
device pointers, unchanged for kernels.

To contain blast radius (the CPU-touching examples/tests must keep working), this
is an **opt-in mode**, not a global flip: a network selects device-residency
(trait/flag); the default stays managed/`mmap`. A `device_ptr` in the default
`SoaView` would break every host `GetField` under CUDA, so the guard lives only
on the device-resident path.

### 3. Atomic count readback
The `AtomicCount` size lives on the device buffer; host reads of `Size()` (grid
sizing, output range, live-count metrics) go through a D2H copy of the count.

### 4. Construction upload + output readback
Construct on the host staging mirror -> one `PushToDevice`. `GetOutput`, loss,
and metrics `PullToHost` only the slice they need (output units are few, so this
is cheap even per step).

### 5. Leak safety -- RAII first, `Defer` for local unwinds
- **Primary: RAII.** `SOAAllocator` stays move-only; `Release()` in the dtor
  `cudaFree`s every device buffer (each field, each back buffer, perm scratch,
  count) and frees each staging buffer; the moved-from view is nulled so
  `Release` is a no-op (unchanged pattern). One owner frees exactly once.
- **`Defer` (C++ has no `defer` keyword; cf. C2y `defer` / Go).**
  `include/plastix/defer.hpp` provides a scope guard: `PLASTIX_DEFER(cudaFree(p));`
  runs at scope exit; `.dismiss()` cancels it. Used for **multi-step allocation
  sequences** in the ctor: allocate buffer 1..N, arm a `Defer` free after each,
  and `dismiss()` them all once every allocation has succeeded -- so a throw or
  early return midway frees what was already allocated (no partial-construction
  leak). This complements, not replaces, the RAII owner.
- **Detection:** run the CUDA suite under
  `compute-sanitizer --tool memcheck --leak-check full` (device leaks at context
  teardown) and `--tool initcheck` (uninitialized device reads -- catches any
  read-before-write the old `cudaMemset` was masking); valgrind for host staging.

## Consequences

- ~2x arena memory (device buffer + host staging). Accepted (Ayrton: not worried
  about over-allocation; the win is determinism, not footprint, at this stage).
- Migration-free, deterministic GPU execution; no per-step page faults.
- Host-deref mistakes become compile errors (`device_ptr`), not runtime segfaults.
- Blast radius contained by opt-in mode; the managed/`mmap` default is unchanged,
  so existing examples/tests keep building.

## Risks / failure modes (see also the pre-ADR analysis)

- **Half-refactor copy cost.** If any per-step host access survives, its D2H/H2D
  can cost as much as the migration we removed. Mitigation: device-only during
  the loop; only construction + tiny output slices cross the bus.
- **Silent staleness.** A host mutation of staging without `PushToDevice` (or a
  `Size()` read without the count D2H) yields stale kernels / wrong bounds with
  no crash. Mitigation: device-only-during-loop discipline + `device_ptr` guard.
- **Eager `cudaMalloc` OOM at extreme scale** -- accepted now, resolved later by
  lazy VMM backing.
- **CPU/GPU divergence** -- the device-resident path is CUDA-only; the CPU
  backend keeps `mmap`. The `device_ptr` guard keeps the two from being conflated.

## Sequenced work
1. `device_ptr.hpp` + `defer.hpp` (utilities) -- this change.
2. Opt-in device-resident allocator mode (buffer + staging + sync) in `alloc.hpp`.
3. Route DEEP R (already fully kernelized incl. the device rewire) onto it;
   confine host loss/metrics to output-slice readback.
4. Validate on Narval: `initcheck`/`memcheck --leak-check`, then the GPU DEEP R
   run; nsys migration counters to confirm the thrashing is gone.
5. Later ADRs: lazy VMM backing; multi-GPU; CPU-backend reconciliation.
