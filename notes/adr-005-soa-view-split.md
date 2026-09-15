# ADR-005: Split the SOA allocator into an owning allocator and a non-owning SoaView

Status: Accepted

## Context

`alloc::SOAAllocator<T, Fields...>` did two incompatible jobs with one type:

1. **Own heap memory** -- it allocates the field arenas (mmap on the host,
   `cudaMallocManaged` under CUDA) and must free them exactly once (RAII, a
   host-only destructor).
2. **Be passed by value into `__global__` kernels** -- the GPU dispatch handed
   the allocator into every kernel by value, so it had to be a cheap,
   device-copyable handle with no host-only destructor riding along.

Bridging those two roles required three hacks: an `Owns` bool, a `PLASTIX_HD`
copy constructor that shallow-copies and clears `Owns`, and a destructor whose
body is guarded by `#ifndef __CUDA_ARCH__`. The kernel-parameter copy still
carried a (host-only) destructor, which is the root of the "cannot call
`__host__` function from a `__global__` function" hazard (review finding
MAJOR-5), and left the ownership model muddy (raw nullable pointers, review
comment 4.6).

## Decision

Split the two jobs into two types.

**`SoaView<T, Fields...>` -- non-owning handle (what kernels take)**
- Holds only the plain data device code reads: `FieldPtrs`, `BackFieldPtrs`,
  `PermScratch`, `CountPtr`, `Capacity`. No `Owns`.
- Owns all per-element / device-callable operations: `Get`/`GetArrayFor`/`Size`/
  `GetCapacity`/`PermutationScratch` (`PLASTIX_HD`), `Allocate` (`PLASTIX_HD`),
  `AllocateMany`, `Gather` (host-only).
- **No destructor**, no owning constructor. It has a `PLASTIX_HD` copy
  constructor so it copies on-device. Because it never owns or frees storage,
  passing it by value into a kernel has no host-in-device-destructor problem --
  that is the MAJOR-5 fix.

**`SOAAllocator<T, Fields...>` -- owner (RAII, host-only)**
- Single member: `SoaView<T, Fields...> View;`.
- Owning constructor allocates each arena and fills `View`'s pointers.
- Destructor frees `View`'s pointers -- plain host-only, no `#ifndef
  __CUDA_ARCH__` guard and no `Owns` check, because the owner is never passed to
  a device.
- **Move-only** (the moved-from view is marked empty so its destructor no-ops);
  copy is deleted. The copyable thing is now the `View`, not the owner.
- `view()` returns the handle to hand to kernels, plus thin forwarding methods
  (`Get`, `Size`, `Allocate`, `Gather`, ...) so host call sites are unchanged.

## Call-site changes

- **Kernel launches** (`dispatch_gpu.hpp`): `kernel<<<...>>>(U, C, ...)` becomes
  `kernel<<<...>>>(U.view(), C.view(), ...)`. Kernel templates deduce the
  allocator type to `SoaView`; kernel bodies are unchanged because every op they
  call exists on the view. `Kahn`/`Proposal` allocators are only accessed via
  `GetArrayFor`, so they are not viewed.
- **`plastix.hpp` `Network`**: holds `SOAAllocator` members and passes them by
  reference into dispatch; host phases use the forwarding methods, so no change.
- **`dispatch_cpu.hpp` / `test_alloc.cpp`**: unchanged via forwarding.
- Removed: every `Owns` reference, the guarded destructor body, and the old
  `PLASTIX_HD` copy constructor.

## Consequences / risks

- **`std::tuple` triviality:** `FieldPtrs`/`BackFieldPtrs` are `std::tuple`s,
  which are not guaranteed trivially copyable -- kernel parameters ideally are.
  `SoaView` keeps a `PLASTIX_HD` copy constructor and copies on-device under
  `--expt-relaxed-constexpr` (the same behaviour the previous allocator relied
  on). A `TODO(plastix-gpu)` on `SoaView` tracks swapping the tuples for a
  trivially-copyable pointer aggregate to make the view strictly trivially
  copyable; kept out of this change as a separable follow-up.
- **`Gather` staleness:** `Gather` swaps the owner's `View` pointers on the
  host; kernels get a fresh `view()` copy per launch, so no staleness -- the GPU
  dispatch never calls `Gather` on a view copy.
- **Behaviour-preserving:** no arithmetic or allocation-order change, so the
  parity/golden vectors are unaffected. Verified via `plastix_tests`
  (`test_alloc` exercises the allocator API directly); the CPU and CUDA builds
  both compile, and the CUDA path was compile-validated with nvcc.
