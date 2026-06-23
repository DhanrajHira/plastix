#ifndef PLASTIX_DEVICE_ATOMICS_HPP
#define PLASTIX_DEVICE_ATOMICS_HPP

// Warp-aggregated atomic add for global reductions in device policies.
//
// A policy that accumulates into a shared GlobalState scalar (e.g. the
// imprinting learner's `G.Tau += ...` summed over every connection that feeds
// the output) issues one atomicAdd per edge. When thousands of edges target
// the same address those atomics serialize and dominate the kernel — profiling
// the on-device 09 update showed this single pattern was ~99% of the phase.
//
// WarpAtomicAdd reduces the contribution within the *coalesced* set of active
// lanes (so it is safe under thread divergence / early returns) and issues a
// single atomicAdd per group — cutting the atomic traffic by up to the warp
// width. Semantics are identical to a plain atomicAdd; only the count of
// hardware atomics changes. Host builds fall back to a plain `+=`.

#include "plastix/macros.hpp"

#ifdef PLASTIX_HAS_CUDA
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#endif

namespace plastix {

PLASTIX_HD void WarpAtomicAdd(float &Dst, float Val) {
#ifdef __CUDA_ARCH__
  namespace cg = cooperative_groups;
  auto Group = cg::coalesced_threads();
  float Sum = cg::reduce(Group, Val, cg::plus<float>());
  if (Group.thread_rank() == 0)
    atomicAdd(&Dst, Sum);
#else
  Dst += Val;
#endif
}

// Keyed (segmented) warp-aggregated scatter-add: lanes that target the *same*
// Key (e.g. the destination unit of a connection) are grouped, reduced, and a
// single atomicAdd is issued per group. This makes a per-edge scatter into a
// per-unit accumulator both full-GPU parallel (one thread per edge) AND
// contention-free even when one unit has a huge in-degree (e.g. one output fed
// by a million inputs — every lane in a warp shares its Key, so it collapses to
// one atomic per warp). Caller must ensure &Dst is the same for equal Keys.
template <typename KeyT>
PLASTIX_HD void WarpAtomicAddKeyed(float &Dst, float Val, KeyT Key) {
#ifdef __CUDA_ARCH__
  namespace cg = cooperative_groups;
  auto Warp = cg::coalesced_threads();
  auto Group =
      cg::labeled_partition(Warp, static_cast<unsigned long long>(Key));
  float Sum = cg::reduce(Group, Val, cg::plus<float>());
  if (Group.thread_rank() == 0)
    atomicAdd(&Dst, Sum);
#else
  Dst += Val;
#endif
}

} // namespace plastix

#endif // PLASTIX_DEVICE_ATOMICS_HPP
