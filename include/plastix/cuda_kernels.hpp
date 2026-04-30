#ifndef PLASTIX_CUDA_KERNELS_HPP
#define PLASTIX_CUDA_KERNELS_HPP

// Per-DoX CUDA kernels. One free function per phase; the network's DoX()
// methods select between the host loop and the kernel launch under
// PLASTIX_HAS_CUDA. Kernels accept the allocators by value (relying on the
// shallow-copy ctor in alloc.hpp) so the device sees the same managed-memory
// pointers without needing the host allocator object itself to be on-device.

#ifdef PLASTIX_HAS_CUDA

#include "plastix/alloc.hpp"
#include "plastix/conn.hpp"
#include "plastix/cuda_primitives.hpp"
#include "plastix/macros.hpp"
#include "plastix/unit_state.hpp"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace plastix {
namespace cuda {

// ---------------------------------------------------------------------------
// Forward pass — Pipeline propagation
// ---------------------------------------------------------------------------
//
// Two kernels per call:
//   1. ForwardConnSweepKernel — one thread per connection. Skips dead conns.
//      Calls FP::Map(...) and atomicAdds the result into ForwardAcc[ToId].
//   2. ForwardUnitApplyKernel — one thread per non-input unit. Calls
//      FP::Apply(...) on the accumulated value, then resets the accumulator.
//
// Constraint (MVP): FP::Combine must be `+` over `float`. atomicAdd<float>
// is the only Combine the kernel knows about today; if a user policy needs a
// non-additive combine, fall back to the host path. We assert this at the
// call site by requiring `FP::Accumulator == float`.

template <typename FP, typename UnitAlloc, typename ConnAlloc, typename Globals>
__global__ void ForwardConnSweepKernel(size_t NumConns, UnitAlloc U,
                                       ConnAlloc C, Globals *G) {
  size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I >= NumConns)
    return;
  if (GetField<DeadTag>(C, I))
    return;
  auto ToId = GetField<ToIdTag>(C, I);
  auto FromId = GetField<FromIdTag>(C, I);
  float V = FP::Map(U, ToId, FromId, C, I, *G);
  atomicAdd(&GetForwardAcc(U, ToId), V);
}

template <typename FP, typename UnitAlloc, typename Globals>
__global__ void ForwardUnitApplyKernel(size_t Begin, size_t End, UnitAlloc U,
                                       Globals *G) {
  size_t Tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t I = Begin + Tid;
  if (I >= End)
    return;
  using Acc = typename FP::Accumulator;
  auto &UAcc = GetForwardAcc(U, I);
  FP::Apply(U, I, *G, UAcc);
  UAcc = Acc{};
}

// ---------------------------------------------------------------------------
// Forward pass — Topological propagation
// ---------------------------------------------------------------------------
//
// Same kernel pair as Pipeline, but the connection sweep is restricted to
// `[Begin, End)` (the level's connection range) and the unit-apply kernel
// applies only to units of the matching level. The level is filtered inside
// the kernel via GetLevel(); a future pass could partition units by level
// at sort time and remove the filter.

template <typename FP, typename UnitAlloc, typename ConnAlloc, typename Globals>
__global__ void ForwardConnSweepLevelKernel(uint32_t Begin, uint32_t End,
                                            UnitAlloc U, ConnAlloc C,
                                            Globals *G) {
  uint32_t Tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t I = Begin + Tid;
  if (I >= End)
    return;
  if (GetField<DeadTag>(C, I))
    return;
  auto ToId = GetField<ToIdTag>(C, I);
  auto FromId = GetField<FromIdTag>(C, I);
  float V = FP::Map(U, ToId, FromId, C, I, *G);
  atomicAdd(&GetForwardAcc(U, ToId), V);
}

template <typename FP, typename UnitAlloc, typename Globals>
__global__ void ForwardUnitApplyLevelKernel(size_t Begin, size_t End,
                                            uint16_t Level, UnitAlloc U,
                                            Globals *G) {
  size_t Tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t I = Begin + Tid;
  if (I >= End)
    return;
  if (GetLevel(U, I) != Level)
    return;
  using Acc = typename FP::Accumulator;
  auto &UAcc = GetForwardAcc(U, I);
  FP::Apply(U, I, *G, UAcc);
  UAcc = Acc{};
}

// ---------------------------------------------------------------------------
// Backward pass — both propagation models. Walks connections in the
// forward direction by index but accumulates into the source unit's
// BackwardAcc instead of the destination's ForwardAcc.
// ---------------------------------------------------------------------------

template <typename BP, typename UnitAlloc, typename ConnAlloc, typename Globals>
__global__ void BackwardConnSweepKernel(size_t NumConns, UnitAlloc U,
                                        ConnAlloc C, Globals *G) {
  size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I >= NumConns)
    return;
  if (GetField<DeadTag>(C, I))
    return;
  auto ToId = GetField<ToIdTag>(C, I);
  auto FromId = GetField<FromIdTag>(C, I);
  float V = BP::Map(U, FromId, ToId, C, I, *G);
  atomicAdd(&GetBackwardAcc(U, FromId), V);
}

template <typename BP, typename UnitAlloc, typename ConnAlloc, typename Globals>
__global__ void BackwardConnSweepLevelKernel(uint32_t Begin, uint32_t End,
                                             UnitAlloc U, ConnAlloc C,
                                             Globals *G) {
  uint32_t Tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t I = Begin + Tid;
  if (I >= End)
    return;
  if (GetField<DeadTag>(C, I))
    return;
  auto ToId = GetField<ToIdTag>(C, I);
  auto FromId = GetField<FromIdTag>(C, I);
  float V = BP::Map(U, FromId, ToId, C, I, *G);
  atomicAdd(&GetBackwardAcc(U, FromId), V);
}

template <typename BP, typename UnitAlloc, typename Globals>
__global__ void BackwardUnitApplyKernel(size_t Begin, size_t End, UnitAlloc U,
                                        Globals *G) {
  size_t Tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t I = Begin + Tid;
  if (I >= End)
    return;
  using Acc = typename BP::Accumulator;
  auto &UAcc = GetBackwardAcc(U, I);
  BP::Apply(U, I, *G, UAcc);
  UAcc = Acc{};
}

template <typename BP, typename UnitAlloc, typename Globals>
__global__ void BackwardUnitApplyLevelKernel(size_t Begin, size_t End,
                                             uint16_t Level, UnitAlloc U,
                                             Globals *G) {
  size_t Tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t I = Begin + Tid;
  if (I >= End)
    return;
  if (GetLevel(U, I) != Level)
    return;
  using Acc = typename BP::Accumulator;
  auto &UAcc = GetBackwardAcc(U, I);
  BP::Apply(U, I, *G, UAcc);
  UAcc = Acc{};
}

// ---------------------------------------------------------------------------
// Update / prune phases — one thread per element, no reduction needed.
// ---------------------------------------------------------------------------

template <typename UP, typename UnitAlloc, typename Globals>
__global__ void UpdateUnitKernel(size_t NumUnits, UnitAlloc U, Globals *G) {
  size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I >= NumUnits)
    return;
  UP::Update(U, I, *G);
}

template <typename UC, typename UnitAlloc, typename ConnAlloc, typename Globals>
__global__ void UpdateConnIncomingKernel(size_t NumConns, UnitAlloc U,
                                         ConnAlloc C, Globals *G) {
  size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I >= NumConns)
    return;
  if (GetField<DeadTag>(C, I))
    return;
  auto ToId = GetField<ToIdTag>(C, I);
  auto FromId = GetField<FromIdTag>(C, I);
  UC::UpdateIncomingConnection(U, ToId, FromId, C, I, *G);
}

template <typename UC, typename UnitAlloc, typename ConnAlloc, typename Globals>
__global__ void UpdateConnOutgoingKernel(size_t NumConns, UnitAlloc U,
                                         ConnAlloc C, Globals *G) {
  size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I >= NumConns)
    return;
  if (GetField<DeadTag>(C, I))
    return;
  auto ToId = GetField<ToIdTag>(C, I);
  auto FromId = GetField<FromIdTag>(C, I);
  UC::UpdateOutgoingConnection(U, FromId, ToId, C, I, *G);
}

template <typename PP, typename UnitAlloc, typename Globals>
__global__ void PruneUnitsKernel(size_t NumUnits, UnitAlloc U, Globals *G) {
  size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I >= NumUnits)
    return;
  GetField<PrunedTag>(U, I) = PP::ShouldPrune(U, I, *G);
}

// Connection-prune kernel handles three cases by template flags so we don't
// have to launch two kernels: unit-prune-only, conn-prune-only, or both.

template <bool HasUnitPrune, bool HasConnPrune, typename CP, typename UnitAlloc,
          typename ConnAlloc, typename Globals>
__global__ void PruneConnectionsKernel(size_t NumConns, UnitAlloc U,
                                       ConnAlloc C, Globals *G) {
  size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I >= NumConns)
    return;
  if (GetField<DeadTag>(C, I))
    return;
  auto ToId = GetField<ToIdTag>(C, I);
  auto FromId = GetField<FromIdTag>(C, I);
  bool Remove = false;
  if constexpr (HasUnitPrune)
    Remove =
        GetField<PrunedTag>(U, ToId) || GetField<PrunedTag>(U, FromId);
  if constexpr (HasConnPrune) {
    if (!Remove)
      Remove = CP::ShouldPrune(U, ToId, FromId, C, I, *G);
  }
  if (Remove)
    GetField<DeadTag>(C, I) = true;
}

// ---------------------------------------------------------------------------
// RecomputeLevels — level-parallel Kahn's algorithm
// ---------------------------------------------------------------------------
//
// Reset non-input unit levels in parallel.

template <typename UnitAlloc>
__global__ void ResetLevelsKernel(size_t Begin, size_t End, UnitAlloc U) {
  size_t Tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t I = Begin + Tid;
  if (I >= End)
    return;
  GetLevel(U, I) = uint16_t{0};
}

// Count in-degree and out-degree per unit (skipping dead connections).
template <typename ConnAlloc>
__global__ void CountDegreesKernel(size_t NumConns, ConnAlloc C,
                                   uint32_t *InDegree, uint32_t *OutOffset) {
  size_t Cid = blockIdx.x * blockDim.x + threadIdx.x;
  if (Cid >= NumConns)
    return;
  if (GetField<DeadTag>(C, Cid))
    return;
  atomicAdd(&OutOffset[GetField<FromIdTag>(C, Cid)], 1u);
  atomicAdd(&InDegree[GetField<ToIdTag>(C, Cid)], 1u);
}

// Scatter live connections into the CSR-style outgoing-edge list. WritePos
// starts as a copy of OutOffset; each thread atomically increments its bucket
// to claim a slot.
template <typename ConnAlloc>
__global__ void ScatterEdgesKernel(size_t NumConns, ConnAlloc C,
                                   uint32_t *WritePos, size_t *OutEdges) {
  size_t Cid = blockIdx.x * blockDim.x + threadIdx.x;
  if (Cid >= NumConns)
    return;
  if (GetField<DeadTag>(C, Cid))
    return;
  uint32_t From = GetField<FromIdTag>(C, Cid);
  uint32_t Pos = atomicAdd(&WritePos[From], 1u);
  OutEdges[Pos] = Cid;
}

template <typename T = void>
__global__ void InitFrontierKernel(size_t NumInput, uint32_t *Frontier) {
  size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I >= NumInput)
    return;
  Frontier[I] = static_cast<uint32_t>(I);
}

// Process one BFS level. Each thread owns one frontier entry, walks its
// outgoing edges, atomically decrements the destination's in-degree, and on
// the decrement-to-zero edge claims a slot in the next frontier and writes
// the level. atomicSub returns the OLD value, so old==1 means this thread
// drove it to zero.
template <typename UnitAlloc, typename ConnAlloc>
__global__ void
ProcessFrontierKernel(uint32_t FrontierSize, uint16_t Level, UnitAlloc U,
                      ConnAlloc C, uint32_t *InDegree,
                      const uint32_t *OutOffset, const size_t *OutEdges,
                      const uint32_t *Frontier, uint32_t *NextFrontier,
                      uint32_t *NextSize) {
  uint32_t F = blockIdx.x * blockDim.x + threadIdx.x;
  if (F >= FrontierSize)
    return;
  uint32_t Unit = Frontier[F];
  uint32_t Lo = OutOffset[Unit];
  uint32_t Hi = OutOffset[Unit + 1];
  for (uint32_t E = Lo; E < Hi; ++E) {
    size_t Cid = OutEdges[E];
    uint32_t To = GetField<ToIdTag>(C, Cid);
    uint32_t Old = atomicSub(&InDegree[To], 1u);
    if (Old == 1u) {
      GetLevel(U, To) = static_cast<uint16_t>(Level + 1);
      uint32_t Pos = atomicAdd(NextSize, 1u);
      NextFrontier[Pos] = To;
    }
  }
}

// ---------------------------------------------------------------------------
// AddConnections — proposal collection
// ---------------------------------------------------------------------------
//
// One thread per ordered (U, V) pair with U != V. Filters out pairs whose
// level difference exceeds Neighbourhood, then evaluates the AddConn
// policy's two predicates and atomically claims slots in the proposal
// buffer. This matches the host loop's semantics: every level pair within
// the window is visited exactly once per direction, so each ordered
// (U, V) combination evaluates `ShouldAddIncoming(U,V)` and
// `ShouldAddOutgoing(U,V)` once.
//
// Slots beyond MaxProposals are dropped silently (capacity bound). The
// caller clamps the final counter after the launch.
// Encodes (From, To) as `From | (To << 32)` to match `CompactEdge::Bits`.
__device__ inline uint64_t PackEdgeBits(uint32_t From, uint32_t To) {
  return static_cast<uint64_t>(From) |
         (static_cast<uint64_t>(To) << 32);
}

template <typename AddConnPolicy, typename UnitAlloc, typename Globals>
__global__ void
CollectProposalsKernel(size_t NumUnits, uint16_t Neighbourhood, UnitAlloc U,
                       Globals *G, uint64_t *Props, uint32_t *Counter,
                       size_t MaxProposals) {
  size_t T = blockIdx.x * blockDim.x + threadIdx.x;
  size_t Total = NumUnits * NumUnits;
  if (T >= Total)
    return;
  size_t Ui = T / NumUnits;
  size_t Vi = T % NumUnits;
  if (Ui == Vi)
    return;
  int32_t Lu = static_cast<int32_t>(GetLevel(U, Ui));
  int32_t Lv = static_cast<int32_t>(GetLevel(U, Vi));
  int32_t Diff = Lv - Lu;
  if (Diff < 0)
    Diff = -Diff;
  if (Diff > static_cast<int32_t>(Neighbourhood))
    return;
  if (AddConnPolicy::ShouldAddIncomingConnection(U, Ui, Vi, *G)) {
    uint32_t Slot = atomicAdd(Counter, 1u);
    if (static_cast<size_t>(Slot) < MaxProposals)
      Props[Slot] = PackEdgeBits(static_cast<uint32_t>(Vi),
                                 static_cast<uint32_t>(Ui));
  }
  if (AddConnPolicy::ShouldAddOutgoingConnection(U, Ui, Vi, *G)) {
    uint32_t Slot = atomicAdd(Counter, 1u);
    if (static_cast<size_t>(Slot) < MaxProposals)
      Props[Slot] = PackEdgeBits(static_cast<uint32_t>(Ui),
                                 static_cast<uint32_t>(Vi));
  }
}

// ---------------------------------------------------------------------------
// AddConnections — dedup commit
// ---------------------------------------------------------------------------
//
// Phase 3 splits in two: a flag kernel that marks the first occurrence of
// each unique key, then (after a scan) a commit kernel that materializes
// kept proposals into the connection allocator. Stream-compaction by scan
// preserves the post-sort order so commit slots are deterministic
// (Begin + WritePos) instead of atomically claimed.

template <typename T = void>
__global__ void DedupFlagKernel(size_t NumProposals, const uint64_t *Props,
                                uint32_t *Flags) {
  size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I > NumProposals)
    return;
  uint32_t F = 0u;
  if (I < NumProposals) {
    if (I == 0)
      F = 1u;
    else
      F = (Props[I] != Props[I - 1]) ? 1u : 0u;
  }
  Flags[I] = F;
}

template <typename AddConnPolicy, typename UnitAlloc, typename ConnAlloc,
          typename Globals>
__global__ void CommitProposalsKernel(size_t NumProposals,
                                      const uint64_t *Props,
                                      const uint32_t *ScannedFlags,
                                      size_t BeginConnId, UnitAlloc U,
                                      ConnAlloc C, Globals *G) {
  size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I >= NumProposals)
    return;
  bool IsKept = (I == 0) || (Props[I] != Props[I - 1]);
  if (!IsKept)
    return;
  size_t ConnId = BeginConnId + ScannedFlags[I];
  uint32_t F = static_cast<uint32_t>(Props[I]);
  uint32_t T = static_cast<uint32_t>(Props[I] >> 32);
  GetField<FromIdTag>(C, ConnId) = F;
  GetField<ToIdTag>(C, ConnId) = T;
  GetField<SrcLevelTag>(C, ConnId) = GetLevel(U, F);
  AddConnPolicy::InitConnection(U, F, T, C, ConnId, *G);
}

// ---------------------------------------------------------------------------
// AddUnits kernel
// ---------------------------------------------------------------------------
//
// One thread per existing parent. The thread evaluates AP::AddUnit; if it
// returns an offset, it claims a new unit via UnitAlloc.Allocate() (atomic
// counter under the hood), writes the clamped level, and calls AP::InitUnit.
// Capacity exhaustion shows up as Allocate returning size_t(-1) and is
// silently dropped, matching the host loop's reliance on capacity bounds.
//
// Constraint: AP::AddUnit and AP::InitUnit must be `PLASTIX_HD`. Policies
// that mutate Globals non-atomically or use host-only APIs (RNG, std::cout)
// should set `Traits::KernelizeAdd = false` on their traits to keep this
// phase on the host loop.
template <typename AddUnitPolicy, typename UnitAlloc, typename Globals>
__global__ void AddUnitsKernel(size_t NumParents, uint16_t MaxLvl, UnitAlloc U,
                               Globals *G) {
  size_t ParentId = blockIdx.x * blockDim.x + threadIdx.x;
  if (ParentId >= NumParents)
    return;
  auto Offset = AddUnitPolicy::AddUnit(U, ParentId, *G);
  if (!Offset.has_value())
    return;
  auto NewId = U.Allocate();
  if (static_cast<size_t>(NewId) == static_cast<size_t>(-1))
    return;
  int32_t Base = static_cast<int32_t>(GetLevel(U, ParentId));
  int32_t Lvl = Base + static_cast<int32_t>(*Offset);
  int32_t Lo = 1;
  int32_t Hi = static_cast<int32_t>(MaxLvl) - 1;
  if (Lvl < Lo)
    Lvl = Lo;
  if (Lvl > Hi)
    Lvl = Hi;
  GetLevel(U, NewId) = static_cast<uint16_t>(Lvl);
  AddUnitPolicy::InitUnit(U, NewId, ParentId, *G);
}

} // namespace cuda
} // namespace plastix

#endif // PLASTIX_HAS_CUDA

#endif // PLASTIX_CUDA_KERNELS_HPP
