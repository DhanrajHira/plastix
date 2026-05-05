#ifndef PLASTIX_DISPATCH_GPU_HPP
#define PLASTIX_DISPATCH_GPU_HPP

// GPU launchers — one free template per phase, mirroring the host paths
// in `dispatch_cpu.hpp`. The function declarations are visible in every
// build so the qualified-name lookup in `if constexpr` branches at the
// dispatch site succeeds; the bodies are guarded by PLASTIX_HAS_CUDA so
// the `<<<>>>` launch syntax and `cuda::` references only have to exist
// when nvcc is doing the parse.
//
// In host builds the bodies are empty stubs — they are never instantiated
// (the dispatch site discards them with `if constexpr (HasCuda && ...)`),
// so no codegen happens and the unused parameters produce no warnings.

#include "plastix/conn.hpp"
#include "plastix/macros.hpp"
#include "plastix/unit_state.hpp"

#ifdef PLASTIX_HAS_CUDA
#include "plastix/cuda_kernels.hpp"
#include "plastix/cuda_primitives.hpp"
#include <cuda_runtime.h>
#endif

#include <cstddef>
#include <cstdint>

namespace plastix {

struct CompactEdge;
struct ProposalTag;

namespace gpu {

#ifdef PLASTIX_HAS_CUDA

// ---------------------------------------------------------------------------
// Forward pass — Topological propagation
// ---------------------------------------------------------------------------

template <typename FP, typename UA, typename CA, typename Globals,
          typename RangesT>
inline void DoForwardTopological(UA &UnitAlloc, CA &ConnAlloc, Globals *G,
                                 size_t NumInput, const RangesT &Ranges,
                                 uint16_t NumLevels) {
  size_t NumUnits = UnitAlloc.Size();
  for (uint16_t L = 1; L <= NumLevels; ++L) {
    uint32_t Begin = Ranges[L - 1].Begin;
    uint32_t End = Ranges[L - 1].End;
    if (End > Begin) {
      unsigned Block = cuda::DefaultBlockSize;
      unsigned Grid =
          static_cast<unsigned>(cuda::GridSize(End - Begin, Block));
      cuda::ForwardConnSweepLevelKernel<FP>
          <<<Grid, Block>>>(Begin, End, UnitAlloc, ConnAlloc, G);
    }
    if (NumUnits > NumInput) {
      unsigned Block = cuda::DefaultBlockSize;
      unsigned Grid = static_cast<unsigned>(
          cuda::GridSize(NumUnits - NumInput, Block));
      cuda::ForwardUnitApplyLevelKernel<FP>
          <<<Grid, Block>>>(NumInput, NumUnits, L, UnitAlloc, G);
    }
  }
  cudaDeviceSynchronize();
}

// ---------------------------------------------------------------------------
// Forward pass — Pipeline propagation
// ---------------------------------------------------------------------------

template <typename FP, typename UA, typename CA, typename Globals>
inline void DoForwardPipeline(UA &UnitAlloc, CA &ConnAlloc, Globals *G,
                              size_t NumInput) {
  size_t NumConns = ConnAlloc.Size();
  if (NumConns > 0) {
    unsigned Block = cuda::DefaultBlockSize;
    unsigned Grid = static_cast<unsigned>(cuda::GridSize(NumConns, Block));
    cuda::ForwardConnSweepKernel<FP>
        <<<Grid, Block>>>(NumConns, UnitAlloc, ConnAlloc, G);
  }
  size_t NumUnits = UnitAlloc.Size();
  if (NumUnits > NumInput) {
    unsigned Block = cuda::DefaultBlockSize;
    unsigned Grid = static_cast<unsigned>(
        cuda::GridSize(NumUnits - NumInput, Block));
    cuda::ForwardUnitApplyKernel<FP>
        <<<Grid, Block>>>(NumInput, NumUnits, UnitAlloc, G);
  }
  cudaDeviceSynchronize();
}

// ---------------------------------------------------------------------------
// Backward pass — Topological propagation
// ---------------------------------------------------------------------------

template <typename BP, typename UA, typename CA, typename Globals,
          typename RangesT>
inline void DoBackwardTopological(UA &UnitAlloc, CA &ConnAlloc, Globals *G,
                                  size_t NumInput, const RangesT &Ranges,
                                  uint16_t NumLevels) {
  size_t NumUnits = UnitAlloc.Size();
  for (uint16_t L = NumLevels; L >= 1; --L) {
    uint32_t Begin = Ranges[L].Begin;
    uint32_t End = Ranges[L].End;
    if (End > Begin) {
      unsigned Block = cuda::DefaultBlockSize;
      unsigned Grid =
          static_cast<unsigned>(cuda::GridSize(End - Begin, Block));
      cuda::BackwardConnSweepLevelKernel<BP>
          <<<Grid, Block>>>(Begin, End, UnitAlloc, ConnAlloc, G);
    }
    if (NumUnits > NumInput) {
      unsigned Block = cuda::DefaultBlockSize;
      unsigned Grid = static_cast<unsigned>(
          cuda::GridSize(NumUnits - NumInput, Block));
      cuda::BackwardUnitApplyLevelKernel<BP>
          <<<Grid, Block>>>(NumInput, NumUnits, L, UnitAlloc, G);
    }
  }
  cudaDeviceSynchronize();
}

// ---------------------------------------------------------------------------
// Backward pass — Pipeline propagation
// ---------------------------------------------------------------------------

template <typename BP, typename UA, typename CA, typename Globals>
inline void DoBackwardPipeline(UA &UnitAlloc, CA &ConnAlloc, Globals *G,
                               size_t /*NumInput*/) {
  size_t NumConns = ConnAlloc.Size();
  if (NumConns > 0) {
    unsigned Block = cuda::DefaultBlockSize;
    unsigned Grid = static_cast<unsigned>(cuda::GridSize(NumConns, Block));
    cuda::BackwardConnSweepKernel<BP>
        <<<Grid, Block>>>(NumConns, UnitAlloc, ConnAlloc, G);
  }
  size_t NumUnits = UnitAlloc.Size();
  if (NumUnits > 0) {
    unsigned Block = cuda::DefaultBlockSize;
    unsigned Grid = static_cast<unsigned>(cuda::GridSize(NumUnits, Block));
    cuda::BackwardUnitApplyKernel<BP>
        <<<Grid, Block>>>(size_t{0}, NumUnits, UnitAlloc, G);
  }
  cudaDeviceSynchronize();
}

// ---------------------------------------------------------------------------
// Per-unit / per-connection update phases
// ---------------------------------------------------------------------------

template <typename UP, typename UA, typename Globals>
inline void DoUpdateUnit(UA &UnitAlloc, Globals *G) {
  size_t NumUnits = UnitAlloc.Size();
  if (NumUnits == 0)
    return;
  unsigned Block = cuda::DefaultBlockSize;
  unsigned Grid = static_cast<unsigned>(cuda::GridSize(NumUnits, Block));
  cuda::UpdateUnitKernel<UP><<<Grid, Block>>>(NumUnits, UnitAlloc, G);
  cudaDeviceSynchronize();
}

template <typename UC, typename UA, typename CA, typename Globals>
inline void DoUpdateConn(UA &UnitAlloc, CA &ConnAlloc, Globals *G) {
  size_t NumConns = ConnAlloc.Size();
  if (NumConns == 0)
    return;
  unsigned Block = cuda::DefaultBlockSize;
  unsigned Grid = static_cast<unsigned>(cuda::GridSize(NumConns, Block));
  cuda::UpdateConnIncomingKernel<UC>
      <<<Grid, Block>>>(NumConns, UnitAlloc, ConnAlloc, G);
  cudaDeviceSynchronize();
  cuda::UpdateConnOutgoingKernel<UC>
      <<<Grid, Block>>>(NumConns, UnitAlloc, ConnAlloc, G);
  cudaDeviceSynchronize();
}

// ---------------------------------------------------------------------------
// Prune phases
// ---------------------------------------------------------------------------

template <typename PP, typename UA, typename Globals>
inline void DoPruneUnits(UA &UnitAlloc, Globals *G) {
  size_t NumUnits = UnitAlloc.Size();
  if (NumUnits == 0)
    return;
  unsigned Block = cuda::DefaultBlockSize;
  unsigned Grid = static_cast<unsigned>(cuda::GridSize(NumUnits, Block));
  cuda::PruneUnitsKernel<PP><<<Grid, Block>>>(NumUnits, UnitAlloc, G);
  cudaDeviceSynchronize();
}

template <bool HasUnitPrune, bool HasConnPrune, typename CP, typename UA,
          typename CA, typename Globals>
inline void DoPruneConnections(UA &UnitAlloc, CA &ConnAlloc, Globals *G) {
  size_t NumConns = ConnAlloc.Size();
  if (NumConns == 0)
    return;
  unsigned Block = cuda::DefaultBlockSize;
  unsigned Grid = static_cast<unsigned>(cuda::GridSize(NumConns, Block));
  cuda::PruneConnectionsKernel<HasUnitPrune, HasConnPrune, CP>
      <<<Grid, Block>>>(NumConns, UnitAlloc, ConnAlloc, G);
  cudaDeviceSynchronize();
}

// ---------------------------------------------------------------------------
// Add phases
// ---------------------------------------------------------------------------

template <typename AP, typename UA, typename Globals>
inline void DoAddUnits(UA &UnitAlloc, Globals *G, uint16_t MaxLvl) {
  size_t NumUnits = UnitAlloc.Size();
  if (NumUnits == 0)
    return;
  unsigned Block = cuda::DefaultBlockSize;
  unsigned Grid = static_cast<unsigned>(cuda::GridSize(NumUnits, Block));
  cuda::AddUnitsKernel<AP><<<Grid, Block>>>(NumUnits, MaxLvl, UnitAlloc, G);
  cudaDeviceSynchronize();
}

// AddConnections runs the entire 3-phase proposal pipeline on device:
// Phase 1 (per-(U,V) collection), Phase 2 (radix sort of packed keys),
// Phase 3 (dedup-flag → exclusive scan → AllocateMany → commit). Returns
// true if any new connections were committed (caller flips NeedsResort).
template <typename AC, typename UA, typename CA, typename PA, typename Globals>
inline bool DoAddConnections(UA &UnitAlloc, CA &ConnAlloc, PA &ProposalAlloc,
                             Globals *G, uint16_t Neighbourhood) {
  size_t NumUnits = UnitAlloc.Size();
  if (NumUnits == 0)
    return false;

  static_assert(sizeof(CompactEdge) == sizeof(uint64_t),
                "CompactEdge must be bit-compatible with uint64_t for "
                "the device proposal pipeline");
  CompactEdge *PropsEdges = ProposalAlloc.template GetArrayFor<ProposalTag>();
  uint64_t *Props = reinterpret_cast<uint64_t *>(PropsEdges);
  size_t MaxProposals = ProposalAlloc.GetCapacity();

  // Phase 1.
  uint32_t *Counter = nullptr;
  cudaMallocManaged(&Counter, sizeof(uint32_t));
  *Counter = 0u;
  size_t Total = NumUnits * NumUnits;
  unsigned Block = cuda::DefaultBlockSize;
  unsigned Grid = static_cast<unsigned>(cuda::GridSize(Total, Block));
  cuda::CollectProposalsKernel<AC><<<Grid, Block>>>(
      NumUnits, Neighbourhood, UnitAlloc, G, Props, Counter, MaxProposals);
  cudaDeviceSynchronize();
  size_t NumProposals = *Counter;
  cudaFree(Counter);
  if (NumProposals > MaxProposals)
    NumProposals = MaxProposals;
  if (NumProposals == 0)
    return false;

  // Phase 2.
  cuda::RadixSort64InPlace(Props, NumProposals);

  // Phase 3.
  bool Committed = false;
  uint32_t *Flags = nullptr;
  cudaMallocManaged(&Flags, (NumProposals + 1) * sizeof(uint32_t));
  unsigned FlagGrid =
      static_cast<unsigned>(cuda::GridSize(NumProposals + 1, Block));
  cuda::DedupFlagKernel<><<<FlagGrid, Block>>>(NumProposals, Props, Flags);
  cuda::ExclusiveScanInPlace(Flags, NumProposals + 1);
  cudaDeviceSynchronize();
  uint32_t KeptCount = Flags[NumProposals];
  if (KeptCount > 0) {
    auto Range = ConnAlloc.AllocateMany(KeptCount);
    if (Range.first != static_cast<size_t>(-1)) {
      unsigned CommitGrid =
          static_cast<unsigned>(cuda::GridSize(NumProposals, Block));
      cuda::CommitProposalsKernel<AC><<<CommitGrid, Block>>>(
          NumProposals, Props, Flags, Range.first, UnitAlloc, ConnAlloc, G);
      cudaDeviceSynchronize();
      Committed = true;
    }
  }
  cudaFree(Flags);
  return Committed;
}

// ---------------------------------------------------------------------------
// Topology rebuild — level-parallel Kahn's algorithm on device
// ---------------------------------------------------------------------------

template <typename UA, typename CA, typename KA>
inline void RecomputeLevels(UA &UnitAlloc, CA &ConnAlloc, KA &KahnAlloc,
                            size_t NumInput) {
  size_t NumUnits = UnitAlloc.Size();
  size_t NumConns = ConnAlloc.Size();

  uint32_t *InDegree = KahnAlloc.template GetArrayFor<InDegreeTag>();
  uint32_t *OutOffset = KahnAlloc.template GetArrayFor<OutOffsetTag>();
  uint32_t *WritePos = KahnAlloc.template GetArrayFor<KahnWritePosTag>();
  uint32_t *Frontier = KahnAlloc.template GetArrayFor<FrontierTag>();
  uint32_t *NextFrontier = KahnAlloc.template GetArrayFor<NextFrontierTag>();
  size_t *OutEdges = ConnAlloc.PermutationScratch();

  if (NumUnits > NumInput) {
    size_t Span = NumUnits - NumInput;
    unsigned Grid = static_cast<unsigned>(cuda::GridSize(Span));
    cuda::ResetLevelsKernel<<<Grid, cuda::DefaultBlockSize>>>(
        NumInput, NumUnits, UnitAlloc);
  }

  if (NumConns == 0) {
    cudaDeviceSynchronize();
    return;
  }

  cudaMemset(InDegree, 0, NumUnits * sizeof(uint32_t));
  cudaMemset(OutOffset, 0, (NumUnits + 1) * sizeof(uint32_t));

  unsigned ConnGrid = static_cast<unsigned>(cuda::GridSize(NumConns));
  cuda::CountDegreesKernel<<<ConnGrid, cuda::DefaultBlockSize>>>(
      NumConns, ConnAlloc, InDegree, OutOffset);

  // Exclusive scan over OutOffset[0..NumUnits]; the slot at index NumUnits
  // is zero-initialized so it ends up holding the total edge count.
  cuda::ExclusiveScanInPlace(OutOffset, NumUnits + 1);

  cudaMemcpy(WritePos, OutOffset, NumUnits * sizeof(uint32_t),
             cudaMemcpyDefault);
  cuda::ScatterEdgesKernel<<<ConnGrid, cuda::DefaultBlockSize>>>(
      NumConns, ConnAlloc, WritePos, OutEdges);

  if (NumInput > 0) {
    unsigned InputGrid = static_cast<unsigned>(cuda::GridSize(NumInput));
    cuda::InitFrontierKernel<>
        <<<InputGrid, cuda::DefaultBlockSize>>>(NumInput, Frontier);
  }

  uint32_t *NextSize = nullptr;
  cudaMallocManaged(&NextSize, sizeof(uint32_t));
  cudaDeviceSynchronize();

  uint32_t FrontierSize = static_cast<uint32_t>(NumInput);
  uint16_t CurrentLevel = 0;
  while (FrontierSize > 0) {
    *NextSize = 0;
    unsigned FrontierGrid =
        static_cast<unsigned>(cuda::GridSize(FrontierSize));
    cuda::ProcessFrontierKernel<<<FrontierGrid, cuda::DefaultBlockSize>>>(
        FrontierSize, CurrentLevel, UnitAlloc, ConnAlloc, InDegree, OutOffset,
        OutEdges, Frontier, NextFrontier, NextSize);
    cudaDeviceSynchronize();
    ++CurrentLevel;
    std::swap(Frontier, NextFrontier);
    FrontierSize = *NextSize;
  }

  cudaFree(NextSize);
}

#else // !PLASTIX_HAS_CUDA

// Host-build stubs: declared with the same signatures so the qualified-
// name lookup at the dispatch site succeeds. Never instantiated because
// the dispatch site discards the call with `if constexpr (HasCuda && ...)`,
// so empty bodies don't generate code or warn about unused parameters.

template <typename FP, typename UA, typename CA, typename Globals,
          typename RangesT>
inline void DoForwardTopological(UA &, CA &, Globals *, size_t,
                                 const RangesT &, uint16_t) {}

template <typename FP, typename UA, typename CA, typename Globals>
inline void DoForwardPipeline(UA &, CA &, Globals *, size_t) {}

template <typename BP, typename UA, typename CA, typename Globals,
          typename RangesT>
inline void DoBackwardTopological(UA &, CA &, Globals *, size_t,
                                  const RangesT &, uint16_t) {}

template <typename BP, typename UA, typename CA, typename Globals>
inline void DoBackwardPipeline(UA &, CA &, Globals *, size_t) {}

template <typename UP, typename UA, typename Globals>
inline void DoUpdateUnit(UA &, Globals *) {}

template <typename UC, typename UA, typename CA, typename Globals>
inline void DoUpdateConn(UA &, CA &, Globals *) {}

template <typename PP, typename UA, typename Globals>
inline void DoPruneUnits(UA &, Globals *) {}

template <bool HasUnitPrune, bool HasConnPrune, typename CP, typename UA,
          typename CA, typename Globals>
inline void DoPruneConnections(UA &, CA &, Globals *) {}

template <typename AP, typename UA, typename Globals>
inline void DoAddUnits(UA &, Globals *, uint16_t) {}

template <typename AC, typename UA, typename CA, typename PA, typename Globals>
inline bool DoAddConnections(UA &, CA &, PA &, Globals *, uint16_t) {
  return false;
}

template <typename UA, typename CA, typename KA>
inline void RecomputeLevels(UA &, CA &, KA &, size_t) {}

#endif // PLASTIX_HAS_CUDA

} // namespace gpu
} // namespace plastix

#endif // PLASTIX_DISPATCH_GPU_HPP
