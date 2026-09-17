#ifndef PLASTIX_DISPATCH_GPU_HPP
#define PLASTIX_DISPATCH_GPU_HPP

// GPU launchers mirroring dispatch_cpu.hpp; always declared so dispatch-site
// `if constexpr` lookup resolves, bodies guarded by PLASTIX_HAS_CUDA (host
// stubs).

#include "plastix/conn.hpp"
#include "plastix/layers.hpp"
#include "plastix/macros.hpp"
#include "plastix/unit_state.hpp"

#include <span>

#ifdef PLASTIX_HAS_CUDA
#include "plastix/cuda_kernels.hpp"
#include "plastix/cuda_primitives.hpp"
#include <cuda_runtime.h>
#endif

#include <cstddef>
#include <cstdint>
#include <vector>

namespace plastix {

struct CompactEdge;
struct ProposalTag;

namespace gpu {

#ifdef PLASTIX_HAS_CUDA

// Device-resident compaction: reclaim tombstoned (DeadTag) connection slots
// entirely on the device buffer (no host round-trip). Flag live slots, exclusive
// scan to compacted positions, scatter a permutation, GatherDevice + recount.
template <typename CV>
__global__ void FlagLiveKernel(CV View, uint32_t *Flag, size_t N) {
  for (size_t C = blockIdx.x * blockDim.x + threadIdx.x; C < N;
       C += gridDim.x * blockDim.x)
    Flag[C] = GetField<DeadTag>(View, C) ? 0u : 1u;
}
template <typename CV>
__global__ void ScatterLivePermKernel(CV View, const uint32_t *Flag,
                                      size_t *Perm, size_t N) {
  for (size_t C = blockIdx.x * blockDim.x + threadIdx.x; C < N;
       C += gridDim.x * blockDim.x)
    if (!GetField<DeadTag>(View, C))
      Perm[Flag[C]] = C;
}

// Returns the live count. Only touches the device buffer + a transient device
// scratch; keeps the host staging count in sync so Size() stays correct.
template <typename CA> inline size_t DoCompactConnectionsDevice(CA &ConnAlloc) {
  const size_t N = ConnAlloc.DeviceCount();
  if (N == 0)
    return 0;
  auto View = ConnAlloc.view();
  const unsigned Block = 256;
  const unsigned Grid = static_cast<unsigned>((N + Block - 1) / Block);
  uint32_t *Flag = nullptr;
  PLASTIX_CUDA_CHECK(cudaMalloc(&Flag, (N + 1) * sizeof(uint32_t)));
  PLASTIX_CUDA_CHECK(cudaMemset(Flag, 0, (N + 1) * sizeof(uint32_t)));
  FlagLiveKernel<<<Grid, Block>>>(View, Flag, N);
  PLASTIX_CUDA_CHECK(cudaGetLastError());
  cuda::ExclusiveScanInPlace(Flag, N + 1);
  uint32_t NumLive = 0;
  PLASTIX_CUDA_CHECK(
      cudaMemcpy(&NumLive, Flag + N, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  if (NumLive < N) {
    ScatterLivePermKernel<<<Grid, Block>>>(View, Flag,
                                           ConnAlloc.DevicePermScratch(), N);
    PLASTIX_CUDA_CHECK(cudaGetLastError());
    PLASTIX_CUDA_CHECK(cudaDeviceSynchronize());
    ConnAlloc.GatherDevice(NumLive);
    ConnAlloc.SetCountDevice(NumLive);
    ConnAlloc.SetCount(NumLive);
  }
  PLASTIX_CUDA_CHECK(cudaFree(Flag));
  return NumLive;
}

// Device-resident resort by source level (device buffer only). Stamp each edge's
// SrcLevel from the unit view, histogram levels, then a per-level atomic scatter
// builds the permutation for GatherDevice. Unit levels are assumed static (no
// device-resident unit growth), so RecomputeLevels is skipped. Order within a
// level is atomic-nondeterministic (accepted: the level sums tolerate it).
template <typename CV, typename UV>
__global__ void StampSrcLevelKernel(CV CView, UV UView, size_t N) {
  for (size_t C = blockIdx.x * blockDim.x + threadIdx.x; C < N;
       C += gridDim.x * blockDim.x)
    GetField<SrcLevelTag>(CView, C) =
        static_cast<uint16_t>(GetLevel(UView, GetField<FromIdTag>(CView, C)));
}
template <typename CV>
__global__ void HistogramLevelsKernel(CV CView, uint32_t *Hist, size_t N) {
  for (size_t C = blockIdx.x * blockDim.x + threadIdx.x; C < N;
       C += gridDim.x * blockDim.x)
    atomicAdd(&Hist[GetField<SrcLevelTag>(CView, C)], 1u);
}
template <typename CV>
__global__ void ScatterByLevelKernel(CV CView, uint32_t *WritePos, size_t *Perm,
                                     size_t N) {
  for (size_t C = blockIdx.x * blockDim.x + threadIdx.x; C < N;
       C += gridDim.x * blockDim.x) {
    uint16_t L = GetField<SrcLevelTag>(CView, C);
    Perm[atomicAdd(&WritePos[L], 1u)] = C;
  }
}

template <typename CA, typename UA, typename RangeT>
inline void SortConnectionsByLevelDevice(CA &ConnAlloc, UA &UnitAlloc,
                                         RangeT *Ranges, uint16_t &NumLevels,
                                         uint16_t MaxLvls) {
  const size_t N = ConnAlloc.DeviceCount();
  if (N == 0) {
    NumLevels = 0;
    return;
  }
  auto CView = ConnAlloc.view();
  auto UView = UnitAlloc.view();
  const unsigned Block = 256;
  const unsigned Grid = static_cast<unsigned>((N + Block - 1) / Block);
  StampSrcLevelKernel<<<Grid, Block>>>(CView, UView, N);
  PLASTIX_CUDA_CHECK(cudaGetLastError());
  uint32_t *Hist = nullptr;
  PLASTIX_CUDA_CHECK(cudaMalloc(&Hist, MaxLvls * sizeof(uint32_t)));
  PLASTIX_CUDA_CHECK(cudaMemset(Hist, 0, MaxLvls * sizeof(uint32_t)));
  HistogramLevelsKernel<<<Grid, Block>>>(CView, Hist, N);
  PLASTIX_CUDA_CHECK(cudaGetLastError());
  PLASTIX_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint32_t> HHist(MaxLvls);
  PLASTIX_CUDA_CHECK(cudaMemcpy(HHist.data(), Hist, MaxLvls * sizeof(uint32_t),
                                cudaMemcpyDeviceToHost));
  std::vector<uint32_t> WBegin(MaxLvls);
  uint32_t Offset = 0;
  NumLevels = 0;
  for (uint16_t L = 0; L < MaxLvls; ++L) {
    Ranges[L].Begin = Offset;
    WBegin[L] = Offset;
    Offset += HHist[L];
    Ranges[L].End = Offset;
    if (HHist[L] > 0)
      NumLevels = static_cast<uint16_t>(L + 1);
  }
  uint32_t *WritePos = nullptr;
  PLASTIX_CUDA_CHECK(cudaMalloc(&WritePos, MaxLvls * sizeof(uint32_t)));
  PLASTIX_CUDA_CHECK(cudaMemcpy(WritePos, WBegin.data(),
                                MaxLvls * sizeof(uint32_t),
                                cudaMemcpyHostToDevice));
  ScatterByLevelKernel<<<Grid, Block>>>(CView, WritePos,
                                        ConnAlloc.DevicePermScratch(), N);
  PLASTIX_CUDA_CHECK(cudaGetLastError());
  PLASTIX_CUDA_CHECK(cudaDeviceSynchronize());
  ConnAlloc.GatherDevice(N);
  PLASTIX_CUDA_CHECK(cudaFree(Hist));
  PLASTIX_CUDA_CHECK(cudaFree(WritePos));
}

// Forward pass -- Topological propagation

template <typename FP, typename UA, typename CA, typename Globals,
          typename RangesT>
inline void DoForwardTopological(UA &UnitAlloc, CA &ConnAlloc, Globals *G,
                                 size_t NumInput, const RangesT &Ranges,
                                 uint16_t NumLevels, void *Stream = nullptr,
                                 bool Sync = true) {
  cudaStream_t S = static_cast<cudaStream_t>(Stream);
  size_t NumUnits = UnitAlloc.Size();
  for (uint16_t L = 1; L <= NumLevels; ++L) {
    uint32_t Begin = Ranges[L - 1].Begin;
    uint32_t End = Ranges[L - 1].End;
    if (End > Begin) {
      unsigned Block = cuda::DefaultBlockSize;
      unsigned Grid = static_cast<unsigned>(cuda::GridSize(End - Begin, Block));
      cuda::ForwardConnSweepLevelKernel<FP><<<Grid, Block, 0, S>>>(
          Begin, End, UnitAlloc.view(), ConnAlloc.view(), G);
      PLASTIX_CUDA_CHECK_KERNEL();
    }
    if (NumUnits > NumInput) {
      unsigned Block = cuda::DefaultBlockSize;
      unsigned Grid =
          static_cast<unsigned>(cuda::GridSize(NumUnits - NumInput, Block));
      cuda::ForwardUnitApplyLevelKernel<FP><<<Grid, Block, 0, S>>>(
          NumInput, NumUnits, L, UnitAlloc.view(), G);
      PLASTIX_CUDA_CHECK_KERNEL();
    }
  }
  if (Sync)
    PLASTIX_CUDA_CHECK(cudaDeviceSynchronize());
}

// Forward pass -- Pipeline propagation

template <typename FP, typename UA, typename CA, typename Globals>
inline void DoForwardPipeline(UA &UnitAlloc, CA &ConnAlloc, Globals *G,
                              size_t NumInput, void *Stream = nullptr,
                              bool Sync = true) {
  cudaStream_t S = static_cast<cudaStream_t>(Stream);
  size_t NumConns = ConnAlloc.Size();
  if (NumConns > 0) {
    unsigned Block = cuda::DefaultBlockSize;
    unsigned Grid = static_cast<unsigned>(cuda::GridSize(NumConns, Block));
    cuda::ForwardConnSweepKernel<FP><<<Grid, Block, 0, S>>>(
        NumConns, UnitAlloc.view(), ConnAlloc.view(), G);
    PLASTIX_CUDA_CHECK_KERNEL();
  }
  size_t NumUnits = UnitAlloc.Size();
  if (NumUnits > NumInput) {
    unsigned Block = cuda::DefaultBlockSize;
    unsigned Grid =
        static_cast<unsigned>(cuda::GridSize(NumUnits - NumInput, Block));
    cuda::ForwardUnitApplyKernel<FP><<<Grid, Block, 0, S>>>(
        NumInput, NumUnits, UnitAlloc.view(), G);
    PLASTIX_CUDA_CHECK_KERNEL();
  }
  if (Sync)
    PLASTIX_CUDA_CHECK(cudaDeviceSynchronize());
}

// Pipeline forward via per-unit reduction over a reverse-adjacency CSR:
// contention-free per-unit tree reduction (one block per unit) + unit-apply.
template <typename FP, typename UA, typename CA, typename Globals>
inline void DoForwardReverseAdj(UA &UnitAlloc, CA &ConnAlloc, Globals *G,
                                size_t NumInput, const std::uint32_t *Offsets,
                                const std::uint32_t *Incoming,
                                void *Stream = nullptr, bool Sync = true) {
  cudaStream_t S = static_cast<cudaStream_t>(Stream);
  size_t NumUnits = UnitAlloc.Size();
  if (NumUnits > NumInput) {
    constexpr unsigned RedBlock = 256; // must match the kernel's __shared__ Sh
    unsigned Grid = static_cast<unsigned>(NumUnits - NumInput);
    cuda::ForwardPerUnitReduceKernel<FP><<<Grid, RedBlock, 0, S>>>(
        NumInput, NumUnits, Offsets, Incoming, UnitAlloc.view(),
        ConnAlloc.view(), G);
    PLASTIX_CUDA_CHECK_KERNEL();
    unsigned Block = cuda::DefaultBlockSize;
    unsigned AGrid =
        static_cast<unsigned>(cuda::GridSize(NumUnits - NumInput, Block));
    cuda::ForwardUnitApplyKernel<FP><<<AGrid, Block, 0, S>>>(
        NumInput, NumUnits, UnitAlloc.view(), G);
    PLASTIX_CUDA_CHECK_KERNEL();
  }
  if (Sync)
    PLASTIX_CUDA_CHECK(cudaDeviceSynchronize());
}

// Backward pass -- Topological propagation

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
      unsigned Grid = static_cast<unsigned>(cuda::GridSize(End - Begin, Block));
      cuda::BackwardConnSweepLevelKernel<BP>
          <<<Grid, Block>>>(Begin, End, UnitAlloc.view(), ConnAlloc.view(), G);
      PLASTIX_CUDA_CHECK_KERNEL();
    }
    if (NumUnits > NumInput) {
      unsigned Block = cuda::DefaultBlockSize;
      unsigned Grid =
          static_cast<unsigned>(cuda::GridSize(NumUnits - NumInput, Block));
      cuda::BackwardUnitApplyLevelKernel<BP>
          <<<Grid, Block>>>(NumInput, NumUnits, L, UnitAlloc.view(), G);
      PLASTIX_CUDA_CHECK_KERNEL();
    }
  }
  PLASTIX_CUDA_CHECK(cudaDeviceSynchronize());
}

// Backward pass -- Pipeline propagation

template <typename BP, typename UA, typename CA, typename Globals>
inline void DoBackwardPipeline(UA &UnitAlloc, CA &ConnAlloc, Globals *G,
                               size_t /*NumInput*/) {
  size_t NumConns = ConnAlloc.Size();
  if (NumConns > 0) {
    unsigned Block = cuda::DefaultBlockSize;
    unsigned Grid = static_cast<unsigned>(cuda::GridSize(NumConns, Block));
    cuda::BackwardConnSweepKernel<BP>
        <<<Grid, Block>>>(NumConns, UnitAlloc.view(), ConnAlloc.view(), G);
    PLASTIX_CUDA_CHECK_KERNEL();
  }
  size_t NumUnits = UnitAlloc.Size();
  if (NumUnits > 0) {
    unsigned Block = cuda::DefaultBlockSize;
    unsigned Grid = static_cast<unsigned>(cuda::GridSize(NumUnits, Block));
    cuda::BackwardUnitApplyKernel<BP>
        <<<Grid, Block>>>(size_t{0}, NumUnits, UnitAlloc.view(), G);
    PLASTIX_CUDA_CHECK_KERNEL();
  }
  PLASTIX_CUDA_CHECK(cudaDeviceSynchronize());
}

// Per-unit / per-connection update phases

template <typename UP, typename UA, typename Globals>
inline void DoUpdateUnit(UA &UnitAlloc, Globals *G) {
  size_t NumUnits = UnitAlloc.Size();
  if (NumUnits == 0)
    return;
  unsigned Block = cuda::DefaultBlockSize;
  unsigned Grid = static_cast<unsigned>(cuda::GridSize(NumUnits, Block));
  cuda::UpdateUnitKernel<UP><<<Grid, Block>>>(NumUnits, UnitAlloc.view(), G);
  PLASTIX_CUDA_CHECK_KERNEL();
  PLASTIX_CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename UC, typename UA, typename CA, typename Globals>
inline void DoUpdateConn(UA &UnitAlloc, CA &ConnAlloc, Globals *G) {
  size_t NumConns = ConnAlloc.Size();
  if (NumConns == 0)
    return;
  unsigned Block = cuda::DefaultBlockSize;
  unsigned Grid = static_cast<unsigned>(cuda::GridSize(NumConns, Block));
  cuda::UpdateConnIncomingKernel<UC>
      <<<Grid, Block>>>(NumConns, UnitAlloc.view(), ConnAlloc.view(), G);
  PLASTIX_CUDA_CHECK_KERNEL();
  PLASTIX_CUDA_CHECK(cudaDeviceSynchronize());
  cuda::UpdateConnOutgoingKernel<UC>
      <<<Grid, Block>>>(NumConns, UnitAlloc.view(), ConnAlloc.view(), G);
  PLASTIX_CUDA_CHECK_KERNEL();
  PLASTIX_CUDA_CHECK(cudaDeviceSynchronize());
}

// Prune phases

template <typename PP, typename UA, typename Globals>
inline void DoPruneUnits(UA &UnitAlloc, Globals *G) {
  size_t NumUnits = UnitAlloc.Size();
  if (NumUnits == 0)
    return;
  unsigned Block = cuda::DefaultBlockSize;
  unsigned Grid = static_cast<unsigned>(cuda::GridSize(NumUnits, Block));
  cuda::PruneUnitsKernel<PP><<<Grid, Block>>>(NumUnits, UnitAlloc.view(), G);
  PLASTIX_CUDA_CHECK_KERNEL();
  PLASTIX_CUDA_CHECK(cudaDeviceSynchronize());
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
      <<<Grid, Block>>>(NumConns, UnitAlloc.view(), ConnAlloc.view(), G);
  PLASTIX_CUDA_CHECK_KERNEL();
  PLASTIX_CUDA_CHECK(cudaDeviceSynchronize());
}

// Add phases

template <typename AP, typename UA, typename Globals>
inline void DoAddUnits(UA &UnitAlloc, Globals *G, uint16_t MaxLvl) {
  size_t NumUnits = UnitAlloc.Size();
  if (NumUnits == 0)
    return;
  unsigned Block = cuda::DefaultBlockSize;
  unsigned Grid = static_cast<unsigned>(cuda::GridSize(NumUnits, Block));
  cuda::AddUnitsKernel<AP>
      <<<Grid, Block>>>(NumUnits, MaxLvl, UnitAlloc.view(), G);
  PLASTIX_CUDA_CHECK_KERNEL();
  PLASTIX_CUDA_CHECK(cudaDeviceSynchronize());
}

// Device add-conn 3-phase pipeline: collect, radix-sort packed keys,
// dedup-scan-commit. True if any committed (caller flips NeedsResort).
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
  PLASTIX_CUDA_CHECK(cudaMallocManaged(&Counter, sizeof(uint32_t)));
  *Counter = 0u;
  size_t Total = NumUnits * NumUnits;
  unsigned Block = cuda::DefaultBlockSize;
  unsigned Grid = static_cast<unsigned>(cuda::GridSize(Total, Block));
  cuda::CollectProposalsKernel<AC><<<Grid, Block>>>(NumUnits, Neighbourhood,
                                                    UnitAlloc.view(), G, Props,
                                                    Counter, MaxProposals);
  PLASTIX_CUDA_CHECK_KERNEL();
  PLASTIX_CUDA_CHECK(cudaDeviceSynchronize());
  size_t NumProposals = *Counter;
  PLASTIX_CUDA_CHECK(cudaFree(Counter));
  if (NumProposals > MaxProposals)
    NumProposals = MaxProposals;
  if (NumProposals == 0)
    return false;

  // Phase 2.
  cuda::RadixSort64InPlace(Props, NumProposals);

  // Phase 3.
  bool Committed = false;
  uint32_t *Flags = nullptr;
  PLASTIX_CUDA_CHECK(
      cudaMallocManaged(&Flags, (NumProposals + 1) * sizeof(uint32_t)));
  unsigned FlagGrid =
      static_cast<unsigned>(cuda::GridSize(NumProposals + 1, Block));
  cuda::DedupFlagKernel<><<<FlagGrid, Block>>>(NumProposals, Props, Flags);
  PLASTIX_CUDA_CHECK_KERNEL();
  cuda::ExclusiveScanInPlace(Flags, NumProposals + 1);
  PLASTIX_CUDA_CHECK(cudaDeviceSynchronize());
  uint32_t KeptCount = Flags[NumProposals];
  if (KeptCount > 0) {
    auto Range = ConnAlloc.AllocateMany(KeptCount);
    if (Range.first != static_cast<size_t>(-1)) {
      unsigned CommitGrid =
          static_cast<unsigned>(cuda::GridSize(NumProposals, Block));
      cuda::CommitProposalsKernel<AC>
          <<<CommitGrid, Block>>>(NumProposals, Props, Flags, Range.first,
                                  UnitAlloc.view(), ConnAlloc.view(), G);
      PLASTIX_CUDA_CHECK_KERNEL();
      PLASTIX_CUDA_CHECK(cudaDeviceSynchronize());
      Committed = true;
    }
  }
  PLASTIX_CUDA_CHECK(cudaFree(Flags));
  return Committed;
}

// Topology rebuild -- level-parallel Kahn's algorithm on device

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
        NumInput, NumUnits, UnitAlloc.view());
    PLASTIX_CUDA_CHECK_KERNEL();
  }

  if (NumConns == 0) {
    PLASTIX_CUDA_CHECK(cudaDeviceSynchronize());
    return;
  }

  PLASTIX_CUDA_CHECK(cudaMemset(InDegree, 0, NumUnits * sizeof(uint32_t)));
  PLASTIX_CUDA_CHECK(
      cudaMemset(OutOffset, 0, (NumUnits + 1) * sizeof(uint32_t)));

  unsigned ConnGrid = static_cast<unsigned>(cuda::GridSize(NumConns));
  cuda::CountDegreesKernel<<<ConnGrid, cuda::DefaultBlockSize>>>(
      NumConns, ConnAlloc.view(), InDegree, OutOffset);
  PLASTIX_CUDA_CHECK_KERNEL();

  // Exclusive scan over OutOffset[0..NumUnits]; the slot at index NumUnits
  // is zero-initialized so it ends up holding the total edge count.
  cuda::ExclusiveScanInPlace(OutOffset, NumUnits + 1);

  PLASTIX_CUDA_CHECK(cudaMemcpy(
      WritePos, OutOffset, NumUnits * sizeof(uint32_t), cudaMemcpyDefault));
  cuda::ScatterEdgesKernel<<<ConnGrid, cuda::DefaultBlockSize>>>(
      NumConns, ConnAlloc.view(), WritePos, OutEdges);
  PLASTIX_CUDA_CHECK_KERNEL();

  if (NumInput > 0) {
    unsigned InputGrid = static_cast<unsigned>(cuda::GridSize(NumInput));
    cuda::InitFrontierKernel<>
        <<<InputGrid, cuda::DefaultBlockSize>>>(NumInput, Frontier);
    PLASTIX_CUDA_CHECK_KERNEL();
  }

  uint32_t *NextSize = nullptr;
  PLASTIX_CUDA_CHECK(cudaMallocManaged(&NextSize, sizeof(uint32_t)));
  PLASTIX_CUDA_CHECK(cudaDeviceSynchronize());

  uint32_t FrontierSize = static_cast<uint32_t>(NumInput);
  uint16_t CurrentLevel = 0;
  while (FrontierSize > 0) {
    *NextSize = 0;
    unsigned FrontierGrid = static_cast<unsigned>(cuda::GridSize(FrontierSize));
    cuda::ProcessFrontierKernel<<<FrontierGrid, cuda::DefaultBlockSize>>>(
        FrontierSize, CurrentLevel, UnitAlloc.view(), ConnAlloc.view(),
        InDegree, OutOffset, OutEdges, Frontier, NextFrontier, NextSize);
    PLASTIX_CUDA_CHECK_KERNEL();
    PLASTIX_CUDA_CHECK(cudaDeviceSynchronize());
    ++CurrentLevel;
    std::swap(Frontier, NextFrontier);
    FrontierSize = *NextSize;
  }

  PLASTIX_CUDA_CHECK(cudaFree(NextSize));
}

// Device-side global-state reset (ADR-006): one thread applies ResetPolicy to
// the device-resident Globals so it is never host-touched during the loop.
template <typename ResetPolicy, typename G>
__global__ void ResetGlobalKernel(G *Globals) {
  ResetPolicy::Reset(*Globals);
}
template <typename ResetPolicy, typename G>
inline void DoResetGlobalOnDevice(G *Globals) {
  ResetGlobalKernel<ResetPolicy><<<1, 1>>>(Globals);
  PLASTIX_CUDA_CHECK_KERNEL();
  PLASTIX_CUDA_CHECK(cudaDeviceSynchronize());
}

// Device-side loss (ADR-007/008): the device-callable loss policy reads
// Activation and writes BackwardAcc in place. One thread runs it -- the policy
// loops the (few) output units internally.
template <typename Loss, typename UV, typename Globals>
__global__ void CalculateLossKernel(UV Unit, UnitRange Out,
                                    const float *Targets, size_t N,
                                    Globals *G) {
  if (blockIdx.x == 0 && threadIdx.x == 0)
    Loss::CalculateLoss(Unit, Out, std::span<const float>(Targets, N), *G);
}

template <typename Loss, typename UV, typename Globals>
inline void DoCalculateLossOnDevice(UV Unit, UnitRange Out,
                                    const float *DeviceTargets, size_t N,
                                    Globals *G, void *Stream = nullptr,
                                    bool Sync = true) {
  cudaStream_t S = static_cast<cudaStream_t>(Stream);
  CalculateLossKernel<Loss><<<1, 1, 0, S>>>(Unit, Out, DeviceTargets, N, G);
  PLASTIX_CUDA_CHECK_KERNEL();
  if (Sync)
    PLASTIX_CUDA_CHECK(cudaDeviceSynchronize());
}

#else // !PLASTIX_HAS_CUDA

// Host-build stubs: same signatures so dispatch-site lookup resolves;
// never instantiated (`if constexpr` discards), so empty bodies suffice.

template <typename FP, typename UA, typename CA, typename Globals,
          typename RangesT>
inline void DoForwardTopological(UA &, CA &, Globals *, size_t, const RangesT &,
                                 uint16_t, void * = nullptr, bool = true) {}

template <typename FP, typename UA, typename CA, typename Globals>
inline void DoForwardPipeline(UA &, CA &, Globals *, size_t, void * = nullptr,
                              bool = true) {}

template <typename FP, typename UA, typename CA, typename Globals>
inline void DoForwardReverseAdj(UA &, CA &, Globals *, size_t,
                                const std::uint32_t *, const std::uint32_t *,
                                void * = nullptr, bool = true) {}

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

template <typename ResetPolicy, typename G>
inline void DoResetGlobalOnDevice(G *) {}

template <typename Loss, typename UV, typename Globals>
inline void DoCalculateLossOnDevice(UV, UnitRange, const float *, size_t,
                                    Globals *, void * = nullptr, bool = true) {}

#endif // PLASTIX_HAS_CUDA

} // namespace gpu
} // namespace plastix

#endif // PLASTIX_DISPATCH_GPU_HPP
