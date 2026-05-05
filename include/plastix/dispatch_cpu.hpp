#ifndef PLASTIX_DISPATCH_CPU_HPP
#define PLASTIX_DISPATCH_CPU_HPP

// Host-only implementations of every per-step phase. Each function is a
// free template that takes the allocators and any extra state by reference
// so it can stand in as a drop-in replacement for the in-class loop body
// it used to live as. Pure C++; no CUDA syntax, no `cuda::` references.
//
// The matching GPU launchers live in `dispatch_gpu.hpp`. The dispatch site
// in `Network<Traits>::Do*` picks one with a single `if constexpr`.

#include "plastix/conn.hpp"
#include "plastix/macros.hpp"
#include "plastix/unit_state.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace plastix {

// CompactEdge / LevelRange are framework-internal types used by both
// dispatch backends. Forward-declared here so dispatch_cpu.hpp doesn't
// need to pull in plastix.hpp (which would form a cycle); the real
// definitions still live next to `Network<Traits>`.
struct LevelRange;
struct CompactEdge;
struct InDegreeTag;
struct OutOffsetTag;
struct KahnWritePosTag;
struct FrontierTag;
struct NextFrontierTag;
struct ProposalTag;

namespace cpu {

// ---------------------------------------------------------------------------
// Forward pass — Topological propagation
// ---------------------------------------------------------------------------
//
// Walks one level at a time: accumulate across every connection in level L
// into the destination unit's ForwardAcc, then `Apply` on every unit at
// level L. A signal crosses the entire network in one call. `Ranges`
// indexes connections by source level (set up by SortConnectionsByLevel).

template <typename FP, typename UA, typename CA, typename Globals,
          typename RangesT>
inline void DoForwardTopological(UA &UnitAlloc, CA &ConnAlloc, Globals *G,
                                 size_t NumInput, const RangesT &Ranges,
                                 uint16_t NumLevels) {
  using Acc = typename FP::Accumulator;
  size_t NumUnits = UnitAlloc.Size();
  for (uint16_t L = 1; L <= NumLevels; ++L) {
    for (uint32_t C = Ranges[L - 1].Begin; C < Ranges[L - 1].End; ++C) {
      if (GetField<DeadTag>(ConnAlloc, C))
        continue;
      auto ToId = GetField<ToIdTag>(ConnAlloc, C);
      auto FromId = GetField<FromIdTag>(ConnAlloc, C);
      auto &UAcc = GetForwardAcc(UnitAlloc, ToId);
      UAcc = FP::Combine(
          UAcc, FP::Map(UnitAlloc, ToId, FromId, ConnAlloc, C, *G));
    }

    for (size_t I = NumInput; I < NumUnits; ++I) {
      if (GetLevel(UnitAlloc, I) == L) {
        auto &UAcc = GetForwardAcc(UnitAlloc, I);
        FP::Apply(UnitAlloc, I, *G, UAcc);
        UAcc = Acc{};
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Forward pass — Pipeline propagation
// ---------------------------------------------------------------------------
//
// One sweep over every live connection (no level partitioning), then one
// `Apply` per non-input unit. A signal advances exactly one layer per call.

template <typename FP, typename UA, typename CA, typename Globals>
inline void DoForwardPipeline(UA &UnitAlloc, CA &ConnAlloc, Globals *G,
                              size_t NumInput) {
  using Acc = typename FP::Accumulator;
  size_t NumConns = ConnAlloc.Size();
  for (size_t C = 0; C < NumConns; ++C) {
    if (GetField<DeadTag>(ConnAlloc, C))
      continue;
    auto ToId = GetField<ToIdTag>(ConnAlloc, C);
    auto FromId = GetField<FromIdTag>(ConnAlloc, C);
    auto &UAcc = GetForwardAcc(UnitAlloc, ToId);
    UAcc = FP::Combine(
        UAcc, FP::Map(UnitAlloc, ToId, FromId, ConnAlloc, C, *G));
  }

  size_t NumUnits = UnitAlloc.Size();
  for (size_t I = NumInput; I < NumUnits; ++I) {
    auto &UAcc = GetForwardAcc(UnitAlloc, I);
    FP::Apply(UnitAlloc, I, *G, UAcc);
    UAcc = Acc{};
  }
}

// ---------------------------------------------------------------------------
// Backward pass — Topological propagation
// ---------------------------------------------------------------------------
//
// Walks levels high-to-low; gradients flow backward, so `Apply` runs on the
// source unit and `Map` reads destination-side state.

template <typename BP, typename UA, typename CA, typename Globals,
          typename RangesT>
inline void DoBackwardTopological(UA &UnitAlloc, CA &ConnAlloc, Globals *G,
                                  size_t NumInput, const RangesT &Ranges,
                                  uint16_t NumLevels) {
  using Acc = typename BP::Accumulator;
  size_t NumUnits = UnitAlloc.Size();
  for (uint16_t L = NumLevels; L >= 1; --L) {
    for (uint32_t C = Ranges[L].Begin; C < Ranges[L].End; ++C) {
      if (GetField<DeadTag>(ConnAlloc, C))
        continue;
      auto ToId = GetField<ToIdTag>(ConnAlloc, C);
      auto FromId = GetField<FromIdTag>(ConnAlloc, C);
      auto &UAcc = GetBackwardAcc(UnitAlloc, FromId);
      UAcc = BP::Combine(
          UAcc, BP::Map(UnitAlloc, FromId, ToId, ConnAlloc, C, *G));
    }

    for (size_t I = NumInput; I < NumUnits; ++I) {
      if (GetLevel(UnitAlloc, I) == L) {
        auto &UAcc = GetBackwardAcc(UnitAlloc, I);
        BP::Apply(UnitAlloc, I, *G, UAcc);
        UAcc = Acc{};
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Backward pass — Pipeline propagation
// ---------------------------------------------------------------------------

template <typename BP, typename UA, typename CA, typename Globals>
inline void DoBackwardPipeline(UA &UnitAlloc, CA &ConnAlloc, Globals *G,
                               size_t /*NumInput*/) {
  using Acc = typename BP::Accumulator;
  size_t NumConns = ConnAlloc.Size();
  for (size_t C = NumConns; C-- > 0;) {
    if (GetField<DeadTag>(ConnAlloc, C))
      continue;
    auto ToId = GetField<ToIdTag>(ConnAlloc, C);
    auto FromId = GetField<FromIdTag>(ConnAlloc, C);
    auto &UAcc = GetBackwardAcc(UnitAlloc, FromId);
    UAcc = BP::Combine(
        UAcc, BP::Map(UnitAlloc, FromId, ToId, ConnAlloc, C, *G));
  }

  size_t NumUnits = UnitAlloc.Size();
  for (size_t I = 0; I < NumUnits; ++I) {
    auto &UAcc = GetBackwardAcc(UnitAlloc, I);
    BP::Apply(UnitAlloc, I, *G, UAcc);
    UAcc = Acc{};
  }
}

// ---------------------------------------------------------------------------
// Per-unit / per-connection update phases
// ---------------------------------------------------------------------------

template <typename UP, typename UA, typename Globals>
inline void DoUpdateUnit(UA &UnitAlloc, Globals *G) {
  size_t NumUnits = UnitAlloc.Size();
  for (size_t I = 0; I < NumUnits; ++I)
    UP::Update(UnitAlloc, I, *G);
}

template <typename UC, typename UA, typename CA, typename Globals>
inline void DoUpdateConn(UA &UnitAlloc, CA &ConnAlloc, Globals *G) {
  size_t NumConns = ConnAlloc.Size();
  for (size_t C = 0; C < NumConns; ++C) {
    if (GetField<DeadTag>(ConnAlloc, C))
      continue;
    auto ToId = GetField<ToIdTag>(ConnAlloc, C);
    auto FromId = GetField<FromIdTag>(ConnAlloc, C);
    UC::UpdateIncomingConnection(UnitAlloc, ToId, FromId, ConnAlloc, C, *G);
  }
  for (size_t C = 0; C < NumConns; ++C) {
    if (GetField<DeadTag>(ConnAlloc, C))
      continue;
    auto ToId = GetField<ToIdTag>(ConnAlloc, C);
    auto FromId = GetField<FromIdTag>(ConnAlloc, C);
    UC::UpdateOutgoingConnection(UnitAlloc, FromId, ToId, ConnAlloc, C, *G);
  }
}

// ---------------------------------------------------------------------------
// Prune phases
// ---------------------------------------------------------------------------

template <typename PP, typename UA, typename Globals>
inline void DoPruneUnits(UA &UnitAlloc, Globals *G) {
  size_t NumUnits = UnitAlloc.Size();
  for (size_t I = 0; I < NumUnits; ++I)
    GetField<PrunedTag>(UnitAlloc, I) = PP::ShouldPrune(UnitAlloc, I, *G);
}

// HasUnitPrune / HasConnPrune are template flags so the unused branch is
// `if constexpr`-discarded, matching the in-class version's behavior.
template <bool HasUnitPrune, bool HasConnPrune, typename CP, typename UA,
          typename CA, typename Globals>
inline void DoPruneConnections(UA &UnitAlloc, CA &ConnAlloc, Globals *G) {
  size_t NumConns = ConnAlloc.Size();
  for (size_t C = 0; C < NumConns; ++C) {
    if (GetField<DeadTag>(ConnAlloc, C))
      continue;
    auto ToId = GetField<ToIdTag>(ConnAlloc, C);
    auto FromId = GetField<FromIdTag>(ConnAlloc, C);

    bool Remove = false;
    if constexpr (HasUnitPrune)
      Remove = GetField<PrunedTag>(UnitAlloc, ToId) ||
               GetField<PrunedTag>(UnitAlloc, FromId);
    if constexpr (HasConnPrune)
      if (!Remove)
        Remove = CP::ShouldPrune(UnitAlloc, ToId, FromId, ConnAlloc, C, *G);

    if (Remove)
      GetField<DeadTag>(ConnAlloc, C) = true;
  }
}

// ---------------------------------------------------------------------------
// Add phases
// ---------------------------------------------------------------------------

template <typename AP, typename UA, typename Globals>
inline void DoAddUnits(UA &UnitAlloc, Globals *G, uint16_t MaxLvl) {
  size_t NumUnits = UnitAlloc.Size();
  for (size_t I = 0; I < NumUnits; ++I) {
    auto Offset = AP::AddUnit(UnitAlloc, I, *G);
    if (Offset.has_value()) {
      int32_t Base = GetLevel(UnitAlloc, I);
      int32_t NewLevel =
          std::clamp(Base + static_cast<int32_t>(*Offset), int32_t{1},
                     static_cast<int32_t>(MaxLvl - 1));
      auto NewId = UnitAlloc.Allocate();
      GetLevel(UnitAlloc, NewId) = static_cast<uint16_t>(NewLevel);
      AP::InitUnit(UnitAlloc, NewId, I, *G);
    }
  }
}

// AddConnections is the only phase whose CPU and GPU pipelines diverge
// algorithmically: the host path indexes units by level into the Kahn
// scratch arrays, walks a rolling level-pair window for proposal
// collection, sorts the packed (From, To) keys, and commits unique
// proposals serially. Returns true when at least one connection was
// committed (caller flips NeedsResort).

template <typename AC, typename UA, typename CA, typename KA, typename PA,
          typename Globals>
inline bool DoAddConnections(UA &UnitAlloc, CA &ConnAlloc, KA &KahnAlloc,
                             PA &ProposalAlloc, Globals *G,
                             uint16_t Neighbourhood) {
  size_t NumUnits = UnitAlloc.Size();

  // Phase 0: per-level unit index into KahnAlloc scratch.
  uint32_t *LevelOffset = KahnAlloc.template GetArrayFor<InDegreeTag>();
  uint32_t *UnitsByLevel = KahnAlloc.template GetArrayFor<FrontierTag>();
  uint32_t *LevelWritePos = KahnAlloc.template GetArrayFor<OutOffsetTag>();

  std::memset(LevelOffset, 0, (NumUnits + 1) * sizeof(uint32_t));
  uint16_t HighestLevel = 0;
  for (size_t I = 0; I < NumUnits; ++I) {
    uint16_t Lvl = GetLevel(UnitAlloc, I);
    ++LevelOffset[Lvl];
    if (Lvl > HighestLevel)
      HighestLevel = Lvl;
  }

  uint32_t Sum = 0;
  for (uint16_t L = 0; L <= HighestLevel; ++L) {
    uint32_t Count = LevelOffset[L];
    LevelOffset[L] = Sum;
    Sum += Count;
  }
  LevelOffset[HighestLevel + 1] = Sum;

  std::memcpy(LevelWritePos, LevelOffset,
              (HighestLevel + 1) * sizeof(uint32_t));
  for (size_t I = 0; I < NumUnits; ++I) {
    uint16_t Lvl = GetLevel(UnitAlloc, I);
    UnitsByLevel[LevelWritePos[Lvl]++] = static_cast<uint32_t>(I);
  }

  // Phase 1: rolling level-pair window collection.
  auto *Props = ProposalAlloc.template GetArrayFor<ProposalTag>();
  size_t NumProposals = 0;
  size_t MaxProposals = ProposalAlloc.GetCapacity();

  // Process one level-pair (La <= Lb). For cross-level pairs, queries
  // both (U,V) and (V,U) perspectives to match the original all-pairs
  // semantics.
  auto ProcessLevelPair = [&](uint16_t La, uint16_t Lb) {
    for (uint32_t Ai = LevelOffset[La]; Ai < LevelOffset[La + 1]; ++Ai) {
      uint32_t U = UnitsByLevel[Ai];
      for (uint32_t Bi = LevelOffset[Lb]; Bi < LevelOffset[Lb + 1]; ++Bi) {
        uint32_t V = UnitsByLevel[Bi];
        if (U == V)
          continue;

        if (NumProposals < MaxProposals &&
            AC::ShouldAddIncomingConnection(UnitAlloc, U, V, *G))
          Props[NumProposals++] = {V, U};
        if (NumProposals < MaxProposals &&
            AC::ShouldAddOutgoingConnection(UnitAlloc, U, V, *G))
          Props[NumProposals++] = {U, V};

        if (La != Lb) {
          if (NumProposals < MaxProposals &&
              AC::ShouldAddIncomingConnection(UnitAlloc, V, U, *G))
            Props[NumProposals++] = {U, V};
          if (NumProposals < MaxProposals &&
              AC::ShouldAddOutgoingConnection(UnitAlloc, V, U, *G))
            Props[NumProposals++] = {V, U};
        }
      }
    }
  };

  uint16_t InitRight =
      (Neighbourhood <= HighestLevel) ? Neighbourhood : HighestLevel;
  for (uint16_t La = 0; La <= InitRight; ++La)
    for (uint16_t Lb = La; Lb <= InitRight; ++Lb)
      ProcessLevelPair(La, Lb);

  for (uint16_t NewRight = Neighbourhood + 1; NewRight <= HighestLevel;
       ++NewRight) {
    uint16_t Left = NewRight - Neighbourhood;
    for (uint16_t WinLevel = Left; WinLevel <= NewRight; ++WinLevel)
      ProcessLevelPair(WinLevel, NewRight);
  }

  if (NumProposals == 0)
    return false;

  // Phase 2: sort packed (From, To) bits ascending. Pure host sort —
  // KernelizeAdd=true callers go through the GPU dispatcher instead.
  std::sort(Props, Props + NumProposals,
            [](const auto &A, const auto &B) { return A.Bits < B.Bits; });

  // Phase 3: commit unique proposals.
  size_t SizeBefore = ConnAlloc.Size();
  uint64_t Prev = UINT64_MAX;
  for (size_t I = 0; I < NumProposals; ++I) {
    if (Props[I].Bits == Prev)
      continue;
    Prev = Props[I].Bits;

    uint32_t F = Props[I].From();
    uint32_t T = Props[I].To();
    auto ConnId = ConnAlloc.Allocate();
    GetField<FromIdTag>(ConnAlloc, ConnId) = F;
    GetField<ToIdTag>(ConnAlloc, ConnId) = T;
    GetField<SrcLevelTag>(ConnAlloc, ConnId) = GetLevel(UnitAlloc, F);
    AC::InitConnection(UnitAlloc, F, T, ConnAlloc, ConnId, *G);
  }

  return ConnAlloc.Size() > SizeBefore;
}

// ---------------------------------------------------------------------------
// Topology rebuild (Kahn's algorithm, level-parallel on host)
// ---------------------------------------------------------------------------
//
// Resets non-input levels to 0, builds a CSR-style outgoing-edge list,
// then BFS-propagates levels from the input frontier outward. `OutEdges`
// is the connection allocator's permutation scratch (re-used here since
// the topology sort is the only other consumer).

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

  for (size_t I = NumInput; I < NumUnits; ++I)
    GetLevel(UnitAlloc, I) = 0;

  if (NumConns == 0)
    return;

  std::memset(InDegree, 0, NumUnits * sizeof(uint32_t));
  std::memset(OutOffset, 0, (NumUnits + 1) * sizeof(uint32_t));

  for (size_t C = 0; C < NumConns; ++C) {
    if (GetField<DeadTag>(ConnAlloc, C))
      continue;
    ++OutOffset[GetField<FromIdTag>(ConnAlloc, C)];
    ++InDegree[GetField<ToIdTag>(ConnAlloc, C)];
  }

  uint32_t Sum = 0;
  for (size_t I = 0; I < NumUnits; ++I) {
    uint32_t Deg = OutOffset[I];
    OutOffset[I] = Sum;
    Sum += Deg;
  }
  OutOffset[NumUnits] = Sum;

  std::memcpy(WritePos, OutOffset, NumUnits * sizeof(uint32_t));
  for (size_t C = 0; C < NumConns; ++C) {
    if (GetField<DeadTag>(ConnAlloc, C))
      continue;
    uint32_t From = GetField<FromIdTag>(ConnAlloc, C);
    OutEdges[WritePos[From]++] = C;
  }

  for (size_t I = 0; I < NumInput; ++I)
    Frontier[I] = static_cast<uint32_t>(I);
  uint32_t FrontierSize = static_cast<uint32_t>(NumInput);

  uint16_t CurrentLevel = 0;
  while (FrontierSize > 0) {
    uint32_t NextSize = 0;
    for (uint32_t F = 0; F < FrontierSize; ++F) {
      uint32_t U = Frontier[F];
      for (uint32_t E = OutOffset[U]; E < OutOffset[U + 1]; ++E) {
        size_t C = OutEdges[E];
        uint32_t To = GetField<ToIdTag>(ConnAlloc, C);
        if (--InDegree[To] == 0) {
          GetLevel(UnitAlloc, To) = static_cast<uint16_t>(CurrentLevel + 1);
          NextFrontier[NextSize++] = To;
        }
      }
    }
    ++CurrentLevel;
    std::swap(Frontier, NextFrontier);
    FrontierSize = NextSize;
  }
}

} // namespace cpu
} // namespace plastix

#endif // PLASTIX_DISPATCH_CPU_HPP
