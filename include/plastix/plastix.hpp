#ifndef PLASTIX_PLASTIX_HPP
#define PLASTIX_PLASTIX_HPP

#include "plastix/conn.hpp"
#include "plastix/layers.hpp"
#include "plastix/traits.hpp"
#include "plastix/unit_state.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <type_traits>

namespace plastix {

/// Returns the library version as a string.
const char *version();

// ---------------------------------------------------------------------------
// Network
// ---------------------------------------------------------------------------

static constexpr uint16_t MaxLevels = 1024;

struct LevelRange {
  uint32_t Begin;
  uint32_t End;
};

struct InDegreeTag {};
struct OutOffsetTag {};
struct KahnWritePosTag {};
struct FrontierTag {};
struct NextFrontierTag {};
struct KahnScratchEntity {};

using KahnScratchAllocator =
    alloc::SOAAllocator<KahnScratchEntity,
                        alloc::SOAField<InDegreeTag, uint32_t>,
                        alloc::SOAField<OutOffsetTag, uint32_t>,
                        alloc::SOAField<KahnWritePosTag, uint32_t>,
                        alloc::SOAField<FrontierTag, uint32_t>,
                        alloc::SOAField<NextFrontierTag, uint32_t>>;

template <NetworkTraits Traits> class Network {
  using UnitAllocator = UnitAllocFor<Traits>;
  using ConnAllocator = typename Traits::ConnAllocator;
  using GlobalState = typename Traits::GlobalState;

public:
  template <typename... Builders>
    requires(sizeof...(Builders) > 0 &&
             (LayerBuilder<Builders, UnitAllocator, ConnAllocator> && ...))
  Network(size_t InputDim, Builders... Layers)
      : NumInput(InputDim), UnitAlloc(256), ConnAlloc(256),
        KahnAlloc(UnitAlloc.GetCapacity() + 1) {
    for (size_t I = 0; I < InputDim; ++I) {
      auto Id = UnitAlloc.Allocate();
      float Y = static_cast<float>(I) - static_cast<float>(InputDim - 1) / 2.0f;
      UnitAlloc.template Get<PositionTag>(Id) = {
          _Float16{0}, static_cast<_Float16>(Y), _Float16{0}, 0};
    }
    UnitRange Prev{0, InputDim};
    ((Prev = Layers(UnitAlloc, ConnAlloc, Prev)), ...);
    OutputRange = Prev;
    SortConnectionsByLevel();
  }

  Network(size_t InputDim, size_t OutputDim = 1)
      : Network(InputDim, FullyConnected{OutputDim}) {}

  size_t GetStep() const { return Step; }

  void DoForwardPass(std::span<const float> Inputs) {
    if (NeedsResort)
      SortConnectionsByLevel();

    using FP = typename Traits::ForwardPass;
    using Acc = typename FP::Accumulator;

    for (size_t I = 0; I < NumInput; ++I)
      UnitAlloc.template Get<ActivationTag>(I) = Inputs[I];

    for (uint16_t L = 1; L <= NumLevels; ++L) {
      for (uint32_t C = Ranges[L - 1].Begin; C < Ranges[L - 1].End; ++C) {
        if (ConnAlloc.template Get<DeadTag>(C))
          continue;
        auto ToId = ConnAlloc.template Get<ToIdTag>(C);
        auto FromId = ConnAlloc.template Get<FromIdTag>(C);
        auto &UAcc = UnitAlloc.template Get<ForwardAccTag>(ToId);
        UAcc = FP::Combine(
            UAcc, FP::Map(UnitAlloc, ToId, FromId, ConnAlloc, C, Globals));
      }

      size_t NumUnits = UnitAlloc.Size();
      for (size_t I = NumInput; I < NumUnits; ++I) {
        if (UnitAlloc.template Get<LevelTag>(I) == L) {
          auto &UAcc = UnitAlloc.template Get<ForwardAccTag>(I);
          FP::Apply(UnitAlloc, I, Globals, UAcc);
          UAcc = Acc{};
        }
      }
    }

    ++Step;
  }

  void DoBackwardPass() {
    if constexpr (std::is_same_v<typename Traits::BackwardPass, NoBackwardPass>)
      return;
    else {
      using BP = typename Traits::BackwardPass;
      using Acc = typename BP::Accumulator;

      for (uint16_t L = NumLevels; L >= 1; --L) {
        for (uint32_t C = Ranges[L].Begin; C < Ranges[L].End; ++C) {
          if (ConnAlloc.template Get<DeadTag>(C))
            continue;
          auto ToId = ConnAlloc.template Get<ToIdTag>(C);
          auto FromId = ConnAlloc.template Get<FromIdTag>(C);
          auto &UAcc = UnitAlloc.template Get<BackwardAccTag>(FromId);
          UAcc = BP::Combine(
              UAcc, BP::Map(UnitAlloc, FromId, ToId, ConnAlloc, C, Globals));
        }

        size_t NumUnits = UnitAlloc.Size();
        for (size_t I = NumInput; I < NumUnits; ++I) {
          if (UnitAlloc.template Get<LevelTag>(I) == L) {
            auto &UAcc = UnitAlloc.template Get<BackwardAccTag>(I);
            BP::Apply(UnitAlloc, I, Globals, UAcc);
            UAcc = Acc{};
          }
        }
      }
    }
  }

  void DoUpdateUnitState() {
    if constexpr (std::is_same_v<typename Traits::UpdateUnit, NoUpdateUnit>)
      return;
    else {
      using UP = typename Traits::UpdateUnit;
      using Partial = typename UP::Partial;

      for (size_t C = 0; C < ConnAlloc.Size(); ++C) {
        if (ConnAlloc.template Get<DeadTag>(C))
          continue;
        auto ToId = ConnAlloc.template Get<ToIdTag>(C);
        auto FromId = ConnAlloc.template Get<FromIdTag>(C);
        auto Weight = ConnAlloc.template Get<WeightTag>(C);
        auto &Acc = UnitAlloc.template Get<UpdateAccTag>(ToId);
        Acc =
            UP::Combine(Acc, UP::Map(UnitAlloc, ToId, FromId, Globals, Weight));
      }

      size_t NumUnits = UnitAlloc.Size();
      for (size_t I = 0; I < NumUnits; ++I) {
        auto &Acc = UnitAlloc.template Get<UpdateAccTag>(I);
        UP::Apply(UnitAlloc, I, Globals, Acc);
        Acc = Partial{};
      }
    }
  }

  void DoUpdateConnectionState() {
    if constexpr (std::is_same_v<typename Traits::UpdateConn, NoUpdateConn>)
      return;
    else {
      using UP = typename Traits::UpdateConn;

      for (size_t C = 0; C < ConnAlloc.Size(); ++C) {
        if (ConnAlloc.template Get<DeadTag>(C))
          continue;
        auto ToId = ConnAlloc.template Get<ToIdTag>(C);
        auto FromId = ConnAlloc.template Get<FromIdTag>(C);
        UP::UpdateIncomingConnection(UnitAlloc, ToId, FromId, ConnAlloc, C,
                                     Globals);
      }

      for (size_t C = 0; C < ConnAlloc.Size(); ++C) {
        if (ConnAlloc.template Get<DeadTag>(C))
          continue;
        auto ToId = ConnAlloc.template Get<ToIdTag>(C);
        auto FromId = ConnAlloc.template Get<FromIdTag>(C);
        UP::UpdateOutgoingConnection(UnitAlloc, FromId, ToId, ConnAlloc, C,
                                     Globals);
      }
    }
  }
  void DoPruneUnits() {
    if constexpr (std::is_same_v<typename Traits::PruneUnit, NoPruneUnit>)
      return;
    else {
      using PP = typename Traits::PruneUnit;
      size_t NumUnits = UnitAlloc.Size();
      for (size_t I = 0; I < NumUnits; ++I)
        UnitAlloc.template Get<PrunedTag>(I) =
            PP::ShouldPrune(UnitAlloc, I, Globals);
    }
  }

  void DoPruneConnections() {
    if constexpr (std::is_same_v<typename Traits::PruneUnit, NoPruneUnit> &&
                  std::is_same_v<typename Traits::PruneConn, NoPruneConn>)
      return;
    else {
      using CP = typename Traits::PruneConn;
      constexpr bool HasUnitPrune =
          !std::is_same_v<typename Traits::PruneUnit, NoPruneUnit>;
      constexpr bool HasConnPrune =
          !std::is_same_v<typename Traits::PruneConn, NoPruneConn>;

      for (size_t C = 0; C < ConnAlloc.Size(); ++C) {
        if (ConnAlloc.template Get<DeadTag>(C))
          continue;
        auto ToId = ConnAlloc.template Get<ToIdTag>(C);
        auto FromId = ConnAlloc.template Get<FromIdTag>(C);

        bool Remove = false;
        if constexpr (HasUnitPrune)
          Remove = UnitAlloc.template Get<PrunedTag>(ToId) ||
                   UnitAlloc.template Get<PrunedTag>(FromId);
        if constexpr (HasConnPrune)
          if (!Remove)
            Remove =
                CP::ShouldPrune(UnitAlloc, ToId, FromId, ConnAlloc, C, Globals);

        if (Remove)
          ConnAlloc.template Get<DeadTag>(C) = true;
      }
    }
  }
  void DoAddUnits() {
    if constexpr (std::is_same_v<typename Traits::AddUnit, NoAddUnit>)
      return;
    else {
      using AP = typename Traits::AddUnit;
      size_t NumUnits = UnitAlloc.Size();
      for (size_t I = 0; I < NumUnits; ++I) {
        auto Pos = AP::AddUnit(UnitAlloc, I, Globals);
        if (Pos) {
          auto NewId = UnitAlloc.Allocate();
          UnitAlloc.template Get<PositionTag>(NewId) = Pos;
        }
      }
    }
  }
  void DoAddConnections() {
    if constexpr (std::is_same_v<typename Traits::AddConn, NoAddConn>)
      return;
    else {
      using AC = typename Traits::AddConn;
      size_t SizeBefore = ConnAlloc.Size();
      size_t NumUnits = UnitAlloc.Size();
      for (size_t Self = 0; Self < NumUnits; ++Self) {
        for (size_t Other = 0; Other < NumUnits; ++Other) {
          if (Self == Other)
            continue;

          auto [AddIn, WeightIn] =
              AC::ShouldAddIncomingConnection(UnitAlloc, Self, Other, Globals);
          if (AddIn) {
            auto ConnId = ConnAlloc.Allocate();
            ConnAlloc.template Get<FromIdTag>(ConnId) =
                static_cast<uint32_t>(Other);
            ConnAlloc.template Get<ToIdTag>(ConnId) =
                static_cast<uint32_t>(Self);
            ConnAlloc.template Get<WeightTag>(ConnId) = WeightIn;
            ConnAlloc.template Get<SrcLevelTag>(ConnId) =
                UnitAlloc.template Get<LevelTag>(Other);
          }

          auto [AddOut, WeightOut] =
              AC::ShouldAddOutgoingConnection(UnitAlloc, Self, Other, Globals);
          if (AddOut) {
            auto ConnId = ConnAlloc.Allocate();
            ConnAlloc.template Get<FromIdTag>(ConnId) =
                static_cast<uint32_t>(Self);
            ConnAlloc.template Get<ToIdTag>(ConnId) =
                static_cast<uint32_t>(Other);
            ConnAlloc.template Get<WeightTag>(ConnId) = WeightOut;
            ConnAlloc.template Get<SrcLevelTag>(ConnId) =
                UnitAlloc.template Get<LevelTag>(Self);
          }
        }
      }
      if (ConnAlloc.Size() > SizeBefore)
        NeedsResort = true;
    }
  }

  void DoStep(std::span<const float> Inputs) {
    DoForwardPass(Inputs);
    DoBackwardPass();
    DoUpdateUnitState();
    DoUpdateConnectionState();
    DoPruneUnits();
    DoPruneConnections();
    DoAddUnits();
    DoAddConnections();
    if (NeedsResort)
      SortConnectionsByLevel();
  }

  std::span<const float> GetOutput() const {
    const float *Base =
        &UnitAlloc.template Get<ActivationTag>(OutputRange.Begin);
    return {Base, OutputRange.Size()};
  }

  auto &GetConnAlloc() { return ConnAlloc; }
  auto &GetUnitAlloc() { return UnitAlloc; }

private:
  void RecomputeLevels() {
    size_t NumUnits = UnitAlloc.Size();
    size_t NumConns = ConnAlloc.Size();

    for (size_t I = NumInput; I < NumUnits; ++I)
      UnitAlloc.template Get<LevelTag>(I) = 0;

    if (NumConns == 0)
      return;

    uint32_t *InDegree = KahnAlloc.template GetArrayFor<InDegreeTag>();
    uint32_t *OutOffset = KahnAlloc.template GetArrayFor<OutOffsetTag>();
    uint32_t *WritePos = KahnAlloc.template GetArrayFor<KahnWritePosTag>();
    uint32_t *Frontier = KahnAlloc.template GetArrayFor<FrontierTag>();
    uint32_t *NextFrontier = KahnAlloc.template GetArrayFor<NextFrontierTag>();
    memset(InDegree, 0, NumUnits * sizeof(uint32_t));
    memset(OutOffset, 0, (NumUnits + 1) * sizeof(uint32_t));

    // Outgoing edge list reuses ConnAlloc's pre-allocated scratch
    size_t *OutEdges = ConnAlloc.PermutationScratch();

    // Count in-degree and out-degree per unit
    for (size_t C = 0; C < NumConns; ++C) {
      if (ConnAlloc.template Get<DeadTag>(C))
        continue;
      ++OutOffset[ConnAlloc.template Get<FromIdTag>(C)];
      ++InDegree[ConnAlloc.template Get<ToIdTag>(C)];
    }

    // Prefix sum: convert out-degrees in OutOffset to cumulative offsets
    uint32_t Sum = 0;
    for (size_t I = 0; I < NumUnits; ++I) {
      uint32_t Deg = OutOffset[I];
      OutOffset[I] = Sum;
      Sum += Deg;
    }
    OutOffset[NumUnits] = Sum;

    // Scatter connections into OutEdges
    memcpy(WritePos, OutOffset, NumUnits * sizeof(uint32_t));
    for (size_t C = 0; C < NumConns; ++C) {
      if (ConnAlloc.template Get<DeadTag>(C))
        continue;
      uint32_t From = ConnAlloc.template Get<FromIdTag>(C);
      OutEdges[WritePos[From]++] = C;
    }

    // Input units [0, NumInput) are always the level-0 frontier
    for (size_t I = 0; I < NumInput; ++I)
      Frontier[I] = static_cast<uint32_t>(I);
    uint32_t FrontierSize = static_cast<uint32_t>(NumInput);

    // Level-parallel Kahn's algorithm
    uint16_t CurrentLevel = 0;
    while (FrontierSize > 0) {
      uint32_t NextSize = 0;
      for (uint32_t F = 0; F < FrontierSize; ++F) {
        uint32_t U = Frontier[F];
        for (uint32_t E = OutOffset[U]; E < OutOffset[U + 1]; ++E) {
          size_t C = OutEdges[E];
          uint32_t To = ConnAlloc.template Get<ToIdTag>(C);
          if (--InDegree[To] == 0) {
            UnitAlloc.template Get<LevelTag>(To) =
                static_cast<uint16_t>(CurrentLevel + 1);
            NextFrontier[NextSize++] = To;
          }
        }
      }
      ++CurrentLevel;
      std::swap(Frontier, NextFrontier);
      FrontierSize = NextSize;
    }
  }

  void SortConnectionsByLevel() {
    RecomputeLevels();

    size_t N = ConnAlloc.Size();
    if (N == 0) {
      NumLevels = 0;
      return;
    }

    for (size_t C = 0; C < N; ++C) {
      auto From = ConnAlloc.template Get<FromIdTag>(C);
      ConnAlloc.template Get<SrcLevelTag>(C) =
          UnitAlloc.template Get<LevelTag>(From);
    }

    uint32_t Histogram[MaxLevels] = {};
    for (size_t C = 0; C < N; ++C)
      ++Histogram[ConnAlloc.template Get<SrcLevelTag>(C)];

    uint32_t Offset = 0;
    NumLevels = 0;
    for (uint32_t L = 0; L < MaxLevels; ++L) {
      Ranges[L].Begin = Offset;
      Offset += Histogram[L];
      Ranges[L].End = Offset;
      if (Histogram[L] > 0)
        NumLevels = L + 1;
    }

    uint32_t WritePos[MaxLevels];
    for (uint32_t L = 0; L < MaxLevels; ++L)
      WritePos[L] = Ranges[L].Begin;

    size_t *Perm = ConnAlloc.PermutationScratch();
    for (size_t C = 0; C < N; ++C) {
      uint16_t Lvl = ConnAlloc.template Get<SrcLevelTag>(C);
      Perm[WritePos[Lvl]++] = C;
    }

    ConnAlloc.Gather(N);
    NeedsResort = false;
  }

  size_t NumInput;
  size_t Step = 0;
  UnitRange OutputRange;
  UnitAllocator UnitAlloc;
  ConnAllocator ConnAlloc;
  GlobalState Globals;
  std::array<LevelRange, MaxLevels> Ranges{};
  uint16_t NumLevels = 0;
  bool NeedsResort = false;
  KahnScratchAllocator KahnAlloc;
};

} // namespace plastix

#endif // PLASTIX_PLASTIX_HPP
