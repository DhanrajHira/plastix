#ifndef PLASTIX_PLASTIX_HPP
#define PLASTIX_PLASTIX_HPP

#include "plastix/conn.hpp"
#include "plastix/layers.hpp"
#include "plastix/traits.hpp"
#include "plastix/unit_state.hpp"
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

namespace plastix {

/// Returns the library version as a string.
const char *version();

// ---------------------------------------------------------------------------
// Network
// ---------------------------------------------------------------------------

template <NetworkTraits Traits> class Network {
  using UnitAllocator =
      MakeUnitAllocator<typename Traits::ForwardPass::Accumulator,
                        typename Traits::BackwardPass::Accumulator,
                        typename Traits::UpdateUnit::Partial>;
  using ConnAllocator = typename Traits::ConnAllocator;
  using GlobalState = typename Traits::GlobalState;

public:
  template <typename... Builders>
    requires(sizeof...(Builders) > 0 &&
             (LayerBuilder<Builders, UnitAllocator, ConnAllocator> && ...))
  Network(size_t InputDim, Builders... Layers)
      : NumInput(InputDim), UnitAlloc(256), ConnAlloc(256) {
    for (size_t I = 0; I < InputDim; ++I)
      UnitAlloc.Allocate();
    UnitRange Prev{0, InputDim};
    ((Prev = Layers(UnitAlloc, ConnAlloc, Prev)), ...);
    OutputRange = Prev;
  }

  Network(size_t InputDim, size_t OutputDim = 1)
      : Network(InputDim, FullyConnected{OutputDim}) {}

  size_t GetStep() const { return Step; }

  void DoForwardPass(std::span<const float> Inputs) {
    using FP = typename Traits::ForwardPass;
    using Acc = typename FP::Accumulator;

    for (size_t I = 0; I < NumInput; ++I)
      UnitAlloc.template Get<ActivationTag>(I) = Inputs[I];

    size_t NumUnits = UnitAlloc.Size();
    for (size_t I = NumInput; I < NumUnits; ++I)
      UnitAlloc.template Get<ForwardAccTag>(I) = Acc{};

    for (size_t P = 0; P < ConnAlloc.Size(); ++P) {
      auto &Page = ConnAlloc.template Get<ConnPageMarker>(P);
      auto &UAcc = UnitAlloc.template Get<ForwardAccTag>(Page.ToUnitIdx);
      for (size_t S = 0; S < Page.Count; ++S) {
        UAcc = FP::Combine(UAcc, FP::Map(UnitAlloc, Page.ToUnitIdx,
                                         Page.Conn[S].first, ConnAlloc, P, S,
                                         Globals));
      }
    }

    for (size_t I = NumInput; I < NumUnits; ++I) {
      auto &UAcc = UnitAlloc.template Get<ForwardAccTag>(I);
      FP::Apply(UnitAlloc, I, Globals, UAcc);
      UAcc = Acc{};
    }

    ++Step;
  }

  void DoBackwardPass() {
    if constexpr (std::is_same_v<typename Traits::BackwardPass, NoBackwardPass>)
      return;
    else {
      using BP = typename Traits::BackwardPass;
      using Acc = typename BP::Accumulator;

      size_t NumUnits = UnitAlloc.Size();
      for (size_t I = 0; I < NumUnits; ++I)
        UnitAlloc.template Get<BackwardAccTag>(I) = Acc{};

      for (size_t P = 0; P < ConnAlloc.Size(); ++P) {
        auto &Page = ConnAlloc.template Get<ConnPageMarker>(P);
        for (size_t S = 0; S < Page.Count; ++S) {
          auto &UAcc =
              UnitAlloc.template Get<BackwardAccTag>(Page.Conn[S].first);
          UAcc = BP::Combine(UAcc,
                             BP::Map(UnitAlloc, Page.Conn[S].first,
                                     Page.ToUnitIdx, ConnAlloc, P, S, Globals));
        }
      }

      for (size_t I = 0; I < NumUnits; ++I) {
        auto &UAcc = UnitAlloc.template Get<BackwardAccTag>(I);
        BP::Apply(UnitAlloc, I, Globals, UAcc);
        UAcc = Acc{};
      }
    }
  }

  void DoUpdateUnitState() {
    if constexpr (std::is_same_v<typename Traits::UpdateUnit, NoUpdateUnit>)
      return;
    else {
      using UP = typename Traits::UpdateUnit;
      using Partial = typename UP::Partial;

      for (size_t P = 0; P < ConnAlloc.Size(); ++P) {
        auto &Page = ConnAlloc.template Get<ConnPageMarker>(P);
        auto &Acc = UnitAlloc.template Get<UpdateAccTag>(Page.ToUnitIdx);
        for (size_t S = 0; S < Page.Count; ++S) {
          auto [SrcId, Weight] = Page.Conn[S];
          Acc = UP::Combine(
              Acc, UP::Map(UnitAlloc, Page.ToUnitIdx, SrcId, Globals, Weight));
        }
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

      for (size_t P = 0; P < ConnAlloc.Size(); ++P) {
        auto &Page = ConnAlloc.template Get<ConnPageMarker>(P);
        for (size_t S = 0; S < Page.Count; ++S)
          UP::UpdateIncomingConnection(UnitAlloc, Page.ToUnitIdx,
                                       Page.Conn[S].first, ConnAlloc, P, S,
                                       Globals);
      }

      for (size_t P = 0; P < ConnAlloc.Size(); ++P) {
        auto &Page = ConnAlloc.template Get<ConnPageMarker>(P);
        for (size_t S = 0; S < Page.Count; ++S)
          UP::UpdateOutgoingConnection(UnitAlloc, Page.Conn[S].first,
                                       Page.ToUnitIdx, ConnAlloc, P, S,
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

      for (size_t P = 0; P < ConnAlloc.Size(); ++P) {
        auto &Page = ConnAlloc.template Get<ConnPageMarker>(P);
        if (Page.Count == 0)
          continue;

        if constexpr (HasUnitPrune) {
          if (UnitAlloc.template Get<PrunedTag>(Page.ToUnitIdx)) {
            Page.Count = 0;
            continue;
          }
        }

        uint32_t Alive = 0;
        for (size_t S = 0; S < Page.Count; ++S) {
          bool Remove = false;
          if constexpr (HasUnitPrune)
            Remove = UnitAlloc.template Get<PrunedTag>(Page.Conn[S].first);
          if constexpr (HasConnPrune)
            Remove = Remove || CP::ShouldPrune(UnitAlloc, Page.ToUnitIdx,
                                               Page.Conn[S].first, ConnAlloc, P,
                                               S, Globals);
          if (!Remove)
            Alive |= (1u << S);
        }
        Page.Count = std::popcount(Alive);
        ConnAlloc.CompactPage(P, Alive);
      }
    }
  }
  void DoAddUnits() {}
  void DoAddConnections() {}

  void DoStep(std::span<const float> Inputs) {
    DoForwardPass(Inputs);
    DoBackwardPass();
    DoUpdateUnitState();
    DoUpdateConnectionState();
    DoPruneUnits();
    DoPruneConnections();
    DoAddUnits();
    DoAddConnections();
  }

  std::span<const float> GetOutput() const {
    const float *Base =
        &UnitAlloc.template Get<ActivationTag>(OutputRange.Begin);
    return {Base, OutputRange.Size()};
  }

  auto &GetConnAlloc() { return ConnAlloc; }
  auto &GetUnitAlloc() { return UnitAlloc; }

private:
  size_t NumInput;
  size_t Step = 0;
  UnitRange OutputRange;
  UnitAllocator UnitAlloc;
  ConnAllocator ConnAlloc;
  GlobalState Globals;
};

} // namespace plastix

#endif // PLASTIX_PLASTIX_HPP
