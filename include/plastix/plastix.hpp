#ifndef PLASTIX_PLASTIX_HPP
#define PLASTIX_PLASTIX_HPP

#include "plastix/conn.hpp"
#include "plastix/layers.hpp"
#include "plastix/traits.hpp"
#include "plastix/unit_state.hpp"
#include <cstddef>
#include <span>
#include <type_traits>

namespace plastix {

/// Returns the library version as a string.
const char *version();

// ---------------------------------------------------------------------------
// Network
// ---------------------------------------------------------------------------

template <NetworkTraits Traits> class Network {
  using UnitAllocator = typename Traits::UnitAllocator;
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

    for (size_t I = 0; I < NumInput; ++I)
      PreviousActivation(I) = Inputs[I];

    size_t CurDst = static_cast<size_t>(-1);
    float Acc = 0.0f;

    for (size_t P = 0; P < ConnAlloc.Size(); ++P) {
      auto &Page = ConnAlloc.template Get<ConnPageMarker>(P);
      if (Page.ToUnitIdx != CurDst) {
        if (CurDst != static_cast<size_t>(-1))
          CurrentActivation(CurDst) =
              FP::CalculateAndApply(UnitAlloc, CurDst, Globals, Acc);
        CurDst = Page.ToUnitIdx;
        Acc = 0.0f;
      }
      for (size_t S = 0; S < Page.Count; ++S) {
        auto [SrcId, Weight] = Page.Conn[S];
        Acc += FP::Accumulate(UnitAlloc, CurDst, Globals, Weight,
                              PreviousActivation(SrcId));
      }
    }
    if (CurDst != static_cast<size_t>(-1))
      CurrentActivation(CurDst) =
          FP::CalculateAndApply(UnitAlloc, CurDst, Globals, Acc);

    ++Step;
  }

  void DoBackwardPass() {
    if constexpr (std::is_same_v<typename Traits::BackwardPass, NoBackwardPass>)
      return;
    else {
      using BP = typename Traits::BackwardPass;

      size_t NumUnits = UnitAlloc.Size();
      for (size_t I = 0; I < NumUnits; ++I)
        UnitAlloc.template Get<BackwardAccTag>(I) = 0.0f;

      for (size_t P = 0; P < ConnAlloc.Size(); ++P) {
        auto &Page = ConnAlloc.template Get<ConnPageMarker>(P);
        float DstAct = PreviousActivation(Page.ToUnitIdx);
        for (size_t S = 0; S < Page.Count; ++S) {
          auto [SrcId, Weight] = Page.Conn[S];
          UnitAlloc.template Get<BackwardAccTag>(SrcId) +=
              BP::Accumulate(UnitAlloc, SrcId, Globals, Weight, DstAct);
        }
      }

      for (size_t I = 0; I < NumUnits; ++I)
        BP::CalculateAndApply(UnitAlloc, I, Globals,
                              UnitAlloc.template Get<BackwardAccTag>(I));
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
          Acc = UP::Combine(Acc, UP::Map(UnitAlloc, Page.ToUnitIdx, SrcId,
                                         Globals, Weight));
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
                                       Page.Conn[S].first, Globals,
                                       Page.Conn[S].second);
      }

      for (size_t P = 0; P < ConnAlloc.Size(); ++P) {
        auto &Page = ConnAlloc.template Get<ConnPageMarker>(P);
        for (size_t S = 0; S < Page.Count; ++S)
          UP::UpdateOutgoingConnection(UnitAlloc, Page.Conn[S].first,
                                       Page.ToUnitIdx, Globals,
                                       Page.Conn[S].second);
      }
    }
  }
  void DoPruneUnits() {
    if constexpr (std::is_same_v<typename Traits::PruneUnit, NoPruneUnit>)
      return;
  }
  void DoPruneConnections() {
    if constexpr (std::is_same_v<typename Traits::PruneConn, NoPruneConn>)
      return;
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
        Step % 2 == 0
            ? &UnitAlloc.template Get<ActivationATag>(OutputRange.Begin)
            : &UnitAlloc.template Get<ActivationBTag>(OutputRange.Begin);
    return {Base, OutputRange.Size()};
  }

  auto &GetConnAlloc() { return ConnAlloc; }
  auto &GetUnitAlloc() { return UnitAlloc; }

private:
  float &CurrentActivation(size_t Id) {
    if (Step % 2 == 0)
      return UnitAlloc.template Get<ActivationBTag>(Id);
    return UnitAlloc.template Get<ActivationATag>(Id);
  }

  float &PreviousActivation(size_t Id) {
    if (Step % 2 == 0)
      return UnitAlloc.template Get<ActivationATag>(Id);
    return UnitAlloc.template Get<ActivationBTag>(Id);
  }

  size_t NumInput;
  size_t Step = 0;
  UnitRange OutputRange;
  UnitAllocator UnitAlloc;
  ConnAllocator ConnAlloc;
  GlobalState Globals;
};

} // namespace plastix

#endif // PLASTIX_PLASTIX_HPP
