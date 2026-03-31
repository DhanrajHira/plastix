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
  using UnitAllocator = typename Traits::UnitAllocator;
  using ConnAllocator = typename Traits::ConnAllocator;
  using GlobalState = typename Traits::GlobalState;

public:
  template <typename... Builders>
    requires(sizeof...(Builders) > 0 &&
             (LayerBuilder<Builders, UnitAllocator, ConnAllocator> && ...))
  Network(size_t InputDim, Builders... Layers)
      : NumInput(InputDim), UnitAlloc(256), ConnAlloc(256) {
    for (size_t I = 0; I < InputDim; ++I) {
      auto Id = UnitAlloc.Allocate();
      float Y = static_cast<float>(I) - static_cast<float>(InputDim - 1) / 2.0f;
      UnitAlloc.template Get<PositionTag>(Id) = {
          _Float16{0}, static_cast<_Float16>(Y), _Float16{0}, 0};
    }
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

    size_t NumUnits = UnitAlloc.Size();
    for (size_t I = NumInput; I < NumUnits; ++I)
      CurrentActivation(I) = 0.0f;

    for (size_t C = 0; C < ConnAlloc.Size(); ++C) {
      if (ConnAlloc.template Get<DeadTag>(C))
        continue;
      auto ToId = ConnAlloc.template Get<ToIdTag>(C);
      auto FromId = ConnAlloc.template Get<FromIdTag>(C);
      auto Weight = ConnAlloc.template Get<WeightTag>(C);
      CurrentActivation(ToId) +=
          FP::Map(UnitAlloc, ToId, Globals, Weight, PreviousActivation(FromId));
    }

    for (size_t I = NumInput; I < NumUnits; ++I)
      CurrentActivation(I) =
          FP::CalculateAndApply(UnitAlloc, I, Globals, CurrentActivation(I));

    ++Step;
  }

  void DoBackwardPass() {
    if constexpr (std::is_same_v<typename Traits::BackwardPass, NoBackwardPass>)
      return;
    else {
      using BP = typename Traits::BackwardPass;

      for (size_t C = 0; C < ConnAlloc.Size(); ++C) {
        if (ConnAlloc.template Get<DeadTag>(C))
          continue;
        auto ToId = ConnAlloc.template Get<ToIdTag>(C);
        auto FromId = ConnAlloc.template Get<FromIdTag>(C);
        auto Weight = ConnAlloc.template Get<WeightTag>(C);
        float DstAct = PreviousActivation(ToId);
        UnitAlloc.template Get<BackwardAccTag>(FromId) +=
            BP::Map(UnitAlloc, FromId, Globals, Weight, DstAct);
      }

      size_t NumUnits = UnitAlloc.Size();
      for (size_t I = 0; I < NumUnits; ++I) {
        BP::CalculateAndApply(UnitAlloc, I, Globals,
                              UnitAlloc.template Get<BackwardAccTag>(I));
        UnitAlloc.template Get<BackwardAccTag>(I) = 0.0f;
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
          }
        }
      }
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
