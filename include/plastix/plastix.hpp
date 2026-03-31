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
//
// Network<Traits> is the central runtime object. It owns the unit and
// connection allocators and drives the per-step execution pipeline by calling
// into the policy structs defined in Traits. All policy dispatch is resolved
// at compile time — there are no virtual calls.
//
// Typical usage:
//   Network<MyTraits> net(InputDim, FullyConnected{HiddenDim}, FullyConnected{OutputDim});
//   net.DoStep(inputs);
//   auto out = net.GetOutput();
// ---------------------------------------------------------------------------

template <NetworkTraits Traits> class Network {
  using UnitAllocator = typename Traits::UnitAllocator;
  using ConnAllocator = typename Traits::ConnAllocator;
  using GlobalState = typename Traits::GlobalState;

public:
  // Variadic layer constructor. Each Builder is a callable that takes
  // (UnitAlloc, ConnAlloc, PrevLayerRange) and returns the UnitRange it
  // created. The fold expression chains them left-to-right, so each layer
  // receives the range produced by the previous one.
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

  // Convenience constructor for single-layer (input → output) networks.
  Network(size_t InputDim, size_t OutputDim = 1)
      : Network(InputDim, FullyConnected{OutputDim}) {}

  size_t GetStep() const { return Step; }

  // Runs one forward pass. Execution is two-phase:
  //
  //   Phase 1 — accumulate: iterate all connection pages in order. Pages are
  //     sorted by ToUnitIdx, so a destination change signals that the current
  //     accumulator is complete and should be flushed to CurrentActivation.
  //     FP::Map is called per slot to compute each connection's contribution.
  //
  //   Phase 2 — activate: iterate all non-input units and call
  //     FP::CalculateAndApply on their accumulated sums to apply the
  //     activation function (and any side effects such as error computation).
  //
  // Activations double-buffer between ActivationA and ActivationB each step
  // (see CurrentActivation/PreviousActivation). Input units are written into
  // PreviousActivation so they are immediately visible as sources.
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
      CurrentActivation(ToId) += FP::Map(UnitAlloc, ToId, Globals, Weight,
                                         PreviousActivation(FromId));
    }

    for (size_t I = NumInput; I < NumUnits; ++I)
      CurrentActivation(I) =
          FP::CalculateAndApply(UnitAlloc, I, Globals, CurrentActivation(I));

    ++Step;
  }

  // Runs one backward pass using the same two-phase structure as DoForwardPass
  // but in reverse signal direction: BP::Map accumulates into BackwardAccTag
  // on the *source* unit, then BP::CalculateAndApply fires per unit.
  // Compiled out entirely when BackwardPass = NoBackwardPass.
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

  // Updates per-unit state via the Map/Combine/Apply reduce pattern from
  // UpdateUnitPolicy. The UpdateAccTag field (in UnitStateAllocator) serves
  // as the per-unit accumulator; it is zero-reset after Apply.
  // Compiled out entirely when UpdateUnit = NoUpdateUnit.
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
        Acc = UP::Combine(Acc, UP::Map(UnitAlloc, ToId, FromId, Globals, Weight));
      }

      size_t NumUnits = UnitAlloc.Size();
      for (size_t I = 0; I < NumUnits; ++I) {
        auto &Acc = UnitAlloc.template Get<UpdateAccTag>(I);
        UP::Apply(UnitAlloc, I, Globals, Acc);
        Acc = Partial{};
      }
    }
  }

  // Updates connection weights (or other per-connection state) via two
  // separate sweeps of all pages — see UpdateConnPolicy for why two sweeps
  // are needed. Both sweeps pass PageId and SlotIdx so the policy can write
  // directly to the weight stored in the connection page.
  // Compiled out entirely when UpdateConn = NoUpdateConn.
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

  // Marks units for removal by writing ShouldPrune() results into PrunedTag.
  // DoPruneConnections() reads these flags to also remove affected connections.
  // Compiled out entirely when PruneUnit = NoPruneUnit.
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

  // Removes connections whose source or destination unit is pruned, or that
  // are independently flagged by PruneConnPolicy. For each page:
  //   1. If the destination unit is pruned, zero the page's Count and skip.
  //   2. Otherwise, build an Alive bitmask over slots, calling ShouldPrune
  //      and checking PrunedTag on source units as appropriate.
  //   3. Call CompactPage to move surviving slots to the front in-place, then
  //      update Count to the number of survivors.
  // The if constexpr guards remove dead-unit and/or dead-connection checks
  // when the corresponding policy is a noop.
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
            Remove = CP::ShouldPrune(UnitAlloc, ToId, FromId, ConnAlloc, C,
                                     Globals);

        if (Remove)
          ConnAlloc.template Get<DeadTag>(C) = true;
      }
    }
  }
  void DoAddUnits() {}
  void DoAddConnections() {}
  std::span<const size_t> GetSOI() const {return std::span<size_t>();}

  // Runs the full per-step pipeline in order:
  // ForwardPass → BackwardPass → UpdateUnit → UpdateConn → PruneUnits →
  // PruneConnections → AddUnits → AddConnections.
  // Noop-policy steps compile to nothing.
  void DoStep(std::span<const float> Inputs) {
    DoForwardPass(Inputs);
    DoBackwardPass();
    DoAddUnits();
    DoAddConnections();
    DoUpdateUnitState();
    DoUpdateConnectionState();
    DoPruneUnits();
    DoPruneConnections();
    DoAddUnits();
    DoAddConnections();
  }

  // Returns the output activations from the last completed forward pass.
  // Reads from PreviousActivation (the buffer written by the most recent
  // DoForwardPass call) because Step has already been incremented.
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
  // Ping-pong activation buffers. ActivationA and ActivationB alternate roles
  // each step: one holds the values being read (PreviousActivation, i.e. the
  // inputs to this step's forward pass) and the other is being written
  // (CurrentActivation). This ensures each unit sees stable source values
  // from the previous step even as downstream units are updated.
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
