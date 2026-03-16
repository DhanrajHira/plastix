#ifndef PLASTIX_PLASTIX_HPP
#define PLASTIX_PLASTIX_HPP

#include "plastix/alloc.hpp"
#include "plastix/traits.hpp"
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <span>
#include <type_traits>

namespace plastix {

#define PLASTIX_SOA_MODE_TAGS
#include "plastix/soa.hpp"
#include "unit_state.inc"

#define PLASTIX_SOA_MODE_ALLOC
#include "plastix/soa.hpp"
#include "unit_state.inc"

#define PLASTIX_SOA_MODE_HANDLE
#include "plastix/soa.hpp"
#include "unit_state.inc"

/// Returns the library version as a string.
const char *version();

constexpr static size_t ConnPageSlotSize = 7;

struct ConnPage {
  uint32_t ToUnitIdx;
  uint32_t Count;
  std::array<std::pair<uint32_t, float>, ConnPageSlotSize> Conn;
};

struct ConnectionState {};
struct ConnPageMarker {};

using ConnStateAllocator =
    alloc::SOAAllocator<ConnectionState,
                        alloc::SOAField<ConnPageMarker, ConnPage>>;

struct UnitRange {
  size_t Begin;
  size_t End;
  size_t Size() const { return End - Begin; }
};

template <typename B, typename UA, typename CA>
concept LayerBuilder = requires(B Builder, UA &U, CA &C, UnitRange R) {
  { Builder(U, C, R) } -> std::same_as<UnitRange>;
};

struct FullyConnected {
  size_t NumUnits;
  float InitWeight = 1.0f;

  template <typename UnitAlloc, typename ConnAlloc>
  UnitRange operator()(UnitAlloc &UA, ConnAlloc &CA,
                       UnitRange PrevLayer) const {
    size_t Begin = UA.Size();
    for (size_t I = 0; I < NumUnits; ++I)
      UA.Allocate();

    for (size_t U = Begin; U < Begin + NumUnits; ++U) {
      size_t SlotIdx = 0;
      auto PageId = CA.Allocate();
      for (size_t Src = PrevLayer.Begin; Src < PrevLayer.End; ++Src) {
        if (SlotIdx == ConnPageSlotSize) {
          PageId = CA.Allocate();
          SlotIdx = 0;
        }
        auto &Page = CA.template Get<ConnPageMarker>(PageId);
        Page.ToUnitIdx = U;
        Page.Count = SlotIdx + 1;
        Page.Conn[SlotIdx] = {static_cast<uint32_t>(Src), InitWeight};
        ++SlotIdx;
      }
    }
    return {Begin, Begin + NumUnits};
  }
};

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
  }

  Network(size_t InputDim, size_t OutputDim = 1)
      : Network(InputDim, FullyConnected{OutputDim}) {}

  size_t GetStep() const { return Step; }

  void DoForwardPass(std::span<const float> Inputs) {
    using FP = typename Traits::ForwardPass;

    for (size_t I = 0; I < NumInput; ++I) {
      CurrentActivation(I) = Inputs[I];
      PreviousActivation(I) = Inputs[I];
    }

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
  }
  void DoUpdateConnectionState() {
    if constexpr (std::is_same_v<typename Traits::UpdateConn, NoUpdateConn>)
      return;
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
  UnitAllocator UnitAlloc;
  ConnAllocator ConnAlloc;
  GlobalState Globals;
};

} // namespace plastix

#endif // PLASTIX_PLASTIX_HPP
