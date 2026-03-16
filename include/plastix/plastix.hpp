#ifndef PLASTIX_PLASTIX_HPP
#define PLASTIX_PLASTIX_HPP

#include "plastix/alloc.hpp"
#include "plastix/traits.hpp"
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

// ---------------------------------------------------------------------------
// Network
// ---------------------------------------------------------------------------

template <NetworkTraits Traits> class Network {
  using UnitAllocator = typename Traits::UnitAllocator;
  using ConnAllocator = typename Traits::ConnAllocator;
  using GlobalState = typename Traits::GlobalState;

public:
  Network(size_t InputDim, size_t OutputDim = 1)
      : NumInput(InputDim), UnitAlloc(256), ConnAlloc(256) {
    size_t NumUnits = InputDim + OutputDim;
    for (auto _ : std::views::iota(size_t{0}, NumUnits)) {
      UnitAlloc.Allocate();
    }
    for (auto OutputIdx : std::views::iota(InputDim, NumUnits)) {
      auto PageId = ConnAlloc.Allocate();
      for (auto InputIdx : std::views::iota(size_t{0}, InputDim)) {
        if (!(InputIdx % ConnPageSlotSize) && InputIdx)
          PageId = ConnAlloc.Allocate();
        auto &Page = ConnAlloc.template Get<ConnPageMarker>(PageId);
        Page.Count += 1;
        Page.ToUnitIdx = OutputIdx;
        Page.Conn[InputIdx % ConnPageSlotSize] = {InputIdx, 1.0f};
      }
    }
  }

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
    if constexpr (std::is_same_v<typename Traits::BackwardPass,
                                 NoBackwardPass>)
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
