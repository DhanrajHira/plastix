#ifndef PLASTIX_LAYERS_HPP
#define PLASTIX_LAYERS_HPP

#include "plastix/conn.hpp"
#include <concepts>
#include <cstddef>
#include <cstdint>

namespace plastix {

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

} // namespace plastix

#endif // PLASTIX_LAYERS_HPP
