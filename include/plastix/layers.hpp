#ifndef PLASTIX_LAYERS_HPP
#define PLASTIX_LAYERS_HPP

#include "plastix/conn.hpp"
#include "plastix/unit_state.hpp"
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

    float LayerX =
        static_cast<float>(UA.template Get<PositionTag>(PrevLayer.Begin).X) +
        1.0f;
    uint16_t NewLevel = UA.template Get<LevelTag>(PrevLayer.Begin) + 1;
    for (size_t I = 0; I < NumUnits; ++I) {
      auto Id = UA.Allocate();
      float Y = static_cast<float>(I) - static_cast<float>(NumUnits - 1) / 2.0f;
      UA.template Get<PositionTag>(Id) = {static_cast<_Float16>(LayerX),
                                          static_cast<_Float16>(Y), _Float16{0},
                                          0};
      UA.template Get<LevelTag>(Id) = NewLevel;
    }

    uint16_t SrcLevel = UA.template Get<LevelTag>(PrevLayer.Begin);
    for (size_t U = Begin; U < Begin + NumUnits; ++U) {
      for (size_t Src = PrevLayer.Begin; Src < PrevLayer.End; ++Src) {
        auto ConnId = CA.Allocate();
        CA.template Get<ToIdTag>(ConnId) = static_cast<uint32_t>(U);
        CA.template Get<FromIdTag>(ConnId) = static_cast<uint32_t>(Src);
        CA.template Get<WeightTag>(ConnId) = InitWeight;
        CA.template Get<SrcLevelTag>(ConnId) = SrcLevel;
      }
    }
    return {Begin, Begin + NumUnits};
  }
};

} // namespace plastix

#endif // PLASTIX_LAYERS_HPP
