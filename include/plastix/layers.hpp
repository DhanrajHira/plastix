#ifndef PLASTIX_LAYERS_HPP
#define PLASTIX_LAYERS_HPP

#include "plastix/conn.hpp"
#include "plastix/unit_state.hpp"
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <utility>

namespace plastix {

struct UnitRange {
  size_t Begin;
  size_t End;
  UnitRange() = default;
  UnitRange(size_t Begin, size_t End) : Begin(Begin), End(End) {}
  UnitRange(std::pair<size_t, size_t> P) : Begin(P.first), End(P.second) {}
  size_t Size() const { return End - Begin; }
  auto Ids() const { return std::views::iota(Begin, End); }
};

template <typename B, typename UA, typename CA>
concept LayerBuilder = requires(B Builder, UA &U, CA &C, UnitRange R) {
  { Builder(U, C, R) } -> std::same_as<UnitRange>;
};

struct NoUnitInit {
  void operator()(auto &, auto) const {}
};

struct NoConnInit {
  void operator()(auto &, auto) const {}
};

template <typename UnitInit = NoUnitInit, typename ConnInit = NoConnInit>
struct FullyConnected {
  size_t NumUnits;
  UnitInit InitUnit = {};
  ConnInit InitConn = {};

  template <typename UnitAlloc, typename ConnAlloc>
  UnitRange operator()(UnitAlloc &UA, ConnAlloc &CA,
                       UnitRange PrevLayer) const {
    uint16_t NewLevel = UA.template Get<LevelTag>(PrevLayer.Begin) + 1;
    UnitRange Units = UA.AllocateMany(NumUnits);
    for (auto Id : Units.Ids()) {
      UA.template Get<LevelTag>(Id) = NewLevel;
      InitUnit(UA, Id);
    }

    uint16_t SrcLevel = UA.template Get<LevelTag>(PrevLayer.Begin);
    for (auto U : Units.Ids()) {
      for (auto Src : PrevLayer.Ids()) {
        auto ConnId = CA.Allocate();
        CA.template Get<ToIdTag>(ConnId) = static_cast<uint32_t>(U);
        CA.template Get<FromIdTag>(ConnId) = static_cast<uint32_t>(Src);
        CA.template Get<SrcLevelTag>(ConnId) = SrcLevel;
        InitConn(CA, ConnId);
      }
    }
    return Units;
  }
};

} // namespace plastix

#endif // PLASTIX_LAYERS_HPP
