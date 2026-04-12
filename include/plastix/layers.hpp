#ifndef PLASTIX_LAYERS_HPP
#define PLASTIX_LAYERS_HPP

#include "plastix/conn.hpp"
#include "plastix/unit_state.hpp"
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <random>
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

struct RandomUniformWeight {
  RandomUniformWeight(uint32_t Seed = 0, float Min = -1.0f, float Max = 1.0f)
      : Rng(Seed), Dist(Min, Max) {}

  void operator()(auto &CA, auto Id) const {
    GetField<WeightTag>(CA, Id) = Dist(Rng);
  }

private:
  mutable std::mt19937 Rng;
  mutable std::uniform_real_distribution<float> Dist;
};

template <typename ConnInit = NoConnInit, typename UnitInit = NoUnitInit>
struct FullyConnected {
  size_t NumUnits;
  ConnInit InitConn = {};
  UnitInit InitUnit = {};

  template <typename UnitAlloc, typename ConnAlloc>
  UnitRange operator()(UnitAlloc &UA, ConnAlloc &CA,
                       UnitRange PrevLayer) const {
    uint16_t NewLevel = GetField<LevelTag>(UA, PrevLayer.Begin) + 1;
    UnitRange Units = UA.AllocateMany(NumUnits);
    for (auto Id : Units.Ids()) {
      GetField<LevelTag>(UA, Id) = NewLevel;
      InitUnit(UA, Id);
    }

    uint16_t SrcLevel = GetField<LevelTag>(UA, PrevLayer.Begin);
    for (auto U : Units.Ids()) {
      for (auto Src : PrevLayer.Ids()) {
        auto ConnId = CA.Allocate();
        GetField<ToIdTag>(CA, ConnId) = static_cast<uint32_t>(U);
        GetField<FromIdTag>(CA, ConnId) = static_cast<uint32_t>(Src);
        GetField<SrcLevelTag>(CA, ConnId) = SrcLevel;
        InitConn(CA, ConnId);
      }
    }
    return Units;
  }
};

} // namespace plastix

#endif // PLASTIX_LAYERS_HPP
