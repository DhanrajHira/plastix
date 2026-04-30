#ifndef PLASTIX_LAYERS_HPP
#define PLASTIX_LAYERS_HPP

#include "plastix/conn.hpp"
#include "plastix/macros.hpp"
#include "plastix/random.hpp"
#include "plastix/unit_state.hpp"
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <utility>

namespace plastix {

// Hand-rolled iota iterator. Replaces std::views::iota so that UnitRange::Ids()
// is callable from device code (libstdc++'s ranges machinery is not yet
// device-compatible) and avoids dragging <ranges> into translation units that
// nvcc compiles. Models random_access_iterator just enough for range-for
// loops and indexed access on both host and device.
struct IotaIterator {
  using value_type = size_t;
  using difference_type = std::ptrdiff_t;
  using iterator_category = std::random_access_iterator_tag;

  size_t Value = 0;

  PLASTIX_HD IotaIterator() = default;
  PLASTIX_HD explicit IotaIterator(size_t V) : Value(V) {}

  PLASTIX_HD size_t operator*() const { return Value; }
  PLASTIX_HD IotaIterator &operator++() {
    ++Value;
    return *this;
  }
  PLASTIX_HD IotaIterator operator++(int) {
    IotaIterator Tmp = *this;
    ++Value;
    return Tmp;
  }
  PLASTIX_HD IotaIterator &operator--() {
    --Value;
    return *this;
  }
  PLASTIX_HD IotaIterator &operator+=(difference_type N) {
    Value += static_cast<size_t>(N);
    return *this;
  }
  PLASTIX_HD IotaIterator &operator-=(difference_type N) {
    Value -= static_cast<size_t>(N);
    return *this;
  }
  PLASTIX_HD friend IotaIterator operator+(IotaIterator I, difference_type N) {
    return IotaIterator{I.Value + static_cast<size_t>(N)};
  }
  PLASTIX_HD friend IotaIterator operator-(IotaIterator I, difference_type N) {
    return IotaIterator{I.Value - static_cast<size_t>(N)};
  }
  PLASTIX_HD friend difference_type operator-(IotaIterator A, IotaIterator B) {
    return static_cast<difference_type>(A.Value) -
           static_cast<difference_type>(B.Value);
  }
  PLASTIX_HD size_t operator[](difference_type N) const {
    return Value + static_cast<size_t>(N);
  }
  PLASTIX_HD friend bool operator==(IotaIterator A, IotaIterator B) {
    return A.Value == B.Value;
  }
  PLASTIX_HD friend auto operator<=>(IotaIterator A, IotaIterator B) {
    return A.Value <=> B.Value;
  }
};

struct IotaRange {
  IotaIterator B;
  IotaIterator E;
  PLASTIX_HD IotaIterator begin() const { return B; }
  PLASTIX_HD IotaIterator end() const { return E; }
};

struct UnitRange {
  size_t Begin;
  size_t End;
  PLASTIX_HD UnitRange() = default;
  PLASTIX_HD UnitRange(size_t Begin, size_t End) : Begin(Begin), End(End) {}
  PLASTIX_HD UnitRange(std::pair<size_t, size_t> P)
      : Begin(P.first), End(P.second) {}
  PLASTIX_HD size_t Size() const { return End - Begin; }
  PLASTIX_HD IotaRange Ids() const {
    return IotaRange{IotaIterator{Begin}, IotaIterator{End}};
  }
};

template <typename B, typename UA, typename CA>
concept LayerBuilder = requires(B Builder, UA &U, CA &C, UnitRange R) {
  { Builder(U, C, R) } -> std::same_as<UnitRange>;
};

struct NoUnitInit {
  PLASTIX_HD void operator()(auto &, auto) const {}
};

struct NoConnInit {
  PLASTIX_HD void operator()(auto &, auto) const {}
};

// Counter-based RNG initializer: draws a deterministic uniform sample keyed on
// the connection id. Bit-identical between host and device (Philox is pure)
// and stateless, so multiple builders sharing a seed don't interfere.
struct RandomUniformWeight {
  uint64_t Seed;
  float Min;
  float Max;

  PLASTIX_HD RandomUniformWeight(uint64_t Seed = 0, float Min = -1.0f,
                                 float Max = 1.0f)
      : Seed(Seed), Min(Min), Max(Max) {}

  PLASTIX_HD void operator()(auto &CA, auto Id) const {
    GetWeight(CA, Id) = UniformReal(Seed, static_cast<uint64_t>(Id), Min, Max);
  }
};

template <typename ConnInit = NoConnInit, typename UnitInit = NoUnitInit>
struct FullyConnected {
  size_t NumUnits;
  ConnInit InitConn = {};
  UnitInit InitUnit = {};

  template <typename UnitAlloc, typename ConnAlloc>
  PLASTIX_HOST UnitRange operator()(UnitAlloc &UA, ConnAlloc &CA,
                                    UnitRange PrevLayer) const {
    uint16_t NewLevel = GetLevel(UA, PrevLayer.Begin) + 1;
    UnitRange Units = UA.AllocateMany(NumUnits);
    for (auto Id : Units.Ids()) {
      GetLevel(UA, Id) = NewLevel;
      InitUnit(UA, Id);
    }

    uint16_t SrcLevel = GetLevel(UA, PrevLayer.Begin);
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
