#ifndef PLASTIX_TRAITS_HPP
#define PLASTIX_TRAITS_HPP

#include <concepts>
#include <cstddef>

namespace plastix {

// ---------------------------------------------------------------------------
// Policy concepts
// ---------------------------------------------------------------------------

template <typename P, typename UnitAlloc, typename Global>
concept PassPolicy =
    requires(UnitAlloc &U, size_t Id, Global &G, float W, float A, float Acc) {
      { P::Accumulate(U, Id, G, W, A) } -> std::convertible_to<float>;
      { P::CalculateAndApply(U, Id, G, Acc) } -> std::convertible_to<float>;
    };

template <typename P, typename UnitAlloc, typename ConnAlloc, typename Global>
concept UpdatePolicy =
    requires(UnitAlloc &U, size_t UId, ConnAlloc &C, size_t CId, Global &G,
             typename P::Partial A, typename P::Partial B) {
      typename P::Partial;
      { P::Map(U, UId, C, CId, G) } -> std::same_as<typename P::Partial>;
      { P::Combine(A, B) } -> std::same_as<typename P::Partial>;
      { P::Apply(U, UId, G, A) } -> std::same_as<void>;
    };

template <typename P, typename UnitAlloc, typename Global>
concept PruneUnitPolicy = requires(UnitAlloc &U, size_t Id, Global &G) {
  { P::ShouldPrune(U, Id, G) } -> std::convertible_to<bool>;
};

template <typename P, typename UnitAlloc, typename ConnAlloc, typename Global>
concept PruneConnPolicy =
    requires(UnitAlloc &U, size_t UId, ConnAlloc &C, size_t CId, Global &G) {
      {
        P::ShouldPruneIncoming(U, UId, C, CId, G)
      } -> std::convertible_to<bool>;
      {
        P::ShouldPruneOutgoing(U, UId, C, CId, G)
      } -> std::convertible_to<bool>;
    };

// ---------------------------------------------------------------------------
// Default and noop policy implementations
// ---------------------------------------------------------------------------

struct DefaultForwardPass {
  static float Accumulate(auto &, size_t, auto &, float Weight,
                          float Activation) {
    return Weight * Activation;
  }
  static float CalculateAndApply(auto &, size_t, auto &, float Accumulated) {
    return Accumulated;
  }
};

// Sentinel noop policies — satisfy their concepts but DoX() methods compile
// out the entire loop body via if constexpr when these are detected.
struct NoBackwardPass {
  static float Accumulate(auto &, size_t, auto &, float, float) {
    return 0.0f;
  }
  static float CalculateAndApply(auto &, size_t, auto &, float) {
    return 0.0f;
  }
};

struct NoUpdateUnit {
  struct Partial {
    float Sum;
  };
  static Partial Map(auto &, size_t, auto &, size_t, auto &) { return {0.0f}; }
  static Partial Combine(Partial A, Partial B) { return {A.Sum + B.Sum}; }
  static void Apply(auto &, size_t, auto &, Partial) {}
};

struct NoUpdateConn {
  struct Partial {
    float Sum;
  };
  static Partial Map(auto &, size_t, auto &, size_t, auto &) { return {0.0f}; }
  static Partial Combine(Partial A, Partial B) { return {A.Sum + B.Sum}; }
  static void Apply(auto &, size_t, auto &, Partial) {}
};

struct NoPruneUnit {
  static bool ShouldPrune(auto &, size_t, auto &) { return false; }
};

struct NoPruneConn {
  static bool ShouldPruneIncoming(auto &, size_t, auto &, size_t, auto &) {
    return false;
  }
  static bool ShouldPruneOutgoing(auto &, size_t, auto &, size_t, auto &) {
    return false;
  }
};

struct EmptyGlobalState {};

// ---------------------------------------------------------------------------
// Default traits base — inherit and override individual policies as needed
// ---------------------------------------------------------------------------

template <typename UnitAlloc, typename ConnAlloc,
          typename Global = EmptyGlobalState>
struct DefaultNetworkTraits {
  using UnitAllocator = UnitAlloc;
  using ConnAllocator = ConnAlloc;
  using GlobalState = Global;
  using ForwardPass = DefaultForwardPass;
  using BackwardPass = NoBackwardPass;
  using UpdateUnit = NoUpdateUnit;
  using UpdateConn = NoUpdateConn;
  using PruneUnit = NoPruneUnit;
  using PruneConn = NoPruneConn;
};

// ---------------------------------------------------------------------------
// NetworkTraits concept — validates that all policies satisfy their concepts
// ---------------------------------------------------------------------------

template <typename T>
concept NetworkTraits =
    requires {
      typename T::UnitAllocator;
      typename T::ConnAllocator;
      typename T::GlobalState;
      typename T::ForwardPass;
      typename T::BackwardPass;
      typename T::UpdateUnit;
      typename T::UpdateConn;
      typename T::PruneUnit;
      typename T::PruneConn;
    } &&
    PassPolicy<typename T::ForwardPass, typename T::UnitAllocator,
               typename T::GlobalState> &&
    PassPolicy<typename T::BackwardPass, typename T::UnitAllocator,
               typename T::GlobalState> &&
    UpdatePolicy<typename T::UpdateUnit, typename T::UnitAllocator,
                 typename T::ConnAllocator, typename T::GlobalState> &&
    UpdatePolicy<typename T::UpdateConn, typename T::UnitAllocator,
                 typename T::ConnAllocator, typename T::GlobalState> &&
    PruneUnitPolicy<typename T::PruneUnit, typename T::UnitAllocator,
                    typename T::GlobalState> &&
    PruneConnPolicy<typename T::PruneConn, typename T::UnitAllocator,
                    typename T::ConnAllocator, typename T::GlobalState>;

} // namespace plastix

#endif // PLASTIX_TRAITS_HPP
