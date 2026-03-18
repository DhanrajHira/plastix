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
      { P::Map(U, Id, G, W, A) } -> std::convertible_to<float>;
      { P::CalculateAndApply(U, Id, G, Acc) } -> std::convertible_to<float>;
    };

template <typename P, typename UnitAlloc, typename Global>
concept UpdateUnitPolicy =
    requires(UnitAlloc &U, size_t DstId, size_t SrcId, Global &G, float W,
             typename P::Partial A, typename P::Partial B) {
      typename P::Partial;
      {
        P::Map(U, DstId, SrcId, G, W)
      } -> std::convertible_to<typename P::Partial>;
      { P::Combine(A, B) } -> std::convertible_to<typename P::Partial>;
      { P::Apply(U, DstId, G, A) } -> std::same_as<void>;
    };

template <typename P, typename UnitAlloc, typename ConnAlloc, typename Global>
concept UpdateConnPolicy =
    requires(UnitAlloc &U, size_t DstId, size_t SrcId, ConnAlloc &C,
             size_t PageId, size_t SlotIdx, Global &G) {
      {
        P::UpdateIncomingConnection(U, DstId, SrcId, C, PageId, SlotIdx, G)
      } -> std::same_as<void>;
      {
        P::UpdateOutgoingConnection(U, SrcId, DstId, C, PageId, SlotIdx, G)
      } -> std::same_as<void>;
    };

template <typename P, typename UnitAlloc, typename Global>
concept PruneUnitPolicy = requires(UnitAlloc &U, size_t Id, Global &G) {
  { P::ShouldPrune(U, Id, G) } -> std::convertible_to<bool>;
};

template <typename P, typename UnitAlloc, typename ConnAlloc, typename Global>
concept PruneConnPolicy =
    requires(UnitAlloc &U, size_t DstId, size_t SrcId, ConnAlloc &C,
             size_t PageId, size_t SlotIdx, Global &G) {
      {
        P::ShouldPrune(U, DstId, SrcId, C, PageId, SlotIdx, G)
      } -> std::convertible_to<bool>;
    };

// ---------------------------------------------------------------------------
// Default and noop policy implementations
// ---------------------------------------------------------------------------

struct DefaultForwardPass {
  static float Map(auto &, size_t, auto &, float Weight, float Activation) {
    return Weight * Activation;
  }
  static float CalculateAndApply(auto &, size_t, auto &, float Accumulated) {
    return Accumulated;
  }
};

// Sentinel noop policies — satisfy their concepts but DoX() methods compile
// out the entire loop body via if constexpr when these are detected.
struct NoBackwardPass {
  static float Map(auto &, size_t, auto &, float, float) { return 0.0f; }
  static float CalculateAndApply(auto &, size_t, auto &, float) { return 0.0f; }
};

struct NoUpdateUnit {
  using Partial = float;
  static Partial Map(auto &, size_t, size_t, auto &, float) { return 0.0f; }
  static Partial Combine(Partial A, Partial B) { return A + B; }
  static void Apply(auto &, size_t, auto &, Partial) {}
};

struct NoUpdateConn {
  static void UpdateIncomingConnection(auto &, size_t, size_t, auto &, size_t,
                                       size_t, auto &) {}
  static void UpdateOutgoingConnection(auto &, size_t, size_t, auto &, size_t,
                                       size_t, auto &) {}
};

struct NoPruneUnit {
  static bool ShouldPrune(auto &, size_t, auto &) { return false; }
};

struct NoPruneConn {
  static bool ShouldPrune(auto &, size_t, size_t, auto &, size_t, size_t,
                          auto &) {
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
    UpdateUnitPolicy<typename T::UpdateUnit, typename T::UnitAllocator,
                     typename T::GlobalState> &&
    UpdateConnPolicy<typename T::UpdateConn, typename T::UnitAllocator,
                     typename T::ConnAllocator, typename T::GlobalState> &&
    PruneUnitPolicy<typename T::PruneUnit, typename T::UnitAllocator,
                    typename T::GlobalState> &&
    PruneConnPolicy<typename T::PruneConn, typename T::UnitAllocator,
                    typename T::ConnAllocator, typename T::GlobalState>;

} // namespace plastix

#endif // PLASTIX_TRAITS_HPP
