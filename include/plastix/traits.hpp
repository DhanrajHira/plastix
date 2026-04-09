#ifndef PLASTIX_TRAITS_HPP
#define PLASTIX_TRAITS_HPP

#include "plastix/unit_state.hpp"

#include <concepts>
#include <cstddef>

namespace plastix {

// Forward declaration for use in default policies.
struct ConnPageMarker;

// ---------------------------------------------------------------------------
// Policy concepts
// ---------------------------------------------------------------------------

template <typename P, typename UnitAlloc, typename ConnAlloc, typename Global>
concept PassPolicy =
    std::default_initializable<typename P::Accumulator> &&
    requires(UnitAlloc &U, size_t A, size_t B, ConnAlloc &C, size_t PageId,
             size_t SlotIdx, Global &G, typename P::Accumulator AccA,
             typename P::Accumulator AccB) {
      typename P::Accumulator;
      {
        P::Map(U, A, B, C, PageId, SlotIdx, G)
      } -> std::convertible_to<typename P::Accumulator>;
      {
        P::Combine(AccA, AccB)
      } -> std::convertible_to<typename P::Accumulator>;
      { P::Apply(U, A, G, AccA) } -> std::same_as<void>;
    };

template <typename P, typename UnitAlloc, typename Global>
concept UpdateUnitPolicy = requires(UnitAlloc &U, size_t Id, Global &G) {
  { P::Update(U, Id, G) } -> std::same_as<void>;
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
  using Accumulator = float;
  static float Map(auto &U, size_t, size_t SrcId, auto &C, size_t PageId,
                   size_t SlotIdx, auto &) {
    return C.template Get<ConnPageMarker>(PageId).GetSlot(SlotIdx).second *
           U.template Get<ActivationTag>(SrcId);
  }
  static float Combine(float A, float B) { return A + B; }
  static void Apply(auto &U, size_t Id, auto &, float Accumulated) {
    U.template Get<ActivationTag>(Id) = Accumulated;
  }
};

// Sentinel noop policies — satisfy their concepts but DoX() methods compile
// out the entire loop body via if constexpr when these are detected.
struct NoBackwardPass {
  using Accumulator = float;
  static float Map(auto &, size_t, size_t, auto &, size_t, size_t, auto &) {
    return 0.0f;
  }
  static float Combine(float A, float B) { return A + B; }
  static void Apply(auto &, size_t, auto &, float) {}
};

struct NoUpdateUnit {
  static void Update(auto &, size_t, auto &) {}
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

template <typename ConnAlloc, typename Global = EmptyGlobalState>
struct DefaultNetworkTraits {
  using ConnAllocator = ConnAlloc;
  using GlobalState = Global;
  using ForwardPass = DefaultForwardPass;
  using BackwardPass = NoBackwardPass;
  using UpdateUnit = NoUpdateUnit;
  using UpdateConn = NoUpdateConn;
  using PruneUnit = NoPruneUnit;
  using PruneConn = NoPruneConn;
  using ExtraUnitFields = UnitFieldList<>;
};

// ---------------------------------------------------------------------------
// NetworkTraits concept — validates that all policies satisfy their concepts
// ---------------------------------------------------------------------------

// Helper: resolve the unit allocator for a given traits type.
template <typename T>
using UnitAllocFor =
    MakeUnitAllocatorFrom<typename T::ForwardPass::Accumulator,
                          typename T::BackwardPass::Accumulator,
                          typename T::ExtraUnitFields>;

template <typename T>
concept NetworkTraits =
    requires {
      typename T::ConnAllocator;
      typename T::GlobalState;
      typename T::ForwardPass;
      typename T::BackwardPass;
      typename T::UpdateUnit;
      typename T::UpdateConn;
      typename T::PruneUnit;
      typename T::PruneConn;
      typename T::ExtraUnitFields;
    } &&
    PassPolicy<typename T::ForwardPass, UnitAllocFor<T>,
               typename T::ConnAllocator, typename T::GlobalState> &&
    PassPolicy<typename T::BackwardPass, UnitAllocFor<T>,
               typename T::ConnAllocator, typename T::GlobalState> &&
    UpdateUnitPolicy<typename T::UpdateUnit, UnitAllocFor<T>,
                     typename T::GlobalState> &&
    UpdateConnPolicy<typename T::UpdateConn, UnitAllocFor<T>,
                     typename T::ConnAllocator, typename T::GlobalState> &&
    PruneUnitPolicy<typename T::PruneUnit, UnitAllocFor<T>,
                    typename T::GlobalState> &&
    PruneConnPolicy<typename T::PruneConn, UnitAllocFor<T>,
                    typename T::ConnAllocator, typename T::GlobalState>;

} // namespace plastix

#endif // PLASTIX_TRAITS_HPP
