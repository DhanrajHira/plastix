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
//
// Each concept defines the interface a policy struct must satisfy. Policies
// are pure collections of static methods — no instances are ever created.
// Network<Traits> calls these methods directly from its Do*() methods.
// ---------------------------------------------------------------------------

// PassPolicy covers both the forward and backward passes. Execution is split
// into two phases that mirror how connections are stored (grouped by
// destination unit):
//
//   Phase 1 — Map: called once per connection (src→dst). Returns the
//     per-connection contribution, e.g. Weight * Activation. Results are
//     accumulated into a running sum for that destination unit.
//
//   Phase 2 — CalculateAndApply: called once per destination unit after all
//     its incoming Map results have been summed. Applies the activation
//     function, stores any side-effects (e.g. error signals), and returns
//     the value written into the activation buffer.
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

// UpdateUnitPolicy uses a map-reduce pattern so that the per-unit update can
// be expressed without knowing how many incoming connections a unit has:
//
//   Map     — produces a Partial value for each incoming connection.
//   Combine — reduces two Partials into one (must be associative).
//   Apply   — receives the fully-reduced Partial and writes the result back
//             to the unit allocator (e.g. update a threshold or bias).
//
// The Partial type alias is part of the interface so Network can zero-init
// accumulators without knowing the concrete type.
template <typename P, typename UnitAlloc, typename Global>
concept UpdateUnitPolicy = requires(UnitAlloc &U, size_t Id, Global &G) {
  { P::Update(U, Id, G) } -> std::same_as<void>;
};

// AddUnitPolicy determines if a unit should be added within some spheroid
// and then adds it if so.
template <typename P, typename UnitAlloc, typename Global>
concept AddUnitPolicy =
    requires(UnitAlloc &U, float SOI,
             size_t PageId, size_t SlotIdx, Global &G) {
      {
        P::ShouldAddUnit(U, SOI, PageId, SlotIdx, G)
      } -> std::convertible_to<bool>;
      {
        P::AddUnit(U, SOI , PageId, SlotIdx, G)
      } -> std::same_as<void>;
    };


// UpdateConnPolicy runs two separate sweeps over every connection page so
// that each endpoint can be updated with full knowledge of its own state:
//
//   UpdateIncomingConnection — called with (DstId, SrcId): the destination
//     unit drives the update (e.g. Hebbian: dst error × src activation).
//   UpdateOutgoingConnection — called with (SrcId, DstId): the source unit
//     drives the update (e.g. eligibility-trace rules keyed on the sender).
//
// Both sweeps iterate pages in the same order, so PageId/SlotIdx give direct
// write access to the weight stored in the connection page.
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

// AddConnPolicy determines if a connection should be added between two units
// and then adds it if so.
template <typename P, typename UnitAlloc, typename ConnAlloc, typename Global>
concept AddConnPolicy =
    requires(UnitAlloc &U, size_t DstId, size_t SrcId, ConnAlloc &C,
             size_t PageId, size_t SlotIdx, Global &G) {
      {
        P::ShouldAddConnection(U, DstId, SrcId, C, PageId, SlotIdx, G)
      } -> std::convertible_to<bool>;
      {
        P::AddConnection(U, DstId, SrcId, C, PageId, SlotIdx, G)
      } -> std::same_as<void>;
    };

// PruneUnitPolicy marks individual units for removal. DoPruneUnits() writes
// the result into PrunedTag; DoPruneConnections() then removes any connection
// whose source or destination unit is marked.
template <typename P, typename UnitAlloc, typename Global>
concept PruneUnitPolicy = requires(UnitAlloc &U, size_t Id, Global &G) {
  { P::ShouldPrune(U, Id, G) } -> std::convertible_to<bool>;
};

// PruneConnPolicy removes individual connections independently of unit
// pruning (e.g. weight-magnitude thresholding). Both unit and connection
// pruning are evaluated together in DoPruneConnections().
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

// DefaultForwardPass: weighted linear summation with identity activation
// (i.e. no nonlinearity). Equivalent to a plain matrix-vector multiply.
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

// Sentinel noop policies — satisfy their concepts but the corresponding
// Do*() methods in Network compile out the entire loop body via if constexpr
// when these exact types are detected. There is no runtime overhead.
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

struct NoAddUnit {
  static bool ShouldAddUnit(auto &, float, size_t, size_t, auto &) {
    return false;
  }
  static void AddUnit(auto &, float, size_t, size_t, auto &) {}
};

struct NoUpdateConn {
  static void UpdateIncomingConnection(auto &, size_t, size_t, auto &, size_t,
                                       size_t, auto &) {}
  static void UpdateOutgoingConnection(auto &, size_t, size_t, auto &, size_t,
                                       size_t, auto &) {}
};

struct NoAddConn {
  static bool ShouldAddConnection(auto &, size_t, size_t, auto &, size_t,
                                  size_t, auto &) {
    return false;
  }
  static void AddConnection(auto &, size_t, size_t, auto &, size_t, size_t,
                             auto &) {}
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

// GlobalState is a user-defined struct passed by reference to every policy
// method. Use it for shared hyperparameters (learning rates, time step, etc.)
// that are not specific to any single unit or connection. EmptyGlobalState is
// the default when no shared state is needed.
struct EmptyGlobalState {};

// ---------------------------------------------------------------------------
// Default traits base — inherit and override individual policies as needed
//
// To define a custom network, inherit DefaultNetworkTraits and replace only
// the policies you need. For example, to add a sigmoid activation:
//
//   struct MyTraits : DefaultNetworkTraits<UnitStateAllocator,
//                                          ConnStateAllocator> {
//     using ForwardPass = MySigmoidForwardPass;
//   };
//
// UnitAlloc and ConnAlloc are the allocator types for units and connection
// pages respectively. See alloc.hpp for how to extend unit state with extra
// SOA fields without modifying any framework files.
// ---------------------------------------------------------------------------

template <typename ConnAlloc, typename Global = EmptyGlobalState>
struct DefaultNetworkTraits {
  using ConnAllocator = ConnAlloc;
  using GlobalState = Global;
  using ForwardPass = DefaultForwardPass;
  using BackwardPass = NoBackwardPass;
  using UpdateUnit = NoUpdateUnit;
  using AddUnit = NoAddUnit;
  using UpdateConn = NoUpdateConn;
  using AddConn = NoAddConn;
  using PruneUnit = NoPruneUnit;
  using PruneConn = NoPruneConn;
  using ExtraUnitFields = UnitFieldList<>;
};

// ---------------------------------------------------------------------------
// NetworkTraits concept — validates that all policies in a traits struct
// satisfy their respective concepts. This fires at the Network<Traits>
// instantiation site, giving a clear error when a policy is malformed.
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
      typename T::AddUnit;
      typename T::UpdateConn;
      typename T::AddConn;
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
    AddUnitPolicy<typename T::AddUnit, UnitAllocFor<T>,
                  typename T::GlobalState> &&
    UpdateConnPolicy<typename T::UpdateConn, UnitAllocFor<T>,
                     typename T::ConnAllocator, typename T::GlobalState> &&
    AddConnPolicy<typename T::AddConn, UnitAllocFor<T>,
                  typename T::ConnAllocator, typename T::GlobalState> &&
    PruneUnitPolicy<typename T::PruneUnit, UnitAllocFor<T>,
                    typename T::GlobalState> &&
    PruneConnPolicy<typename T::PruneConn, UnitAllocFor<T>,
                    typename T::ConnAllocator, typename T::GlobalState>;

} // namespace plastix

#endif // PLASTIX_TRAITS_HPP
