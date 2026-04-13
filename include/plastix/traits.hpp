#ifndef PLASTIX_TRAITS_HPP
#define PLASTIX_TRAITS_HPP

#include "plastix/conn.hpp"
#include "plastix/layers.hpp"
#include "plastix/unit_state.hpp"
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>

namespace plastix {

// ---------------------------------------------------------------------------
// Propagation model
// ---------------------------------------------------------------------------

enum class Propagation { Topological, Pipeline };

// ---------------------------------------------------------------------------
// Policy concepts
//
// Each concept defines the interface a policy struct must satisfy. Policies
// are pure collections of static methods — no instances are ever created.
// Network<Traits> calls these methods directly from its Do*() methods.
// ---------------------------------------------------------------------------

template <typename P, typename UnitAlloc, typename ConnAlloc, typename Global>
concept PassPolicy =
    std::default_initializable<typename P::Accumulator> &&
    requires(UnitAlloc &U, size_t A, size_t B, ConnAlloc &C, size_t ConnId,
             Global &G, typename P::Accumulator AccA,
             typename P::Accumulator AccB) {
      typename P::Accumulator;
      {
        P::Map(U, A, B, C, ConnId, G)
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
concept UpdateConnPolicy = requires(UnitAlloc &U, size_t DstId, size_t SrcId,
                                    ConnAlloc &C, size_t ConnId, Global &G) {
  {
    P::UpdateIncomingConnection(U, DstId, SrcId, C, ConnId, G)
  } -> std::same_as<void>;
  {
    P::UpdateOutgoingConnection(U, SrcId, DstId, C, ConnId, G)
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
concept PruneConnPolicy = requires(UnitAlloc &U, size_t DstId, size_t SrcId,
                                   ConnAlloc &C, size_t ConnId, Global &G) {
  {
    P::ShouldPrune(U, DstId, SrcId, C, ConnId, G)
  } -> std::convertible_to<bool>;
};

template <typename P, typename UnitAlloc, typename Global>
concept AddUnitPolicy =
    requires(UnitAlloc &U, size_t ParentId, size_t NewId, Global &G) {
      { P::AddUnit(U, ParentId, G) } -> std::same_as<std::optional<int16_t>>;
      { P::InitUnit(U, NewId, ParentId, G) } -> std::same_as<void>;
    };

template <typename P, typename UnitAlloc, typename ConnAlloc, typename Global>
concept AddConnPolicy =
    requires(UnitAlloc &U, size_t SelfId, size_t CandidateId, ConnAlloc &C,
             size_t ConnId, Global &G) {
      {
        P::ShouldAddIncomingConnection(U, SelfId, CandidateId, G)
      } -> std::convertible_to<bool>;
      {
        P::ShouldAddOutgoingConnection(U, SelfId, CandidateId, G)
      } -> std::convertible_to<bool>;
      {
        P::InitConnection(U, SelfId, CandidateId, C, ConnId, G)
      } -> std::same_as<void>;
    };

template <typename P, typename Global>
concept ResetGlobalStatePolicy = requires(Global &G) {
  { P::Reset(G) } -> std::same_as<void>;
};

template <typename P, typename UnitAlloc, typename Global>
concept LossPolicy =
    requires(UnitAlloc &U, UnitRange R, std::span<const float> T, Global &G) {
      { P::CalculateLoss(U, R, T, G) } -> std::same_as<void>;
    };

// ---------------------------------------------------------------------------
// Default and noop policy implementations
// ---------------------------------------------------------------------------

// DefaultForwardPass: weighted linear summation with identity activation
// (i.e. no nonlinearity). Equivalent to a plain matrix-vector multiply.
struct DefaultForwardPass {
  using Accumulator = float;
  static float Map(auto &U, size_t, size_t SrcId, auto &C, size_t ConnId,
                   auto &) {
    return GetWeight(C, ConnId) * GetActivation(U, SrcId);
  }
  static float Combine(float A, float B) { return A + B; }
  static void Apply(auto &U, size_t Id, auto &, float Accumulated) {
    GetActivation(U, Id) = Accumulated;
  }
};

// Sentinel noop policies — satisfy their concepts but the corresponding
// Do*() methods in Network compile out the entire loop body via if constexpr
// when these exact types are detected. There is no runtime overhead.
struct NoBackwardPass {
  using Accumulator = float;
  static float Map(auto &, size_t, size_t, auto &, size_t, auto &) {
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
                                       auto &) {}
  static void UpdateOutgoingConnection(auto &, size_t, size_t, auto &, size_t,
                                       auto &) {}
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
  static bool ShouldPrune(auto &, size_t, size_t, auto &, size_t, auto &) {
    return false;
  }
};

struct NoAddUnit {
  static std::optional<int16_t> AddUnit(auto &, size_t, auto &) {
    return std::nullopt;
  }
  static void InitUnit(auto &, size_t, size_t, auto &) {}
};

struct NoAddConn {
  static bool ShouldAddIncomingConnection(auto &, size_t, size_t, auto &) {
    return false;
  }
  static bool ShouldAddOutgoingConnection(auto &, size_t, size_t, auto &) {
    return false;
  }
  static void InitConnection(auto &, size_t, size_t, auto &, size_t, auto &) {}
};

struct NoResetGlobalState {
  static void Reset(auto &) {}
};

// Loss policies stage the upstream gradient dL/dActivation into
// `BackwardAccTag` on each output unit, so the backward pass can propagate it
// through the network. Convention: `BackwardAcc` holds the gradient with
// respect to the output activation, and weight updates subtract
// `learning_rate * grad * input`.
struct NoLoss {
  static void CalculateLoss(auto &, UnitRange, std::span<const float>, auto &) {
  }
};

// Mean squared error: L = 0.5 * sum((pred - target)^2).
// dL/dpred_i = pred_i - target_i.
struct MSELoss {
  static void CalculateLoss(auto &U, UnitRange Outputs,
                            std::span<const float> Targets, auto &) {
    size_t I = 0;
    for (size_t Id : Outputs.Ids()) {
      float Pred = GetActivation(U, Id);
      GetBackwardAcc(U, Id) = Pred - Targets[I++];
    }
  }
};

// Root mean squared error: L = sqrt(mean((pred - target)^2)).
// dL/dpred_i = (pred_i - target_i) / (N * L). Epsilon keeps the gradient
// finite when predictions match targets exactly.
struct RMSLoss {
  static void CalculateLoss(auto &U, UnitRange Outputs,
                            std::span<const float> Targets, auto &) {
    float SumSq = 0.0f;
    size_t I = 0;
    for (size_t Id : Outputs.Ids()) {
      float Diff = GetActivation(U, Id) - Targets[I++];
      SumSq += Diff * Diff;
    }
    float N = static_cast<float>(Outputs.Size());
    float Rms = std::sqrt(SumSq / N);
    float Denom = N * (Rms + 1e-8f);
    I = 0;
    for (size_t Id : Outputs.Ids()) {
      float Pred = GetActivation(U, Id);
      GetBackwardAcc(U, Id) = (Pred - Targets[I++]) / Denom;
    }
  }
};

// Softmax over outputs + cross-entropy against a target distribution
// (one-hot or soft). L = -sum target_i * log(softmax(pred)_i).
// dL/dpred_i = softmax(pred)_i - target_i.
struct SoftmaxCrossEntropyLoss {
  static void CalculateLoss(auto &U, UnitRange Outputs,
                            std::span<const float> Targets, auto &) {
    float Max = -std::numeric_limits<float>::infinity();
    for (size_t Id : Outputs.Ids()) {
      float P = GetActivation(U, Id);
      if (P > Max)
        Max = P;
    }
    float Sum = 0.0f;
    for (size_t Id : Outputs.Ids())
      Sum += std::exp(GetActivation(U, Id) - Max);
    size_t I = 0;
    for (size_t Id : Outputs.Ids()) {
      float Soft = std::exp(GetActivation(U, Id) - Max) / Sum;
      GetBackwardAcc(U, Id) = Soft - Targets[I++];
    }
  }
};

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

template <typename Global = EmptyGlobalState> struct DefaultNetworkTraits {
  using GlobalState = Global;
  using ForwardPass = DefaultForwardPass;
  using BackwardPass = NoBackwardPass;
  using Loss = NoLoss;
  using UpdateUnit = NoUpdateUnit;
  using AddUnit = NoAddUnit;
  using UpdateConn = NoUpdateConn;
  using AddConn = NoAddConn;
  using PruneUnit = NoPruneUnit;
  using PruneConn = NoPruneConn;
  using AddUnit = NoAddUnit;
  using AddConn = NoAddConn;
  using ResetGlobal = NoResetGlobalState;
  using ExtraUnitFields = UnitFieldList<>;
  using ExtraConnFields = ConnFieldList<alloc::SOAField<WeightTag, float>>;
  static constexpr uint16_t Neighbourhood = 1;
  static constexpr Propagation Model = Propagation::Topological;
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

// Helper: resolve the connection allocator for a given traits type.
template <typename T>
using ConnAllocFor = MakeConnAllocatorFrom<typename T::ExtraConnFields>;

template <typename T>
concept NetworkTraits =
    requires {
      typename T::GlobalState;
      typename T::ForwardPass;
      typename T::ForwardPass::Accumulator;
      typename T::BackwardPass;
      typename T::BackwardPass::Accumulator;
      typename T::Loss;
      typename T::UpdateUnit;
      typename T::UpdateConn;
      typename T::AddConn;
      typename T::PruneUnit;
      typename T::PruneConn;
      typename T::AddUnit;
      typename T::AddConn;
      typename T::ResetGlobal;
      typename T::ExtraUnitFields;
      typename T::ExtraConnFields;
      { T::Neighbourhood } -> std::convertible_to<uint16_t>;
      { T::Model } -> std::convertible_to<Propagation>;
    } &&
    PassPolicy<typename T::ForwardPass, UnitAllocFor<T>, ConnAllocFor<T>,
               typename T::GlobalState> &&
    PassPolicy<typename T::BackwardPass, UnitAllocFor<T>, ConnAllocFor<T>,
               typename T::GlobalState> &&
    LossPolicy<typename T::Loss, UnitAllocFor<T>, typename T::GlobalState> &&
    UpdateUnitPolicy<typename T::UpdateUnit, UnitAllocFor<T>,
                     typename T::GlobalState> &&
    UpdateConnPolicy<typename T::UpdateConn, UnitAllocFor<T>, ConnAllocFor<T>,
                     typename T::GlobalState> &&
    PruneUnitPolicy<typename T::PruneUnit, UnitAllocFor<T>,
                    typename T::GlobalState> &&
    PruneConnPolicy<typename T::PruneConn, UnitAllocFor<T>, ConnAllocFor<T>,
                    typename T::GlobalState> &&
    AddUnitPolicy<typename T::AddUnit, UnitAllocFor<T>,
                  typename T::GlobalState> &&
    AddConnPolicy<typename T::AddConn, UnitAllocFor<T>, ConnAllocFor<T>,
                  typename T::GlobalState> &&
    ResetGlobalStatePolicy<typename T::ResetGlobal, typename T::GlobalState>;

} // namespace plastix

#endif // PLASTIX_TRAITS_HPP
