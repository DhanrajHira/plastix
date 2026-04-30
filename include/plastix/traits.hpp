#ifndef PLASTIX_TRAITS_HPP
#define PLASTIX_TRAITS_HPP

#include "plastix/conn.hpp"
#include "plastix/layers.hpp"
#include "plastix/macros.hpp"
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

template <typename P, typename UnitAlloc, typename Global>
concept UpdateUnitPolicy = requires(UnitAlloc &U, size_t Id, Global &G) {
  { P::Update(U, Id, G) } -> std::same_as<void>;
};

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

template <typename P, typename UnitAlloc, typename Global>
concept PruneUnitPolicy = requires(UnitAlloc &U, size_t Id, Global &G) {
  { P::ShouldPrune(U, Id, G) } -> std::convertible_to<bool>;
};

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

struct DefaultForwardPass {
  using Accumulator = float;
  PLASTIX_HD static float Map(auto &U, size_t, size_t SrcId, auto &C,
                              size_t ConnId, auto &) {
    return GetWeight(C, ConnId) * GetActivation(U, SrcId);
  }
  PLASTIX_HD static float Combine(float A, float B) { return A + B; }
  PLASTIX_HD static void Apply(auto &U, size_t Id, auto &, float Accumulated) {
    GetActivation(U, Id) = Accumulated;
  }
};

// Sentinel noop policies — satisfy their concepts but DoX() methods compile
// out the entire loop body via if constexpr when these are detected.
struct NoBackwardPass {
  using Accumulator = float;
  PLASTIX_HD static float Map(auto &, size_t, size_t, auto &, size_t, auto &) {
    return 0.0f;
  }
  PLASTIX_HD static float Combine(float A, float B) { return A + B; }
  PLASTIX_HD static void Apply(auto &, size_t, auto &, float) {}
};

struct NoUpdateUnit {
  PLASTIX_HD static void Update(auto &, size_t, auto &) {}
};

struct NoUpdateConn {
  PLASTIX_HD static void UpdateIncomingConnection(auto &, size_t, size_t,
                                                  auto &, size_t, auto &) {}
  PLASTIX_HD static void UpdateOutgoingConnection(auto &, size_t, size_t,
                                                  auto &, size_t, auto &) {}
};

struct NoPruneUnit {
  PLASTIX_HD static bool ShouldPrune(auto &, size_t, auto &) { return false; }
};

struct NoPruneConn {
  PLASTIX_HD static bool ShouldPrune(auto &, size_t, size_t, auto &, size_t,
                                     auto &) {
    return false;
  }
};

struct NoAddUnit {
  PLASTIX_HD static std::optional<int16_t> AddUnit(auto &, size_t, auto &) {
    return std::nullopt;
  }
  PLASTIX_HD static void InitUnit(auto &, size_t, size_t, auto &) {}
};

struct NoAddConn {
  PLASTIX_HD static bool ShouldAddIncomingConnection(auto &, size_t, size_t,
                                                     auto &) {
    return false;
  }
  PLASTIX_HD static bool ShouldAddOutgoingConnection(auto &, size_t, size_t,
                                                     auto &) {
    return false;
  }
  PLASTIX_HD static void InitConnection(auto &, size_t, size_t, auto &, size_t,
                                        auto &) {}
};

struct NoResetGlobalState {
  PLASTIX_HD static void Reset(auto &) {}
};

// Loss policies stage the upstream gradient dL/dActivation into
// `BackwardAccTag` on each output unit, so the backward pass can propagate it
// through the network. Convention: `BackwardAcc` holds the gradient with
// respect to the output activation, and weight updates subtract
// `learning_rate * grad * input`.
struct NoLoss {
  PLASTIX_HD static void CalculateLoss(auto &, UnitRange,
                                       std::span<const float>, auto &) {}
};

// Mean squared error: L = 0.5 * sum((pred - target)^2).
// dL/dpred_i = pred_i - target_i.
struct MSELoss {
  PLASTIX_HD static void CalculateLoss(auto &U, UnitRange Outputs,
                                       std::span<const float> Targets,
                                       auto &) {
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
  PLASTIX_HD static void CalculateLoss(auto &U, UnitRange Outputs,
                                       std::span<const float> Targets,
                                       auto &) {
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
  PLASTIX_HD static void CalculateLoss(auto &U, UnitRange Outputs,
                                       std::span<const float> Targets,
                                       auto &) {
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
// ---------------------------------------------------------------------------

template <typename Global = EmptyGlobalState> struct DefaultNetworkTraits {
  using GlobalState = Global;
  using ForwardPass = DefaultForwardPass;
  using BackwardPass = NoBackwardPass;
  using Loss = NoLoss;
  using UpdateUnit = NoUpdateUnit;
  using UpdateConn = NoUpdateConn;
  using PruneUnit = NoPruneUnit;
  using PruneConn = NoPruneConn;
  using AddUnit = NoAddUnit;
  using AddConn = NoAddConn;
  using ResetGlobal = NoResetGlobalState;
  using ExtraUnitFields = UnitFieldList<>;
  using ExtraConnFields = ConnFieldList<alloc::SOAField<WeightTag, float>>;
  static constexpr uint16_t Neighbourhood = 1;
  static constexpr Propagation Model = Propagation::Topological;
  // Opt-out flags for policies that are not device-safe (host-only state,
  // unsafe reductions like `G.Tau += ...` in update). When false, the
  // corresponding DoX phase always runs the host loop, even on a CUDA
  // build. Default true: most policies are pure SOA reads/writes and safe
  // to parallelize.
  static constexpr bool KernelizeUpdate = true;
  static constexpr bool KernelizePrune = true;
  static constexpr bool KernelizeAdd = true;
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
