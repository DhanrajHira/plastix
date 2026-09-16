#ifndef PLASTIX_OPTIM_HPP
#define PLASTIX_OPTIM_HPP

// Optimizer bundles: ready-made UpdateConn policies plus the per-connection
// state columns they need.
//
// A *bundle* is two things a traits class wires in together:
//
//   * `StateFields` -- the extra per-connection SoA columns the rule carries
//     (Adam's moments, momentum's velocity, ...). Because they are ordinary
//     connection fields they shard with the edges, survive compaction, and
//     cost nothing when the rule needs none.
//   * the `UpdateConn` hooks themselves.
//
// `WithOptimizer` (traits.hpp) wires both in one line:
//
//     struct MyTraits : plastix::WithOptimizer<MyBase,
//                                              plastix::optim::Adam<GradTag>> {};
//
// Two design points carried over from the plastax bundles these mirror, both
// deliberate rather than incidental:
//
// **The gradient is the delta rule, not a stored column.** Every bundle forms
// `dL/dw = dL/dz[dst] * activation[src]` from two local reads. That is exact
// for any weighted-sum layer, dense or sparse, so no gradient buffer is ever
// materialised. `GradTag` names the per-unit field holding `dL/dz` at the
// destination -- typically the one a backward pass wrote (the examples call it
// `GradPreActTag`), or `BackwardAccTag` directly for a single-layer network
// where the loss's staged `dL/dActivation` already is `dL/dz`.
//
// **Adam's step counter is per connection, not global.** On a dense network
// every edge updates every step, so `opt/t` equals a global count and the
// optax correspondence is exact. On a sparse one, a regrown edge restarts its
// own bias-correction schedule, which is the behaviour dynamic-sparse training
// wants. It also keeps the bundle shard-safe with no collective.
//
// PARITY: these mirror plastax's `plastax.optim`, which is itself validated
// against `optax`. The formulas below are transcribed from it deliberately --
// plastax is the reference implementation and plastix must agree with it
// (tests/golden/optim_*.json). Do not "improve" the arithmetic here without
// changing plastax first; see notes/parity/02-optimizers.md.

#include "plastix/conn.hpp"
#include "plastix/macros.hpp"
#include "plastix/unit_state.hpp"

#include <cmath>

namespace plastix::optim {

// ---------------------------------------------------------------------------
// Per-connection optimizer state
// ---------------------------------------------------------------------------
//
// Named to match plastax's "opt/..." column prefix so the two are recognisably
// the same state. A traits class carries at most one optimizer; two bundles in
// one traits class would collide on these tags, which
// `ConnFieldTagsAreUniqueV` turns into a compile error rather than a silent
// aliasing bug.

struct OptMTag {}; // Adam first moment            ("opt/m")
struct OptVTag {}; // second moment / velocity     ("opt/v")
struct OptTTag {}; // Adam step count, per edge    ("opt/t")

// ---------------------------------------------------------------------------
// Hyperparameters
// ---------------------------------------------------------------------------
//
// Params expose plain `static constexpr float` members; override by inheriting
// and shadowing:
//
//     struct FastSgd : plastix::optim::SgdParams {
//       static constexpr float Lr = 0.5f;
//     };
//
// For a schedule, `WithGlobalLr` swaps the learning rate for a runtime read
// from GlobalState (mutable through `Network::Global()`); everything else
// stays constant.

struct SgdParams {
  static constexpr float Lr = 0.01f;
};

struct MomentumParams {
  static constexpr float Lr = 0.01f;
  static constexpr float Momentum = 0.9f;
};

struct RmsPropParams {
  static constexpr float Lr = 0.01f;
  static constexpr float Decay = 0.9f;
  static constexpr float Eps = 1e-8f; // inside the sqrt
};

struct AdamParams {
  static constexpr float Lr = 1e-3f;
  static constexpr float B1 = 0.9f;
  static constexpr float B2 = 0.999f;
  static constexpr float Eps = 1e-8f; // outside the sqrt
  static constexpr float WeightDecay = 0.0f;
};

struct AdamWParams : AdamParams {
  static constexpr float WeightDecay = 1e-4f;
};

// Read the learning rate from `GlobalState::*Member` instead of a constant, so
// a host loop can anneal it between steps via `Net.Global()`. The member
// pointer is named explicitly by the traits class -- there is no duck-typed
// discovery of a magic field name.
//
//     using UpdateConn = optim::Sgd<GradTag, optim::WithGlobalLr<&MyGlobals::Lr>>;
template <auto Member, typename Base = SgdParams> struct WithGlobalLr : Base {
  PLASTIX_HD static float Lr(const auto &G) { return G.*Member; }
};

namespace detail {

// Params state the learning rate either as a constant or, under WithGlobalLr,
// as a function of the global state. One accessor serves both so the optimizer
// bodies below never branch on which form they were given.
template <typename Params, typename Globals>
PLASTIX_HD float ReadLr(const Globals &G) {
  if constexpr (requires { Params::Lr(G); })
    return Params::Lr(G);
  else
    return Params::Lr;
}

// dL/dw for one edge, by the delta rule (see the header comment).
template <typename GradTag>
PLASTIX_HD float EdgeGradient(auto &U, size_t DstId, size_t SrcId) {
  return GetField<GradTag>(U, DstId) * GetActivation(U, SrcId);
}

// Every bundle here does all its work from the destination's perspective, so
// the outgoing hook is a genuine no-op rather than an oversight. Inherit it
// instead of repeating the signature five times.
struct NoOutgoingUpdate {
  PLASTIX_HD static void UpdateOutgoingConnection(auto &, size_t, size_t,
                                                  auto &, size_t, auto &) {}
};

} // namespace detail

// ---------------------------------------------------------------------------
// Bundles
// ---------------------------------------------------------------------------

// w <- w - lr * g
//
// Stateless: the minimal bundle, and the one every example's hand-written
// gradient-descent rule collapses to.
template <typename GradTag, typename Params = SgdParams>
struct Sgd : detail::NoOutgoingUpdate {
  using StateFields = ConnFieldList<>;

  PLASTIX_HD static void UpdateIncomingConnection(auto &U, size_t DstId,
                                                  size_t SrcId, auto &C,
                                                  size_t ConnId, auto &G) {
    const float Grad = detail::EdgeGradient<GradTag>(U, DstId, SrcId);
    GetWeight(C, ConnId) -= detail::ReadLr<Params>(G) * Grad;
  }
};

// v <- momentum * v + g;  w <- w - lr * v
//
// Classic heavy-ball momentum (optax.sgd(lr, momentum=...) with
// nesterov=False). `momentum = 0` recovers plain SGD exactly.
template <typename GradTag, typename Params = MomentumParams>
struct Momentum : detail::NoOutgoingUpdate {
  using StateFields = ConnFieldList<alloc::SOAField<OptVTag, float>>;

  PLASTIX_HD static void UpdateIncomingConnection(auto &U, size_t DstId,
                                                  size_t SrcId, auto &C,
                                                  size_t ConnId, auto &G) {
    const float Grad = detail::EdgeGradient<GradTag>(U, DstId, SrcId);
    float &Velocity = GetField<OptVTag>(C, ConnId);
    Velocity = Params::Momentum * Velocity + Grad;
    GetWeight(C, ConnId) -= detail::ReadLr<Params>(G) * Velocity;
  }
};

// v <- decay * v + (1 - decay) * g^2;  w <- w - lr * g / sqrt(v + eps)
//
// Uncentered RMSProp with eps INSIDE the sqrt, matching
// optax.rmsprop(learning_rate, decay, eps). No first moment, no bias
// correction. The eps placement is not interchangeable with Adam's -- it
// changes the update at small v, which is exactly where it matters.
template <typename GradTag, typename Params = RmsPropParams>
struct RmsProp : detail::NoOutgoingUpdate {
  using StateFields = ConnFieldList<alloc::SOAField<OptVTag, float>>;

  PLASTIX_HD static void UpdateIncomingConnection(auto &U, size_t DstId,
                                                  size_t SrcId, auto &C,
                                                  size_t ConnId, auto &G) {
    const float Grad = detail::EdgeGradient<GradTag>(U, DstId, SrcId);
    float &MeanSq = GetField<OptVTag>(C, ConnId);
    MeanSq = Params::Decay * MeanSq + (1.0f - Params::Decay) * (Grad * Grad);
    GetWeight(C, ConnId) -=
        detail::ReadLr<Params>(G) * Grad / std::sqrt(MeanSq + Params::Eps);
  }
};

// t <- t + 1
// m <- b1*m + (1-b1)*g;   v <- b2*v + (1-b2)*g^2
// m_hat = m / (1 - b1^t);  v_hat = v / (1 - b2^t)
// w <- w - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * w)
//
// Matches optax.adam(lr, b1, b2, eps) with eps OUTSIDE the sqrt and
// eps_root = 0; a non-zero `Params::WeightDecay` makes it optax.adamw's
// decoupled decay (applied to the pre-update weight, not folded into the
// gradient). Use the `AdamW` alias below rather than setting it by hand.
//
// The two `std::pow` calls are per edge per step and are the bundle's only
// non-trivial cost. A multiplicative `b1^t` column would be cheaper, but it
// accumulates rounding differently from plastax's `b1 ** t` and would put the
// parity vectors at risk for a saving nobody has measured a need for. Leave it
// unless a profile says otherwise.
template <typename GradTag, typename Params = AdamParams>
struct Adam : detail::NoOutgoingUpdate {
  using StateFields =
      ConnFieldList<alloc::SOAField<OptMTag, float>,
                    alloc::SOAField<OptVTag, float>,
                    alloc::SOAField<OptTTag, float>>;

  PLASTIX_HD static void UpdateIncomingConnection(auto &U, size_t DstId,
                                                  size_t SrcId, auto &C,
                                                  size_t ConnId, auto &G) {
    const float Grad = detail::EdgeGradient<GradTag>(U, DstId, SrcId);
    float &Weight = GetWeight(C, ConnId);
    float &M = GetField<OptMTag>(C, ConnId);
    float &V = GetField<OptVTag>(C, ConnId);
    float &T = GetField<OptTTag>(C, ConnId);

    const float Step = T + 1.0f;
    M = Params::B1 * M + (1.0f - Params::B1) * Grad;
    V = Params::B2 * V + (1.0f - Params::B2) * (Grad * Grad);
    T = Step;

    const float MHat = M / (1.0f - std::pow(Params::B1, Step));
    const float VHat = V / (1.0f - std::pow(Params::B2, Step));
    const float Direction = MHat / (std::sqrt(VHat) + Params::Eps);
    // Read the pre-update weight for the decay term, as optax.adamw does.
    Weight -= detail::ReadLr<Params>(G) *
              (Direction + Params::WeightDecay * Weight);
  }
};

// Adam with decoupled weight decay (optax.adamw).
template <typename GradTag, typename Params = AdamWParams>
using AdamW = Adam<GradTag, Params>;

} // namespace plastix::optim

#endif // PLASTIX_OPTIM_HPP
