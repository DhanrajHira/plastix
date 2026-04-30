// SwiftTD reinforcement-learning algorithm — Plastix port.
//
// SwiftTD is TD(λ) with per-feature meta-learned step sizes (beta = log(alpha)
// updated via the IDBD/Auto-step style trick), an effective step-size cap
// (eta), and auxiliary traces h, h_old, h_temp, z_bar, p.
//
// We model SwiftTD as a single linear layer in Plastix (N feature units →
// 1 value unit). All per-feature state (w, z, z_delta, delta_w, h, h_old,
// h_temp, beta, z_bar, p) lives on the connection. The original step is
// split across the Loss / UpdateConn hooks:
//
//   Forward                = compute v = Σ w_i f_i, stash in Globals.V
//   Loss (CalculateLoss)   = compute δ, snapshot v_delta_prev, reset Tau/B
//   UpdateIncomingConn[i]  = SwiftTD "phase 1" — w/beta/h updates + decay,
//                             plus reductions Tau, B over the new beta and
//                             the decayed z (which phase 2 needs).
//   UpdateOutgoingConn[i]  = SwiftTD "phase 2" — refresh z, p, z_bar, h_temp
//                             using Tau and B, accumulate v_delta.
//   ResetGlobal            = v_old ← v

#include <plastix/plastix.hpp>

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <span>

namespace {

// --- per-connection SwiftTD state ------------------------------------------
// (plastix::WeightTag is reused for w_i.)
struct ZTag {};
struct ZDeltaTag {};
struct DeltaWTag {};
struct HTag {};
struct HOldTag {};
struct HTempTag {};
struct ZBarTag {};
struct PTag {};
struct BetaTag {}; // beta = log(alpha)

// --- global state -----------------------------------------------------------
// Network owns Globals by value and provides no public setter, so the
// hyperparameters live as default member initializers here.
struct SwiftTDGlobals {
  float Gamma = 1.0f;
  float Lambda = 0.9f;
  float Eta = 0.05f;
  float EtaMin = 1e-7f;
  float Decay = 0.999f;
  float MetaStepSize = 1e-3f;

  float V = 0.0f;
  float VOld = 0.0f;
  float Delta = 0.0f;
  float VDeltaPrev = 0.0f; // v_delta from previous step (used in phase 1)
  float VDelta = 0.0f;     // v_delta accumulated in phase 2
  float Tau = 0.0f;        // Σ exp(beta_i) * f_i^2 (computed in phase 1)
  float B = 0.0f;          // Σ z_i * f_i           (computed in phase 1)
};

// --- ConnInit functor: only beta needs an initial value; placement-new
// already zero-inits the other POD fields.
struct SwiftTDConnInit {
  float AlphaInit;
  void operator()(auto &CA, auto ConnId) const {
    plastix::GetField<BetaTag>(CA, ConnId) = std::log(AlphaInit);
  }
};

// --- forward: v = Σ w_i f_i; expose v through Globals so Loss can read it.
struct SwiftTDForward {
  using Accumulator = float;
  PLASTIX_HD static float Map(auto &U, size_t, size_t SrcId, auto &C,
                              size_t ConnId, auto &) {
    return plastix::GetWeight(C, ConnId) * plastix::GetActivation(U, SrcId);
  }
  PLASTIX_HD static float Combine(float A, float B) { return A + B; }
  PLASTIX_HD static void Apply(auto &U, size_t Id, auto &G, float Acc) {
    plastix::GetActivation(U, Id) = Acc;
    G.V = Acc;
  }
};

// --- Loss runs after forward / before UpdateConn. We hijack it to compute
// δ from (reward, v, v_old) and to reset the per-step reductions.
// Targets[0] is interpreted as the reward observed at this step.
struct SwiftTDLoss {
  static void CalculateLoss(auto &, plastix::UnitRange,
                            std::span<const float> Targets, auto &G) {
    float Reward = Targets[0];
    G.Delta = Reward + G.Gamma * G.V - G.VOld;
    G.VDeltaPrev = G.VDelta;
    G.VDelta = 0.0f;
    G.Tau = 0.0f;
    G.B = 0.0f;
  }
};

// --- per-connection SwiftTD update, split across the incoming/outgoing hooks
// so that the reductions Tau and B (computed in phase 1) are visible to the
// phase 2 loop.
struct SwiftTDUpdate {
  static void UpdateIncomingConnection(auto &U, size_t /*DstId*/, size_t SrcId,
                                       auto &C, size_t ConnId, auto &G) {
    using namespace plastix;
    const float F = GetActivation(U, SrcId);

    float &W = GetWeight(C, ConnId);
    float &Z = GetField<ZTag>(C, ConnId);
    float &Zd = GetField<ZDeltaTag>(C, ConnId);
    float &Dw = GetField<DeltaWTag>(C, ConnId);
    float &H = GetField<HTag>(C, ConnId);
    float &HOld = GetField<HOldTag>(C, ConnId);
    float &HTemp = GetField<HTempTag>(C, ConnId);
    float &Beta = GetField<BetaTag>(C, ConnId);
    float &ZBar = GetField<ZBarTag>(C, ConnId);
    float &P = GetField<PTag>(C, ConnId);

    Dw = G.Delta * Z - Zd * G.VDeltaPrev;
    W += Dw;

    Beta += G.MetaStepSize / std::exp(Beta) * (G.Delta - G.VDeltaPrev) * P;
    float ExpBeta = std::exp(Beta);
    if (ExpBeta > G.Eta || std::isinf(ExpBeta)) {
      Beta = std::log(G.Eta);
      ExpBeta = G.Eta;
    }
    if (ExpBeta < G.EtaMin) {
      Beta = std::log(G.EtaMin);
      ExpBeta = G.EtaMin;
    }

    HOld = H;
    H = HTemp + G.Delta * ZBar - Zd * G.VDeltaPrev;
    HTemp = H;
    Zd = 0.0f;

    const float TraceDecay = G.Gamma * G.Lambda;
    Z *= TraceDecay;
    P *= TraceDecay;
    ZBar *= TraceDecay;

    // Reductions consumed in phase 2 (use the post-update beta and the
    // post-decay z).
    G.Tau += ExpBeta * F * F;
    G.B += Z * F;
  }

  static void UpdateOutgoingConnection(auto &U, size_t SrcId, size_t /*DstId*/,
                                       auto &C, size_t ConnId, auto &G) {
    using namespace plastix;
    const float F = GetActivation(U, SrcId);

    float &Z = GetField<ZTag>(C, ConnId);
    float &Zd = GetField<ZDeltaTag>(C, ConnId);
    float &Dw = GetField<DeltaWTag>(C, ConnId);
    float &H = GetField<HTag>(C, ConnId);
    float &HOld = GetField<HOldTag>(C, ConnId);
    float &HTemp = GetField<HTempTag>(C, ConnId);
    float &Beta = GetField<BetaTag>(C, ConnId);
    float &ZBar = GetField<ZBarTag>(C, ConnId);
    float &P = GetField<PTag>(C, ConnId);

    G.VDelta += Dw * F;

    float Multiplier = 1.0f;
    if (G.Tau > 0.0f && G.Eta / G.Tau < 1.0f)
      Multiplier = G.Eta / G.Tau;
    Zd = Multiplier * std::exp(Beta) * F;

    Z += Zd * (1.0f - G.B);
    P += HOld * F;
    ZBar += Zd * (1.0f - G.B - ZBar * F);
    HTemp = H - HOld * F * (Z - Zd) - H * Zd * F;

    if (G.Tau > G.Eta) {
      HTemp = 0.0f;
      H = 0.0f;
      HOld = 0.0f;
      ZBar = 0.0f;
      Beta += std::log(G.Decay) * F * F;
    }
  }
};

struct SwiftTDResetGlobal {
  static void Reset(auto &G) { G.VOld = G.V; }
};

struct SwiftTDTraits : plastix::DefaultNetworkTraits<SwiftTDGlobals> {
  using ForwardPass = SwiftTDForward;
  using Loss = SwiftTDLoss;
  using UpdateConn = SwiftTDUpdate;
  using ResetGlobal = SwiftTDResetGlobal;
  // SwiftTDUpdate writes `G.Tau += ...` and `G.B += ...` from every thread —
  // an unsafe reduction under parallel execution. Keep this on the host.
  static constexpr bool KernelizeUpdate = false;

  using ExtraConnFields = plastix::ConnFieldList<
      plastix::alloc::SOAField<plastix::WeightTag, float>,
      plastix::alloc::SOAField<ZTag, float>,
      plastix::alloc::SOAField<ZDeltaTag, float>,
      plastix::alloc::SOAField<DeltaWTag, float>,
      plastix::alloc::SOAField<HTag, float>,
      plastix::alloc::SOAField<HOldTag, float>,
      plastix::alloc::SOAField<HTempTag, float>,
      plastix::alloc::SOAField<ZBarTag, float>,
      plastix::alloc::SOAField<PTag, float>,
      plastix::alloc::SOAField<BetaTag, float>>;
};

using ImprintingLearner = plastix::Network<SwiftTDTraits>;

} // namespace

// ---------------------------------------------------------------------------
// Episodic 5-state random walk (Sutton & Barto §6.2). States 1..5 are
// non-terminal, with absorbing terminals on either side. The right terminal
// pays reward 1; everything else is 0. Features are one-hot for the current
// non-terminal state and the all-zeros vector for the terminal. Under γ=1
// the true value function is V*(s_i) = i / 6.
//
// At the start of each episode we seed v_old by running a forward pass and
// triggering ResetGlobalState — that bypasses the spurious δ that would
// otherwise be computed against the previous episode's terminal v_old=0.
// ---------------------------------------------------------------------------
int main() {
  std::cout << "Plastix SwiftTD example (5-state random walk)\n"
               "==============================================\n";

  constexpr size_t NumStates = 5;
  constexpr float AlphaInit = 0.05f;
  constexpr size_t NumEpisodes = 5000;

  using FC = plastix::FullyConnected<SwiftTDConnInit>;
  ImprintingLearner Net(NumStates, FC{1, SwiftTDConnInit{AlphaInit}});

  std::mt19937 Rng(42);
  std::bernoulli_distribution Coin(0.5);

  std::array<float, NumStates> Features{};
  std::array<float, 1> Reward{0.0f};

  auto SetFeaturesForState = [&](int State) {
    Features.fill(0.0f);
    if (State >= 1 && State <= int(NumStates))
      Features[State - 1] = 1.0f;
  };

  for (size_t Ep = 0; Ep < NumEpisodes; ++Ep) {
    int State = 3;
    SetFeaturesForState(State);
    Net.DoForwardPass(Features);
    Net.DoResetGlobalState();

    while (true) {
      int Next = State + (Coin(Rng) ? 1 : -1);
      bool Terminal = (Next == 0 || Next == 6);
      float R = (Next == 6) ? 1.0f : 0.0f;

      if (Terminal)
        Features.fill(0.0f);
      else
        SetFeaturesForState(Next);
      Reward[0] = R;
      Net.DoStep(Features, Reward);

      if (Terminal)
        break;
      State = Next;
    }
  }

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "\nLearned V vs reference V*(i) = i/6:\n";
  float TotalErr = 0.0f;
  for (int S = 1; S <= int(NumStates); ++S) {
    SetFeaturesForState(S);
    Net.DoForwardPass(Features);
    float Pred = Net.GetOutput()[0];
    float Truth = float(S) / 6.0f;
    TotalErr += (Pred - Truth) * (Pred - Truth);
    std::cout << "  state " << S << ": V=" << std::setw(7) << Pred
              << "  V*=" << std::setw(7) << Truth << "  err=" << std::setw(8)
              << (Pred - Truth) << "\n";
  }
  std::cout << "\nRMSE = " << std::sqrt(TotalErr / NumStates) << "\n";
  return 0;
}
