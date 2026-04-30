#include "plastix/alloc.hpp"
#include "plastix/traits.hpp"
#include "plastix/unit_state.hpp"
#include <plastix/plastix.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <span>

namespace {

// --- per-connection state ------------------------------------------
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

// --- per-feature state ------------------------------------------
enum UnitKind : uint8_t { UK_Pattern, UK_Memory, UK_Output };
enum TenureStatus : uint8_t { TS_Idle, TS_TenureTrack, TS_Tenured };
struct UKindTag {};
struct UFlagsTags {};
struct Pulse {};
struct Delay {};
struct ActThreshold {};
struct Tenure {};

struct UnitFlags {
  bool InputsAdded = false;
  bool OutputsAdded = false;
};

namespace hp {
// SwiftTD Params
// Old HP were as follow:
constexpr static float Gamma = 1.0f;
constexpr static float Lambda = 0.9f;
constexpr static float Eta = 0.05f;
constexpr static float EtaMin = 1e-7f;
constexpr static float Decay = 0.999f;
constexpr static float MetaStepSize = 1e-3f;
constexpr static float AlphaInit = 0.01f;
// constexpr static float Gamma = 0.9f;
// constexpr static float Lambda = 0.9f;
// constexpr static float Eta = 0.1f;
// constexpr static float EtaMin = 1e-4f;
// constexpr static float Decay = 0.99f;
// constexpr static float MetaStepSize = 0.01f;
// constexpr static float AlphaInit = 0.01f;

constexpr static float TenureThreshold = 0.5f;
constexpr static float TenureTrackThreshold = 0.1f;
constexpr static int MaxGenerationsPerStep = 5;
}; // namespace hp

// --- global state -----------------------------------------------------------
// Network owns Globals by value and provides no public setter, so the
// hyperparameters live as default member initializers here.
struct ImprintingLearnerGlobals {
  std::mt19937 Rng{42};

  float V = 0.0f;
  float VOld = 0.0f;
  float Delta = 0.0f;
  float VDeltaPrev = 0.0f; // v_delta from previous step (used in phase 1)
  float VDelta = 0.0f;     // v_delta accumulated in phase 2
  float Tau = 0.0f;        // Σ exp(beta_i) * f_i^2 (computed in phase 1)
  float B = 0.0f;          // Σ z_i * f_i           (computed in phase 1)
                           //
  int GenerationLeft = hp::MaxGenerationsPerStep;
  bool WasActive = false;
};

struct ImprintingLearnerOutputUnitInit {
  void operator()(auto &UA, auto Id) const {
    plastix::GetField<UKindTag>(UA, Id) = UK_Output;
  }
};

// --- ConnInit functor: only beta needs an initial value; placement-new
// already zero-inits the other POD fields.
struct ImprintingLearnerConnInit {
  void operator()(auto &CA, auto ConnId) const {
    plastix::GetField<BetaTag>(CA, ConnId) = std::log(hp::AlphaInit);
  }
};

// --- forward: v = Σ w_i f_i; expose v through Globals so Loss can read it.
struct ImprintingLearnerForward {
  struct Acc {
    float Activation = 0;
    int NumConns = 0;
  };
  using Accumulator = Acc;
  static Acc Map(auto &U, size_t, size_t SrcId, auto &C, size_t ConnId,
                 auto &) {
    return {plastix::GetWeight(C, ConnId) * plastix::GetActivation(U, SrcId),
            1};
  }
  static Acc Combine(Acc A, Acc B) {
    return {A.Activation + B.Activation, A.NumConns + B.NumConns};
  }
  static void Apply(auto &U, size_t Id, auto &G, Acc Acc) {
    UnitKind Kind = plastix::GetField<UKindTag>(U, Id);
    if (Kind == UK_Output) {
      ApplyOutput(U, Id, Acc, G);
    } else if (Kind == UK_Pattern) {
      ApplyPattern(U, Id, Acc, G);
    } else {
      ApplyMemory(U, Id, Acc, G);
    }
  }

  static void ApplyMemory(auto &U, size_t Id, Acc Acc,
                          ImprintingLearnerGlobals &G) {
    uint32_t &P = GetField<Pulse>(U, Id);
    uint32_t D = GetField<Delay>(U, Id);

    // Should we activate this step?
    plastix::GetActivation(U, Id) = P & 0x1;
    P = (P >> 1);

    // Should we set up a new trigger?
    if (Acc.Activation != 0) {
      G.WasActive |= true;
      P |= 0x1 << D;
    }
  }

  static void ApplyOutput(auto &U, size_t Id, Acc Acc,
                          ImprintingLearnerGlobals &G) {

    plastix::GetActivation(U, Id) = Acc.Activation;
    G.V = Acc.Activation;
  }

  static void ApplyPattern(auto &U, size_t Id, Acc Acc,
                           ImprintingLearnerGlobals &G) {
    auto Threshold = GetField<ActThreshold>(U, Id);
    // Activate if our acc is larger than our threshold.
    int Act = static_cast<int>((Acc.Activation / Acc.NumConns) > Threshold);
    G.WasActive |= (Act != 0);
    plastix::GetActivation(U, Id) = Act;
  }
};

// --- Loss runs after forward / before UpdateConn. We hijack it to compute
// δ from (reward, v, v_old) and to reset the per-step reductions.
// Targets[0] is interpreted as the reward observed at this step.
struct ImprintingLearnerLoss {
  static void CalculateLoss(auto &, plastix::UnitRange,
                            std::span<const float> Targets,
                            ImprintingLearnerGlobals &G) {
    float Reward = Targets[0];
    G.Delta = Reward + hp::Gamma * G.V - G.VOld;
    G.VDeltaPrev = G.VDelta;
    G.VDelta = 0.0f;
    G.B = 0.0f;
  }
};

TenureStatus DetermineTenureState(float Weight) {
  if (Weight > hp::TenureThreshold)
    return TS_Tenured;
  else if (Weight > hp::TenureTrackThreshold)
    return TS_TenureTrack;
  else
    return TS_Idle;
}

struct ImprintingLearnerUnitUpdate {
  static void Update(auto &UA, size_t Id, auto &) {
    if (plastix::GetField<UKindTag>(UA, Id) == UK_Pattern)
      plastix::GetField<UFlagsTags>(UA, Id).InputsAdded = true;
  }
};

// --- per-connection SwiftTD update, split across the incoming/outgoing hooks
// so that the reductions Tau and B (computed in phase 1) are visible to the
// phase 2 loop.
struct ImprintingLearnerConnUpdate {
  static void UpdateIncomingConnection(auto &U, size_t DstId, size_t SrcId,
                                       auto &C, size_t ConnId,
                                       ImprintingLearnerGlobals &G) {
    using namespace plastix;
    if (GetField<UKindTag>(U, DstId) != UK_Output)
      return;

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
    GetField<Tenure>(U, SrcId) = DetermineTenureState(W);

    Beta += hp::MetaStepSize / std::exp(Beta) * (G.Delta - G.VDeltaPrev) * P;
    float ExpBeta = std::exp(Beta);
    if (ExpBeta > hp::Eta || std::isinf(ExpBeta)) {
      Beta = std::log(hp::Eta);
      ExpBeta = hp::Eta;
    }
    if (ExpBeta < hp::EtaMin) {
      Beta = std::log(hp::EtaMin);
      ExpBeta = hp::EtaMin;
    }

    HOld = H;
    H = HTemp + G.Delta * ZBar - Zd * G.VDeltaPrev;
    HTemp = H;
    Zd = 0.0f;

    const float TraceDecay = hp::Gamma * hp::Lambda;
    Z *= TraceDecay;
    P *= TraceDecay;
    ZBar *= TraceDecay;

    // Reductions consumed in phase 2 (use the post-update beta and the
    // post-decay z).
    G.Tau += ExpBeta * F * F;
    G.B += Z * F;
  }

  static void UpdateOutgoingConnection(auto &U, size_t SrcId, size_t DstId,
                                       auto &C, size_t ConnId,
                                       ImprintingLearnerGlobals &G) {
    using namespace plastix;
    if (GetField<UKindTag>(U, DstId) != UK_Output)
      return;
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
    if (G.Tau > 0.0f && hp::Eta / G.Tau < 1.0f)
      Multiplier = hp::Eta / G.Tau;
    Zd = Multiplier * std::exp(Beta) * F;

    Z += Zd * (1.0f - G.B);
    P += HOld * F;
    ZBar += Zd * (1.0f - G.B - ZBar * F);
    HTemp = H - HOld * F * (Z - Zd) - H * Zd * F;

    if (G.Tau > hp::Eta) {
      HTemp = 0.0f;
      H = 0.0f;
      HOld = 0.0f;
      ZBar = 0.0f;
      Beta += std::log(hp::Decay) * F * F;
    }
  }
};

struct ImprintingLearnerAddUnit {
  static std::optional<int16_t> AddUnit(auto & /*U*/, size_t /*Id*/,
                                        ImprintingLearnerGlobals &G) {
    if (!G.GenerationLeft || !G.WasActive || G.Tau + hp::AlphaInit > hp::Eta)
      return std::nullopt;
    --G.GenerationLeft;
    G.Tau += hp::AlphaInit;
    return 0;
  }

  static void InitUnit(auto &U, size_t Id, size_t /*Parent*/,
                       ImprintingLearnerGlobals &G) {
    std::cout << "Init Units";
    auto Sampled =
        std::bernoulli_distribution{0.5}(G.Rng) ? UK_Pattern : UK_Memory;
    GetField<UKindTag>(U, Id) = Sampled;
    if (Sampled == UK_Pattern) {
      constexpr float Thresholds[] = {0.6f, 0.7f, 0.8f, 0.9f, 1.0f};
      float Threshold =
          Thresholds[std::uniform_int_distribution<int>{0, 4}(G.Rng)];
      GetField<ActThreshold>(U, Id) = Threshold;
      plastix::GetActivation(U, Id) = 1.0f;
    } else {
      int D = std::uniform_int_distribution<int>{1, 20}(G.Rng);
      GetField<Delay>(U, Id) = D;
      auto &P = GetField<Pulse>(U, Id);
      P = P | 0x1 << D;
    }
  }
};

struct ImprintingLearnerAddConn {
  static bool ShouldAddIncomingConnection(auto &U, size_t Self,
                                          size_t Candidate,
                                          ImprintingLearnerGlobals &G) {
    UnitKind SelfKind = GetField<UKindTag>(U, Self);
    if (SelfKind == UK_Output) {
      return HandleConnsToOutput(U, Candidate);
    }

    UnitFlags &Flags = GetField<UFlagsTags>(U, Self);
    if (Flags.InputsAdded)
      return false;

    UnitKind CandidateKind = GetField<UKindTag>(U, Candidate);
    if (CandidateKind == UK_Output)
      return false;

    auto CandidateWasActive = plastix::GetActivation(U, Candidate) != 0;

    bool CandidateWasTenured = GetField<Tenure>(U, Candidate) == TS_Tenured;

    if (!CandidateWasActive || !CandidateWasTenured)
      return false;

    if (SelfKind == UK_Memory) {
      // NOTE: This approach is heavily biased towards connections that
      // have their query run first.
      auto ShouldConnectToThis = std::bernoulli_distribution{0.2}(G.Rng);
      if (ShouldConnectToThis) {
        Flags.InputsAdded = true;
        return true;
      } else {
        return false;
      }
    } else if (SelfKind == UK_Pattern) {
      return true;
    }
    return false;
  }

  static bool HandleConnsToOutput(auto &U, size_t Candidate) {
    UnitFlags &CandidateFlags = GetField<UFlagsTags>(U, Candidate);
    if (!CandidateFlags.OutputsAdded) {
      std::cout << "Adding connection to output from " << Candidate << "\n";
      CandidateFlags.OutputsAdded = true;
      return true;
    } else {
      return false;
    }
  };

  static bool ShouldAddOutgoingConnection(auto &, size_t, size_t,
                                          ImprintingLearnerGlobals &) {
    return false;
  }

  static void InitConnection(auto &UA, size_t, size_t To, auto &CA, size_t Conn,
                             ImprintingLearnerGlobals &) {
    UnitKind ToKind = GetField<UKindTag>(UA, To);
    if (ToKind == UK_Output) {
      plastix::GetWeight(CA, Conn) = 0.0;
    } else {
      plastix::GetWeight(CA, Conn) = 1.0;
    }
  }
};

struct ImprintingLearnerResetGlobal {
  static void Reset(auto &G) {
    G.VOld = G.V;
    G.GenerationLeft = hp::MaxGenerationsPerStep;
    G.WasActive = false;
    G.Tau = 0.0f;
  }
};

struct ImprintingLearnerTraits
    : plastix::DefaultNetworkTraits<ImprintingLearnerGlobals> {
  using ForwardPass = ImprintingLearnerForward;
  using Loss = ImprintingLearnerLoss;
  using UpdateUnit = ImprintingLearnerUnitUpdate;
  using UpdateConn = ImprintingLearnerConnUpdate;
  using ResetGlobal = ImprintingLearnerResetGlobal;
  using AddUnit = ImprintingLearnerAddUnit;
  using AddConn = ImprintingLearnerAddConn;
  // ImprintingLearnerConnUpdate writes `G.Tau`, `G.B`, `G.VDelta` from every
  // thread (unsafe reduction); InitUnit / HandleConnsToOutput use std::cout
  // which isn't device-callable. Keep update on the host. AddUnit mutates
  // `G.GenerationLeft` / `G.Tau` non-atomically and InitUnit prints + draws
  // RNG from a host-side `std::bernoulli_distribution`, so add stays on
  // the host as well.
  static constexpr bool KernelizeUpdate = false;
  static constexpr bool KernelizeAdd = false;

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

  using ExtraUnitFields =
      plastix::UnitFieldList<plastix::alloc::SOAField<UKindTag, UnitKind>,
                             plastix::alloc::SOAField<UFlagsTags, UnitFlags>,
                             plastix::alloc::SOAField<Pulse, uint32_t>,
                             plastix::alloc::SOAField<Delay, uint16_t>,
                             plastix::alloc::SOAField<ActThreshold, float>,
                             plastix::alloc::SOAField<Tenure, TenureStatus>>;
  static constexpr plastix::Propagation Model = plastix::Propagation::Pipeline;
};

using ImprintingLearner = plastix::Network<ImprintingLearnerTraits>;

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
  constexpr size_t NumEpisodes = 5000;

  using FC = plastix::FullyConnected<ImprintingLearnerConnInit,
                                     ImprintingLearnerOutputUnitInit>;
  auto InputInit = [](auto &U, size_t Id) {
    plastix::GetField<UFlagsTags>(U, Id).InputsAdded = true;
    plastix::GetField<UFlagsTags>(U, Id).OutputsAdded = true;
  };
  ImprintingLearner Net(NumStates, InputInit,
                        FC{1, ImprintingLearnerConnInit{}});

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
