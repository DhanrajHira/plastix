// ===========================================================================
// Streaming iPC — Multi-Layer Regression Example (v2)
// ===========================================================================
//
// This example implements Streaming incremental Predictive Coding (iPC)
// using the generalized pass API on the dbhira-generalized-passes branch.
//
// Key improvements over v1:
//   - No ping-pong buffer sync (the pre-step block is eliminated)
//   - ActivationTag serves directly as the persistent value node x^(l)
//   - Two extra fields (ErrorTag, BottomUpTag) are added via ExtraUnitFields
//   - Error and bottom-up signals are relayed through extra tags between passes
//
// ---------------------------------------------------------------------------
// Algorithm overview
// ---------------------------------------------------------------------------
//
// A predictive coding network has L+1 layers of value nodes x^(0)..x^(L)
// connected by L weight matrices theta^(0)..theta^(L-1). The generative
// (top-down) convention is:
//
//   Layer L  (top)    = input layer,  clamped to y_in
//   Layer 0  (bottom) = output layer, clamped to y_target
//   Layers 1..L-1     = hidden layers with *persistent* value nodes
//
// Each weight matrix theta^(l) predicts layer l from layer l+1:
//
//   prediction:  mu^(l)  = theta^(l) * f(x^(l+1))
//   error:       eps^(l) = x^(l) - mu^(l)
//
// Per observation, the network runs T simultaneous iPC steps. Each step:
//
//   (a) Compute predictions and errors at all layers:
//         mu^(l)  = theta^(l) * f(x^(l+1))
//         eps^(l) = x^(l) - mu^(l)
//
//   (b) Update hidden value nodes (layers 1..L-1):
//         x^(l) += gamma * ( -eps^(l)
//                             + f'(x^(l)) . [theta^(l-1)^T * eps^(l-1)] )
//
//   (c) Update all weights with a local Hebbian rule:
//         theta^(l) += alpha * eps^(l) * f(x^(l+1))^T
//
// Value nodes are *never reset* between observations (streaming variant).
//
// ---------------------------------------------------------------------------
// Mapping to Plastix (v2 — generalized passes)
// ---------------------------------------------------------------------------
//
// ActivationTag is the persistent value node x^(l). ErrorTag is an extra
// per-unit field added via ExtraUnitFields to store eps^(l).
//
// Each iPC step maps to:
//
//   1. DoForwardPass  [iPCForwardPass]
//        Map:     reads weight from ConnAlloc, reads f(x_src) from
//                 ActivationTag(SrcId), returns weight * f(x_src)
//        Combine: sum
//        Apply:   mu = accumulated sum
//                 eps = ActivationTag(Id) - mu
//                 writes eps to ErrorTag(Id)
//
//   2. DoBackwardPass  [iPCBackwardPass]
//        Map:     reads weight and ErrorTag(DstId), returns weight * eps
//        Combine: sum
//        Apply:   for hidden units: bottom_up = f'(x) * BackwardAcc
//                 writes bottom_up to BottomUpTag(Id)
//
//   3. DoUpdateConnectionState  [iPCUpdateConn]
//        reads ErrorTag(DstId) and f(ActivationTag(SrcId))
//        writes theta += alpha * eps * f(x_src)
//
//   4. DoUpdateUnitState  [iPCUpdateUnit]
//        for hidden units:
//        ActivationTag(I) += gamma * (-eps + bottom_up)
//        where eps from ErrorTag, bottom_up from BottomUpTag.
// ===========================================================================

#include <plastix/plastix.hpp>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

// ---------------------------------------------------------------------------
// Extra per-unit fields for prediction error and bottom-up signal
// ---------------------------------------------------------------------------

struct ErrorTag {};
struct BottomUpTag {};

// ---------------------------------------------------------------------------
// Topology & hyperparameters
// ---------------------------------------------------------------------------

constexpr size_t NumInputs = 3;
constexpr size_t HiddenSize = 16;
constexpr size_t OutputSize = 1;
constexpr size_t HiddenBegin = NumInputs;
constexpr size_t HiddenEnd = HiddenBegin + HiddenSize;
constexpr size_t OutputBegin = HiddenEnd;
constexpr size_t OutputEnd = OutputBegin + OutputSize;
constexpr size_t NumLayers = 3; // input, hidden, output
constexpr size_t StepsPerObs = 2 * NumLayers;

constexpr float Alpha = 0.005f;
constexpr float Gamma = 0.1f;

// ---------------------------------------------------------------------------
// Activation function (tanh) and derivative
// ---------------------------------------------------------------------------

inline float Act(float X) { return std::tanh(X); }
inline float ActDeriv(float X) {
  float T = std::tanh(X);
  return 1.0f - T * T;
}

// ---------------------------------------------------------------------------
// iPC forward pass: accumulate μ = Σ w · f(x_src), then ε = x - μ
// ---------------------------------------------------------------------------

struct iPCForwardPass {
  using Accumulator = float;

  static float Map(auto &U, size_t, size_t SrcId, auto &C, size_t PageId,
                   size_t SlotIdx, auto &) {
    float W =
        C.template Get<plastix::ConnPageMarker>(PageId).GetSlot(SlotIdx).second;
    float X = U.template Get<plastix::ActivationTag>(SrcId);
    float Fx = (SrcId < NumInputs) ? X : Act(X);
    return W * Fx;
  }

  static float Combine(float A, float B) { return A + B; }

  static void Apply(auto &U, size_t Id, auto &, float Mu) {
    float X = U.template Get<plastix::ActivationTag>(Id);
    float Eps = X - Mu;
    U.template Get<ErrorTag>(Id) = Eps;
  }
};

// ---------------------------------------------------------------------------
// iPC backward pass: accumulate θᵀε, then compute bottom-up signal
// ---------------------------------------------------------------------------

struct iPCBackwardPass {
  using Accumulator = float;

  static float Map(auto &U, size_t, size_t DstId, auto &C, size_t PageId,
                   size_t SlotIdx, auto &) {
    float W =
        C.template Get<plastix::ConnPageMarker>(PageId).GetSlot(SlotIdx).second;
    float Eps = U.template Get<ErrorTag>(DstId);
    return W * Eps;
  }

  static float Combine(float A, float B) { return A + B; }

  static void Apply(auto &U, size_t Id, auto &, float BackwardAcc) {
    if (Id >= HiddenBegin && Id < HiddenEnd) {
      float X = U.template Get<plastix::ActivationTag>(Id);
      float BottomUp = ActDeriv(X) * BackwardAcc;
      U.template Get<BottomUpTag>(Id) = BottomUp;
    }
  }
};

// ---------------------------------------------------------------------------
// iPC weight update: θ += α · ε_dst · f(x_src)
// ---------------------------------------------------------------------------

struct iPCUpdateConn {
  static void UpdateIncomingConnection(auto &U, size_t DstId, size_t SrcId,
                                       auto &C, size_t PageId, size_t SlotIdx,
                                       auto &) {
    float Eps = U.template Get<ErrorTag>(DstId);
    float X = U.template Get<plastix::ActivationTag>(SrcId);
    float Fx = (SrcId < NumInputs) ? X : Act(X);
    auto &Page = C.template Get<plastix::ConnPageMarker>(PageId);
    Page.Conn[SlotIdx].second += Alpha * Eps * Fx;
  }
  static void UpdateOutgoingConnection(auto &, size_t, size_t, auto &, size_t,
                                       size_t, auto &) {}
};

// ---------------------------------------------------------------------------
// iPC value-node update: x += γ · (-ε + bottom_up)
// ---------------------------------------------------------------------------

struct iPCUpdateUnit {
  static void Update(auto &U, size_t Id, auto &) {
    if (Id >= HiddenBegin && Id < HiddenEnd) {
      float Eps = U.template Get<ErrorTag>(Id);
      float BottomUp = U.template Get<BottomUpTag>(Id);
      U.template Get<plastix::ActivationTag>(Id) += Gamma * (-Eps + BottomUp);
      U.template Get<BottomUpTag>(Id) = 0.0f;
    }
  }
};

// ---------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------

struct iPCTraits
    : plastix::DefaultNetworkTraits<plastix::ConnStateAllocator> {
  using ForwardPass = iPCForwardPass;
  using BackwardPass = iPCBackwardPass;
  using UpdateUnit = iPCUpdateUnit;
  using UpdateConn = iPCUpdateConn;
  using ExtraUnitFields = plastix::UnitFieldList<
      plastix::alloc::SOAField<ErrorTag, float>,
      plastix::alloc::SOAField<BottomUpTag, float>>;
};

using iPCNet = plastix::Network<iPCTraits>;
using FC = plastix::FullyConnected;

// ---------------------------------------------------------------------------
// One iPC inference+learning step
// ---------------------------------------------------------------------------

void iPCStep(iPCNet &Net, std::span<const float> Input) {
  // Forward pass: computes ε = x - μ, stores in ErrorTag.
  Net.DoForwardPass(Input);

  // Backward pass: propagates errors, stores bottom-up signal in UpdateAccTag.
  Net.DoBackwardPass();

  // Weight update: θ += α · ε · f(x_src).
  Net.DoUpdateConnectionState();

  // Value-node update for hidden units: x += γ · (-ε + bottom_up).
  Net.DoUpdateUnitState();
}

// ---------------------------------------------------------------------------

int main() {
  std::cout << "Plastix iPC Multi-Layer Regression (v2)" << std::endl;
  std::cout << "========================================" << std::endl;

  iPCNet net(NumInputs, FC{HiddenSize, 0.0f}, FC{OutputSize, 0.0f});

  // LeCun-uniform init: U(-sqrt(3/fan_in), sqrt(3/fan_in))
  std::mt19937 Rng(42);
  {
    float BoundHidden = std::sqrt(3.0f / NumInputs);
    float BoundOutput = std::sqrt(3.0f / HiddenSize);
    std::uniform_real_distribution<float> HiddenDist(-BoundHidden, BoundHidden);
    std::uniform_real_distribution<float> OutputDist(-BoundOutput, BoundOutput);

    auto &CA = net.GetConnAlloc();
    for (size_t P = 0; P < CA.Size(); ++P) {
      auto &Page = CA.Get<plastix::ConnPageMarker>(P);
      bool IsOutput = Page.ToUnitIdx >= OutputBegin;
      for (size_t S = 0; S < Page.Count; ++S)
        Page.Conn[S].second = IsOutput ? OutputDist(Rng) : HiddenDist(Rng);
    }
  }

  std::uniform_real_distribution<float> Dist(-1.0f, 1.0f);

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "Target: y = 2*x1 - x2 + 0.5*x3" << std::endl;
  std::cout << "Steps per observation: " << StepsPerObs << "\n" << std::endl;

  constexpr size_t TotalSteps = 2000;
  constexpr size_t PrintInterval = 100;

  float AvgLoss = 0.0f;
  float AvgBaseline = 0.0f;
  float LastAvgLoss = 0.0f;
  float LastAvgBaseline = 0.0f;
  float PrevY = 0.0f;
  float Y = 0.0f;
  float Input[3] = {};

  for (size_t T = 0; T < TotalSteps; ++T) {
    // Sample a new observation every StepsPerObs steps.
    if (T % StepsPerObs == 0) {
      float X1 = Dist(Rng), X2 = Dist(Rng), X3 = Dist(Rng);
      PrevY = Y;
      Y = 2.0f * X1 - 1.0f * X2 + 0.5f * X3;
      Input[0] = X1;
      Input[1] = X2;
      Input[2] = X3;

      // Clamp output value node (input is clamped by DoForwardPass).
      auto &U = net.GetUnitAlloc();
      U.Get<plastix::ActivationTag>(OutputBegin) = Y;
    }

    // Run one iPC step.
    iPCStep(net, Input);

    // Compute prediction every step: μ = x - ε.
    auto &U = net.GetUnitAlloc();
    float X = U.Get<plastix::ActivationTag>(OutputBegin);
    float Eps = U.Get<ErrorTag>(OutputBegin);
    float Pred = X - Eps;
    float Loss = (Pred - Y) * (Pred - Y);
    float BaselineLoss = (PrevY - Y) * (PrevY - Y);
    AvgLoss += Loss;
    AvgBaseline += BaselineLoss;

    if ((T + 1) % PrintInterval == 0) {
      AvgLoss /= static_cast<float>(PrintInterval);
      AvgBaseline /= static_cast<float>(PrintInterval);
      std::cout << "Step " << std::setw(6) << T + 1
                << "  avg error^2: " << std::setw(8) << AvgLoss
                << "  baseline: " << std::setw(8) << AvgBaseline << std::endl;
      LastAvgLoss = AvgLoss;
      LastAvgBaseline = AvgBaseline;
      AvgLoss = 0.0f;
      AvgBaseline = 0.0f;
    }
  }

  std::cout << "\nFinal window  iPC: " << LastAvgLoss
            << "  baseline: " << LastAvgBaseline << std::endl;

  // Print learned weights grouped by destination unit.
  auto &CA = net.GetConnAlloc();
  std::cout << "\nInput -> Hidden weights (theta^1):";
  for (size_t P = 0; P < CA.Size(); ++P) {
    auto &Page = CA.Get<plastix::ConnPageMarker>(P);
    if (Page.ToUnitIdx >= OutputBegin)
      break;
    std::cout << "\n  unit " << std::setw(2) << Page.ToUnitIdx << ":";
    for (size_t S = 0; S < Page.Count; ++S)
      std::cout << std::setw(8) << Page.Conn[S].second;
  }
  std::cout << "\n\nHidden -> Output weights (theta^0):";
  for (size_t P = 0; P < CA.Size(); ++P) {
    auto &Page = CA.Get<plastix::ConnPageMarker>(P);
    if (Page.ToUnitIdx < OutputBegin)
      continue;
    std::cout << "\n  unit " << std::setw(2) << Page.ToUnitIdx << ":";
    for (size_t S = 0; S < Page.Count; ++S)
      std::cout << std::setw(8) << Page.Conn[S].second;
  }
  std::cout << std::endl;

  std::cout << "\n"
            << (LastAvgLoss < LastAvgBaseline ? "PASS (beats baseline)"
                                              : "FAIL (worse than baseline)")
            << std::endl;

  return LastAvgLoss < LastAvgBaseline ? 0 : 1;
}
