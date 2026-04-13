// ===========================================================================
// Streaming iPC — Multi-Layer Regression Example
// ===========================================================================
//
// This example implements the full Streaming incremental Predictive Coding
// (Streaming iPC) algorithm in Plastix: a multi-layer network with persistent
// hidden value nodes, simultaneous inference and learning, and pipelined
// signal propagation.
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
//       The two terms are:
//         -eps^(l)                              top-down correction
//         f'(x^(l)) . theta^(l-1)^T * eps^(l-1) bottom-up correction
//
//   (c) Update all weights with a local Hebbian rule:
//         theta^(l) += alpha * eps^(l) * f(x^(l+1))^T
//
// Value nodes are *never reset* between observations — this is the defining
// property of the streaming variant. The network state evolves continuously
// as new data arrives.
//
// ---------------------------------------------------------------------------
// Mapping to Plastix
// ---------------------------------------------------------------------------
//
// Plastix connections go from source (higher layer) to destination (lower
// layer), matching the top-down prediction direction. Three extra per-unit
// SOA fields (ValueNodeTag, ErrorTag, BottomUpTag) are added via
// ExtraUnitFields in the traits — no framework source changes needed.
//
// Pipeline propagation mode is used so that the forward pass accumulates all
// connections in one sweep (reading current Activation values) before
// applying any unit. This ensures every prediction uses the same snapshot
// of f(ValueNode), matching the simultaneous-computation semantics of iPC.
//
// Each iPC step maps to:
//
//   0. Pre-step: copy f(ValueNode) into Activation for non-input units.
//      DoForwardPass clamps input units from the Input span.
//
//   1. DoForwardPass  [iPCForwardPass — PassPolicy, Pipeline mode]
//        Map:     Weight * Activation(SrcId)  (source already has f(x))
//        Combine: sums contributions
//        Apply:   eps = ValueNode - ForwardAcc; stores in ErrorTag and
//                 Activation
//
//   2. DoBackwardPass  [iPCBackwardPass — PassPolicy, Pipeline mode]
//        Map:     Weight * ErrorTag(ToId)  (eps at destination)
//          -> accumulated into BackwardAcc(FromId) = theta^T * eps
//        Apply:   for hidden units, BottomUpTag = f'(x) * BackwardAcc
//
//   3. DoUpdateConnectionState  [iPCUpdateConn — UpdateConnPolicy]
//        UpdateIncomingConnection:
//          reads  ErrorTag(DstId)     = eps at destination
//          reads  ValueNodeTag(SrcId) = x at source, applies f()
//          writes theta += alpha * eps * f(x_src)
//
//   4. Manual value-node update (hidden units only):
//        x += gamma * (-eps + BottomUp)
//
// Because Pipeline mode processes all connections in one sweep per DoStep,
// each observation is held for StepsPerObs = 2 * NumLayers iPC steps to
// let signals propagate fully through the network.
// ===========================================================================

#include <plastix/plastix.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

// ---------------------------------------------------------------------------
// iPC-specific SOA fields
// ---------------------------------------------------------------------------

struct ValueNodeTag {};
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
// iPC forward pass: compute prediction error eps = ValueNode - mu
// ---------------------------------------------------------------------------

struct iPCForwardPass {
  using Accumulator = float;

  static float Map(auto &U, size_t, size_t SrcId, auto &C, size_t ConnId,
                   auto &) {
    return plastix::GetWeight(C, ConnId) * plastix::GetActivation(U, SrcId);
  }

  static float Combine(float A, float B) { return A + B; }

  static void Apply(auto &U, size_t Id, auto &, float Mu) {
    float X = plastix::GetField<ValueNodeTag>(U, Id);
    plastix::GetField<ErrorTag>(U, Id) = X - Mu;
  }
};

// ---------------------------------------------------------------------------
// iPC backward pass: propagate errors, compute value-node gradient
// ---------------------------------------------------------------------------

struct iPCBackwardPass {
  using Accumulator = float;

  // Accumulates Weight * eps(destination) into BackwardAcc(source).
  static float Map(auto &U, size_t, size_t ToId, auto &C, size_t ConnId,
                   auto &) {
    return plastix::GetWeight(C, ConnId) * plastix::GetField<ErrorTag>(U, ToId);
  }

  static float Combine(float A, float B) { return A + B; }

  // For hidden units: store bottom-up signal f'(x) * BackwardAcc.
  // Input/output units are clamped — skip them.
  static void Apply(auto &U, size_t Id, auto &, float BackwardAcc) {
    if (Id >= HiddenBegin && Id < HiddenEnd) {
      float X = plastix::GetField<ValueNodeTag>(U, Id);
      plastix::GetField<BottomUpTag>(U, Id) = ActDeriv(X) * BackwardAcc;
    }
  }
};

// ---------------------------------------------------------------------------
// iPC weight update: theta += alpha * eps_dst * f(x_src)
// ---------------------------------------------------------------------------

struct iPCUpdateConn {
  static void UpdateIncomingConnection(auto &U, size_t DstId, size_t SrcId,
                                       auto &C, size_t ConnId, auto &) {
    float Eps = plastix::GetField<ErrorTag>(U, DstId);
    float XSrc = plastix::GetField<ValueNodeTag>(U, SrcId);
    // Input units use identity activation; hidden units use tanh.
    float Fx = (SrcId < NumInputs) ? XSrc : Act(XSrc);
    plastix::GetWeight(C, ConnId) += Alpha * Eps * Fx;
  }

  static void UpdateOutgoingConnection(auto &, size_t, size_t, auto &, size_t,
                                       auto &) {}
};

// ---------------------------------------------------------------------------
// Traits — Pipeline mode for simultaneous prediction across all layers
// ---------------------------------------------------------------------------

struct iPCTraits : plastix::DefaultNetworkTraits<> {
  using ForwardPass = iPCForwardPass;
  using BackwardPass = iPCBackwardPass;
  using UpdateConn = iPCUpdateConn;
  using ExtraUnitFields =
      plastix::UnitFieldList<plastix::alloc::SOAField<ValueNodeTag, float>,
                             plastix::alloc::SOAField<ErrorTag, float>,
                             plastix::alloc::SOAField<BottomUpTag, float>>;
  static constexpr plastix::Propagation Model = plastix::Propagation::Pipeline;
};

using iPCNet = plastix::Network<iPCTraits>;
using FC = plastix::FullyConnected<>;

// ---------------------------------------------------------------------------
// One iPC inference+learning step
// ---------------------------------------------------------------------------

void iPCStep(iPCNet &Net, std::span<const float> Input) {
  auto &U = Net.GetUnitAlloc();
  size_t NumUnits = U.Size();

  // Pre-step: copy f(ValueNode) into Activation for non-input units.
  // DoForwardPass clamps input units from the Input span.
  for (size_t I = NumInputs; I < NumUnits; ++I) {
    float X = plastix::GetField<ValueNodeTag>(U, I);
    float Fx = (I >= HiddenBegin && I < HiddenEnd) ? Act(X) : X;
    plastix::GetActivation(U, I) = Fx;
  }

  // Forward pass: computes eps = ValueNode - mu, stores in ErrorTag.
  Net.DoForwardPass(Input);

  // Backward pass: propagates errors, stores bottom-up signal in BottomUpTag.
  Net.DoBackwardPass();

  // Weight update: theta += alpha * eps * f(x_src).
  Net.DoUpdateConnectionState();

  // Value-node update for hidden units: x += gamma * (-eps + bottom_up).
  for (size_t I = HiddenBegin; I < HiddenEnd; ++I) {
    float Eps = plastix::GetField<ErrorTag>(U, I);
    float BU = plastix::GetField<BottomUpTag>(U, I);
    plastix::GetField<ValueNodeTag>(U, I) += Gamma * (-Eps + BU);
    plastix::GetField<BottomUpTag>(U, I) = 0.0f;
  }
}

// ---------------------------------------------------------------------------

int main() {
  std::cout << "Plastix iPC Multi-Layer Regression\n";
  std::cout << "===================================\n";

  iPCNet Net(NumInputs, FC{HiddenSize}, FC{OutputSize});

  // LeCun-uniform init: U(-sqrt(3/fan_in), sqrt(3/fan_in)).
  std::mt19937 Rng(42);
  {
    float BoundHidden = std::sqrt(3.0f / NumInputs);
    float BoundOutput = std::sqrt(3.0f / HiddenSize);
    std::uniform_real_distribution<float> HiddenDist(-BoundHidden, BoundHidden);
    std::uniform_real_distribution<float> OutputDist(-BoundOutput, BoundOutput);

    auto &CA = Net.GetConnAlloc();
    for (size_t C = 0; C < CA.Size(); ++C) {
      uint32_t ToId = plastix::GetField<plastix::ToIdTag>(CA, C);
      bool IsOutput = ToId >= OutputBegin;
      plastix::GetWeight(CA, C) = IsOutput ? OutputDist(Rng) : HiddenDist(Rng);
    }
  }

  std::uniform_real_distribution<float> Dist(-1.0f, 1.0f);

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "Target: y = 2*x1 - x2 + 0.5*x3\n";
  std::cout << "Steps per observation: " << StepsPerObs << "\n\n";

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

      // Clamp boundary value nodes.
      auto &U = Net.GetUnitAlloc();
      for (size_t I = 0; I < NumInputs; ++I)
        plastix::GetField<ValueNodeTag>(U, I) = Input[I];
      plastix::GetField<ValueNodeTag>(U, OutputBegin) = Y;
    }

    // Run one iPC step.
    iPCStep(Net, Input);

    // Compute prediction every step: mu = ValueNode - eps.
    auto &U = Net.GetUnitAlloc();
    float Pred = plastix::GetField<ValueNodeTag>(U, OutputBegin) -
                 plastix::GetField<ErrorTag>(U, OutputBegin);
    float Loss = (Pred - Y) * (Pred - Y);
    float BaselineLoss = (PrevY - Y) * (PrevY - Y);
    AvgLoss += Loss;
    AvgBaseline += BaselineLoss;

    if ((T + 1) % PrintInterval == 0) {
      AvgLoss /= static_cast<float>(PrintInterval);
      AvgBaseline /= static_cast<float>(PrintInterval);
      std::cout << "Step " << std::setw(6) << T + 1
                << "  avg error^2: " << std::setw(8) << AvgLoss
                << "  baseline: " << std::setw(8) << AvgBaseline << "\n";
      LastAvgLoss = AvgLoss;
      LastAvgBaseline = AvgBaseline;
      AvgLoss = 0.0f;
      AvgBaseline = 0.0f;
    }
  }

  std::cout << "\nFinal window  iPC: " << LastAvgLoss
            << "  baseline: " << LastAvgBaseline << "\n";

  // Print learned weights grouped by destination unit.
  auto &CA = Net.GetConnAlloc();
  std::cout << "\nInput -> Hidden weights (theta^1):";
  uint32_t LastTo = UINT32_MAX;
  for (size_t C = 0; C < CA.Size(); ++C) {
    uint32_t To = plastix::GetField<plastix::ToIdTag>(CA, C);
    if (To >= OutputBegin)
      break;
    if (To != LastTo) {
      std::cout << "\n  unit " << std::setw(2) << To << ":";
      LastTo = To;
    }
    std::cout << std::setw(8) << plastix::GetWeight(CA, C);
  }
  std::cout << "\n\nHidden -> Output weights (theta^0):";
  LastTo = UINT32_MAX;
  for (size_t C = 0; C < CA.Size(); ++C) {
    uint32_t To = plastix::GetField<plastix::ToIdTag>(CA, C);
    if (To < OutputBegin)
      continue;
    if (To != LastTo) {
      std::cout << "\n  unit " << std::setw(2) << To << ":";
      LastTo = To;
    }
    std::cout << std::setw(8) << plastix::GetWeight(CA, C);
  }
  std::cout << "\n";

  std::cout << "\n"
            << (LastAvgLoss < LastAvgBaseline ? "PASS (beats baseline)"
                                              : "FAIL (worse than baseline)")
            << "\n";

  return LastAvgLoss < LastAvgBaseline ? 0 : 1;
}
