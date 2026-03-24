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
// layer), matching the top-down prediction direction. Two extra per-unit SOA
// fields (ValueNodeTag, ErrorTag) are added via a custom allocator — no
// framework source changes are needed.
//
// Each iPC step maps to the following Plastix operations:
//
//   0. Pre-step: copy f(ValueNode) into the PreviousActivation ping-pong
//      buffer (selected via net.GetStep() % 2) so that DoForwardPass reads
//      the correct source activations.
//
//   1. DoForwardPass  [iPCForwardPass — PassPolicy]
//        Map:              Weight * Activation  (source already has f(x))
//        CalculateAndApply:
//          mu  = accumulated weighted sum
//          eps = ValueNode(Id) - mu
//          stores eps in ErrorTag(Id)
//          returns eps -> written to CurrentActivation
//        After Step++, PreviousActivation now holds eps values.
//
//   2. DoBackwardPass  [iPCBackwardPass — PassPolicy]
//        Map:  Weight * DstAct  (DstAct = eps at destination, from step 1)
//          -> accumulated into BackwardAccTag(SrcId) = theta^T * eps
//        CalculateAndApply:
//          for hidden units:
//            bottom_up = f'(x) * BackwardAcc
//            stores bottom_up in UpdateAccTag(Id)
//          BackwardAccTag is cleared by the framework after this call.
//
//   3. DoUpdateConnectionState  [iPCUpdateConn — UpdateConnPolicy]
//        UpdateIncomingConnection:
//          reads  ErrorTag(DstId)     = eps at destination
//          reads  ValueNodeTag(SrcId) = x at source, applies f()
//          writes theta += alpha * eps * f(x_src)
//        This runs before the value-node update, so both weight and
//        backward-pass computations use pre-update values of theta and x.
//
//   4. Manual value-node update (hidden units only):
//        x += gamma * (-eps + bottom_up)
//      where eps is read from ErrorTag and bottom_up from UpdateAccTag.
//      UpdateAccTag is then cleared for the next step.
//
// Because Plastix is pipelined (forward pass reads from the *previous*
// step's activation buffer), each observation is held for 2*NumLayers
// iPC steps to let signals propagate fully through the network.
// ===========================================================================

#include <plastix/plastix.hpp>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

// ---------------------------------------------------------------------------
// iPC-specific SOA fields (no framework changes required)
// ---------------------------------------------------------------------------

struct ValueNodeTag {};
struct ErrorTag {};

using iPCUnitAllocator = plastix::alloc::SOAAllocator<
    plastix::UnitState,
    plastix::alloc::SOAField<plastix::ActivationATag, float>,
    plastix::alloc::SOAField<plastix::ActivationBTag, float>,
    plastix::alloc::SOAField<plastix::BackwardAccTag, float>,
    plastix::alloc::SOAField<plastix::UpdateAccTag, float>,
    plastix::alloc::SOAField<plastix::PrunedTag, bool>,
    plastix::alloc::SOAField<ValueNodeTag, float>,
    plastix::alloc::SOAField<ErrorTag, float>>;

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
// iPC forward pass: compute prediction error ε = x - μ
// ---------------------------------------------------------------------------

struct iPCForwardPass {
  static float Map(auto &, size_t, auto &, float Weight, float Activation) {
    return Weight * Activation;
  }

  static float CalculateAndApply(auto &U, size_t Id, auto &, float Mu) {
    float X = U.template Get<ValueNodeTag>(Id);
    float Eps = X - Mu;
    U.template Get<ErrorTag>(Id) = Eps;
    return Eps;
  }
};

// ---------------------------------------------------------------------------
// iPC backward pass: propagate errors, compute value-node gradient
// ---------------------------------------------------------------------------

struct iPCBackwardPass {
  // Accumulates Weight * ε_dst into BackwardAccTag(SrcId).
  static float Map(auto &, size_t, auto &, float Weight, float DstError) {
    return Weight * DstError;
  }

  // For hidden units: store bottom-up signal f'(x) · BackwardAcc in
  // UpdateAccTag. Input/output units are clamped — skip them.
  static float CalculateAndApply(auto &U, size_t Id, auto &,
                                 float BackwardAcc) {
    if (Id >= HiddenBegin && Id < HiddenEnd) {
      float X = U.template Get<ValueNodeTag>(Id);
      float BottomUp = ActDeriv(X) * BackwardAcc;
      U.template Get<plastix::UpdateAccTag>(Id) = BottomUp;
    }
    return 0.0f;
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
    float XSrc = U.template Get<ValueNodeTag>(SrcId);
    // Input units use identity activation; hidden units use tanh.
    float Fx = (SrcId < NumInputs) ? XSrc : Act(XSrc);
    auto &Page = C.template Get<plastix::ConnPageMarker>(PageId);
    Page.Conn[SlotIdx].second += Alpha * Eps * Fx;
  }
  static void UpdateOutgoingConnection(auto &, size_t, size_t, auto &, size_t,
                                       size_t, auto &) {}
};

// ---------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------

struct iPCTraits
    : plastix::DefaultNetworkTraits<iPCUnitAllocator,
                                    plastix::ConnStateAllocator> {
  using ForwardPass = iPCForwardPass;
  using BackwardPass = iPCBackwardPass;
  using UpdateConn = iPCUpdateConn;
};

using iPCNet = plastix::Network<iPCTraits>;
using FC = plastix::FullyConnected;

// ---------------------------------------------------------------------------
// One iPC inference+learning step
// ---------------------------------------------------------------------------

void iPCStep(iPCNet &Net, std::span<const float> Input) {
  auto &U = Net.GetUnitAlloc();
  size_t NumUnits = U.Size();

  // Pre-step: copy f(ValueNode) into PreviousActivation for non-input units.
  {
    size_t Step = Net.GetStep();
    for (size_t I = NumInputs; I < NumUnits; ++I) {
      float X = U.Get<ValueNodeTag>(I);
      float Fx = (I >= HiddenBegin && I < HiddenEnd) ? Act(X) : X;
      if (Step % 2 == 0)
        U.Get<plastix::ActivationATag>(I) = Fx;
      else
        U.Get<plastix::ActivationBTag>(I) = Fx;
    }
  }

  // Forward pass: computes ε = ValueNode - μ, stores in ErrorTag.
  Net.DoForwardPass(Input);

  // Backward pass: propagates errors, stores bottom-up signal in UpdateAccTag.
  Net.DoBackwardPass();

  // Weight update: θ += α · ε · f(x_src).
  Net.DoUpdateConnectionState();

  // Value-node update for hidden units:
  //   x += γ · (-ε + bottom_up)
  for (size_t I = HiddenBegin; I < HiddenEnd; ++I) {
    float Eps = U.Get<ErrorTag>(I);
    float BottomUp = U.Get<plastix::UpdateAccTag>(I);
    U.Get<ValueNodeTag>(I) += Gamma * (-Eps + BottomUp);
    U.Get<plastix::UpdateAccTag>(I) = 0.0f;
  }
}

// ---------------------------------------------------------------------------

int main() {
  std::cout << "Plastix iPC Multi-Layer Regression" << std::endl;
  std::cout << "===================================" << std::endl;

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

      // Clamp boundary value nodes.
      auto &U = net.GetUnitAlloc();
      for (size_t I = 0; I < NumInputs; ++I)
        U.Get<ValueNodeTag>(I) = Input[I];
      U.Get<ValueNodeTag>(OutputBegin) = Y;
    }

    // Run one iPC step.
    iPCStep(net, Input);

    // Compute prediction every step: μ = x_output - ε_output.
    auto &U = net.GetUnitAlloc();
    float Pred = U.Get<ValueNodeTag>(OutputBegin) -
                 U.Get<ErrorTag>(OutputBegin);
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
