// ===========================================================================
// Streaming iPC — 1-Layer Linear Debug Example
// ===========================================================================
//
// This example implements the simplest possible case of Streaming incremental
// Predictive Coding (Streaming iPC) in Plastix: a single-layer linear network
// with no hidden units. In this regime iPC reduces to ordinary online gradient
// descent for linear regression, so the example doubles as a correctness check
// against the standard linear_regression example.
//
// ---------------------------------------------------------------------------
// Algorithm overview
// ---------------------------------------------------------------------------
//
// A predictive coding network maintains *value nodes* x^(l) at each layer and
// learns by minimising local prediction errors. In the generative (top-down)
// convention the network predicts lower layers from higher ones:
//
//   prediction:  mu^(l)  = theta^(l) * f(x^(l+1))
//   error:       eps^(l) = x^(l) - mu^(l)
//
// Weights are updated with a local Hebbian rule:
//
//   theta^(l) += alpha * eps^(l) * f(x^(l+1))^T
//
// For a 1-layer network (input -> output) with f = identity, this becomes:
//
//   mu    = theta * x_input
//   eps   = y_target - mu
//   theta += alpha * eps * x_input^T
//
// which is exactly the delta rule / online SGD for linear regression.
//
// ---------------------------------------------------------------------------
// Mapping to Plastix
// ---------------------------------------------------------------------------
//
// Plastix connection pages store directed edges from source units (input
// layer, higher index in iPC) to destination units (output layer, lower
// index in iPC). The execution flow per observation is:
//
//   0. Clamp value nodes:
//        ValueNode[input]  = x_input
//        ValueNode[output] = y_target
//
//   1. Pre-step: copy f(ValueNode) into the ping-pong PreviousActivation
//      buffer so DoForwardPass reads the correct source activations.
//      (For 1-layer with identity activation this is a no-op since
//      DoForwardPass clamps input units directly.)
//
//   2. DoForwardPass  [iPCForwardPass policy]
//        Map:              returns Weight * Activation  (= theta * x_input)
//        CalculateAndApply: accumulates mu per destination unit, then
//                           computes eps = ValueNode - mu, stores eps in
//                           ErrorTag, and returns eps into the activation
//                           buffer.
//
//   3. DoUpdateConnectionState  [iPCUpdateConn policy]
//        UpdateIncomingConnection:
//          reads  ErrorTag(DstId)     = eps at destination
//          reads  ValueNodeTag(SrcId) = f(x) at source  (identity here)
//          writes theta += alpha * eps * f(x_src)
//
// No backward pass or value-node update is needed because there are no
// hidden layers.
//
// The custom allocator iPCUnitAllocator extends the default unit state with
// two extra SOA fields (ValueNodeTag, ErrorTag) without modifying any
// framework source files.
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
constexpr size_t OutputSize = 1;
constexpr size_t OutputBegin = NumInputs;
constexpr size_t OutputEnd = OutputBegin + OutputSize;
constexpr float Alpha = 0.01f;

// ---------------------------------------------------------------------------
// iPC forward pass: compute prediction error ε = x - μ
// ---------------------------------------------------------------------------

struct iPCForwardPass {
  // f = identity for the linear case.
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
// iPC weight update: θ += α · ε_dst · f(x_src)
// ---------------------------------------------------------------------------

struct iPCUpdateConn {
  static void UpdateIncomingConnection(auto &U, size_t DstId, size_t SrcId,
                                       auto &C, size_t PageId, size_t SlotIdx,
                                       auto &) {
    float Eps = U.template Get<ErrorTag>(DstId);
    float Fx = U.template Get<ValueNodeTag>(SrcId); // f = identity
    auto &Page = C.template Get<plastix::ConnPageMarker>(PageId);
    Page.Conn[SlotIdx].second += Alpha * Eps * Fx;
  }
  static void UpdateOutgoingConnection(auto &, size_t, size_t, auto &, size_t,
                                       size_t, auto &) {}
};

// ---------------------------------------------------------------------------
// Traits — 1-layer, no backward pass, no hidden value-node updates
// ---------------------------------------------------------------------------

struct iPCLinearTraits
    : plastix::DefaultNetworkTraits<iPCUnitAllocator,
                                    plastix::ConnStateAllocator> {
  using ForwardPass = iPCForwardPass;
  using UpdateConn = iPCUpdateConn;
};

using iPCNet = plastix::Network<iPCLinearTraits>;
using FC = plastix::FullyConnected;

// ---------------------------------------------------------------------------

int main() {
  std::cout << "Plastix iPC Linear Regression (1-layer debug)" << std::endl;
  std::cout << "===============================================" << std::endl;

  // 3 inputs -> 1 output, weights initialised to 0.
  // True relationship: y = 2*x1 - x2 + 0.5*x3
  iPCNet net(NumInputs, FC{OutputSize, 0.0f});

  std::mt19937 Rng(42);
  std::uniform_real_distribution<float> Dist(-1.0f, 1.0f);

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "Target weights: [2.0, -1.0, 0.5]\n" << std::endl;

  constexpr size_t NumSteps = 2000;
  constexpr size_t PrintInterval = 200;

  float FinalLoss = 0.0f;
  float AvgLoss = 0.0f;
  for (size_t T = 0; T < NumSteps; ++T) {
    float X1 = Dist(Rng), X2 = Dist(Rng), X3 = Dist(Rng);
    float Y = 2.0f * X1 - 1.0f * X2 + 0.5f * X3;
    float Input[3] = {X1, X2, X3};

    auto &U = net.GetUnitAlloc();

    // Step 0: Clamp value nodes.
    for (size_t I = 0; I < NumInputs; ++I)
      U.Get<ValueNodeTag>(I) = Input[I];
    U.Get<ValueNodeTag>(OutputBegin) = Y;

    // Step 1: Pre-step — copy f(ValueNode) into PreviousActivation.
    // For non-input units (output unit here), the forward pass accumulation
    // needs f(x) in PreviousActivation for sources. But sources are input
    // units here, and DoForwardPass clamps those. The output unit's
    // PreviousActivation is unused as a source, so this is a no-op for the
    // 1-layer case. We include it for consistency.
    {
      size_t Step = net.GetStep();
      for (size_t I = NumInputs; I < U.Size(); ++I) {
        float Fx = U.Get<ValueNodeTag>(I); // f = identity
        if (Step % 2 == 0)
          U.Get<plastix::ActivationATag>(I) = Fx;
        else
          U.Get<plastix::ActivationBTag>(I) = Fx;
      }
    }

    // Step 2: Forward pass — computes ε = ValueNode - μ, stores in ErrorTag.
    net.DoForwardPass(Input);

    // Step 3: No backward pass (no hidden layers).

    // Step 4: Weight update — θ += α · ε · f(x_src).
    net.DoUpdateConnectionState();

    // Step 5: No hidden-unit value-node update (no hidden layers).

    float Pred = U.Get<ValueNodeTag>(OutputBegin) -
                 U.Get<ErrorTag>(OutputBegin); // μ = x - ε
    float Loss = (Pred - Y) * (Pred - Y);
    AvgLoss += Loss;
    if ((T + 1) % PrintInterval == 0) {
      AvgLoss /= static_cast<float>(PrintInterval);
      std::cout << "Step " << std::setw(5) << T + 1
                << "  avg error^2: " << std::setw(8) << AvgLoss << std::endl;
      AvgLoss = 0.0f;
    }
  }

  // Print learned weights.
  auto &Page = net.GetConnAlloc().Get<plastix::ConnPageMarker>(0);
  std::cout << "\nLearned weights: [";
  for (size_t S = 0; S < Page.Count; ++S) {
    if (S > 0)
      std::cout << ", ";
    std::cout << Page.Conn[S].second;
  }
  std::cout << "]" << std::endl;
}
