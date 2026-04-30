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
// Two extra per-unit SOA fields (ValueNodeTag, ErrorTag) are added via
// ExtraUnitFields in the traits — no framework source changes needed.
//
// Each observation maps to:
//
//   0. Clamp value nodes:
//        ValueNode[input]  = x_input
//        ValueNode[output] = y_target
//
//   1. DoForwardPass  [iPCForwardPass policy]
//        Map:     returns Weight * Activation(SrcId)  (= theta * x_input)
//        Combine: sums contributions into ForwardAcc per destination unit
//        Apply:   eps = ValueNode - ForwardAcc, stores in ErrorTag
//
//   2. DoUpdateConnectionState  [iPCUpdateConn policy]
//        UpdateIncomingConnection:
//          reads  ErrorTag(DstId)     = eps at destination
//          reads  ValueNodeTag(SrcId) = x at source (f = identity)
//          writes theta += alpha * eps * x_src
//
// No backward pass or value-node update is needed because there are no
// hidden layers.
// ===========================================================================

#include <plastix/plastix.hpp>

#include <iomanip>
#include <iostream>
#include <random>

// ---------------------------------------------------------------------------
// iPC-specific SOA fields
// ---------------------------------------------------------------------------

struct ValueNodeTag {};
struct ErrorTag {};

// ---------------------------------------------------------------------------
// Topology & hyperparameters
// ---------------------------------------------------------------------------

constexpr size_t NumInputs = 3;
constexpr size_t OutputSize = 1;
constexpr size_t OutputBegin = NumInputs;
constexpr float Alpha = 0.01f;

// ---------------------------------------------------------------------------
// iPC forward pass: compute prediction error eps = ValueNode - mu
// ---------------------------------------------------------------------------

struct iPCForwardPass {
  using Accumulator = float;

  PLASTIX_HD static float Map(auto &U, size_t, size_t SrcId, auto &C,
                              size_t ConnId, auto &) {
    return plastix::GetWeight(C, ConnId) * plastix::GetActivation(U, SrcId);
  }

  PLASTIX_HD static float Combine(float A, float B) { return A + B; }

  PLASTIX_HD static void Apply(auto &U, size_t Id, auto &, float Mu) {
    float X = plastix::GetField<ValueNodeTag>(U, Id);
    plastix::GetField<ErrorTag>(U, Id) = X - Mu;
  }
};

// ---------------------------------------------------------------------------
// iPC weight update: theta += alpha * eps_dst * f(x_src)
// ---------------------------------------------------------------------------

struct iPCUpdateConn {
  PLASTIX_HD static void UpdateIncomingConnection(auto &U, size_t DstId,
                                                  size_t SrcId, auto &C,
                                                  size_t ConnId, auto &) {
    float Eps = plastix::GetField<ErrorTag>(U, DstId);
    float Fx = plastix::GetField<ValueNodeTag>(U, SrcId); // f = identity
    plastix::GetWeight(C, ConnId) += Alpha * Eps * Fx;
  }

  PLASTIX_HD static void UpdateOutgoingConnection(auto &, size_t, size_t,
                                                  auto &, size_t, auto &) {}
};

// ---------------------------------------------------------------------------
// Traits — 1-layer, no backward pass, no hidden value-node updates
// ---------------------------------------------------------------------------

struct iPCLinearTraits : plastix::DefaultNetworkTraits<> {
  using ForwardPass = iPCForwardPass;
  using UpdateConn = iPCUpdateConn;
  using ExtraUnitFields =
      plastix::UnitFieldList<plastix::alloc::SOAField<ValueNodeTag, float>,
                             plastix::alloc::SOAField<ErrorTag, float>>;
};

using iPCNet = plastix::Network<iPCLinearTraits>;
using FC = plastix::FullyConnected<>;

// ---------------------------------------------------------------------------

int main() {
  std::cout << "Plastix iPC Linear Regression (1-layer debug)\n";
  std::cout << "===============================================\n";

  // 3 inputs -> 1 output, weights initialised to 0.
  // True relationship: y = 2*x1 - x2 + 0.5*x3
  iPCNet Net(NumInputs, FC{OutputSize});

  std::mt19937 Rng(42);
  std::uniform_real_distribution<float> Dist(-1.0f, 1.0f);

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "Target weights: [2.0, -1.0, 0.5]\n\n";

  constexpr size_t NumSteps = 2000;
  constexpr size_t PrintInterval = 200;

  float AvgLoss = 0.0f;
  for (size_t T = 0; T < NumSteps; ++T) {
    float X1 = Dist(Rng), X2 = Dist(Rng), X3 = Dist(Rng);
    float Y = 2.0f * X1 - 1.0f * X2 + 0.5f * X3;
    float Input[3] = {X1, X2, X3};

    auto &U = Net.GetUnitAlloc();

    // Clamp value nodes.
    for (size_t I = 0; I < NumInputs; ++I)
      plastix::GetField<ValueNodeTag>(U, I) = Input[I];
    plastix::GetField<ValueNodeTag>(U, OutputBegin) = Y;

    // Forward pass — computes eps = ValueNode - mu, stores in ErrorTag.
    Net.DoForwardPass(Input);

    // Weight update — theta += alpha * eps * f(x_src).
    Net.DoUpdateConnectionState();

    // mu = ValueNode - eps (reconstruct prediction from error).
    float Pred = plastix::GetField<ValueNodeTag>(U, OutputBegin) -
                 plastix::GetField<ErrorTag>(U, OutputBegin);
    float Loss = (Pred - Y) * (Pred - Y);
    AvgLoss += Loss;
    if ((T + 1) % PrintInterval == 0) {
      AvgLoss /= static_cast<float>(PrintInterval);
      std::cout << "Step " << std::setw(5) << T + 1
                << "  avg error^2: " << std::setw(8) << AvgLoss << "\n";
      AvgLoss = 0.0f;
    }
  }

  // Print learned weights. FullyConnected allocates one connection per
  // (dst, src) pair; the sort is stable within a single source level,
  // so connections 0..2 correspond to inputs 0..2.
  auto &CA = Net.GetConnAlloc();
  std::cout << "\nLearned weights: [";
  for (size_t C = 0; C < CA.Size(); ++C) {
    if (C > 0)
      std::cout << ", ";
    std::cout << plastix::GetWeight(CA, C);
  }
  std::cout << "]\n";
}
