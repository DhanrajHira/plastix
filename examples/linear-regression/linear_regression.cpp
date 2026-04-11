#include <plastix/plastix.hpp>

#include <iomanip>
#include <iostream>
#include <random>

constexpr float LearningRate = 0.01f;
constexpr size_t NumInputs = 3;

// Identity activation — pure linear output.
struct LinearForwardPass {
  using Accumulator = float;

  static float Map(auto &U, size_t, size_t SrcId, auto &C, size_t ConnId,
                   auto &) {
    return plastix::GetField<plastix::WeightTag>(C, ConnId) *
           plastix::GetField<plastix::ActivationTag>(U, SrcId);
  }

  static float Combine(float A, float B) { return A + B; }

  static void Apply(auto &U, size_t Id, auto &, float Accumulated) {
    plastix::GetField<plastix::ActivationTag>(U, Id) = Accumulated;
  }
};

// Per-connection weight update: w += lr * error * input.
// The error for the destination unit is staged by the outer loop into
// BackwardAcc(OutputId); the source activation still holds the input
// value from the forward pass, so we can read it directly off the
// source unit — no dedicated "update accumulator" field needed.
struct GradientDescentConn {
  static void UpdateIncomingConnection(auto &U, size_t DstId, size_t SrcId,
                                       auto &C, size_t ConnId, auto &) {
    float Error = plastix::GetField<plastix::BackwardAccTag>(U, DstId);
    float Input = plastix::GetField<plastix::ActivationTag>(U, SrcId);
    plastix::GetField<plastix::WeightTag>(C, ConnId) +=
        LearningRate * Error * Input;
  }

  static void UpdateOutgoingConnection(auto &, size_t, size_t, auto &, size_t,
                                       auto &) {}
};

struct LRTraits : plastix::DefaultNetworkTraits<> {
  using ForwardPass = LinearForwardPass;
  using UpdateConn = GradientDescentConn;
};

using LRNetwork = plastix::Network<LRTraits>;
using FC = plastix::FullyConnected<>;

int main() {
  std::cout << "Plastix Linear Regression Example\n";
  std::cout << "==================================\n";

  // 3 inputs -> 1 output. NoConnInit leaves WeightTag at its default-
  // constructed value of 0.0f (placement-new zero-inits POD fields).
  // True relationship: y = 2*x1 - x2 + 0.5*x3.
  LRNetwork Net(NumInputs, FC{1});

  std::mt19937 Rng(42);
  std::uniform_real_distribution<float> Dist(-1.0f, 1.0f);

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "Target weights: [2.0, -1.0, 0.5]\n\n";

  constexpr size_t NumSteps = 2000;
  constexpr size_t PrintInterval = 200;
  constexpr size_t OutputId = NumInputs; // Output unit follows inputs.

  float FinalLoss = 0.0f;
  for (size_t T = 0; T < NumSteps; ++T) {
    float X1 = Dist(Rng), X2 = Dist(Rng), X3 = Dist(Rng);
    float Y = 2.0f * X1 - 1.0f * X2 + 0.5f * X3;
    float Input[3] = {X1, X2, X3};

    Net.DoForwardPass(Input);

    // Stage the error on the output unit for the update step to consume.
    auto &U = Net.GetUnitAlloc();
    float Pred = Net.GetOutput()[0];
    plastix::GetField<plastix::BackwardAccTag>(U, OutputId) = Y - Pred;

    Net.DoUpdateConnectionState();

    float Loss = (Pred - Y) * (Pred - Y);
    if (T % PrintInterval == 0)
      std::cout << "Step " << std::setw(5) << T << "  error^2: " << std::setw(8)
                << Loss << "\n";
    FinalLoss = Loss;
  }

  // Print learned weights. FullyConnected allocates one connection per
  // (dst, src) pair in source-id order, and the level-sort is stable
  // within a source level, so connections 0..2 correspond to inputs
  // 0..2 feeding the single output unit.
  auto &CA = Net.GetConnAlloc();
  std::cout << "\nLearned weights: [";
  for (size_t C = 0; C < CA.Size(); ++C) {
    if (C > 0)
      std::cout << ", ";
    std::cout << plastix::GetField<plastix::WeightTag>(CA, C);
  }
  std::cout << "]\n";

  std::cout << "\n" << (FinalLoss < 0.01f ? "PASS" : "FAIL") << "\n";
  return FinalLoss < 0.01f ? 0 : 1;
}
