#include <plastix/plastix.hpp>

#include <iomanip>
#include <iostream>
#include <random>

constexpr float LearningRate = 0.01f;
constexpr size_t NumInputs = 3;

// Identity activation — pure linear output.
struct LinearForwardPass {
  using Accumulator = float;

  PLASTIX_HD static float Map(auto &U, size_t, size_t SrcId, auto &C,
                              size_t ConnId, auto &) {
    return plastix::GetWeight(C, ConnId) * plastix::GetActivation(U, SrcId);
  }

  PLASTIX_HD static float Combine(float A, float B) { return A + B; }

  PLASTIX_HD static void Apply(auto &U, size_t Id, auto &, float Accumulated) {
    plastix::GetActivation(U, Id) = Accumulated;
  }
};

// Per-connection weight update: w -= lr * grad * input.
// MSELoss staged dL/dpred = (pred - target) into BackwardAcc(DstId) during
// DoCalculateLoss; the source activation still holds the input value from
// the forward pass, so we read it directly off the source unit.
struct GradientDescentConn {
  PLASTIX_HD static void UpdateIncomingConnection(auto &U, size_t DstId,
                                                  size_t SrcId, auto &C,
                                                  size_t ConnId, auto &) {
    float Grad = plastix::GetBackwardAcc(U, DstId);
    float Input = plastix::GetActivation(U, SrcId);
    plastix::GetWeight(C, ConnId) -= LearningRate * Grad * Input;
  }

  PLASTIX_HD static void UpdateOutgoingConnection(auto &, size_t, size_t,
                                                  auto &, size_t, auto &) {}
};

struct LRTraits : plastix::DefaultNetworkTraits<> {
  using ForwardPass = LinearForwardPass;
  using Loss = plastix::MSELoss;
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

  float FinalLoss = 0.0f;
  for (size_t T = 0; T < NumSteps; ++T) {
    float X1 = Dist(Rng), X2 = Dist(Rng), X3 = Dist(Rng);
    float Y = 2.0f * X1 - 1.0f * X2 + 0.5f * X3;
    float Input[3] = {X1, X2, X3};
    float Target[1] = {Y};

    Net.DoStep(Input, Target);

    float Pred = Net.GetOutput()[0];
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
    std::cout << plastix::GetWeight(CA, C);
  }
  std::cout << "]\n";

  std::cout << "\n" << (FinalLoss < 0.01f ? "PASS" : "FAIL") << "\n";
  return FinalLoss < 0.01f ? 0 : 1;
}
