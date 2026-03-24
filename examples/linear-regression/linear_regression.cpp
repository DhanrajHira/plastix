#include <plastix/plastix.hpp>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <random>

constexpr float LearningRate = 0.01f;
constexpr size_t NumInputs = 3;

// Identity activation — pure linear output.
struct LinearForwardPass {
  static float Map(auto &, size_t, auto &, float Weight, float Activation) {
    return Weight * Activation;
  }
  static float CalculateAndApply(auto &, size_t, auto &, float Accumulated) {
    return Accumulated;
  }
};

// Update each weight: w += lr * error * input.
// Error is pre-computed in BackwardAcc, input values in UpdateAcc.
struct GradientDescentConn {
  static void UpdateIncomingConnection(auto &U, size_t DstId, size_t SrcId,
                                       auto &C, size_t PageId, size_t SlotIdx,
                                       auto &) {
    float Error = U.template Get<plastix::BackwardAccTag>(DstId);
    float Input = U.template Get<plastix::UpdateAccTag>(SrcId);
    auto &Page = C.template Get<plastix::ConnPageMarker>(PageId);
    Page.Conn[SlotIdx].second += LearningRate * Error * Input;
  }
  static void UpdateOutgoingConnection(auto &, size_t, size_t, auto &, size_t,
                                       size_t, auto &) {}
};

struct LRTraits
    : plastix::DefaultNetworkTraits<plastix::UnitStateAllocator,
                                    plastix::ConnStateAllocator> {
  using ForwardPass = LinearForwardPass;
  using UpdateConn = GradientDescentConn;
};

using LRNetwork = plastix::Network<LRTraits>;
using FC = plastix::FullyConnected;

int main() {
  std::cout << "Plastix Linear Regression Example" << std::endl;
  std::cout << "==================================" << std::endl;

  // 3 inputs -> 1 output, weights initialized to 0.
  // True relationship: y = 2*x1 - x2 + 0.5*x3
  LRNetwork net(NumInputs, FC{1, 0.0f});

  std::mt19937 Rng(42);
  std::uniform_real_distribution<float> Dist(-1.0f, 1.0f);

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "Target weights: [2.0, -1.0, 0.5]\n" << std::endl;

  constexpr size_t NumSteps = 2000;
  constexpr size_t PrintInterval = 200;
  constexpr size_t OutputId = NumInputs; // Output unit follows input units.

  float FinalLoss = 0.0f;
  for (size_t T = 0; T < NumSteps; ++T) {
    float X1 = Dist(Rng), X2 = Dist(Rng), X3 = Dist(Rng);
    float Y = 2.0f * X1 - 1.0f * X2 + 0.5f * X3;
    float Input[3] = {X1, X2, X3};

    net.DoForwardPass(Input);

    // Compute error and stage values for the connection update.
    auto &U = net.GetUnitAlloc();
    float Pred = net.GetOutput()[0];
    U.Get<plastix::BackwardAccTag>(OutputId) = Y - Pred;
    for (size_t I = 0; I < NumInputs; ++I)
      U.Get<plastix::UpdateAccTag>(I) = Input[I];

    net.DoUpdateConnectionState();

    float Loss = (Pred - Y) * (Pred - Y);
    if (T % PrintInterval == 0)
      std::cout << "Step " << std::setw(5) << T
                << "  error^2: " << std::setw(8) << Loss << std::endl;
    FinalLoss = Loss;
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

  std::cout << "\n" << (FinalLoss < 0.01f ? "PASS" : "FAIL") << std::endl;

  return FinalLoss < 0.01f ? 0 : 1;
}
