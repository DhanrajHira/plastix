#include <plastix/plastix.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

// Sigmoid forward pass: weighted sum of inputs, squashed through sigmoid.
struct SigmoidForwardPass {
  using Accumulator = float;

  static float Map(auto &U, size_t, size_t SrcId, auto &C, size_t ConnId,
                   auto &) {
    return plastix::GetWeight(C, ConnId) * plastix::GetActivation(U, SrcId);
  }

  static float Combine(float A, float B) { return A + B; }

  static void Apply(auto &U, size_t Id, auto &, float Accumulated) {
    plastix::GetActivation(U, Id) = 1.0f / (1.0f + std::exp(-Accumulated));
  }
};

struct FccTraits : plastix::DefaultNetworkTraits<> {
  using ForwardPass = SigmoidForwardPass;
};

// Weight initializers for the FullyConnected layer builder.
template <int Scaled> struct ConstWeightInit {
  void operator()(auto &CA, auto Id) const {
    plastix::GetWeight(CA, Id) = Scaled / 100.0f;
  }
};

using FccNetwork = plastix::Network<FccTraits>;
using HiddenLayer = plastix::FullyConnected<ConstWeightInit<50>>;
using OutputLayer = plastix::FullyConnected<ConstWeightInit<-20>>;

int main() {
  std::cout << "Plastix FCC-NN Example\n";
  std::cout << "======================\n";

  // 2 inputs -> 4 hidden -> 1 output. Default Topological propagation
  // evaluates all levels in a single DoStep, so the output is ready after
  // one call.
  FccNetwork Net(2, HiddenLayer{4}, OutputLayer{1});

  std::vector<std::vector<float>> Inputs = {
      {0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}};

  std::cout << std::fixed << std::setprecision(4);

  for (const auto &In : Inputs) {
    std::cout << "Input: [" << In[0] << ", " << In[1] << "] ";
    Net.DoStep(In);
    auto Output = Net.GetOutput();
    std::cout << "=> Output: " << Output[0] << "\n";
  }

  return 0;
}
