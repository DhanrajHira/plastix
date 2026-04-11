#include <plastix/plastix.hpp>

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

// Tanh forward pass.
struct TanhForwardPass {
  using Accumulator = float;

  static float Map(auto &U, size_t, size_t SrcId, auto &C, size_t ConnId,
                   auto &) {
    return C.template Get<plastix::WeightTag>(ConnId) *
           U.template Get<plastix::ActivationTag>(SrcId);
  }

  static float Combine(float A, float B) { return A + B; }

  static void Apply(auto &U, size_t Id, auto &, float Accumulated) {
    U.template Get<plastix::ActivationTag>(Id) = std::tanh(Accumulated);
  }
};

// A custom LayerBuilder that allocates units and connections by hand,
// mirroring what plastix::FullyConnected does internally. Demonstrates
// the raw allocator interface for users who need more control than the
// built-in layer builders provide.
struct ManualLayerBuilder {
  size_t NumUnits;
  float InitialWeight;

  template <typename UnitAlloc, typename ConnAlloc>
  plastix::UnitRange operator()(UnitAlloc &UA, ConnAlloc &CA,
                                plastix::UnitRange PrevLayer) const {
    uint16_t SrcLevel = UA.template Get<plastix::LevelTag>(PrevLayer.Begin);
    uint16_t NewLevel = SrcLevel + 1;

    plastix::UnitRange Units = UA.AllocateMany(NumUnits);
    for (auto Id : Units.Ids())
      UA.template Get<plastix::LevelTag>(Id) = NewLevel;

    for (auto Dst : Units.Ids()) {
      for (auto Src : PrevLayer.Ids()) {
        auto ConnId = CA.Allocate();
        CA.template Get<plastix::FromIdTag>(ConnId) =
            static_cast<uint32_t>(Src);
        CA.template Get<plastix::ToIdTag>(ConnId) = static_cast<uint32_t>(Dst);
        CA.template Get<plastix::SrcLevelTag>(ConnId) = SrcLevel;
        CA.template Get<plastix::WeightTag>(ConnId) = InitialWeight;
      }
    }

    return Units;
  }
};

struct ManualFccTraits : plastix::DefaultNetworkTraits<> {
  using ForwardPass = TanhForwardPass;
};

using ManualNetwork = plastix::Network<ManualFccTraits>;

int main() {
  std::cout << "Plastix Manual FCC-NN Example\n";
  std::cout << "=============================\n";

  // 2 inputs -> 4 hidden -> 1 output, built via a user-provided
  // LayerBuilder instead of plastix::FullyConnected.
  ManualNetwork Net(2, ManualLayerBuilder{4, 0.5f},
                    ManualLayerBuilder{1, -0.3f});

  std::vector<std::vector<float>> Inputs = {
      {0.1f, 0.2f}, {0.5f, -0.5f}, {1.0f, 1.0f}};

  std::cout << std::fixed << std::setprecision(6);

  for (const auto &In : Inputs) {
    std::cout << "Input: [" << In[0] << ", " << In[1] << "] ";
    Net.DoStep(In);
    auto Output = Net.GetOutput();
    std::cout << "=> Output (Tanh): " << Output[0] << "\n";
  }

  return 0;
}
