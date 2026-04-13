#include <plastix/plastix.hpp>

#include <iomanip>
#include <iostream>
#include <vector>

// Identity forward pass — easier to trace numerically than sigmoid/tanh
// when following signal propagation layer by layer.
struct IdentityForwardPass {
  using Accumulator = float;

  static float Map(auto &U, size_t, size_t SrcId, auto &C, size_t ConnId,
                   auto &) {
    return plastix::GetWeight(C, ConnId) * plastix::GetActivation(U, SrcId);
  }

  static float Combine(float A, float B) { return A + B; }

  static void Apply(auto &U, size_t Id, auto &, float Accumulated) {
    plastix::GetActivation(U, Id) = Accumulated;
  }
};

// Pipeline propagation: each DoStep walks every live connection exactly
// once and applies every non-input unit in one sweep. A signal therefore
// advances exactly one layer per DoStep, versus the default Topological
// model which flushes the whole network in a single call.
struct PipelineFccTraits : plastix::DefaultNetworkTraits<> {
  using ForwardPass = IdentityForwardPass;
  static constexpr plastix::Propagation Model = plastix::Propagation::Pipeline;
};

struct HalfWeightInit {
  void operator()(auto &CA, auto Id) const {
    plastix::GetWeight(CA, Id) = 0.5f;
  }
};

using PipelineNetwork = plastix::Network<PipelineFccTraits>;
using FC = plastix::FullyConnected<HalfWeightInit>;

static void PrintActivations(PipelineNetwork &Net, size_t NumInput,
                             size_t NumHidden, size_t NumOutput) {
  auto &UA = Net.GetUnitAlloc();
  std::cout << "  inputs  = [";
  for (size_t I = 0; I < NumInput; ++I) {
    std::cout << plastix::GetActivation(UA, I);
    if (I + 1 < NumInput)
      std::cout << ", ";
  }
  std::cout << "]\n  hidden  = [";
  for (size_t I = 0; I < NumHidden; ++I) {
    std::cout << plastix::GetActivation(UA, NumInput + I);
    if (I + 1 < NumHidden)
      std::cout << ", ";
  }
  std::cout << "]\n  outputs = [";
  for (size_t I = 0; I < NumOutput; ++I) {
    std::cout << plastix::GetActivation(UA, NumInput + NumHidden + I);
    if (I + 1 < NumOutput)
      std::cout << ", ";
  }
  std::cout << "]\n";
}

int main() {
  std::cout << "Plastix Pipeline FCC Example\n";
  std::cout << "============================\n\n";

  // 2 inputs -> 3 hidden -> 1 output. All weights initialized to 0.5.
  // Hand-computable: if every input is 1.0, hidden units see 2 * 0.5 = 1.0,
  // and the output sees 3 * 0.5 = 1.5. But because the model is Pipeline,
  // that value only appears at the output after the signal has had time
  // to walk across both layers.
  constexpr size_t NumInput = 2;
  constexpr size_t NumHidden = 3;
  constexpr size_t NumOutput = 1;
  PipelineNetwork Net(NumInput, FC{NumHidden}, FC{NumOutput});

  std::cout << std::fixed << std::setprecision(4);

  // ---------------------------------------------------------------------
  // Part 1: hold a single input constant and watch the signal march
  // forward one layer per step.
  // ---------------------------------------------------------------------
  std::cout << "Part 1: constant input [1.0, 1.0]\n";
  std::cout << "Every DoStep advances the signal by exactly one layer.\n\n";

  std::vector<float> In = {1.0f, 1.0f};

  std::cout << "Before any step (fresh network):\n";
  PrintActivations(Net, NumInput, NumHidden, NumOutput);

  // Step 1: the forward pass reads current activations of every unit,
  // accumulates into ForwardAcc(ToId) across all connections, then writes
  // the result into ActivationTag for every non-input unit. Inputs are
  // written at the start of DoForwardPass. After this step:
  //   - hidden units now hold 0.5*1 + 0.5*1 = 1.0 (reading the inputs)
  //   - output unit still holds 0 because hidden units were zero when
  //     the forward pass read them.
  Net.DoStep(In);
  std::cout << "\nAfter step 1 (hidden layer activated):\n";
  PrintActivations(Net, NumInput, NumHidden, NumOutput);

  // Step 2: the forward pass now sees hidden = 1.0, so the output picks
  // up 3 * 0.5 * 1.0 = 1.5. Hidden units also refresh from the (still
  // constant) inputs, keeping their value at 1.0.
  Net.DoStep(In);
  std::cout << "\nAfter step 2 (output layer activated):\n";
  PrintActivations(Net, NumInput, NumHidden, NumOutput);

  // ---------------------------------------------------------------------
  // Part 2: feed a new input every step and watch the pipeline delay.
  // Output at step N reflects input from step N - (NumLayers - 1) = N - 1
  // here, because we have two non-input layers (hidden, output).
  // ---------------------------------------------------------------------
  std::cout << "\nPart 2: streaming inputs through the pipeline\n";
  std::cout << "Each row feeds one new input and prints the output that\n";
  std::cout << "emerges the same step. The output trails the input by one\n";
  std::cout << "step because the signal has to cross the hidden layer.\n\n";

  std::vector<std::vector<float>> Stream = {
      {2.0f, 2.0f}, {0.0f, 0.0f}, {4.0f, 4.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}};

  std::cout << "  step | input        | output\n";
  std::cout << "  -----+--------------+--------\n";
  for (size_t S = 0; S < Stream.size(); ++S) {
    Net.DoStep(Stream[S]);
    auto Out = Net.GetOutput();
    std::cout << "   " << S + 1 << "   | [" << Stream[S][0] << ", "
              << Stream[S][1] << "] | " << Out[0] << "\n";
  }

  return 0;
}
