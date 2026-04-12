#include <plastix/plastix.hpp>
#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip>

// Simple Sigmoid activation function policy
struct SigmoidForwardPass {
    static float Map(auto&, size_t, auto&, float Weight, float Activation) {
        return Weight * Activation;
    }

    static float CalculateAndApply(auto&, size_t, auto&, float Accumulated) {
        return 1.0f / (1.0f + std::exp(-Accumulated));
    }
};

// Traits using our custom ForwardPass
struct FccTraits : plastix::DefaultNetworkTraits<
    plastix::UnitStateAllocator,
    plastix::ConnStateAllocator
> {
    using ForwardPass = SigmoidForwardPass;
};

using FccNetwork = plastix::Network<FccTraits>;
using FC = plastix::FullyConnected;

int main() {
    std::cout << "Plastix FCC-NN Example" << std::endl;
    std::cout << "======================" << std::endl;

    // Create a simple network: 2 inputs -> 4 hidden -> 1 output
    // We use different initial weights for variety
    FccNetwork net(2, FC{4, 0.5f}, FC{1, -0.2f});

    // Example inputs (XOR-like data)
    std::vector<std::vector<float>> inputs = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };

    std::cout << std::fixed << std::setprecision(4);

    // In a pipelined network, it takes N steps for the signal to reach the output,
    // where N is the number of layers (including input).
    // Here: Input -> Hidden (1 step) -> Output (2 steps)

    for (const auto& in : inputs) {
        std::cout << "Input: [" << in[0] << ", " << in[1] << "] ";

        // Step 1: Input propagates to Hidden
        net.DoStep(in);
        net.DoStep(in);

        auto output = net.GetOutput();
        std::cout << "=> Output: " << output[0] << std::endl;
    }

    return 0;
}
