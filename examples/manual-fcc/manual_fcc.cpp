#include <plastix/plastix.hpp>
#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip>

// ---------------------------------------------------------------------------
// 1. Manually defined ForwardPass policy implementing PassPolicy concept
// ---------------------------------------------------------------------------
struct TanhForwardPass {
    // Map function for weights and activations (standard product)
    static float Map(auto&, size_t, auto&, float Weight, float Activation) {
        return Weight * Activation;
    }

    // Tanh activation function
    static float CalculateAndApply(auto&, size_t, auto&, float Accumulated) {
        return std::tanh(Accumulated);
    }
};

// ---------------------------------------------------------------------------
// 2. Custom LayerBuilder — Manually implementing the fully connected logic
// ---------------------------------------------------------------------------
struct ManualLayerBuilder {
    size_t NumUnits;
    float InitialWeight;

    // The builder is a callable that allocates units and sets up connections
    template <typename UnitAlloc, typename ConnAlloc>
    plastix::UnitRange operator()(UnitAlloc &UA, ConnAlloc &CA, plastix::UnitRange PrevLayer) const {
        size_t Begin = UA.Size();

        // Step A: Manually allocate the units for this layer
        for (size_t I = 0; I < NumUnits; ++I) {
            UA.Allocate();
        }

        // Step B: Connect each new unit to every unit in the previous layer
        for (size_t TargetIdx = Begin; TargetIdx < Begin + NumUnits; ++TargetIdx) {
            size_t SlotIdx = 0;
            auto PageId = CA.Allocate(); // Initial page for this unit's incoming conns

            for (size_t SrcIdx = PrevLayer.Begin; SrcIdx < PrevLayer.End; ++SrcIdx) {
                // If a connection page is full, allocate another one
                if (SlotIdx == plastix::ConnPageSlotSize) {
                    PageId = CA.Allocate();
                    SlotIdx = 0;
                }

                auto &Page = CA.template Get<plastix::ConnPageMarker>(PageId);
                Page.ToUnitIdx = TargetIdx;
                Page.Count = SlotIdx + 1;
                Page.Conn[SlotIdx] = {static_cast<uint32_t>(SrcIdx), InitialWeight};

                ++SlotIdx;
            }
        }

        return {Begin, Begin + NumUnits};
    }
};

// ---------------------------------------------------------------------------
// 3. Network configuration using traits and custom components
// ---------------------------------------------------------------------------
struct ManualFccTraits : plastix::DefaultNetworkTraits<
    plastix::UnitStateAllocator,
    plastix::ConnStateAllocator
> {
    using ForwardPass = TanhForwardPass;
};

using ManualNetwork = plastix::Network<ManualFccTraits>;

int main() {
    std::cout << "Plastix Manual FCC-NN Example" << std::endl;
    std::cout << "=============================" << std::endl;

    // Instantiate network: 2 inputs -> 4 hidden -> 1 output
    // Using our custom ManualLayerBuilder instead of plastix::FullyConnected
    ManualNetwork net(2,
        ManualLayerBuilder{4, 0.5f},
        ManualLayerBuilder{1, -0.3f}
    );

    std::vector<std::vector<float>> inputs = {
        {0.1f, 0.2f},
        {0.5f, -0.5f},
        {1.0f, 1.0f}
    };

    std::cout << std::fixed << std::setprecision(6);

    // In this architecture, it takes N steps to propagate through N layers
    // Input -> Layer 1 (Hidden) -> Layer 2 (Output) = 2 steps
    for (const auto& in : inputs) {
        std::cout << "Input: [" << in[0] << ", " << in[1] << "] ";

        // Propagate through the network
        net.DoStep(in); // Hidden layer becomes active
        net.DoStep(in); // Output layer becomes active

        auto output = net.GetOutput();
        std::cout << "=> Output (Tanh): " << output[0] << std::endl;
    }

    return 0;
}
