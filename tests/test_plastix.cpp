#include <gtest/gtest.h>

#include "plastix/plastix.hpp"
#include <array>
#include <cmath>

TEST(PlastixTest, Version) { EXPECT_STREQ(plastix::version(), "0.1.0"); }

namespace plastix_test {

using TestTraits = plastix::DefaultNetworkTraits<plastix::UnitStateAllocator,
                                                 plastix::ConnStateAllocator>;
using TestNetwork = plastix::Network<TestTraits>;

TEST(NetworkTest, ConnPageDefaultInitialization) {
  plastix::ConnStateAllocator Alloc(4);
  auto Id = Alloc.Allocate();
  auto &Page = Alloc.Get<plastix::ConnPageMarker>(Id);

  EXPECT_EQ(Page.Count, 0u);
  EXPECT_EQ(Page.ToUnitIdx, 0u);
  for (size_t I = 0; I < plastix::ConnPageSlotSize; ++I) {
    EXPECT_EQ(Page.Conn[I].first, 0u);
    EXPECT_FLOAT_EQ(Page.Conn[I].second, 0.0f);
  }
}

TEST(NetworkTest, SingleLayerPerceptronConnections) {
  constexpr size_t InputDim = 3;
  constexpr size_t OutputDim = 2;
  TestNetwork Net(InputDim, OutputDim);

  auto &ConnAlloc = Net.GetConnAlloc();
  auto &UnitAlloc = Net.GetUnitAlloc();

  EXPECT_EQ(UnitAlloc.Size(), InputDim + OutputDim);
  EXPECT_EQ(ConnAlloc.Size(), OutputDim);

  for (size_t Out = 0; Out < OutputDim; ++Out) {
    auto &Page = ConnAlloc.Get<plastix::ConnPageMarker>(Out);
    EXPECT_EQ(Page.ToUnitIdx, InputDim + Out);
    EXPECT_EQ(Page.Count, InputDim);

    for (size_t In = 0; In < InputDim; ++In) {
      EXPECT_EQ(Page.Conn[In].first, In);
      EXPECT_FLOAT_EQ(Page.Conn[In].second, 1.0f);
    }
  }
}

TEST(NetworkTest, SingleLayerPerceptronMultiplePages) {
  constexpr size_t InputDim = 10;
  constexpr size_t OutputDim = 1;
  TestNetwork Net(InputDim, OutputDim);

  auto &ConnAlloc = Net.GetConnAlloc();

  EXPECT_EQ(ConnAlloc.Size(), 2u);

  auto &Page0 = ConnAlloc.Get<plastix::ConnPageMarker>(0);
  EXPECT_EQ(Page0.ToUnitIdx, InputDim);
  EXPECT_EQ(Page0.Count, plastix::ConnPageSlotSize);
  for (size_t I = 0; I < plastix::ConnPageSlotSize; ++I) {
    EXPECT_EQ(Page0.Conn[I].first, I);
    EXPECT_FLOAT_EQ(Page0.Conn[I].second, 1.0f);
  }

  auto &Page1 = ConnAlloc.Get<plastix::ConnPageMarker>(1);
  EXPECT_EQ(Page1.ToUnitIdx, InputDim);
  EXPECT_EQ(Page1.Count, InputDim - plastix::ConnPageSlotSize);
  for (size_t I = 0; I < InputDim - plastix::ConnPageSlotSize; ++I) {
    EXPECT_EQ(Page1.Conn[I].first, plastix::ConnPageSlotSize + I);
    EXPECT_FLOAT_EQ(Page1.Conn[I].second, 1.0f);
  }
}

struct ScaledForwardPass {
  static float Accumulate(auto &, size_t, auto &, float Weight,
                          float Activation) {
    return 2.0f * Weight * Activation;
  }
  static float CalculateAndApply(auto &, size_t, auto &, float Accumulated) {
    return std::tanh(Accumulated);
  }
};

struct CustomForwardTraits
    : plastix::DefaultNetworkTraits<plastix::UnitStateAllocator,
                                    plastix::ConnStateAllocator> {
  using ForwardPass = ScaledForwardPass;
};

TEST(NetworkTraitsTest, CustomForwardPass) {
  static_assert(plastix::NetworkTraits<CustomForwardTraits>);
  plastix::Network<CustomForwardTraits> Net(3, 1);
  EXPECT_EQ(Net.GetUnitAlloc().Size(), 4u);
}

struct TestGlobalState {
  float LearningRate = 0.01f;
  float DropoutRate = 0.4f;
};

struct CustomGlobalTraits
    : plastix::DefaultNetworkTraits<plastix::UnitStateAllocator,
                                    plastix::ConnStateAllocator,
                                    TestGlobalState> {};

TEST(NetworkTraitsTest, CustomGlobalState) {
  static_assert(plastix::NetworkTraits<CustomGlobalTraits>);
  plastix::Network<CustomGlobalTraits> Net(3, 1);
  EXPECT_EQ(Net.GetUnitAlloc().Size(), 4u);
}

// ---------------------------------------------------------------------------
// Forward / Backward pass tests
// ---------------------------------------------------------------------------

TEST(PassTest, StepCounter) {
  TestNetwork Net(2, 1);
  EXPECT_EQ(Net.GetStep(), 0u);

  std::array<float, 2> In = {1.0f, 2.0f};
  Net.DoForwardPass(In);
  EXPECT_EQ(Net.GetStep(), 1u);

  // Backward does not increment step — it doesn't swap activation buffers.
  Net.DoBackwardPass();
  EXPECT_EQ(Net.GetStep(), 1u);
}

TEST(PassTest, ForwardPassIdentity) {
  // 2 inputs, 1 output, default weights=1.0
  // Output = sum of inputs = 3.0 + 5.0 = 8.0
  TestNetwork Net(2, 1);
  std::array<float, 2> In = {3.0f, 5.0f};
  Net.DoForwardPass(In);

  auto &UA = Net.GetUnitAlloc();
  // After step 0->1 (odd step), current buffer was B (even step when written).
  // Step is now 1. Previous = B (the buffer just written).
  // But we want the values written during the pass — those are in the buffer
  // that was "current" during the pass (step=0, even => B).
  float Out = UA.Get<plastix::ActivationBTag>(2);
  EXPECT_FLOAT_EQ(Out, 8.0f);

  // Inputs should also be in the current (B) buffer.
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationBTag>(0), 3.0f);
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationBTag>(1), 5.0f);
}

TEST(PassTest, ForwardPassCustomPolicy) {
  // ScaledForwardPass: Accumulate = 2 * W * A, Apply = tanh(Acc)
  // 2 inputs with weights=1.0, inputs={1.0, 2.0}
  // Acc = 2*1*1 + 2*1*2 = 6.0, output = tanh(6.0)
  plastix::Network<CustomForwardTraits> Net(2, 1);
  std::array<float, 2> In = {1.0f, 2.0f};
  Net.DoForwardPass(In);

  auto &UA = Net.GetUnitAlloc();
  float Out = UA.Get<plastix::ActivationBTag>(2);
  EXPECT_FLOAT_EQ(Out, std::tanh(6.0f));
}

TEST(PassTest, ForwardPassMultiplePages) {
  // 10 inputs (each=1.0), 1 output — spans 2 conn pages
  // Output = sum of 10 * 1.0 = 10.0
  TestNetwork Net(10, 1);
  std::array<float, 10> In;
  In.fill(1.0f);
  Net.DoForwardPass(In);

  auto &UA = Net.GetUnitAlloc();
  float Out = UA.Get<plastix::ActivationBTag>(10);
  EXPECT_FLOAT_EQ(Out, 10.0f);
}

TEST(PassTest, ConsecutiveForwardPasses) {
  // Two forward passes with different inputs — verify buffer swap.
  TestNetwork Net(2, 1);

  // First pass (step 0, writes to B)
  std::array<float, 2> In1 = {1.0f, 2.0f};
  Net.DoForwardPass(In1);
  auto &UA = Net.GetUnitAlloc();
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationBTag>(2), 3.0f);

  // Second pass (step 1, writes to A, reads from B)
  std::array<float, 2> In2 = {4.0f, 5.0f};
  Net.DoForwardPass(In2);
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationATag>(2), 9.0f);
  // Inputs in A buffer
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationATag>(0), 4.0f);
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationATag>(1), 5.0f);
  // Previous buffer (B) has new inputs (overwritten) but old output
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationBTag>(0), 4.0f);
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationBTag>(1), 5.0f);
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationBTag>(2), 3.0f);
}

struct GradientBackwardPass {
  static float Accumulate(auto &, size_t, auto &, float Weight,
                          float Activation) {
    return Weight * Activation;
  }
  static float CalculateAndApply(auto &, size_t, auto &, float) { return 0.0f; }
};

struct GradientBackwardTraits
    : plastix::DefaultNetworkTraits<plastix::UnitStateAllocator,
                                    plastix::ConnStateAllocator> {
  using BackwardPass = GradientBackwardPass;
};

TEST(PassTest, BackwardPassBasic) {
  // Forward: 2 inputs, 1 output. inputs={3,5}, output=8.
  // Backward: reads output activation (8) from previous buffer,
  // accumulates weight*8 into BackwardAcc for source units.
  plastix::Network<GradientBackwardTraits> Net(2, 1);
  std::array<float, 2> In = {3.0f, 5.0f};
  Net.DoForwardPass(In);
  EXPECT_EQ(Net.GetStep(), 1u);

  Net.DoBackwardPass();
  // Step unchanged — backward doesn't swap buffers.
  EXPECT_EQ(Net.GetStep(), 1u);

  auto &UA = Net.GetUnitAlloc();
  // BackwardAcc[0] = 1.0 * 8.0 = 8.0
  // BackwardAcc[1] = 1.0 * 8.0 = 8.0
  // BackwardAcc[2] = 0.0 (output has no incoming backward connections)
  EXPECT_FLOAT_EQ(UA.Get<plastix::BackwardAccTag>(0), 8.0f);
  EXPECT_FLOAT_EQ(UA.Get<plastix::BackwardAccTag>(1), 8.0f);
  EXPECT_FLOAT_EQ(UA.Get<plastix::BackwardAccTag>(2), 0.0f);

  // Activations are untouched by backward pass.
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationBTag>(0), 3.0f);
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationBTag>(1), 5.0f);
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationBTag>(2), 8.0f);
}

// ---------------------------------------------------------------------------
// Layer builder tests
// ---------------------------------------------------------------------------

using FC = plastix::FullyConnected;

TEST(LayerBuilderTest, MultiLayerBuilder) {
  // 3 inputs -> 5 hidden -> 1 output = 9 units
  // Layer 1: 5 units, each connected to 3 inputs = 5 pages (3 < 7 slots)
  // Layer 2: 1 unit connected to 5 hidden = 1 page (5 < 7 slots)
  // Total: 6 pages
  TestNetwork Net(3, FC{5}, FC{1});

  auto &UA = Net.GetUnitAlloc();
  auto &CA = Net.GetConnAlloc();

  EXPECT_EQ(UA.Size(), 9u);
  EXPECT_EQ(CA.Size(), 6u);

  // Hidden layer pages (units 3-7, each with 1 page of 3 connections)
  for (size_t I = 0; I < 5; ++I) {
    auto &Page = CA.Get<plastix::ConnPageMarker>(I);
    EXPECT_EQ(Page.ToUnitIdx, 3 + I);
    EXPECT_EQ(Page.Count, 3u);
    for (size_t S = 0; S < 3; ++S) {
      EXPECT_EQ(Page.Conn[S].first, S);
      EXPECT_FLOAT_EQ(Page.Conn[S].second, 1.0f);
    }
  }

  // Output layer page (unit 8, connected to units 3-7)
  auto &OutPage = CA.Get<plastix::ConnPageMarker>(5);
  EXPECT_EQ(OutPage.ToUnitIdx, 8u);
  EXPECT_EQ(OutPage.Count, 5u);
  for (size_t S = 0; S < 5; ++S) {
    EXPECT_EQ(OutPage.Conn[S].first, 3 + S);
    EXPECT_FLOAT_EQ(OutPage.Conn[S].second, 1.0f);
  }
}

TEST(LayerBuilderTest, MultiLayerMultiplePages) {
  // 10 inputs -> 2 hidden -> 1 output = 13 units
  // Layer 1: 2 units, each connected to 10 inputs = 2*2 = 4 pages
  // Layer 2: 1 unit connected to 2 hidden = 1 page
  // Total: 5 pages
  TestNetwork Net(10, FC{2}, FC{1});

  auto &UA = Net.GetUnitAlloc();
  auto &CA = Net.GetConnAlloc();

  EXPECT_EQ(UA.Size(), 13u);
  EXPECT_EQ(CA.Size(), 5u);
}

TEST(LayerBuilderTest, CustomInitWeight) {
  TestNetwork Net(2, FC{1, 0.5f});

  auto &CA = Net.GetConnAlloc();
  EXPECT_EQ(CA.Size(), 1u);

  auto &Page = CA.Get<plastix::ConnPageMarker>(0);
  EXPECT_EQ(Page.Count, 2u);
  for (size_t S = 0; S < 2; ++S)
    EXPECT_FLOAT_EQ(Page.Conn[S].second, 0.5f);
}

TEST(LayerBuilderTest, ThreeLayerBuilder) {
  // 4 inputs -> 3 hidden1 -> 2 hidden2 -> 1 output = 10 units
  TestNetwork Net(4, FC{3}, FC{2}, FC{1});

  auto &UA = Net.GetUnitAlloc();
  EXPECT_EQ(UA.Size(), 10u);
}

TEST(LayerBuilderTest, MultiLayerForwardPass) {
  // 2 inputs -> 2 hidden -> 1 output, all weights=1
  // Pipelined semantics: signals propagate one layer per DoForwardPass call.
  TestNetwork Net(2, FC{2}, FC{1});

  auto &UA = Net.GetUnitAlloc();

  // Step 0: inputs={1,1}. Hidden reads previous buffer (zeros) => hidden=0.
  // Output reads previous buffer (zeros) => output=0.
  std::array<float, 2> In = {1.0f, 1.0f};
  Net.DoForwardPass(In);
  // Step was 0 (even), wrote to B buffer.
  // Hidden units (2,3) get prev activations of inputs (A buffer = 0 initially).
  // Actually inputs are written to BOTH buffers, so prev=A has 1.0.
  // Hidden[2] = 1+1 = 2, Hidden[3] = 1+1 = 2
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationBTag>(2), 2.0f);
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationBTag>(3), 2.0f);
  // Output[4] reads previous activation of hidden (A buffer = 0 initially)
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationBTag>(4), 0.0f);

  // Step 1: same inputs. Hidden reads prev (B) = {1,1} => 2 again.
  // Output reads prev (B) hidden = {2,2} => 4.
  Net.DoForwardPass(In);
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationATag>(4), 4.0f);
}

} // namespace plastix_test
