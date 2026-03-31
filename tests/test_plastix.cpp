#include <gtest/gtest.h>

#include "plastix/plastix.hpp"
#include <array>
#include <cmath>

TEST(PlastixTest, Version) { EXPECT_STREQ(plastix::version(), "0.1.0"); }

namespace plastix_test {

using TestTraits = plastix::DefaultNetworkTraits<plastix::UnitStateAllocator,
                                                 plastix::ConnStateAllocator>;
using TestNetwork = plastix::Network<TestTraits>;

TEST(NetworkTest, ConnDefaultInitialization) {
  plastix::ConnStateAllocator Alloc(4);
  auto Id = Alloc.Allocate();

  EXPECT_EQ(Alloc.Get<plastix::FromIdTag>(Id), 0u);
  EXPECT_EQ(Alloc.Get<plastix::ToIdTag>(Id), 0u);
  EXPECT_FLOAT_EQ(Alloc.Get<plastix::WeightTag>(Id), 0.0f);
  EXPECT_FALSE(Alloc.Get<plastix::DeadTag>(Id));
}

TEST(NetworkTest, SingleLayerPerceptronConnections) {
  constexpr size_t InputDim = 3;
  constexpr size_t OutputDim = 2;
  TestNetwork Net(InputDim, OutputDim);

  auto &ConnAlloc = Net.GetConnAlloc();
  auto &UnitAlloc = Net.GetUnitAlloc();

  EXPECT_EQ(UnitAlloc.Size(), InputDim + OutputDim);
  // 2 outputs * 3 inputs = 6 individual connections
  EXPECT_EQ(ConnAlloc.Size(), InputDim * OutputDim);

  // FullyConnected allocates: for each output unit, for each source.
  // Output unit 0 (id=3): connections 0,1,2 from inputs 0,1,2
  // Output unit 1 (id=4): connections 3,4,5 from inputs 0,1,2
  for (size_t Out = 0; Out < OutputDim; ++Out) {
    for (size_t In = 0; In < InputDim; ++In) {
      size_t C = Out * InputDim + In;
      EXPECT_EQ(ConnAlloc.Get<plastix::ToIdTag>(C), InputDim + Out);
      EXPECT_EQ(ConnAlloc.Get<plastix::FromIdTag>(C), In);
      EXPECT_FLOAT_EQ(ConnAlloc.Get<plastix::WeightTag>(C), 1.0f);
    }
  }
}

TEST(NetworkTest, SingleLayerPerceptronManyConnections) {
  constexpr size_t InputDim = 10;
  constexpr size_t OutputDim = 1;
  TestNetwork Net(InputDim, OutputDim);

  auto &ConnAlloc = Net.GetConnAlloc();

  // 10 individual connections (was 2 pages)
  EXPECT_EQ(ConnAlloc.Size(), 10u);

  for (size_t I = 0; I < 10; ++I) {
    EXPECT_EQ(ConnAlloc.Get<plastix::FromIdTag>(I), I);
    EXPECT_EQ(ConnAlloc.Get<plastix::ToIdTag>(I), InputDim);
    EXPECT_FLOAT_EQ(ConnAlloc.Get<plastix::WeightTag>(I), 1.0f);
  }
}

struct ScaledForwardPass {
  static float Map(auto &, size_t, auto &, float Weight, float Activation) {
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
  // Step 0 (even): inputs written to A (previous), output written to B
  // (current).
  float Out = UA.Get<plastix::ActivationBTag>(2);
  EXPECT_FLOAT_EQ(Out, 8.0f);

  // Inputs are in the previous (A) buffer only.
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationATag>(0), 3.0f);
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationATag>(1), 5.0f);
}

TEST(PassTest, ForwardPassCustomPolicy) {
  // ScaledForwardPass: Map = 2 * W * A, Apply = tanh(Acc)
  // 2 inputs with weights=1.0, inputs={1.0, 2.0}
  // Acc = 2*1*1 + 2*1*2 = 6.0, output = tanh(6.0)
  plastix::Network<CustomForwardTraits> Net(2, 1);
  std::array<float, 2> In = {1.0f, 2.0f};
  Net.DoForwardPass(In);

  auto &UA = Net.GetUnitAlloc();
  float Out = UA.Get<plastix::ActivationBTag>(2);
  EXPECT_FLOAT_EQ(Out, std::tanh(6.0f));
}

TEST(PassTest, ForwardPassManyConnections) {
  // 10 inputs (each=1.0), 1 output — 10 individual connections
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
  // Inputs written to previous (B) only.
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationBTag>(0), 4.0f);
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationBTag>(1), 5.0f);
  // Previous output in B untouched by this pass.
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationBTag>(2), 3.0f);
}

struct GradientBackwardPass {
  static float Map(auto &, size_t, auto &, float Weight, float Activation) {
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
  // BackwardAcc is reset to 0 after CalculateAndApply (same pattern as
  // UpdateAcc), so the accumulated values are consumed and then cleared.
  EXPECT_FLOAT_EQ(UA.Get<plastix::BackwardAccTag>(0), 0.0f);
  EXPECT_FLOAT_EQ(UA.Get<plastix::BackwardAccTag>(1), 0.0f);
  EXPECT_FLOAT_EQ(UA.Get<plastix::BackwardAccTag>(2), 0.0f);

  // Activations are untouched by backward pass.
  // Inputs in A (previous at step 0), output in B (current at step 0).
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationATag>(0), 3.0f);
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationATag>(1), 5.0f);
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationBTag>(2), 8.0f);
}

// ---------------------------------------------------------------------------
// Layer builder tests
// ---------------------------------------------------------------------------

using FC = plastix::FullyConnected;

TEST(LayerBuilderTest, MultiLayerBuilder) {
  // 3 inputs -> 5 hidden -> 1 output = 9 units
  // Layer 1: 5 units * 3 inputs = 15 connections
  // Layer 2: 1 unit * 5 hidden = 5 connections
  // Total: 20 connections
  TestNetwork Net(3, FC{5}, FC{1});

  auto &UA = Net.GetUnitAlloc();
  auto &CA = Net.GetConnAlloc();

  EXPECT_EQ(UA.Size(), 9u);
  EXPECT_EQ(CA.Size(), 20u);

  // Hidden layer connections (units 3-7, each with 3 connections from inputs)
  for (size_t I = 0; I < 5; ++I) {
    for (size_t S = 0; S < 3; ++S) {
      size_t C = I * 3 + S;
      EXPECT_EQ(CA.Get<plastix::ToIdTag>(C), 3 + I);
      EXPECT_EQ(CA.Get<plastix::FromIdTag>(C), S);
      EXPECT_FLOAT_EQ(CA.Get<plastix::WeightTag>(C), 1.0f);
    }
  }

  // Output layer connections (unit 8, connected to units 3-7)
  for (size_t S = 0; S < 5; ++S) {
    size_t C = 15 + S;
    EXPECT_EQ(CA.Get<plastix::ToIdTag>(C), 8u);
    EXPECT_EQ(CA.Get<plastix::FromIdTag>(C), 3 + S);
    EXPECT_FLOAT_EQ(CA.Get<plastix::WeightTag>(C), 1.0f);
  }
}

TEST(LayerBuilderTest, MultiLayerManyConnections) {
  // 10 inputs -> 2 hidden -> 1 output = 13 units
  // Layer 1: 2 * 10 = 20 connections
  // Layer 2: 1 * 2 = 2 connections
  // Total: 22 connections
  TestNetwork Net(10, FC{2}, FC{1});

  auto &UA = Net.GetUnitAlloc();
  auto &CA = Net.GetConnAlloc();

  EXPECT_EQ(UA.Size(), 13u);
  EXPECT_EQ(CA.Size(), 22u);
}

TEST(LayerBuilderTest, CustomInitWeight) {
  TestNetwork Net(2, FC{1, 0.5f});

  auto &CA = Net.GetConnAlloc();
  EXPECT_EQ(CA.Size(), 2u);

  for (size_t C = 0; C < 2; ++C)
    EXPECT_FLOAT_EQ(CA.Get<plastix::WeightTag>(C), 0.5f);
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

  // Step 0 (even): inputs written to A (previous), accumulation writes to B.
  // Hidden reads prev inputs from A = {1,1}. Hidden[2]=Hidden[3]=2.
  // Output reads prev hidden from A (still 0) => Output[4]=0.
  std::array<float, 2> In = {1.0f, 1.0f};
  Net.DoForwardPass(In);
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationBTag>(2), 2.0f);
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationBTag>(3), 2.0f);
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationBTag>(4), 0.0f);

  // Step 1 (odd): inputs written to B (previous), accumulation writes to A.
  // Hidden reads prev inputs from B = {1,1} => 2. Output reads prev hidden
  // from B = {2,2} => 4. Signal has now propagated through both layers.
  Net.DoForwardPass(In);
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationATag>(4), 4.0f);
}

// ---------------------------------------------------------------------------
// GetOutput tests
// ---------------------------------------------------------------------------

TEST(OutputTest, SingleLayerGetOutput) {
  // 2 inputs, 1 output. Output = 3+5 = 8.
  TestNetwork Net(2, 1);
  std::array<float, 2> In = {3.0f, 5.0f};
  Net.DoForwardPass(In);

  auto Out = Net.GetOutput();
  ASSERT_EQ(Out.size(), 1u);
  EXPECT_FLOAT_EQ(Out[0], 8.0f);
}

TEST(OutputTest, MultipleOutputUnits) {
  // 3 inputs, 2 outputs. Each output = sum of inputs = 1+2+3 = 6.
  TestNetwork Net(3, 2);
  std::array<float, 3> In = {1.0f, 2.0f, 3.0f};
  Net.DoForwardPass(In);

  auto Out = Net.GetOutput();
  ASSERT_EQ(Out.size(), 2u);
  EXPECT_FLOAT_EQ(Out[0], 6.0f);
  EXPECT_FLOAT_EQ(Out[1], 6.0f);
}

TEST(OutputTest, MultiLayerGetOutput) {
  // 2 inputs -> 2 hidden -> 1 output, weights=1.
  // Pipelined: output is 0 after first step, 4 after second.
  TestNetwork Net(2, FC{2}, FC{1});
  std::array<float, 2> In = {1.0f, 1.0f};

  Net.DoForwardPass(In);
  auto Out1 = Net.GetOutput();
  ASSERT_EQ(Out1.size(), 1u);
  EXPECT_FLOAT_EQ(Out1[0], 0.0f);

  Net.DoForwardPass(In);
  auto Out2 = Net.GetOutput();
  ASSERT_EQ(Out2.size(), 1u);
  EXPECT_FLOAT_EQ(Out2[0], 4.0f);
}

TEST(OutputTest, GetOutputIsConst) {
  const TestNetwork Net(2, 1);
  auto Out = Net.GetOutput();
  EXPECT_EQ(Out.size(), 1u);
}

TEST(OutputTest, DoStepGetOutput) {
  // Verify GetOutput works after DoStep (full pipeline).
  TestNetwork Net(2, 1);
  std::array<float, 2> In = {2.0f, 3.0f};
  Net.DoStep(In);

  auto Out = Net.GetOutput();
  ASSERT_EQ(Out.size(), 1u);
  EXPECT_FLOAT_EQ(Out[0], 5.0f);
}

// ---------------------------------------------------------------------------
// Update unit state tests
// ---------------------------------------------------------------------------

struct SumWeightsUpdateUnit {
  using Partial = float;
  static Partial Map(auto &, size_t, size_t, auto &, float W) { return W; }
  static Partial Combine(Partial A, Partial B) { return A + B; }
  static void Apply(auto &U, size_t Id, auto &, Partial Acc) {
    U.template Get<plastix::BackwardAccTag>(Id) = Acc;
  }
};

struct UpdateUnitTraits
    : plastix::DefaultNetworkTraits<plastix::UnitStateAllocator,
                                    plastix::ConnStateAllocator> {
  using UpdateUnit = SumWeightsUpdateUnit;
};

TEST(UpdateTest, UpdateUnitStateMapReduce) {
  // 2 inputs, 1 output, weights=1.0.
  // Map returns weight, Combine sums, Apply writes to BackwardAcc.
  // Output unit (id=2) has 2 incoming connections with weight=1 each => Acc=2.
  // Input units (id=0,1) have no incoming connections => Acc=0.
  plastix::Network<UpdateUnitTraits> Net(2, 1);
  std::array<float, 2> In = {1.0f, 1.0f};
  Net.DoForwardPass(In);
  Net.DoUpdateUnitState();

  auto &UA = Net.GetUnitAlloc();
  EXPECT_FLOAT_EQ(UA.Get<plastix::BackwardAccTag>(0), 0.0f);
  EXPECT_FLOAT_EQ(UA.Get<plastix::BackwardAccTag>(1), 0.0f);
  EXPECT_FLOAT_EQ(UA.Get<plastix::BackwardAccTag>(2), 2.0f);
}

TEST(UpdateTest, UpdateUnitStateManyConnections) {
  // 10 inputs, 1 output (10 connections), weights=1.0.
  // Output unit accumulates 10 weights => Acc=10.
  plastix::Network<UpdateUnitTraits> Net(10, 1);
  std::array<float, 10> In;
  In.fill(1.0f);
  Net.DoForwardPass(In);
  Net.DoUpdateUnitState();

  auto &UA = Net.GetUnitAlloc();
  EXPECT_FLOAT_EQ(UA.Get<plastix::BackwardAccTag>(10), 10.0f);
  for (size_t I = 0; I < 10; ++I)
    EXPECT_FLOAT_EQ(UA.Get<plastix::BackwardAccTag>(I), 0.0f);
}

// ---------------------------------------------------------------------------
// Update connection state tests
// ---------------------------------------------------------------------------

struct WeightDecayUpdateConn {
  static void UpdateIncomingConnection(auto &, size_t, size_t, auto &C,
                                       size_t ConnId, auto &) {
    C.template Get<plastix::WeightTag>(ConnId) *= 0.5f;
  }
  static void UpdateOutgoingConnection(auto &, size_t, size_t, auto &, size_t,
                                       auto &) {}
};

struct UpdateConnTraits
    : plastix::DefaultNetworkTraits<plastix::UnitStateAllocator,
                                    plastix::ConnStateAllocator> {
  using UpdateConn = WeightDecayUpdateConn;
};

TEST(UpdateTest, UpdateConnStateWeightDecay) {
  // 2 inputs, 1 output, weights=1.0.
  // UpdateIncomingConnection halves weight => all weights become 0.5.
  plastix::Network<UpdateConnTraits> Net(2, 1);
  std::array<float, 2> In = {1.0f, 1.0f};
  Net.DoForwardPass(In);
  Net.DoUpdateConnectionState();

  auto &CA = Net.GetConnAlloc();
  EXPECT_FLOAT_EQ(CA.Get<plastix::WeightTag>(0), 0.5f);
  EXPECT_FLOAT_EQ(CA.Get<plastix::WeightTag>(1), 0.5f);
}

// ---------------------------------------------------------------------------
// DoStep with update policies
// ---------------------------------------------------------------------------

struct DoStepUpdateTraits
    : plastix::DefaultNetworkTraits<plastix::UnitStateAllocator,
                                    plastix::ConnStateAllocator> {
  using UpdateUnit = SumWeightsUpdateUnit;
  using UpdateConn = WeightDecayUpdateConn;
};

TEST(UpdateTest, DoStepWithUpdate) {
  // Full pipeline: forward + update unit + update conn.
  // After DoStep: BackwardAcc[2]=2 (sum of weights), weights halved to 0.5.
  plastix::Network<DoStepUpdateTraits> Net(2, 1);
  std::array<float, 2> In = {3.0f, 4.0f};
  Net.DoStep(In);

  auto &UA = Net.GetUnitAlloc();
  EXPECT_FLOAT_EQ(UA.Get<plastix::BackwardAccTag>(2), 2.0f);

  auto &CA = Net.GetConnAlloc();
  EXPECT_FLOAT_EQ(CA.Get<plastix::WeightTag>(0), 0.5f);
  EXPECT_FLOAT_EQ(CA.Get<plastix::WeightTag>(1), 0.5f);

  // Output should still be correct from forward pass: 3+4=7.
  auto Out = Net.GetOutput();
  ASSERT_EQ(Out.size(), 1u);
  EXPECT_FLOAT_EQ(Out[0], 7.0f);
}

// ---------------------------------------------------------------------------
// Prune tests
// ---------------------------------------------------------------------------

// Helper: count alive (non-dead) connections.
static size_t CountAlive(auto &ConnAlloc) {
  size_t Count = 0;
  for (size_t C = 0; C < ConnAlloc.Size(); ++C)
    if (!ConnAlloc.template Get<plastix::DeadTag>(C))
      ++Count;
  return Count;
}

// Policy that prunes a specific unit by index.
struct PruneUnitById {
  static bool ShouldPrune(auto &, size_t Id, auto &) {
    return Id == 2; // prune unit 2
  }
};

struct PruneUnitTraits
    : plastix::DefaultNetworkTraits<plastix::UnitStateAllocator,
                                    plastix::ConnStateAllocator> {
  using PruneUnit = PruneUnitById;
};

TEST(PruneTest, PruneUnitMarksFlag) {
  // 3 inputs, 1 output. Unit 2 should be marked pruned.
  plastix::Network<PruneUnitTraits> Net(3, 1);
  Net.DoPruneUnits();

  auto &UA = Net.GetUnitAlloc();
  EXPECT_FALSE(UA.Get<plastix::PrunedTag>(0));
  EXPECT_FALSE(UA.Get<plastix::PrunedTag>(1));
  EXPECT_TRUE(UA.Get<plastix::PrunedTag>(2));
  EXPECT_FALSE(UA.Get<plastix::PrunedTag>(3));
}

TEST(PruneTest, PruneUnitMarksSourceConnectionsDead) {
  // 3 inputs -> 1 output (unit 3). Sources are units 0,1,2.
  // Prune unit 2 => connection from unit 2 is marked dead.
  // 2 connections survive.
  plastix::Network<PruneUnitTraits> Net(3, 1);
  Net.DoPruneUnits();
  Net.DoPruneConnections();

  auto &CA = Net.GetConnAlloc();
  // Total connections unchanged (tombstone)
  EXPECT_EQ(CA.Size(), 3u);
  // 2 alive
  EXPECT_EQ(CountAlive(CA), 2u);

  // Verify the connection from unit 2 is dead
  for (size_t C = 0; C < CA.Size(); ++C) {
    if (CA.Get<plastix::FromIdTag>(C) == 2)
      EXPECT_TRUE(CA.Get<plastix::DeadTag>(C));
    else
      EXPECT_FALSE(CA.Get<plastix::DeadTag>(C));
  }
}

// Policy that prunes the output unit (last unit).
struct PruneDestUnit {
  static size_t TargetId;
  static bool ShouldPrune(auto &, size_t Id, auto &) { return Id == TargetId; }
};
size_t PruneDestUnit::TargetId = 0;

struct PruneDestTraits
    : plastix::DefaultNetworkTraits<plastix::UnitStateAllocator,
                                    plastix::ConnStateAllocator> {
  using PruneUnit = PruneDestUnit;
};

TEST(PruneTest, PruneDestinationMarksAllDead) {
  // 2 inputs -> 1 output (unit 2). Prune the output unit.
  // All connections targeting unit 2 should be marked dead.
  PruneDestUnit::TargetId = 2;
  plastix::Network<PruneDestTraits> Net(2, 1);
  Net.DoPruneUnits();
  Net.DoPruneConnections();

  auto &CA = Net.GetConnAlloc();
  EXPECT_EQ(CA.Size(), 2u);
  EXPECT_EQ(CountAlive(CA), 0u);
}

// Connection-level pruning policy.
struct PruneSmallWeight {
  static bool ShouldPrune(auto &, size_t, size_t, auto &C, size_t ConnId,
                          auto &) {
    return C.template Get<plastix::WeightTag>(ConnId) < 0.5f;
  }
};

struct PruneConnTraits
    : plastix::DefaultNetworkTraits<plastix::UnitStateAllocator,
                                    plastix::ConnStateAllocator> {
  using PruneConn = PruneSmallWeight;
};

TEST(PruneTest, PruneConnMarksAllDeadWhenAllPruned) {
  // 2 inputs -> 1 output, weights=1.0 initially.
  // Manually set weights below threshold, then prune.
  plastix::Network<PruneConnTraits> Net(2, 1);

  auto &CA = Net.GetConnAlloc();
  CA.Get<plastix::WeightTag>(0) = 0.1f;
  CA.Get<plastix::WeightTag>(1) = 0.2f;

  Net.DoPruneConnections();
  EXPECT_EQ(CA.Size(), 2u);
  EXPECT_EQ(CountAlive(CA), 0u);
}

TEST(PruneTest, PruneConnMarksPartialDead) {
  // 2 inputs -> 1 output. Conn[0] weight below threshold, Conn[1] survives.
  plastix::Network<PruneConnTraits> Net(2, 1);

  auto &CA = Net.GetConnAlloc();
  CA.Get<plastix::WeightTag>(0) = 0.1f; // below threshold — pruned
  CA.Get<plastix::WeightTag>(1) = 1.0f; // above threshold — survives

  Net.DoPruneConnections();
  EXPECT_EQ(CA.Size(), 2u);
  EXPECT_EQ(CountAlive(CA), 1u);

  // Connection 0 is dead, connection 1 is alive
  EXPECT_TRUE(CA.Get<plastix::DeadTag>(0));
  EXPECT_FALSE(CA.Get<plastix::DeadTag>(1));
  EXPECT_FLOAT_EQ(CA.Get<plastix::WeightTag>(1), 1.0f);
}

TEST(PruneTest, DoStepWithPrune) {
  // Full pipeline: forward, then prune destination unit.
  PruneDestUnit::TargetId = 2;
  plastix::Network<PruneDestTraits> Net(2, 1);
  std::array<float, 2> In = {1.0f, 2.0f};
  Net.DoStep(In);

  // After DoStep, all connections to the output unit are dead.
  auto &CA = Net.GetConnAlloc();
  EXPECT_EQ(CountAlive(CA), 0u);
}

} // namespace plastix_test
