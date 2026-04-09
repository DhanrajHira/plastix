#include <gtest/gtest.h>

#include "plastix/plastix.hpp"
#include <array>
#include <cmath>
#include <optional>

TEST(PlastixTest, Version) { EXPECT_STREQ(plastix::version(), "0.1.0"); }

namespace plastix_test {

using TestTraits = plastix::DefaultNetworkTraits<>;
using TestNetwork = plastix::Network<TestTraits>;

struct WeightOneInit {
  void operator()(auto &CA, auto Id) const {
    CA.template Get<plastix::WeightTag>(Id) = 1.0f;
  }
};
using FC = plastix::FullyConnected<plastix::NoUnitInit, WeightOneInit>;

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
  TestNetwork Net(InputDim, FC{OutputDim});

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
  TestNetwork Net(InputDim, FC{OutputDim});

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
  using Accumulator = float;
  static float Map(auto &U, size_t, size_t SrcId, auto &C, size_t ConnId,
                   auto &) {
    return 2.0f * C.template Get<plastix::WeightTag>(ConnId) *
           U.template Get<plastix::ActivationTag>(SrcId);
  }
  static float Combine(float A, float B) { return A + B; }
  static void Apply(auto &U, size_t Id, auto &, float Accumulated) {
    U.template Get<plastix::ActivationTag>(Id) = std::tanh(Accumulated);
  }
};

struct CustomForwardTraits : plastix::DefaultNetworkTraits<> {
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

struct CustomGlobalTraits : plastix::DefaultNetworkTraits<TestGlobalState> {};

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

  // Backward does not increment step.
  Net.DoBackwardPass();
  EXPECT_EQ(Net.GetStep(), 1u);
}

TEST(PassTest, ForwardPassIdentity) {
  // 2 inputs, 1 output, weights=1.0
  // Output = sum of inputs = 3.0 + 5.0 = 8.0
  TestNetwork Net(2, FC{1});
  std::array<float, 2> In = {3.0f, 5.0f};
  Net.DoForwardPass(In);

  auto &UA = Net.GetUnitAlloc();
  float Out = UA.Get<plastix::ActivationTag>(2);
  EXPECT_FLOAT_EQ(Out, 8.0f);

  // Inputs written to Activation.
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationTag>(0), 3.0f);
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationTag>(1), 5.0f);
}

TEST(PassTest, ForwardPassCustomPolicy) {
  // ScaledForwardPass: Map = 2 * W * A, Apply = tanh(Acc)
  // 2 inputs with weights=1.0, inputs={1.0, 2.0}
  // Acc = 2*1*1 + 2*1*2 = 6.0, output = tanh(6.0)
  plastix::Network<CustomForwardTraits> Net(2, FC{1});
  std::array<float, 2> In = {1.0f, 2.0f};
  Net.DoForwardPass(In);

  auto &UA = Net.GetUnitAlloc();
  float Out = UA.Get<plastix::ActivationTag>(2);
  EXPECT_FLOAT_EQ(Out, std::tanh(6.0f));
}

TEST(PassTest, ForwardPassManyConnections) {
  // 10 inputs (each=1.0), 1 output — 10 individual connections
  // Output = sum of 10 * 1.0 = 10.0
  TestNetwork Net(10, FC{1});
  std::array<float, 10> In;
  In.fill(1.0f);
  Net.DoForwardPass(In);

  auto &UA = Net.GetUnitAlloc();
  float Out = UA.Get<plastix::ActivationTag>(10);
  EXPECT_FLOAT_EQ(Out, 10.0f);
}

TEST(PassTest, ConsecutiveForwardPasses) {
  // Two forward passes with different inputs — second overwrites first.
  TestNetwork Net(2, FC{1});

  // First pass: output = 1+2 = 3
  std::array<float, 2> In1 = {1.0f, 2.0f};
  Net.DoForwardPass(In1);
  auto &UA = Net.GetUnitAlloc();
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationTag>(2), 3.0f);

  // Second pass: output = 4+5 = 9.
  std::array<float, 2> In2 = {4.0f, 5.0f};
  Net.DoForwardPass(In2);
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationTag>(2), 9.0f);
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationTag>(0), 4.0f);
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationTag>(1), 5.0f);
}

struct GradientBackwardPass {
  using Accumulator = float;
  static float Map(auto &U, size_t, size_t ToId, auto &C, size_t ConnId,
                   auto &) {
    return C.template Get<plastix::WeightTag>(ConnId) *
           U.template Get<plastix::ActivationTag>(ToId);
  }
  static float Combine(float A, float B) { return A + B; }
  static void Apply(auto &, size_t, auto &, float) {}
};

struct GradientBackwardTraits : plastix::DefaultNetworkTraits<> {
  using BackwardPass = GradientBackwardPass;
};

TEST(PassTest, BackwardPassBasic) {
  // Forward: 2 inputs, 1 output. inputs={3,5}, output=8.
  // Backward: reads output activation (8) from Activation,
  // accumulates weight*8 into BackwardAcc for source units.
  plastix::Network<GradientBackwardTraits> Net(2, FC{1});
  std::array<float, 2> In = {3.0f, 5.0f};
  Net.DoForwardPass(In);
  EXPECT_EQ(Net.GetStep(), 1u);

  Net.DoBackwardPass();
  // Step unchanged — backward doesn't increment.
  EXPECT_EQ(Net.GetStep(), 1u);

  auto &UA = Net.GetUnitAlloc();
  // BackwardAcc is reset to 0 after Apply, so the accumulated
  // values are consumed and then cleared.
  EXPECT_FLOAT_EQ(UA.Get<plastix::BackwardAccTag>(0), 0.0f);
  EXPECT_FLOAT_EQ(UA.Get<plastix::BackwardAccTag>(1), 0.0f);
  EXPECT_FLOAT_EQ(UA.Get<plastix::BackwardAccTag>(2), 0.0f);

  // Activations are untouched by backward pass.
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationTag>(0), 3.0f);
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationTag>(1), 5.0f);
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationTag>(2), 8.0f);
}

// ---------------------------------------------------------------------------
// Layer builder tests
// ---------------------------------------------------------------------------

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

struct HalfWeightInit {
  void operator()(auto &CA, auto Id) const {
    CA.template Get<plastix::WeightTag>(Id) = 0.5f;
  }
};

TEST(LayerBuilderTest, CustomConnInit) {
  using HalfFC = plastix::FullyConnected<plastix::NoUnitInit, HalfWeightInit>;
  TestNetwork Net(2, HalfFC{1});

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
  // Level-based: all layers evaluated in a single DoForwardPass call.
  TestNetwork Net(2, FC{2}, FC{1});

  auto &UA = Net.GetUnitAlloc();

  // Hidden = {1+1, 1+1} = {2, 2}. Output = 2+2 = 4.
  std::array<float, 2> In = {1.0f, 1.0f};
  Net.DoForwardPass(In);
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationTag>(2), 2.0f);
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationTag>(3), 2.0f);
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationTag>(4), 4.0f);
}

// ---------------------------------------------------------------------------
// GetOutput tests
// ---------------------------------------------------------------------------

TEST(OutputTest, SingleLayerGetOutput) {
  // 2 inputs, 1 output. Output = 3+5 = 8.
  TestNetwork Net(2, FC{1});
  std::array<float, 2> In = {3.0f, 5.0f};
  Net.DoForwardPass(In);

  auto Out = Net.GetOutput();
  ASSERT_EQ(Out.size(), 1u);
  EXPECT_FLOAT_EQ(Out[0], 8.0f);
}

TEST(OutputTest, MultipleOutputUnits) {
  // 3 inputs, 2 outputs. Each output = sum of inputs = 1+2+3 = 6.
  TestNetwork Net(3, FC{2});
  std::array<float, 3> In = {1.0f, 2.0f, 3.0f};
  Net.DoForwardPass(In);

  auto Out = Net.GetOutput();
  ASSERT_EQ(Out.size(), 2u);
  EXPECT_FLOAT_EQ(Out[0], 6.0f);
  EXPECT_FLOAT_EQ(Out[1], 6.0f);
}

TEST(OutputTest, MultiLayerGetOutput) {
  // 2 inputs -> 2 hidden -> 1 output, weights=1.
  // Level-based: output is 4 after first step.
  TestNetwork Net(2, FC{2}, FC{1});
  std::array<float, 2> In = {1.0f, 1.0f};

  Net.DoForwardPass(In);
  auto Out1 = Net.GetOutput();
  ASSERT_EQ(Out1.size(), 1u);
  EXPECT_FLOAT_EQ(Out1[0], 4.0f);
}

TEST(OutputTest, GetOutputIsConst) {
  const TestNetwork Net(2, 1);
  auto Out = Net.GetOutput();
  EXPECT_EQ(Out.size(), 1u);
}

TEST(OutputTest, DoStepGetOutput) {
  // Verify GetOutput works after DoStep (full pipeline).
  TestNetwork Net(2, FC{1});
  std::array<float, 2> In = {2.0f, 3.0f};
  Net.DoStep(In);

  auto Out = Net.GetOutput();
  ASSERT_EQ(Out.size(), 1u);
  EXPECT_FLOAT_EQ(Out[0], 5.0f);
}

// ---------------------------------------------------------------------------
// Update unit state tests
// ---------------------------------------------------------------------------

struct CopyActivationToBackwardAcc {
  static void Update(auto &U, size_t Id, auto &) {
    U.template Get<plastix::BackwardAccTag>(Id) =
        U.template Get<plastix::ActivationTag>(Id);
  }
};

struct UpdateUnitTraits : plastix::DefaultNetworkTraits<> {
  using UpdateUnit = CopyActivationToBackwardAcc;
};

TEST(UpdateTest, UpdateUnitState) {
  // 2 inputs, 1 output, weights=1.0.
  // Update copies Activation into BackwardAcc for every unit.
  plastix::Network<UpdateUnitTraits> Net(2, FC{1});
  std::array<float, 2> In = {3.0f, 4.0f};
  Net.DoForwardPass(In);
  Net.DoUpdateUnitState();

  auto &UA = Net.GetUnitAlloc();
  EXPECT_FLOAT_EQ(UA.Get<plastix::BackwardAccTag>(0), 3.0f);
  EXPECT_FLOAT_EQ(UA.Get<plastix::BackwardAccTag>(1), 4.0f);
  // Output activation = 3*1 + 4*1 = 7
  EXPECT_FLOAT_EQ(UA.Get<plastix::BackwardAccTag>(2), 7.0f);
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

struct UpdateConnTraits : plastix::DefaultNetworkTraits<> {
  using UpdateConn = WeightDecayUpdateConn;
};

TEST(UpdateTest, UpdateConnStateWeightDecay) {
  // 2 inputs, 1 output, weights=1.0.
  // UpdateIncomingConnection halves weight => all weights become 0.5.
  plastix::Network<UpdateConnTraits> Net(2, FC{1});
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

struct DoStepUpdateTraits : plastix::DefaultNetworkTraits<> {
  using UpdateUnit = CopyActivationToBackwardAcc;
  using UpdateConn = WeightDecayUpdateConn;
};

TEST(UpdateTest, DoStepWithUpdate) {
  // Full pipeline: forward + update unit + update conn.
  // After DoStep: BackwardAcc copies activation, weights halved to 0.5.
  plastix::Network<DoStepUpdateTraits> Net(2, FC{1});
  std::array<float, 2> In = {3.0f, 4.0f};
  Net.DoStep(In);

  auto &UA = Net.GetUnitAlloc();
  // Output activation = 3*1 + 4*1 = 7, copied to BackwardAcc.
  EXPECT_FLOAT_EQ(UA.Get<plastix::BackwardAccTag>(2), 7.0f);

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

struct PruneUnitTraits : plastix::DefaultNetworkTraits<> {
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

struct PruneDestTraits : plastix::DefaultNetworkTraits<> {
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

struct PruneConnTraits : plastix::DefaultNetworkTraits<> {
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

// ---------------------------------------------------------------------------
// AddUnit tests
// ---------------------------------------------------------------------------

// Policy that adds a new unit at level 5 for unit 0 only.
struct AddOneUnit {
  static std::optional<uint16_t> AddUnit(auto &, size_t Id, auto &) {
    if (Id == 0)
      return uint16_t{5};
    return std::nullopt;
  }
};

struct AddUnitTraits : plastix::DefaultNetworkTraits<> {
  using AddUnit = AddOneUnit;
};

TEST(AddUnitTest, NoopDoesNotAddUnits) {
  // Default traits have NoAddUnit — DoAddUnits is a no-op.
  TestNetwork Net(2, 1);
  EXPECT_EQ(Net.GetUnitAlloc().Size(), 3u);
  Net.DoAddUnits();
  EXPECT_EQ(Net.GetUnitAlloc().Size(), 3u);
}

TEST(AddUnitTest, AddsUnitAtReturnedLevel) {
  // AddOneUnit returns level 5 for unit 0 only => 1 new unit.
  plastix::Network<AddUnitTraits> Net(2, 1);
  EXPECT_EQ(Net.GetUnitAlloc().Size(), 3u);

  Net.DoAddUnits();
  EXPECT_EQ(Net.GetUnitAlloc().Size(), 4u);

  // New unit (id=3) should be at level 5.
  auto &UA = Net.GetUnitAlloc();
  EXPECT_EQ(UA.Get<plastix::LevelTag>(3), uint16_t{5});
}

TEST(AddUnitTest, NulloptReturnDoesNotAdd) {
  // AddOneUnit returns nullopt for all units except unit 0.
  // With 2 inputs + 1 output = 3 units, only unit 0 triggers an add.
  plastix::Network<AddUnitTraits> Net(2, 1);
  Net.DoAddUnits();
  // Exactly 1 new unit added (from unit 0).
  EXPECT_EQ(Net.GetUnitAlloc().Size(), 4u);
}

TEST(AddUnitTest, DoesNotIterateNewlyAddedUnits) {
  // The loop captures Size() before iterating, so newly added units
  // in this call are not visited.
  plastix::Network<AddUnitTraits> Net(2, 1);
  Net.DoAddUnits(); // adds 1 unit (from unit 0)
  EXPECT_EQ(Net.GetUnitAlloc().Size(), 4u);

  // Second call: now 4 units exist, but only unit 0 triggers an add.
  Net.DoAddUnits();
  EXPECT_EQ(Net.GetUnitAlloc().Size(), 5u);
}

TEST(AddUnitTest, NewUnitDefaultActivation) {
  // Newly added unit should have zero activation.
  plastix::Network<AddUnitTraits> Net(2, 1);
  Net.DoAddUnits();

  auto &UA = Net.GetUnitAlloc();
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationTag>(3), 0.0f);
  EXPECT_FALSE(UA.Get<plastix::PrunedTag>(3));
}

// ---------------------------------------------------------------------------
// AddConn tests
// ---------------------------------------------------------------------------

// Policy that adds one incoming connection: unit 0 accepts from unit 1.
struct AddOneIncoming {
  static bool ShouldAddIncomingConnection(auto &, size_t Self, size_t Candidate,
                                          auto &) {
    return Self == 0 && Candidate == 1;
  }
  static bool ShouldAddOutgoingConnection(auto &, size_t, size_t, auto &) {
    return false;
  }
  static void InitConnection(auto &, size_t, size_t, auto &C, size_t ConnId,
                             auto &) {
    C.template Get<plastix::WeightTag>(ConnId) = 0.5f;
  }
};

struct AddIncomingTraits : plastix::DefaultNetworkTraits<> {
  using AddConn = AddOneIncoming;
};

// Policy that adds one outgoing connection: unit 1 sends to unit 0.
struct AddOneOutgoing {
  static bool ShouldAddIncomingConnection(auto &, size_t, size_t, auto &) {
    return false;
  }
  static bool ShouldAddOutgoingConnection(auto &, size_t Self, size_t Candidate,
                                          auto &) {
    return Self == 1 && Candidate == 0;
  }
  static void InitConnection(auto &, size_t, size_t, auto &C, size_t ConnId,
                             auto &) {
    C.template Get<plastix::WeightTag>(ConnId) = 0.75f;
  }
};

struct AddOutgoingTraits : plastix::DefaultNetworkTraits<> {
  using AddConn = AddOneOutgoing;
};

// Policy that adds via both methods: incoming 1→0, outgoing 2→0.
struct AddBothDirections {
  static bool ShouldAddIncomingConnection(auto &, size_t Self, size_t Candidate,
                                          auto &) {
    return Self == 0 && Candidate == 1;
  }
  static bool ShouldAddOutgoingConnection(auto &, size_t Self, size_t Candidate,
                                          auto &) {
    return Self == 2 && Candidate == 0;
  }
  static void InitConnection(auto &, size_t SrcId, size_t, auto &C,
                             size_t ConnId, auto &) {
    // Incoming 1→0 gets 0.3, outgoing 2→0 gets 0.7. Distinguish by SrcId.
    C.template Get<plastix::WeightTag>(ConnId) = (SrcId == 1) ? 0.3f : 0.7f;
  }
};

struct AddBothTraits : plastix::DefaultNetworkTraits<> {
  using AddConn = AddBothDirections;
};

TEST(AddConnTest, NoopDoesNotAddConnections) {
  TestNetwork Net(2, 1);
  EXPECT_EQ(Net.GetConnAlloc().Size(), 2u);
  Net.DoAddConnections();
  EXPECT_EQ(Net.GetConnAlloc().Size(), 2u);
}

TEST(AddConnTest, AddsIncomingConnection) {
  // 2 inputs (0,1) + 1 output (2) = 3 units, 2 initial connections.
  // Policy adds incoming to unit 0 from unit 1 with weight 0.5.
  plastix::Network<AddIncomingTraits> Net(2, 1);
  EXPECT_EQ(Net.GetConnAlloc().Size(), 2u);

  Net.DoAddConnections();

  auto &CA = Net.GetConnAlloc();
  EXPECT_EQ(CA.Size(), 3u);

  // New connection (id=2): From=1, To=0, Weight=0.5
  EXPECT_EQ(CA.Get<plastix::FromIdTag>(2), 1u);
  EXPECT_EQ(CA.Get<plastix::ToIdTag>(2), 0u);
  EXPECT_FLOAT_EQ(CA.Get<plastix::WeightTag>(2), 0.5f);
  EXPECT_FALSE(CA.Get<plastix::DeadTag>(2));
}

TEST(AddConnTest, AddsOutgoingConnection) {
  // Policy adds outgoing from unit 1 to unit 0 with weight 0.75.
  plastix::Network<AddOutgoingTraits> Net(2, 1);
  Net.DoAddConnections();

  auto &CA = Net.GetConnAlloc();
  EXPECT_EQ(CA.Size(), 3u);

  // New connection (id=2): From=1, To=0, Weight=0.75
  EXPECT_EQ(CA.Get<plastix::FromIdTag>(2), 1u);
  EXPECT_EQ(CA.Get<plastix::ToIdTag>(2), 0u);
  EXPECT_FLOAT_EQ(CA.Get<plastix::WeightTag>(2), 0.75f);
}

TEST(AddConnTest, BothIncomingAndOutgoing) {
  // Policy adds two connections: incoming 1→0 (0.3), outgoing 2→0 (0.7).
  plastix::Network<AddBothTraits> Net(2, 1);
  Net.DoAddConnections();

  auto &CA = Net.GetConnAlloc();
  EXPECT_EQ(CA.Size(), 4u);

  // Find the two new connections by weight.
  bool FoundIncoming = false, FoundOutgoing = false;
  for (size_t C = 2; C < CA.Size(); ++C) {
    auto From = CA.Get<plastix::FromIdTag>(C);
    auto To = CA.Get<plastix::ToIdTag>(C);
    auto W = CA.Get<plastix::WeightTag>(C);
    if (From == 1 && To == 0 && std::abs(W - 0.3f) < 1e-6f)
      FoundIncoming = true;
    if (From == 2 && To == 0 && std::abs(W - 0.7f) < 1e-6f)
      FoundOutgoing = true;
  }
  EXPECT_TRUE(FoundIncoming);
  EXPECT_TRUE(FoundOutgoing);
}

TEST(AddConnTest, CalledTwiceAddsAgain) {
  // Policy fires every call — no dedup, so two calls = two connections.
  plastix::Network<AddIncomingTraits> Net(2, 1);
  Net.DoAddConnections();
  EXPECT_EQ(Net.GetConnAlloc().Size(), 3u);
  Net.DoAddConnections();
  EXPECT_EQ(Net.GetConnAlloc().Size(), 4u);
}

// Policy that adds a connection 0→2 with weight 0.5 (targets the output unit).
struct AddConnToOutput {
  static bool ShouldAddIncomingConnection(auto &, size_t Self, size_t Candidate,
                                          auto &) {
    return Self == 2 && Candidate == 0;
  }
  static bool ShouldAddOutgoingConnection(auto &, size_t, size_t, auto &) {
    return false;
  }
  static void InitConnection(auto &, size_t, size_t, auto &C, size_t ConnId,
                             auto &) {
    C.template Get<plastix::WeightTag>(ConnId) = 0.5f;
  }
};

struct AddConnToOutputTraits : plastix::DefaultNetworkTraits<> {
  using AddConn = AddConnToOutput;
};

TEST(AddConnTest, NewConnectionParticipatesInForwardPass) {
  // 2 inputs (0,1), 1 output (2). Initial: 0→2 (w=1), 1→2 (w=1).
  // inputs={2, 3} => output = 1*2 + 1*3 = 5.
  plastix::Network<AddConnToOutputTraits> Net(2, FC{1});
  std::array<float, 2> In = {2.0f, 3.0f};
  Net.DoForwardPass(In);
  EXPECT_FLOAT_EQ(Net.GetOutput()[0], 5.0f);

  // Add connection 0→2 with weight 0.5.
  Net.DoAddConnections();

  // Now output = 1*2 + 1*3 + 0.5*2 = 6.
  Net.DoForwardPass(In);
  EXPECT_FLOAT_EQ(Net.GetOutput()[0], 6.0f);
}

// ---------------------------------------------------------------------------
// Level and sort tests
// ---------------------------------------------------------------------------

TEST(LevelTest, InputsAtLevelZero) {
  TestNetwork Net(3, 1);
  auto &UA = Net.GetUnitAlloc();
  for (size_t I = 0; I < 3; ++I)
    EXPECT_EQ(UA.Get<plastix::LevelTag>(I), 0u);
}

TEST(LevelTest, FCLayerLevelProgression) {
  // 2 inputs (level 0) -> 3 hidden (level 1) -> 1 output (level 2)
  TestNetwork Net(2, FC{3}, FC{1});
  auto &UA = Net.GetUnitAlloc();

  EXPECT_EQ(UA.Get<plastix::LevelTag>(0), 0u); // input 0
  EXPECT_EQ(UA.Get<plastix::LevelTag>(1), 0u); // input 1
  EXPECT_EQ(UA.Get<plastix::LevelTag>(2), 1u); // hidden 0
  EXPECT_EQ(UA.Get<plastix::LevelTag>(3), 1u); // hidden 1
  EXPECT_EQ(UA.Get<plastix::LevelTag>(4), 1u); // hidden 2
  EXPECT_EQ(UA.Get<plastix::LevelTag>(5), 2u); // output
}

TEST(LevelTest, ConnectionsSortedBySrcLevel) {
  // 2 inputs -> 2 hidden -> 1 output
  // After sort: level-0 source connections first, then level-1.
  TestNetwork Net(2, FC{2}, FC{1});
  auto &CA = Net.GetConnAlloc();

  uint16_t PrevLevel = 0;
  for (size_t C = 0; C < CA.Size(); ++C) {
    uint16_t Lvl = CA.Get<plastix::SrcLevelTag>(C);
    EXPECT_GE(Lvl, PrevLevel);
    PrevLevel = Lvl;
  }
}

TEST(LevelTest, ThreeLayerSinglePassPropagation) {
  // 3 inputs -> 2 hidden -> 2 hidden -> 1 output, all weights=1
  // Input = {1, 1, 1}
  // Hidden layer 1: each = 1+1+1 = 3 (2 units)
  // Hidden layer 2: each = 3+3 = 6 (2 units)
  // Output: 6+6 = 12
  TestNetwork Net(3, FC{2}, FC{2}, FC{1});
  std::array<float, 3> In = {1.0f, 1.0f, 1.0f};
  Net.DoForwardPass(In);
  EXPECT_FLOAT_EQ(Net.GetOutput()[0], 12.0f);
}

TEST(LevelTest, AddConnectionTriggersResort) {
  // After DoAddConnections, a subsequent DoForwardPass should see the new
  // connection (which requires re-sorting).
  plastix::Network<AddConnToOutputTraits> Net(2, FC{1});
  std::array<float, 2> In = {2.0f, 3.0f};

  Net.DoForwardPass(In);
  EXPECT_FLOAT_EQ(Net.GetOutput()[0], 5.0f);

  Net.DoAddConnections(); // adds 0→2 with w=0.5
  Net.DoForwardPass(In);  // should re-sort and include new connection
  EXPECT_FLOAT_EQ(Net.GetOutput()[0], 6.0f);
}

// ---------------------------------------------------------------------------
// Neighbourhood filtering tests
// ---------------------------------------------------------------------------

// Policy that tries to add incoming connections between all pairs.
struct AddAllIncoming {
  static bool ShouldAddIncomingConnection(auto &, size_t Self, size_t Candidate,
                                          auto &) {
    return Self > Candidate;
  }
  static bool ShouldAddOutgoingConnection(auto &, size_t, size_t, auto &) {
    return false;
  }
  static void InitConnection(auto &, size_t, size_t, auto &C, size_t ConnId,
                             auto &) {
    C.template Get<plastix::WeightTag>(ConnId) = 1.0f;
  }
};

struct ZeroNeighbourhoodTraits : plastix::DefaultNetworkTraits<> {
  using AddConn = AddOneIncoming;
  static constexpr uint16_t Neighbourhood = 0;
};

TEST(NeighbourhoodTest, ZeroNeighbourhoodBlocksCrossLevelConnections) {
  // 2 inputs (level 0), 1 output (level 1).
  // AddOneIncoming: unit 0 accepts from unit 1 (same level, dist=0, allowed).
  // The policy also checks pairs like (2,0) and (2,1) which have dist=1,
  // but those are blocked by Neighbourhood=0.
  plastix::Network<ZeroNeighbourhoodTraits> Net(2, 1);
  Net.DoAddConnections();

  auto &CA = Net.GetConnAlloc();
  // Original 2 connections + 1 new (0 from 1, same level) = 3
  EXPECT_EQ(CA.Size(), 3u);
}

struct LargeNeighbourhoodTraits : plastix::DefaultNetworkTraits<> {
  using AddConn = AddConnToOutput;
  static constexpr uint16_t Neighbourhood = 1024;
};

TEST(NeighbourhoodTest, LargeNeighbourhoodAllowsAllPairs) {
  // Same behavior as unlimited — no filtering.
  plastix::Network<LargeNeighbourhoodTraits> Net(2, FC{1});
  std::array<float, 2> In = {2.0f, 3.0f};
  Net.DoForwardPass(In);
  EXPECT_FLOAT_EQ(Net.GetOutput()[0], 5.0f);

  Net.DoAddConnections();
  Net.DoForwardPass(In);
  EXPECT_FLOAT_EQ(Net.GetOutput()[0], 6.0f);
}

struct FilteredNeighbourhoodTraits : plastix::DefaultNetworkTraits<> {
  using AddConn = AddAllIncoming;
  static constexpr uint16_t Neighbourhood = 1;
};

TEST(NeighbourhoodTest, NeighbourhoodFiltersDistantLevels) {
  // 2 inputs (L0) -> 2 hidden (L1) -> 1 output (L2).
  // Neighbourhood=1: input-to-output (distance 2) should be filtered.
  plastix::Network<FilteredNeighbourhoodTraits> Net(2, FC{2}, FC{1});
  size_t ConnsBefore = Net.GetConnAlloc().Size();

  Net.DoAddConnections();

  // Verify no new connection has level distance > 1.
  auto &CA = Net.GetConnAlloc();
  auto &UA = Net.GetUnitAlloc();
  for (size_t C = ConnsBefore; C < CA.Size(); ++C) {
    auto From = CA.Get<plastix::FromIdTag>(C);
    auto To = CA.Get<plastix::ToIdTag>(C);
    uint16_t FL = UA.Get<plastix::LevelTag>(From);
    uint16_t TL = UA.Get<plastix::LevelTag>(To);
    uint16_t Dist = (FL >= TL) ? (FL - TL) : (TL - FL);
    EXPECT_LE(Dist, 1u);
  }

  // Also verify that some connections were added (policy is not vacuous).
  EXPECT_GT(CA.Size(), ConnsBefore);
}

} // namespace plastix_test
