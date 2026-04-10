#include <gtest/gtest.h>

#include "plastix/plastix.hpp"
#include <array>
#include <cmath>
#include <optional>

namespace plastix_pipeline_test {

// ---------------------------------------------------------------------------
// Pipeline traits and helpers
// ---------------------------------------------------------------------------

struct PipelineTraits : plastix::DefaultNetworkTraits<> {
  static constexpr plastix::Propagation Model = plastix::Propagation::Pipeline;
};

using TestNetwork = plastix::Network<PipelineTraits>;

struct WeightOneInit {
  void operator()(auto &CA, auto Id) const {
    CA.template Get<plastix::WeightTag>(Id) = 1.0f;
  }
};
using FC = plastix::FullyConnected<plastix::NoUnitInit, WeightOneInit>;

static size_t CountAlive(auto &ConnAlloc) {
  size_t Count = 0;
  for (size_t C = 0; C < ConnAlloc.Size(); ++C)
    if (!ConnAlloc.template Get<plastix::DeadTag>(C))
      ++Count;
  return Count;
}

// ---------------------------------------------------------------------------
// Connection initialization tests
// ---------------------------------------------------------------------------

TEST(PipelineNetworkTest, ConnDefaultInitialization) {
  plastix::ConnStateAllocator Alloc(4);
  auto Id = Alloc.Allocate();

  EXPECT_EQ(Alloc.Get<plastix::FromIdTag>(Id), 0u);
  EXPECT_EQ(Alloc.Get<plastix::ToIdTag>(Id), 0u);
  EXPECT_FLOAT_EQ(Alloc.Get<plastix::WeightTag>(Id), 0.0f);
  EXPECT_FALSE(Alloc.Get<plastix::DeadTag>(Id));
}

TEST(PipelineNetworkTest, SingleLayerPerceptronConnections) {
  constexpr size_t InputDim = 3;
  constexpr size_t OutputDim = 2;
  TestNetwork Net(InputDim, FC{OutputDim});

  auto &ConnAlloc = Net.GetConnAlloc();
  auto &UnitAlloc = Net.GetUnitAlloc();

  EXPECT_EQ(UnitAlloc.Size(), InputDim + OutputDim);
  EXPECT_EQ(ConnAlloc.Size(), InputDim * OutputDim);

  for (size_t Out = 0; Out < OutputDim; ++Out) {
    for (size_t In = 0; In < InputDim; ++In) {
      size_t C = Out * InputDim + In;
      EXPECT_EQ(ConnAlloc.Get<plastix::ToIdTag>(C), InputDim + Out);
      EXPECT_EQ(ConnAlloc.Get<plastix::FromIdTag>(C), In);
      EXPECT_FLOAT_EQ(ConnAlloc.Get<plastix::WeightTag>(C), 1.0f);
    }
  }
}

TEST(PipelineNetworkTest, SingleLayerPerceptronManyConnections) {
  constexpr size_t InputDim = 10;
  constexpr size_t OutputDim = 1;
  TestNetwork Net(InputDim, FC{OutputDim});

  auto &ConnAlloc = Net.GetConnAlloc();
  EXPECT_EQ(ConnAlloc.Size(), 10u);

  for (size_t I = 0; I < 10; ++I) {
    EXPECT_EQ(ConnAlloc.Get<plastix::FromIdTag>(I), I);
    EXPECT_EQ(ConnAlloc.Get<plastix::ToIdTag>(I), InputDim);
    EXPECT_FLOAT_EQ(ConnAlloc.Get<plastix::WeightTag>(I), 1.0f);
  }
}

// ---------------------------------------------------------------------------
// Custom forward pass
// ---------------------------------------------------------------------------

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
  static constexpr plastix::Propagation Model = plastix::Propagation::Pipeline;
};

TEST(PipelineTraitsTest, CustomForwardPass) {
  static_assert(plastix::NetworkTraits<CustomForwardTraits>);
  plastix::Network<CustomForwardTraits> Net(3, 1);
  EXPECT_EQ(Net.GetUnitAlloc().Size(), 4u);
}

// ---------------------------------------------------------------------------
// Custom global state
// ---------------------------------------------------------------------------

struct TestGlobalState {
  float LearningRate = 0.01f;
  float DropoutRate = 0.4f;
};

struct CustomGlobalTraits : plastix::DefaultNetworkTraits<TestGlobalState> {
  static constexpr plastix::Propagation Model = plastix::Propagation::Pipeline;
};

TEST(PipelineTraitsTest, CustomGlobalState) {
  static_assert(plastix::NetworkTraits<CustomGlobalTraits>);
  plastix::Network<CustomGlobalTraits> Net(3, 1);
  EXPECT_EQ(Net.GetUnitAlloc().Size(), 4u);
}

// ---------------------------------------------------------------------------
// Forward / Backward pass tests
// ---------------------------------------------------------------------------

TEST(PipelinePassTest, StepCounter) {
  TestNetwork Net(2, 1);
  EXPECT_EQ(Net.GetStep(), 0u);

  std::array<float, 2> In = {1.0f, 2.0f};
  Net.DoForwardPass(In);
  EXPECT_EQ(Net.GetStep(), 1u);

  Net.DoBackwardPass();
  EXPECT_EQ(Net.GetStep(), 1u);
}

TEST(PipelinePassTest, ForwardPassIdentity) {
  // 2 inputs, 1 output, weights=1.0
  // Single layer: output = sum of inputs = 3.0 + 5.0 = 8.0
  TestNetwork Net(2, FC{1});
  std::array<float, 2> In = {3.0f, 5.0f};
  Net.DoForwardPass(In);

  auto &UA = Net.GetUnitAlloc();
  float Out = UA.Get<plastix::ActivationTag>(2);
  EXPECT_FLOAT_EQ(Out, 8.0f);

  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationTag>(0), 3.0f);
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationTag>(1), 5.0f);
}

TEST(PipelinePassTest, ForwardPassCustomPolicy) {
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

TEST(PipelinePassTest, ForwardPassManyConnections) {
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

TEST(PipelinePassTest, ConsecutiveForwardPasses) {
  TestNetwork Net(2, FC{1});

  std::array<float, 2> In1 = {1.0f, 2.0f};
  Net.DoForwardPass(In1);
  auto &UA = Net.GetUnitAlloc();
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationTag>(2), 3.0f);

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
  static constexpr plastix::Propagation Model = plastix::Propagation::Pipeline;
};

TEST(PipelinePassTest, BackwardPassBasic) {
  plastix::Network<GradientBackwardTraits> Net(2, FC{1});
  std::array<float, 2> In = {3.0f, 5.0f};
  Net.DoForwardPass(In);
  EXPECT_EQ(Net.GetStep(), 1u);

  Net.DoBackwardPass();
  EXPECT_EQ(Net.GetStep(), 1u);

  auto &UA = Net.GetUnitAlloc();
  // BackwardAcc is reset to 0 after Apply.
  EXPECT_FLOAT_EQ(UA.Get<plastix::BackwardAccTag>(0), 0.0f);
  EXPECT_FLOAT_EQ(UA.Get<plastix::BackwardAccTag>(1), 0.0f);
  EXPECT_FLOAT_EQ(UA.Get<plastix::BackwardAccTag>(2), 0.0f);

  // Activations are untouched by backward pass.
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationTag>(0), 3.0f);
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationTag>(1), 5.0f);
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationTag>(2), 8.0f);
}

// ---------------------------------------------------------------------------
// Pipeline-specific: multi-layer pipelined semantics
// ---------------------------------------------------------------------------

TEST(PipelinePassTest, MultiLayerForwardPassPipelined) {
  // 2 inputs -> 2 hidden -> 1 output, all weights=1
  // Pipeline semantics: signals take multiple steps to propagate.
  TestNetwork Net(2, FC{2}, FC{1});

  auto &UA = Net.GetUnitAlloc();

  // Step 0: hidden reads inputs {1,1} => hidden=2 each.
  // Output reads old hidden (0) => output=0.
  std::array<float, 2> In = {1.0f, 1.0f};
  Net.DoForwardPass(In);
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationTag>(2), 2.0f);
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationTag>(3), 2.0f);
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationTag>(4), 0.0f);

  // Step 1: hidden reads inputs {1,1} => hidden=2 again.
  // Output reads hidden (2,2) => output=4.
  Net.DoForwardPass(In);
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationTag>(4), 4.0f);
}

TEST(PipelinePassTest, ThreeLayerPipelined) {
  // 3 inputs -> 2 hidden1 -> 2 hidden2 -> 1 output, all weights=1
  // Input = {1, 1, 1}
  // Step 0: hidden1 = 3 each, hidden2 = 0 each, output = 0
  // Step 1: hidden1 = 3, hidden2 = 6 each, output = 0
  // Step 2: hidden1 = 3, hidden2 = 6, output = 12
  TestNetwork Net(3, FC{2}, FC{2}, FC{1});
  std::array<float, 3> In = {1.0f, 1.0f, 1.0f};

  Net.DoForwardPass(In);
  EXPECT_FLOAT_EQ(Net.GetOutput()[0], 0.0f);

  Net.DoForwardPass(In);
  EXPECT_FLOAT_EQ(Net.GetOutput()[0], 0.0f);

  Net.DoForwardPass(In);
  EXPECT_FLOAT_EQ(Net.GetOutput()[0], 12.0f);
}

// ---------------------------------------------------------------------------
// Layer builder tests
// ---------------------------------------------------------------------------

TEST(PipelineLayerTest, MultiLayerBuilder) {
  // 3 inputs -> 5 hidden -> 1 output = 9 units, 20 connections
  TestNetwork Net(3, FC{5}, FC{1});

  auto &UA = Net.GetUnitAlloc();
  auto &CA = Net.GetConnAlloc();

  EXPECT_EQ(UA.Size(), 9u);
  EXPECT_EQ(CA.Size(), 20u);

  for (size_t I = 0; I < 5; ++I) {
    for (size_t S = 0; S < 3; ++S) {
      size_t C = I * 3 + S;
      EXPECT_EQ(CA.Get<plastix::ToIdTag>(C), 3 + I);
      EXPECT_EQ(CA.Get<plastix::FromIdTag>(C), S);
      EXPECT_FLOAT_EQ(CA.Get<plastix::WeightTag>(C), 1.0f);
    }
  }

  for (size_t S = 0; S < 5; ++S) {
    size_t C = 15 + S;
    EXPECT_EQ(CA.Get<plastix::ToIdTag>(C), 8u);
    EXPECT_EQ(CA.Get<plastix::FromIdTag>(C), 3 + S);
    EXPECT_FLOAT_EQ(CA.Get<plastix::WeightTag>(C), 1.0f);
  }
}

TEST(PipelineLayerTest, MultiLayerManyConnections) {
  // 10 inputs -> 2 hidden -> 1 output = 13 units, 22 connections
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

TEST(PipelineLayerTest, CustomConnInit) {
  using HalfFC = plastix::FullyConnected<plastix::NoUnitInit, HalfWeightInit>;
  TestNetwork Net(2, HalfFC{1});

  auto &CA = Net.GetConnAlloc();
  EXPECT_EQ(CA.Size(), 2u);

  for (size_t C = 0; C < 2; ++C)
    EXPECT_FLOAT_EQ(CA.Get<plastix::WeightTag>(C), 0.5f);
}

TEST(PipelineLayerTest, ThreeLayerBuilder) {
  // 4 inputs -> 3 hidden1 -> 2 hidden2 -> 1 output = 10 units
  TestNetwork Net(4, FC{3}, FC{2}, FC{1});

  auto &UA = Net.GetUnitAlloc();
  EXPECT_EQ(UA.Size(), 10u);
}

// ---------------------------------------------------------------------------
// GetOutput tests
// ---------------------------------------------------------------------------

TEST(PipelineOutputTest, SingleLayerGetOutput) {
  TestNetwork Net(2, FC{1});
  std::array<float, 2> In = {3.0f, 5.0f};
  Net.DoForwardPass(In);

  auto Out = Net.GetOutput();
  ASSERT_EQ(Out.size(), 1u);
  EXPECT_FLOAT_EQ(Out[0], 8.0f);
}

TEST(PipelineOutputTest, MultipleOutputUnits) {
  TestNetwork Net(3, FC{2});
  std::array<float, 3> In = {1.0f, 2.0f, 3.0f};
  Net.DoForwardPass(In);

  auto Out = Net.GetOutput();
  ASSERT_EQ(Out.size(), 2u);
  EXPECT_FLOAT_EQ(Out[0], 6.0f);
  EXPECT_FLOAT_EQ(Out[1], 6.0f);
}

TEST(PipelineOutputTest, MultiLayerGetOutput) {
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

TEST(PipelineOutputTest, GetOutputIsConst) {
  const TestNetwork Net(2, 1);
  auto Out = Net.GetOutput();
  EXPECT_EQ(Out.size(), 1u);
}

TEST(PipelineOutputTest, DoStepGetOutput) {
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
  static constexpr plastix::Propagation Model = plastix::Propagation::Pipeline;
};

TEST(PipelineUpdateTest, UpdateUnitState) {
  plastix::Network<UpdateUnitTraits> Net(2, FC{1});
  std::array<float, 2> In = {3.0f, 4.0f};
  Net.DoForwardPass(In);
  Net.DoUpdateUnitState();

  auto &UA = Net.GetUnitAlloc();
  EXPECT_FLOAT_EQ(UA.Get<plastix::BackwardAccTag>(0), 3.0f);
  EXPECT_FLOAT_EQ(UA.Get<plastix::BackwardAccTag>(1), 4.0f);
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
  static constexpr plastix::Propagation Model = plastix::Propagation::Pipeline;
};

TEST(PipelineUpdateTest, UpdateConnStateWeightDecay) {
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
  static constexpr plastix::Propagation Model = plastix::Propagation::Pipeline;
};

TEST(PipelineUpdateTest, DoStepWithUpdate) {
  plastix::Network<DoStepUpdateTraits> Net(2, FC{1});
  std::array<float, 2> In = {3.0f, 4.0f};
  Net.DoStep(In);

  auto &UA = Net.GetUnitAlloc();
  EXPECT_FLOAT_EQ(UA.Get<plastix::BackwardAccTag>(2), 7.0f);

  auto &CA = Net.GetConnAlloc();
  EXPECT_FLOAT_EQ(CA.Get<plastix::WeightTag>(0), 0.5f);
  EXPECT_FLOAT_EQ(CA.Get<plastix::WeightTag>(1), 0.5f);

  auto Out = Net.GetOutput();
  ASSERT_EQ(Out.size(), 1u);
  EXPECT_FLOAT_EQ(Out[0], 7.0f);
}

// ---------------------------------------------------------------------------
// Prune tests
// ---------------------------------------------------------------------------

struct PruneUnitById {
  static bool ShouldPrune(auto &, size_t Id, auto &) { return Id == 2; }
};

struct PruneUnitTraits : plastix::DefaultNetworkTraits<> {
  using PruneUnit = PruneUnitById;
  static constexpr plastix::Propagation Model = plastix::Propagation::Pipeline;
};

TEST(PipelinePruneTest, PruneUnitMarksFlag) {
  plastix::Network<PruneUnitTraits> Net(3, 1);
  Net.DoPruneUnits();

  auto &UA = Net.GetUnitAlloc();
  EXPECT_FALSE(UA.Get<plastix::PrunedTag>(0));
  EXPECT_FALSE(UA.Get<plastix::PrunedTag>(1));
  EXPECT_TRUE(UA.Get<plastix::PrunedTag>(2));
  EXPECT_FALSE(UA.Get<plastix::PrunedTag>(3));
}

TEST(PipelinePruneTest, PruneUnitMarksSourceConnectionsDead) {
  // 3 inputs -> 1 output (unit 3). Prune unit 2 => connection from
  // unit 2 is marked dead. 2 connections survive (tombstone-based).
  plastix::Network<PruneUnitTraits> Net(3, 1);
  Net.DoPruneUnits();
  Net.DoPruneConnections();

  auto &CA = Net.GetConnAlloc();
  EXPECT_EQ(CA.Size(), 3u);
  EXPECT_EQ(CountAlive(CA), 2u);

  for (size_t C = 0; C < CA.Size(); ++C) {
    if (CA.Get<plastix::FromIdTag>(C) == 2)
      EXPECT_TRUE(CA.Get<plastix::DeadTag>(C));
    else
      EXPECT_FALSE(CA.Get<plastix::DeadTag>(C));
  }
}

struct PruneDestUnit {
  static size_t TargetId;
  static bool ShouldPrune(auto &, size_t Id, auto &) { return Id == TargetId; }
};
size_t PruneDestUnit::TargetId = 0;

struct PruneDestTraits : plastix::DefaultNetworkTraits<> {
  using PruneUnit = PruneDestUnit;
  static constexpr plastix::Propagation Model = plastix::Propagation::Pipeline;
};

TEST(PipelinePruneTest, PruneDestinationMarksAllDead) {
  PruneDestUnit::TargetId = 2;
  plastix::Network<PruneDestTraits> Net(2, 1);
  Net.DoPruneUnits();
  Net.DoPruneConnections();

  auto &CA = Net.GetConnAlloc();
  EXPECT_EQ(CA.Size(), 2u);
  EXPECT_EQ(CountAlive(CA), 0u);
}

struct PruneSmallWeight {
  static bool ShouldPrune(auto &, size_t, size_t, auto &C, size_t ConnId,
                          auto &) {
    return C.template Get<plastix::WeightTag>(ConnId) < 0.5f;
  }
};

struct PruneConnTraits : plastix::DefaultNetworkTraits<> {
  using PruneConn = PruneSmallWeight;
  static constexpr plastix::Propagation Model = plastix::Propagation::Pipeline;
};

TEST(PipelinePruneTest, PruneConnMarksAllDeadWhenAllPruned) {
  plastix::Network<PruneConnTraits> Net(2, 1);

  auto &CA = Net.GetConnAlloc();
  CA.Get<plastix::WeightTag>(0) = 0.1f;
  CA.Get<plastix::WeightTag>(1) = 0.2f;

  Net.DoPruneConnections();
  EXPECT_EQ(CA.Size(), 2u);
  EXPECT_EQ(CountAlive(CA), 0u);
}

TEST(PipelinePruneTest, PruneConnMarksPartialDead) {
  plastix::Network<PruneConnTraits> Net(2, 1);

  auto &CA = Net.GetConnAlloc();
  CA.Get<plastix::WeightTag>(0) = 0.1f;
  CA.Get<plastix::WeightTag>(1) = 1.0f;

  Net.DoPruneConnections();
  EXPECT_EQ(CA.Size(), 2u);
  EXPECT_EQ(CountAlive(CA), 1u);

  EXPECT_TRUE(CA.Get<plastix::DeadTag>(0));
  EXPECT_FALSE(CA.Get<plastix::DeadTag>(1));
  EXPECT_FLOAT_EQ(CA.Get<plastix::WeightTag>(1), 1.0f);
}

TEST(PipelinePruneTest, DoStepWithPrune) {
  PruneDestUnit::TargetId = 2;
  plastix::Network<PruneDestTraits> Net(2, 1);
  std::array<float, 2> In = {1.0f, 2.0f};
  Net.DoStep(In);

  auto &CA = Net.GetConnAlloc();
  EXPECT_EQ(CountAlive(CA), 0u);
}

// ---------------------------------------------------------------------------
// AddUnit tests
// ---------------------------------------------------------------------------

struct AddOneUnit {
  static std::optional<int16_t> AddUnit(auto &, size_t Id, auto &) {
    if (Id == 0)
      return int16_t{1};
    return std::nullopt;
  }
};

struct AddUnitTraits : plastix::DefaultNetworkTraits<> {
  using AddUnit = AddOneUnit;
  static constexpr plastix::Propagation Model = plastix::Propagation::Pipeline;
};

TEST(PipelineAddUnitTest, NoopDoesNotAddUnits) {
  TestNetwork Net(2, 1);
  EXPECT_EQ(Net.GetUnitAlloc().Size(), 3u);
  Net.DoAddUnits();
  EXPECT_EQ(Net.GetUnitAlloc().Size(), 3u);
}

TEST(PipelineAddUnitTest, AddsUnitAtOffsetLevel) {
  plastix::Network<AddUnitTraits> Net(2, 1);
  EXPECT_EQ(Net.GetUnitAlloc().Size(), 3u);

  Net.DoAddUnits();
  EXPECT_EQ(Net.GetUnitAlloc().Size(), 4u);

  auto &UA = Net.GetUnitAlloc();
  EXPECT_EQ(UA.Get<plastix::LevelTag>(3), uint16_t{1});
}

TEST(PipelineAddUnitTest, DoesNotIterateNewlyAddedUnits) {
  plastix::Network<AddUnitTraits> Net(2, 1);
  Net.DoAddUnits();
  EXPECT_EQ(Net.GetUnitAlloc().Size(), 4u);

  Net.DoAddUnits();
  EXPECT_EQ(Net.GetUnitAlloc().Size(), 5u);
}

// ---------------------------------------------------------------------------
// AddConn tests
// ---------------------------------------------------------------------------

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
  static constexpr plastix::Propagation Model = plastix::Propagation::Pipeline;
};

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
  static constexpr plastix::Propagation Model = plastix::Propagation::Pipeline;
};

TEST(PipelineAddConnTest, NoopDoesNotAddConnections) {
  TestNetwork Net(2, 1);
  EXPECT_EQ(Net.GetConnAlloc().Size(), 2u);
  Net.DoAddConnections();
  EXPECT_EQ(Net.GetConnAlloc().Size(), 2u);
}

TEST(PipelineAddConnTest, AddsIncomingConnection) {
  plastix::Network<AddIncomingTraits> Net(2, 1);
  EXPECT_EQ(Net.GetConnAlloc().Size(), 2u);

  Net.DoAddConnections();

  auto &CA = Net.GetConnAlloc();
  EXPECT_EQ(CA.Size(), 3u);

  EXPECT_EQ(CA.Get<plastix::FromIdTag>(2), 1u);
  EXPECT_EQ(CA.Get<plastix::ToIdTag>(2), 0u);
  EXPECT_FLOAT_EQ(CA.Get<plastix::WeightTag>(2), 0.5f);
  EXPECT_FALSE(CA.Get<plastix::DeadTag>(2));
}

TEST(PipelineAddConnTest, NewConnectionParticipatesInForwardPass) {
  // 2 inputs (0,1), 1 output (2). Initial: 0->2 (w=1), 1->2 (w=1).
  // inputs={2, 3} => output = 1*2 + 1*3 = 5.
  plastix::Network<AddConnToOutputTraits> Net(2, FC{1});
  std::array<float, 2> In = {2.0f, 3.0f};
  Net.DoForwardPass(In);
  EXPECT_FLOAT_EQ(Net.GetOutput()[0], 5.0f);

  // Add connection 0->2 with weight 0.5.
  Net.DoAddConnections();

  // Now output = 1*2 + 1*3 + 0.5*2 = 6.
  Net.DoForwardPass(In);
  EXPECT_FLOAT_EQ(Net.GetOutput()[0], 6.0f);
}

// ---------------------------------------------------------------------------
// Deduplication tests
// ---------------------------------------------------------------------------

struct AddDuplicateEdge {
  static bool ShouldAddIncomingConnection(auto &, size_t Self, size_t Candidate,
                                          auto &) {
    return Self == 0 && Candidate == 1;
  }
  static bool ShouldAddOutgoingConnection(auto &, size_t Self, size_t Candidate,
                                          auto &) {
    return Self == 1 && Candidate == 0;
  }
  static void InitConnection(auto &, size_t, size_t, auto &C, size_t ConnId,
                             auto &) {
    C.template Get<plastix::WeightTag>(ConnId) = 0.25f;
  }
};

struct DedupTraits : plastix::DefaultNetworkTraits<> {
  using AddConn = AddDuplicateEdge;
  static constexpr plastix::Propagation Model = plastix::Propagation::Pipeline;
};

TEST(PipelineDeduplicationTest, DuplicateProposalsDeduped) {
  plastix::Network<DedupTraits> Net(2, 1);
  EXPECT_EQ(Net.GetConnAlloc().Size(), 2u);

  Net.DoAddConnections();

  EXPECT_EQ(Net.GetConnAlloc().Size(), 3u);

  auto &CA = Net.GetConnAlloc();
  EXPECT_EQ(CA.Get<plastix::FromIdTag>(2), 1u);
  EXPECT_EQ(CA.Get<plastix::ToIdTag>(2), 0u);
  EXPECT_FLOAT_EQ(CA.Get<plastix::WeightTag>(2), 0.25f);
}

} // namespace plastix_pipeline_test
