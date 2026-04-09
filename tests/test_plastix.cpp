#include <gtest/gtest.h>

#include "plastix/plastix.hpp"
#include <array>
#include <cmath>

TEST(PlastixTest, Version) { EXPECT_STREQ(plastix::version(), "0.1.0"); }

namespace plastix_test {

using TestTraits = plastix::DefaultNetworkTraits<plastix::ConnStateAllocator>;
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
  using Accumulator = float;
  static float Map(auto &U, size_t, size_t SrcId, auto &C, size_t PageId,
                   size_t SlotIdx, auto &) {
    return 2.0f *
           C.template Get<plastix::ConnPageMarker>(PageId)
               .GetSlot(SlotIdx)
               .second *
           U.template Get<plastix::ActivationTag>(SrcId);
  }
  static float Combine(float A, float B) { return A + B; }
  static void Apply(auto &U, size_t Id, auto &, float Accumulated) {
    U.template Get<plastix::ActivationTag>(Id) = std::tanh(Accumulated);
  }
};

struct CustomForwardTraits
    : plastix::DefaultNetworkTraits<plastix::ConnStateAllocator> {
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
    : plastix::DefaultNetworkTraits<plastix::ConnStateAllocator,
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

  // Backward does not increment step.
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
  plastix::Network<CustomForwardTraits> Net(2, 1);
  std::array<float, 2> In = {1.0f, 2.0f};
  Net.DoForwardPass(In);

  auto &UA = Net.GetUnitAlloc();
  float Out = UA.Get<plastix::ActivationTag>(2);
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
  float Out = UA.Get<plastix::ActivationTag>(10);
  EXPECT_FLOAT_EQ(Out, 10.0f);
}

TEST(PassTest, ConsecutiveForwardPasses) {
  // Two forward passes with different inputs.
  TestNetwork Net(2, 1);

  // First pass: output = 1+2 = 3
  std::array<float, 2> In1 = {1.0f, 2.0f};
  Net.DoForwardPass(In1);
  auto &UA = Net.GetUnitAlloc();
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationTag>(2), 3.0f);

  // Second pass: output = 4+5 = 9
  std::array<float, 2> In2 = {4.0f, 5.0f};
  Net.DoForwardPass(In2);
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationTag>(2), 9.0f);
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationTag>(0), 4.0f);
  EXPECT_FLOAT_EQ(UA.Get<plastix::ActivationTag>(1), 5.0f);
}

struct GradientBackwardPass {
  using Accumulator = float;
  static float Map(auto &U, size_t, size_t ToId, auto &C, size_t PageId,
                   size_t SlotIdx, auto &) {
    return C.template Get<plastix::ConnPageMarker>(PageId)
               .GetSlot(SlotIdx)
               .second *
           U.template Get<plastix::ActivationTag>(ToId);
  }
  static float Combine(float A, float B) { return A + B; }
  static void Apply(auto &, size_t, auto &, float) {}
};

struct GradientBackwardTraits
    : plastix::DefaultNetworkTraits<plastix::ConnStateAllocator> {
  using BackwardPass = GradientBackwardPass;
};

TEST(PassTest, BackwardPassBasic) {
  // Forward: 2 inputs, 1 output. inputs={3,5}, output=8.
  // Backward: reads output activation (8),
  // accumulates weight*8 into BackwardAcc for source units.
  // Apply is a noop, so BackwardAcc is reset to 0.
  plastix::Network<GradientBackwardTraits> Net(2, 1);
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
  // Pipelined semantics: signals take multiple steps to propagate.
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

struct CopyActivationToBackwardAcc {
  static void Update(auto &U, size_t Id, auto &) {
    U.template Get<plastix::BackwardAccTag>(Id) =
        U.template Get<plastix::ActivationTag>(Id);
  }
};

struct UpdateUnitTraits
    : plastix::DefaultNetworkTraits<plastix::ConnStateAllocator> {
  using UpdateUnit = CopyActivationToBackwardAcc;
};

TEST(UpdateTest, UpdateUnitState) {
  // 2 inputs, 1 output, weights=1.0.
  // Update copies Activation into BackwardAcc for every unit.
  plastix::Network<UpdateUnitTraits> Net(2, 1);
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
                                       size_t PageId, size_t SlotIdx, auto &) {
    C.template Get<plastix::ConnPageMarker>(PageId).WriteSlot(SlotIdx).second *=
        0.5f;
  }
  static void UpdateOutgoingConnection(auto &, size_t, size_t, auto &, size_t,
                                       size_t, auto &) {}
};

struct UpdateConnTraits
    : plastix::DefaultNetworkTraits<plastix::ConnStateAllocator> {
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
  auto &Page = CA.Get<plastix::ConnPageMarker>(0);
  EXPECT_FLOAT_EQ(Page.Conn[0].second, 0.5f);
  EXPECT_FLOAT_EQ(Page.Conn[1].second, 0.5f);
}

// ---------------------------------------------------------------------------
// DoStep with update policies
// ---------------------------------------------------------------------------

struct DoStepUpdateTraits
    : plastix::DefaultNetworkTraits<plastix::ConnStateAllocator> {
  using UpdateUnit = CopyActivationToBackwardAcc;
  using UpdateConn = WeightDecayUpdateConn;
};

TEST(UpdateTest, DoStepWithUpdate) {
  // Full pipeline: forward + update unit + update conn.
  // After DoStep: BackwardAcc copies activation, weights halved to 0.5.
  plastix::Network<DoStepUpdateTraits> Net(2, 1);
  std::array<float, 2> In = {3.0f, 4.0f};
  Net.DoStep(In);

  auto &UA = Net.GetUnitAlloc();
  // Output activation = 3*1 + 4*1 = 7, copied to BackwardAcc.
  EXPECT_FLOAT_EQ(UA.Get<plastix::BackwardAccTag>(2), 7.0f);

  auto &CA = Net.GetConnAlloc();
  auto &Page = CA.Get<plastix::ConnPageMarker>(0);
  EXPECT_FLOAT_EQ(Page.Conn[0].second, 0.5f);
  EXPECT_FLOAT_EQ(Page.Conn[1].second, 0.5f);

  // Output should still be correct from forward pass: 3+4=7.
  auto Out = Net.GetOutput();
  ASSERT_EQ(Out.size(), 1u);
  EXPECT_FLOAT_EQ(Out[0], 7.0f);
}

// ---------------------------------------------------------------------------
// Prune tests
// ---------------------------------------------------------------------------

// Policy that prunes a specific unit by index.
struct PruneUnitById {
  static bool ShouldPrune(auto &, size_t Id, auto &) {
    return Id == 2; // prune unit 2
  }
};

struct PruneUnitTraits
    : plastix::DefaultNetworkTraits<plastix::ConnStateAllocator> {
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

TEST(PruneTest, PruneUnitCompactsSourceConnections) {
  // 3 inputs -> 1 output (unit 3). Sources are units 0,1,2.
  // Prune unit 2 => its connection is removed, page compacted to 2.
  // Surviving connections (from units 0,1) should be at slots 0,1.
  plastix::Network<PruneUnitTraits> Net(3, 1);
  Net.DoPruneUnits();
  Net.DoPruneConnections();

  auto &CA = Net.GetConnAlloc();
  auto &Page = CA.Get<plastix::ConnPageMarker>(0);
  EXPECT_EQ(Page.Count, 2u);
  EXPECT_EQ(Page.Conn[0].first, 0u);
  EXPECT_EQ(Page.Conn[1].first, 1u);
}

// Policy that prunes the output unit (last unit).
struct PruneDestUnit {
  static size_t TargetId;
  static bool ShouldPrune(auto &, size_t Id, auto &) { return Id == TargetId; }
};
size_t PruneDestUnit::TargetId = 0;

struct PruneDestTraits
    : plastix::DefaultNetworkTraits<plastix::ConnStateAllocator> {
  using PruneUnit = PruneDestUnit;
};

TEST(PruneTest, PruneDestinationClearsPage) {
  // 2 inputs -> 1 output (unit 2). Prune the output unit.
  // The page targets unit 2, so it should be cleared.
  PruneDestUnit::TargetId = 2;
  plastix::Network<PruneDestTraits> Net(2, 1);
  Net.DoPruneUnits();
  Net.DoPruneConnections();

  auto &CA = Net.GetConnAlloc();
  auto &Page = CA.Get<plastix::ConnPageMarker>(0);
  EXPECT_EQ(Page.Count, 0u);
}

// Connection-level pruning policy.
struct PruneSmallWeight {
  static bool ShouldPrune(auto &, size_t, size_t, auto &C, size_t PageId,
                          size_t SlotIdx, auto &) {
    return C.template Get<plastix::ConnPageMarker>(PageId)
               .GetSlot(SlotIdx)
               .second < 0.5f;
  }
};

struct PruneConnTraits
    : plastix::DefaultNetworkTraits<plastix::ConnStateAllocator> {
  using PruneConn = PruneSmallWeight;
};

TEST(PruneTest, PruneConnClearsPageWhenAllPruned) {
  // 2 inputs -> 1 output, weights=1.0 initially.
  // Halve weights via update, then prune connections < 0.5.
  // After halving, weights=0.5, so ShouldPrune (< 0.5) is false => page stays.
  plastix::Network<PruneConnTraits> Net(2, 1);

  // Manually set weights below threshold.
  auto &CA = Net.GetConnAlloc();
  auto &Page = CA.Get<plastix::ConnPageMarker>(0);
  Page.Conn[0].second = 0.1f;
  Page.Conn[1].second = 0.2f;

  Net.DoPruneConnections();
  EXPECT_EQ(Page.Count, 0u);
}

TEST(PruneTest, PruneConnCompactsPartialPage) {
  // 2 inputs -> 1 output. Conn[0] weight below threshold, Conn[1] survives.
  // After compaction: Count=1, slot 0 holds the surviving connection.
  plastix::Network<PruneConnTraits> Net(2, 1);

  auto &CA = Net.GetConnAlloc();
  auto &Page = CA.Get<plastix::ConnPageMarker>(0);
  Page.Conn[0].second = 0.1f; // below threshold — pruned
  Page.Conn[1].second = 1.0f; // above threshold — survives

  Net.DoPruneConnections();
  EXPECT_EQ(Page.Count, 1u);
  // Surviving connection (was slot 1, src=unit 1, weight=1.0) compacted to slot
  // 0.
  EXPECT_EQ(Page.Conn[0].first, 1u);
  EXPECT_FLOAT_EQ(Page.Conn[0].second, 1.0f);
}

TEST(PruneTest, DoStepWithPrune) {
  // Full pipeline: forward, then prune destination unit.
  PruneDestUnit::TargetId = 2;
  plastix::Network<PruneDestTraits> Net(2, 1);
  std::array<float, 2> In = {1.0f, 2.0f};
  Net.DoStep(In);

  // After DoStep, the output unit's connections are cleared.
  auto &CA = Net.GetConnAlloc();
  auto &Page = CA.Get<plastix::ConnPageMarker>(0);
  EXPECT_EQ(Page.Count, 0u);
}

} // namespace plastix_test
