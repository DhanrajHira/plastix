// Unit tests for the plastix::optim bundles.
//
// The conformance vectors (tests/golden/mlp_optim_*.json) already pin every
// bundle's *arithmetic* against plastax, which is itself validated against
// optax -- so nothing here re-checks a formula. What these cover is the
// framework wiring the vectors cannot see:
//
//   * that a bundle reproduces the hand-written rule it replaces,
//   * that its state columns are real columns, and survive the things that
//     move connections around,
//   * that the per-edge step counter behaves per edge,
//   * that WithGlobalLr is genuinely read at runtime.
//
// See notes/parity/02-optimizers.md.

#include <plastix/optim.hpp>
#include <plastix/plastix.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

// Targeted usings instead of `using namespace plastix`: the blanket directive
// pulls plastix::cuda (the in-house GPU-primitives namespace) into scope, which
// collides with the CUDA toolkit's ::cuda namespace (libcu++ / cooperative
// groups) when this TU is compiled as CUDA -- "cuda is ambiguous". Naming only
// what the test uses avoids that.
namespace alloc = plastix::alloc;
namespace optim = plastix::optim;
using plastix::ConnFieldList;
using plastix::ConnFieldTagsAreUniqueV;
using plastix::DeadTag;
using plastix::DefaultNetworkTraits;
using plastix::FullyConnected;
using plastix::GetActivation;
using plastix::GetField;
using plastix::GetWeight;
using plastix::Network;
using plastix::NetworkTraits;
using plastix::RandomUniformWeight;
using plastix::UnitFieldList;
using plastix::WeightTag;
using plastix::WithOptimizer;

struct GradTag {};

struct Globals {
  float Lr = 0.1f;
};

// Identity forward so a weight change shows up in the activation unchanged,
// and the arithmetic under test stays visible.
struct LinearForward {
  using Accumulator = float;
  PLASTIX_HD static float Map(auto &U, size_t, size_t SrcId, auto &C,
                              size_t ConnId, auto &) {
    return GetWeight(C, ConnId) * GetActivation(U, SrcId);
  }
  PLASTIX_HD static float Combine(float A, float B) { return A + B; }
  PLASTIX_HD static void Apply(auto &U, size_t Id, auto &, float Acc) {
    GetActivation(U, Id) = Acc;
  }
};

// Stages a fixed dL/dz on every non-input unit, so an update runs with a known
// gradient and no loss or backward pass is needed. Keeps each test about the
// optimizer rather than about the surrounding phases.
struct StageConstantGrad {
  PLASTIX_HD static void Update(auto &U, size_t Id, auto &) {
    GetField<GradTag>(U, Id) = 1.0f;
  }
};

struct Base : DefaultNetworkTraits<Globals> {
  using ForwardPass = LinearForward;
  using UpdateUnit = StageConstantGrad;
  using ExtraUnitFields = UnitFieldList<alloc::SOAField<GradTag, float>>;
};

constexpr float Lr = 0.1f;

struct FixedLr : optim::SgdParams {
  static constexpr float Lr = ::Lr;
};

// The hand-written rule every example carries, and the thing optim::Sgd is
// meant to replace verbatim.
struct HandWrittenSgd {
  PLASTIX_HD static void UpdateIncomingConnection(auto &U, size_t DstId,
                                                  size_t SrcId, auto &C,
                                                  size_t ConnId, auto &) {
    const float Grad = GetField<GradTag>(U, DstId);
    GetWeight(C, ConnId) -= Lr * Grad * GetActivation(U, SrcId);
  }
  PLASTIX_HD static void UpdateOutgoingConnection(auto &, size_t, size_t,
                                                  auto &, size_t, auto &) {}
};

struct HandWrittenTraits : Base {
  using UpdateConn = HandWrittenSgd;
};

struct BundleTraits : WithOptimizer<Base, optim::Sgd<GradTag, FixedLr>> {};

// Collect every live weight in allocation order.
std::vector<float> Weights(auto &Net) {
  std::vector<float> Out;
  auto &CA = Net.GetConnAlloc();
  for (size_t C = 0, N = CA.Size(); C < N; ++C)
    if (!GetField<DeadTag>(CA, C))
      Out.push_back(GetWeight(CA, C));
  return Out;
}

using FC = FullyConnected<RandomUniformWeight>;

// A bundle must be a drop-in for the rule it generalizes -- not merely close.
// Both nets are seeded identically, so this is an exact comparison.
TEST(Optim, SgdMatchesTheHandWrittenRule) {
  Network<HandWrittenTraits> Hand(3, FC{2, RandomUniformWeight{7}});
  Network<BundleTraits> Bundle(3, FC{2, RandomUniformWeight{7}});

  const float Inputs[3] = {0.5f, -0.25f, 1.0f};
  for (int Step = 0; Step < 20; ++Step) {
    Hand.DoStep(Inputs);
    Bundle.DoStep(Inputs);
  }

  const auto Expected = Weights(Hand);
  const auto Actual = Weights(Bundle);
  ASSERT_EQ(Expected.size(), Actual.size());
  for (size_t I = 0; I < Expected.size(); ++I)
    EXPECT_FLOAT_EQ(Actual[I], Expected[I]) << "weight " << I;
}

// momentum == 0 degenerates to v = g, i.e. plain SGD. A bundle whose special
// case does not reduce to the simpler rule has an arithmetic error somewhere.
struct ZeroMomentum : optim::MomentumParams {
  static constexpr float Lr = ::Lr;
  static constexpr float Momentum = 0.0f;
};
struct ZeroMomentumTraits
    : WithOptimizer<Base, optim::Momentum<GradTag, ZeroMomentum>> {};

TEST(Optim, ZeroMomentumRecoversSgd) {
  Network<BundleTraits> Sgd(3, FC{2, RandomUniformWeight{11}});
  Network<ZeroMomentumTraits> Mom(3, FC{2, RandomUniformWeight{11}});

  const float Inputs[3] = {1.0f, 0.5f, -0.5f};
  for (int Step = 0; Step < 10; ++Step) {
    Sgd.DoStep(Inputs);
    Mom.DoStep(Inputs);
  }

  const auto Expected = Weights(Sgd);
  const auto Actual = Weights(Mom);
  ASSERT_EQ(Expected.size(), Actual.size());
  for (size_t I = 0; I < Expected.size(); ++I)
    EXPECT_FLOAT_EQ(Actual[I], Expected[I]) << "weight " << I;
}

// The state columns must exist as columns and actually accumulate. A bundle
// whose fields were dropped by WithOptimizer would still run -- reading and
// writing a weight column that exists -- and only the numbers would be wrong.
struct AdamFixedLr : optim::AdamParams {
  static constexpr float Lr = ::Lr;
};
struct AdamTraits : WithOptimizer<Base, optim::Adam<GradTag, AdamFixedLr>> {};

TEST(Optim, AdamCarriesItsStateColumns) {
  Network<AdamTraits> Net(3, FC{2, RandomUniformWeight{3}});
  auto &CA = Net.GetConnAlloc();

  for (size_t C = 0, N = CA.Size(); C < N; ++C) {
    EXPECT_EQ(GetField<optim::OptTTag>(CA, C), 0.0f) << "conn " << C;
    EXPECT_EQ(GetField<optim::OptMTag>(CA, C), 0.0f) << "conn " << C;
  }

  const float Inputs[3] = {1.0f, 1.0f, 1.0f};
  Net.DoStep(Inputs);

  for (size_t C = 0, N = CA.Size(); C < N; ++C) {
    EXPECT_EQ(GetField<optim::OptTTag>(CA, C), 1.0f)
        << "conn " << C << ": step counter did not advance";
    EXPECT_NE(GetField<optim::OptMTag>(CA, C), 0.0f)
        << "conn " << C << ": first moment stayed at zero";
  }
}

// The step counter is deliberately per edge rather than global (optim.hpp's
// header comment): an edge that has not been updated must not inherit another
// edge's schedule. Prune one edge, then check the survivors counted their own
// steps and the arena did not smear state across slots.
struct PruneOne {
  PLASTIX_HD static bool ShouldPrune(auto &, size_t, size_t, auto &C,
                                     size_t ConnId, auto &) {
    // Prune exactly one edge, on the first step only -- once it is dead the
    // predicate is not consulted for it again.
    return ConnId == 1 && GetField<optim::OptTTag>(C, ConnId) == 1.0f;
  }
};
struct AdamPruneTraits : AdamTraits {
  using PruneConn = PruneOne;
};

// Live (non-dead) edge count. The sharded NumLiveEdgesTotal() has no
// single-process equivalent, so count directly like Weights() above.
static size_t LiveEdges(auto &Net) {
  auto &CA = Net.GetConnAlloc();
  size_t N = 0;
  for (size_t C = 0, M = CA.Size(); C < M; ++C)
    if (!GetField<DeadTag>(CA, C))
      ++N;
  return N;
}

TEST(Optim, StateSurvivesPruneAndCompaction) {
  Network<AdamPruneTraits> Net(3, FC{2, RandomUniformWeight{5}});
  const size_t Before = LiveEdges(Net);

  const float Inputs[3] = {1.0f, 0.5f, 0.25f};
  Net.DoStep(Inputs); // updates all edges (t -> 1), then prunes conn 1
  ASSERT_EQ(LiveEdges(Net), Before - 1)
      << "the fixture is meant to prune exactly one edge";

  // Compaction moves surviving connections to new slots. Every field moves
  // together or the optimizer state would be silently reattached to the wrong
  // edge -- which nothing else in the suite would notice.
  for (int Step = 0; Step < 3; ++Step)
    Net.DoStep(Inputs);

  auto &CA = Net.GetConnAlloc();
  for (size_t C = 0, N = CA.Size(); C < N; ++C) {
    if (GetField<DeadTag>(CA, C))
      continue;
    EXPECT_EQ(GetField<optim::OptTTag>(CA, C), 4.0f)
        << "conn " << C << " lost its step count across compaction";
  }
}

// WithGlobalLr must be read every step, not baked in at construction --
// otherwise a schedule silently does nothing.
struct GlobalLrTraits
    : WithOptimizer<Base, optim::Sgd<GradTag, optim::WithGlobalLr<&Globals::Lr>>> {
};

TEST(Optim, WithGlobalLrIsReadAtRuntime) {
  Network<GlobalLrTraits> Net(3, FC{1, RandomUniformWeight{9}});
  const float Inputs[3] = {1.0f, 0.0f, 0.0f};

  Net.Global().Lr = 0.0f;
  Net.DoStep(Inputs);
  const auto Frozen = Weights(Net);

  Net.Global().Lr = 0.5f;
  Net.DoStep(Inputs);
  const auto Moved = Weights(Net);

  ASSERT_EQ(Frozen.size(), Moved.size());
  bool AnyChanged = false;
  for (size_t I = 0; I < Frozen.size(); ++I)
    AnyChanged |= Frozen[I] != Moved[I];
  EXPECT_TRUE(AnyChanged)
      << "raising the learning rate mid-run changed nothing; WithGlobalLr is "
         "not being read per step";
}

TEST(Optim, ZeroLearningRateIsANoOp) {
  Network<GlobalLrTraits> Net(3, FC{2, RandomUniformWeight{13}});
  Net.Global().Lr = 0.0f;

  const auto Initial = Weights(Net);
  const float Inputs[3] = {1.0f, -1.0f, 0.5f};
  for (int Step = 0; Step < 5; ++Step)
    Net.DoStep(Inputs);

  const auto After = Weights(Net);
  ASSERT_EQ(Initial.size(), After.size());
  for (size_t I = 0; I < Initial.size(); ++I)
    EXPECT_FLOAT_EQ(After[I], Initial[I]) << "weight " << I << " moved at lr=0";
}

// A stateless bundle must not add columns, so wiring one in costs nothing.
static_assert(std::is_same_v<BundleTraits::ExtraConnFields,
                             ConnFieldList<alloc::SOAField<WeightTag, float>>>);
static_assert(std::is_same_v<
              AdamTraits::ExtraConnFields,
              ConnFieldList<alloc::SOAField<WeightTag, float>,
                            alloc::SOAField<optim::OptMTag, float>,
                            alloc::SOAField<optim::OptVTag, float>,
                            alloc::SOAField<optim::OptTTag, float>>>);

// The uniqueness check WithOptimizer relies on must actually reject a repeat;
// a check that always passes is worse than none.
static_assert(ConnFieldTagsAreUniqueV<
              ConnFieldList<alloc::SOAField<WeightTag, float>,
                            alloc::SOAField<optim::OptMTag, float>>>);
static_assert(!ConnFieldTagsAreUniqueV<
              ConnFieldList<alloc::SOAField<WeightTag, float>,
                            alloc::SOAField<WeightTag, float>>>);

static_assert(NetworkTraits<BundleTraits>);
static_assert(NetworkTraits<AdamTraits>);
static_assert(NetworkTraits<GlobalLrTraits>);

} // namespace
