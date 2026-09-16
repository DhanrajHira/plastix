// Every optimizer bundle learning XOR through one set of traits.
//
// The point of the example is what does NOT change between the runs below.
// The forward pass, backward pass, loss, unit fields and topology are declared
// once; swapping the learning rule is a single template argument, because a
// `plastix::optim` bundle carries both its UpdateConn policy and the
// per-connection state columns that rule needs. `WithOptimizer` splices them
// into the traits together, so Adam's three `opt/...` columns appear (and
// SGD's zero columns do not) without the traits class mentioning them.
//
// Contrast examples/mlp-xor, which writes its gradient-descent rule by hand.
// That one is still the place to look for *how* to author an UpdateConn
// policy; this one is how to reach for a ready-made rule.
//
// The bundles mirror plastax's `plastax.optim`, which is validated against
// optax, and plastix is pinned to them by tests/golden/mlp_optim_*.json --
// so the numbers these produce are the same numbers optax would.

#include <plastix/optim.hpp>
#include <plastix/plastix.hpp>

#include <cmath>
#include <cstdio>

namespace {

constexpr size_t NumEpochs = 4000;

// Per-unit pre-activation gradient dL/dz. The backward pass writes it and the
// optimizer reads it at the destination -- the `GradTag` every bundle takes.
// It has to be a persistent unit field rather than the framework's BackwardAcc
// because BackwardAcc is cleared right after each per-level Apply, so it
// cannot carry a value from one level to the next.
struct GradPreActTag {};

struct SigmoidForward {
  using Accumulator = float;
  PLASTIX_HD static float Map(auto &U, size_t, size_t SrcId, auto &C,
                              size_t ConnId, auto &) {
    return plastix::GetWeight(C, ConnId) * plastix::GetActivation(U, SrcId);
  }
  PLASTIX_HD static float Combine(float A, float B) { return A + B; }
  PLASTIX_HD static void Apply(auto &U, size_t Id, auto &, float Acc) {
    plastix::GetActivation(U, Id) = 1.0f / (1.0f + std::exp(-Acc));
  }
};

struct SigmoidBackward {
  using Accumulator = float;
  PLASTIX_HD static float Map(auto &U, size_t, size_t ToId, auto &C,
                              size_t ConnId, auto &) {
    return plastix::GetWeight(C, ConnId) *
           plastix::GetField<GradPreActTag>(U, ToId);
  }
  PLASTIX_HD static float Combine(float A, float B) { return A + B; }
  PLASTIX_HD static void Apply(auto &U, size_t Id, auto &, float Acc) {
    const float A = plastix::GetActivation(U, Id);
    plastix::GetField<GradPreActTag>(U, Id) = Acc * A * (1.0f - A);
  }
};

// Everything except the learning rule. Each optimizer below varies only this
// base's UpdateConn and connection fields, via WithOptimizer.
struct XorBase : plastix::DefaultNetworkTraits<> {
  using ForwardPass = SigmoidForward;
  using BackwardPass = SigmoidBackward;
  using Loss = plastix::MSELoss;
  using ExtraUnitFields =
      plastix::UnitFieldList<plastix::alloc::SOAField<GradPreActTag, float>>;
};

// Learning rates differ by rule, not by preference: SGD and momentum step
// along a raw gradient, while Adam and RMSProp divide by a running scale and
// so want a smaller one. Each was picked to make the rule converge on this
// problem, which is all the example claims -- these are not tuned results and
// the losses below are not a ranking of the optimizers.
//
// RMSProp is the fussy one here, and instructively so: it solves XOR at 0.02
// and 0.05 but diverges at 0.1, because dividing by sqrt(v) keeps the step
// near `lr` regardless of gradient magnitude, so too large an `lr` overshoots
// on every step instead of just the early ones.
struct SgdLr : plastix::optim::SgdParams {
  static constexpr float Lr = 0.5f;
};
struct MomentumLr : plastix::optim::MomentumParams {
  static constexpr float Lr = 0.1f;
};
struct RmsPropLr : plastix::optim::RmsPropParams {
  static constexpr float Lr = 0.02f;
};
struct AdamLr : plastix::optim::AdamParams {
  static constexpr float Lr = 0.05f;
};
struct AdamWLr : plastix::optim::AdamWParams {
  static constexpr float Lr = 0.05f;
};

const float Inputs[4][3] = {
    {0.0f, 0.0f, 1.0f}, // the constant third input is the bias, so hidden and
    {0.0f, 1.0f, 1.0f}, // output units learn a bias through ordinary incoming
    {1.0f, 0.0f, 1.0f}, // weights with no extra plumbing
    {1.0f, 1.0f, 1.0f},
};
const float Targets[4] = {0.0f, 1.0f, 1.0f, 0.0f};

// Train one optimizer and report where it got to. Templated on the bundle, so
// the body below is literally the same code for all five.
template <typename Opt> bool Run(const char *Name) {
  struct Traits : plastix::WithOptimizer<XorBase, Opt> {};
  plastix::Network<Traits> Net(
      3, plastix::FullyConnected{4, plastix::RandomUniformWeight{1u}},
      plastix::FullyConnected{1, plastix::RandomUniformWeight{2u}});

  float Loss = 0.0f;
  for (size_t Epoch = 0; Epoch < NumEpochs; ++Epoch) {
    Loss = 0.0f;
    for (size_t S = 0; S < 4; ++S) {
      const float Target[1] = {Targets[S]};
      Net.DoStep(Inputs[S], Target);
      const float Diff = Net.GetOutput()[0] - Targets[S];
      Loss += 0.5f * Diff * Diff;
    }
  }

  bool Learned = true;
  for (size_t S = 0; S < 4; ++S) {
    Net.DoForwardPass(Inputs[S]);
    const int Predicted = Net.GetOutput()[0] > 0.5f ? 1 : 0;
    if (Predicted != static_cast<int>(Targets[S]))
      Learned = false;
  }

  std::printf("  %-10s  final loss %8.6f   %s\n", Name, static_cast<double>(Loss),
              Learned ? "solved" : "DID NOT SOLVE");
  return Learned;
}

} // namespace

int main() {
  std::printf("Plastix optimizer bundles / XOR\n");
  std::printf("===============================\n");
  std::printf("Same traits, same seeds, %zu epochs -- only the rule differs.\n\n",
              NumEpochs);

  namespace optim = plastix::optim;
  bool AllLearned = true;
  AllLearned &= Run<optim::Sgd<GradPreActTag, SgdLr>>("sgd");
  AllLearned &= Run<optim::Momentum<GradPreActTag, MomentumLr>>("momentum");
  AllLearned &= Run<optim::RmsProp<GradPreActTag, RmsPropLr>>("rmsprop");
  AllLearned &= Run<optim::Adam<GradPreActTag, AdamLr>>("adam");
  AllLearned &= Run<optim::AdamW<GradPreActTag, AdamWLr>>("adamw");

  std::printf("\n%s\n", AllLearned ? "PASS" : "FAIL");
  return AllLearned ? 0 : 1;
}
