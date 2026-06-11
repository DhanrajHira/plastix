// Phase-isolated Plastix microbenchmark.
//
// Companion to bench_spmv_crossover.cpp. Whereas the crossover bench
// times one fused `DoForwardPass`, this one drills into every per-step
// phase the framework owns:
//
//   Forward, Loss, Backward, UpdateConn, PruneConn, AddConn,
//   CompactConn, Reset
//
// Each phase is timed in isolation on the *same* underlying network so
// the costs are directly comparable and the per-phase breakdowns in
// `traditional/_results/runs.csv` (which fuse them inside `DoStep`) can
// be reproduced and audited.
//
// Args per phase: (In, Hidden, density_bps)
//   density_bps = density in basis points (1 bp = 0.01%).
//
// All phases use the same toy MLP:
//   inputs -> 1 hidden layer (linear activation) -> 1 output (linear)
// with a deterministic sparse mask shared with the crossover bench.
//
// Validation: a tiny (In=4, Hidden=3, Out=1) instance with hand-set
// weights and a hand-computed forward+backward+update step. Asserted
// before any phase is timed.

#include <benchmark/benchmark.h>

#include "_bench_common.hpp"
#include "plastix/plastix.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

using bench_common::EdgeExists;
using bench_common::EdgeWeight;
using bench_common::RandomVector;

namespace {

// ---------------------------------------------------------------------------
// Policy set: linear MLP with SGD + threshold prune + Bernoulli add.
// ---------------------------------------------------------------------------

struct GradPreActTag {};

struct PhaseGlobals {
  float LearningRate = 0.01f;
  float PruneThreshold = 0.05f;
  uint32_t AddSeed = 0xA5A5A5A5u;
  uint32_t StepCounter = 0;
  float AddRate = 1e-6f;
};

struct LinearForward {
  using Accumulator = float;
  PLASTIX_HD static float Map(auto &U, size_t, size_t SrcId, auto &C,
                              size_t Cid, auto &) {
    return plastix::GetWeight(C, Cid) * plastix::GetActivation(U, SrcId);
  }
  PLASTIX_HD static float Combine(float A, float B) { return A + B; }
  PLASTIX_HD static void Apply(auto &U, size_t Id, auto &, float Acc) {
    plastix::GetActivation(U, Id) = Acc;
  }
};

// Linear activation: grad-pre-act == grad-activation. The output level
// gets its starting gradient from the MSE Loss policy (BackwardAcc).
struct LinearBackward {
  using Accumulator = float;
  PLASTIX_HD static float Map(auto &U, size_t, size_t ToId, auto &C,
                              size_t Cid, auto &) {
    return plastix::GetWeight(C, Cid) *
           plastix::GetField<GradPreActTag>(U, ToId);
  }
  PLASTIX_HD static float Combine(float A, float B) { return A + B; }
  PLASTIX_HD static void Apply(auto &U, size_t Id, auto &, float Acc) {
    plastix::GetField<GradPreActTag>(U, Id) = Acc;
  }
};

struct SGDUpdateConn {
  PLASTIX_HD static void
  UpdateIncomingConnection(auto &U, size_t Dst, size_t Src, auto &C,
                           size_t Cid, PhaseGlobals &G) {
    float Grad = plastix::GetField<GradPreActTag>(U, Dst);
    float In = plastix::GetActivation(U, Src);
    plastix::GetWeight(C, Cid) -= G.LearningRate * Grad * In;
  }
  PLASTIX_HD static void UpdateOutgoingConnection(auto &, size_t, size_t,
                                                  auto &, size_t,
                                                  PhaseGlobals &) {}
};

struct ThresholdPruneConn {
  PLASTIX_HD static bool ShouldPrune(auto &, size_t, size_t, auto &C,
                                     size_t Cid, PhaseGlobals &G) {
    float W = plastix::GetWeight(C, Cid);
    return std::abs(W) < G.PruneThreshold;
  }
};

struct BernoulliAddConn {
  PLASTIX_HD static bool ShouldAddIncomingConnection(auto &, size_t Self,
                                                     size_t Cand,
                                                     PhaseGlobals &G) {
    uint64_t Ctr = (static_cast<uint64_t>(Self) << 32) | Cand;
    return plastix::Bernoulli(G.AddSeed + G.StepCounter, Ctr, G.AddRate);
  }
  PLASTIX_HD static bool ShouldAddOutgoingConnection(auto &, size_t, size_t,
                                                     PhaseGlobals &) {
    return false;
  }
  PLASTIX_HD static void InitConnection(auto &, size_t, size_t, auto &C,
                                        size_t Cid, PhaseGlobals &) {
    plastix::GetWeight(C, Cid) = 0.1f;
  }
};

struct ZeroActivationsReset {
  static void Reset(PhaseGlobals &G) { ++G.StepCounter; }
};

struct PhaseTraits : plastix::DefaultNetworkTraits<PhaseGlobals> {
  using ForwardPass = LinearForward;
  using BackwardPass = LinearBackward;
  using Loss = plastix::MSELoss;
  using UpdateConn = SGDUpdateConn;
  using PruneConn = ThresholdPruneConn;
  using AddConn = BernoulliAddConn;
  using ResetGlobal = ZeroActivationsReset;
  using ExtraUnitFields =
      plastix::UnitFieldList<plastix::alloc::SOAField<GradPreActTag, float>>;
  static constexpr size_t UnitCapacity = 64 * 1024;
  static constexpr size_t ConnCapacity = 32 * 1024 * 1024;
};

using PhaseNet = plastix::Network<PhaseTraits>;

// ---------------------------------------------------------------------------
// Layer builder: emits a deterministic random sparse layer, sharing the
// same EdgeExists / EdgeWeight functions as bench_spmv_crossover so the
// connection counts and weights match across files.
// ---------------------------------------------------------------------------

struct SparseLinearLayer {
  size_t NumUnits;
  float Density;
  size_t *NnzOut;

  template <typename UnitAlloc, typename ConnAlloc>
  plastix::UnitRange operator()(UnitAlloc &UA, ConnAlloc &CA,
                                plastix::UnitRange Prev) const {
    uint16_t SrcLevel = plastix::GetLevel(UA, Prev.Begin);
    uint16_t NewLevel = SrcLevel + 1;
    plastix::UnitRange Units = UA.AllocateMany(NumUnits);
    for (auto Id : Units.Ids())
      plastix::GetLevel(UA, Id) = NewLevel;

    size_t In = Prev.End - Prev.Begin;
    size_t Out = NumUnits;
    size_t Nnz = 0;
    for (size_t Src = 0; Src < In; ++Src) {
      for (size_t Dst = 0; Dst < Out; ++Dst) {
        if (!EdgeExists(Src, Dst, Out, Density))
          continue;
        auto Cid = CA.Allocate();
        plastix::GetField<plastix::FromIdTag>(CA, Cid) =
            static_cast<uint32_t>(Prev.Begin + Src);
        plastix::GetField<plastix::ToIdTag>(CA, Cid) =
            static_cast<uint32_t>(Units.Begin + Dst);
        plastix::GetField<plastix::SrcLevelTag>(CA, Cid) = SrcLevel;
        plastix::GetWeight(CA, Cid) = EdgeWeight(Src, Dst, Out);
        ++Nnz;
      }
    }
    if (NnzOut)
      *NnzOut = Nnz;
    return Units;
  }
};

// ---------------------------------------------------------------------------
// Argument parsing / counter annotation.
// ---------------------------------------------------------------------------

struct Args {
  size_t In;
  size_t Hidden;
  float Density;
};

Args ParseArgs(const benchmark::State &S) {
  Args A;
  A.In = static_cast<size_t>(S.range(0));
  A.Hidden = static_cast<size_t>(S.range(1));
  A.Density = static_cast<float>(S.range(2)) / 10000.0f;
  return A;
}

void AnnotateCounters(benchmark::State &S, const Args &A, size_t Nnz,
                      size_t Units, size_t Conns) {
  S.counters["In"] = static_cast<double>(A.In);
  S.counters["Hidden"] = static_cast<double>(A.Hidden);
  S.counters["density"] = A.Density;
  S.counters["nnz_init"] = static_cast<double>(Nnz);
  S.counters["units"] = static_cast<double>(Units);
  S.counters["conns_live"] = static_cast<double>(Conns);
}

// Build a 1-hidden-layer MLP with a sparse In->Hidden mask and a fully
// dense Hidden->1 head.
std::unique_ptr<PhaseNet> BuildNet(const Args &A, size_t *NnzOut) {
  *NnzOut = 0;
  auto Net = std::make_unique<PhaseNet>(
      A.In,
      SparseLinearLayer{A.Hidden, A.Density, NnzOut},
      SparseLinearLayer{1, 1.0f, nullptr});
  return Net;
}

// ---------------------------------------------------------------------------
// Per-phase benchmarks. The fixed-topology phases (Forward, Backward,
// Loss, UpdateConn, Reset) build the network once and iterate. The
// topology-changing phases (PruneConn, AddConn, CompactConn) either pair
// with the inverse operation or rebuild via Pause/Resume so each timed
// iteration runs on a known starting state.
// ---------------------------------------------------------------------------

void BM_Forward(benchmark::State &S) {
  auto A = ParseArgs(S);
  size_t Nnz = 0;
  auto Net = BuildNet(A, &Nnz);
  auto Input = RandomVector(A.In);

  for (auto _ : S) {
    Net->DoForwardPass(std::span<const float>(Input));
    auto Out = Net->GetOutput();
    benchmark::DoNotOptimize(Out.data());
  }
  AnnotateCounters(S, A, Nnz, Net->GetUnitAlloc().Size(),
                   Net->GetConnAlloc().Size());
}

void BM_Loss(benchmark::State &S) {
  auto A = ParseArgs(S);
  size_t Nnz = 0;
  auto Net = BuildNet(A, &Nnz);
  auto Input = RandomVector(A.In);
  Net->DoForwardPass(std::span<const float>(Input));
  std::vector<float> Tgt(1, 0.5f);

  for (auto _ : S) {
    Net->DoCalculateLoss(std::span<const float>(Tgt));
  }
  AnnotateCounters(S, A, Nnz, Net->GetUnitAlloc().Size(),
                   Net->GetConnAlloc().Size());
}

void BM_Backward(benchmark::State &S) {
  auto A = ParseArgs(S);
  size_t Nnz = 0;
  auto Net = BuildNet(A, &Nnz);
  auto Input = RandomVector(A.In);
  std::vector<float> Tgt(1, 0.5f);
  Net->DoForwardPass(std::span<const float>(Input));
  Net->DoCalculateLoss(std::span<const float>(Tgt));

  for (auto _ : S) {
    // Need to re-prime the loss-side BackwardAcc on every iteration;
    // DoBackwardPass zeroes BackwardAcc as it propagates.
    S.PauseTiming();
    Net->DoCalculateLoss(std::span<const float>(Tgt));
    S.ResumeTiming();
    Net->DoBackwardPass();
  }
  AnnotateCounters(S, A, Nnz, Net->GetUnitAlloc().Size(),
                   Net->GetConnAlloc().Size());
}

void BM_UpdateConn(benchmark::State &S) {
  auto A = ParseArgs(S);
  size_t Nnz = 0;
  auto Net = BuildNet(A, &Nnz);
  auto Input = RandomVector(A.In);
  std::vector<float> Tgt(1, 0.5f);
  Net->DoForwardPass(std::span<const float>(Input));
  Net->DoCalculateLoss(std::span<const float>(Tgt));
  Net->DoBackwardPass();

  for (auto _ : S) {
    Net->DoUpdateConnectionState();
    benchmark::ClobberMemory();
  }
  AnnotateCounters(S, A, Nnz, Net->GetUnitAlloc().Size(),
                   Net->GetConnAlloc().Size());
}

// PruneConn timing. Each timed iteration prunes from a fresh state with
// all DeadTag=false; Pause/Resume cleans up the dead flags from the prior
// iteration without counting that cleanup in the measured time.
void BM_PruneConn(benchmark::State &S) {
  auto A = ParseArgs(S);
  size_t Nnz = 0;
  auto Net = BuildNet(A, &Nnz);
  auto &CA = Net->GetConnAlloc();

  for (auto _ : S) {
    S.PauseTiming();
    size_t N = CA.Size();
    for (size_t C = 0; C < N; ++C)
      plastix::GetField<plastix::DeadTag>(CA, C) = false;
    S.ResumeTiming();
    Net->DoPruneConnections();
  }
  AnnotateCounters(S, A, Nnz, Net->GetUnitAlloc().Size(),
                   Net->GetConnAlloc().Size());
}

// CompactConn timing. Pre-mark a configurable fraction of connections as
// dead, time the compaction call, then rebuild for the next iteration.
void BM_CompactConn(benchmark::State &S) {
  auto A = ParseArgs(S);
  size_t Nnz = 0;
  auto Net = BuildNet(A, &Nnz);
  auto &CA = Net->GetConnAlloc();

  // S.range(3) = dead-fraction in percent (0-100).
  float DeadFrac = static_cast<float>(S.range(3)) / 100.0f;
  size_t TotalLive = CA.Size();

  for (auto _ : S) {
    S.PauseTiming();
    // Reset live; flag a deterministic subset as dead.
    size_t N = CA.Size();
    for (size_t C = 0; C < N; ++C)
      plastix::GetField<plastix::DeadTag>(CA, C) = false;
    for (size_t C = 0; C < TotalLive; ++C) {
      if (plastix::Bernoulli(0xDEADC0DEull, C, DeadFrac))
        plastix::GetField<plastix::DeadTag>(CA, C) = true;
    }
    S.ResumeTiming();
    Net->DoCompactConnections();
    S.PauseTiming();
    // Compaction shrinks Size(); the test rebuilds before the next
    // iteration so the input shape stays comparable across iterations.
    if (CA.Size() != TotalLive) {
      Net.reset();
      Net = BuildNet(A, &Nnz);
      TotalLive = Net->GetConnAlloc().Size();
    }
    S.ResumeTiming();
  }
  S.counters["dead_frac"] = DeadFrac;
  AnnotateCounters(S, A, Nnz, Net->GetUnitAlloc().Size(),
                   Net->GetConnAlloc().Size());
}

void BM_AddConn(benchmark::State &S) {
  auto A = ParseArgs(S);
  size_t Nnz = 0;
  auto Net = BuildNet(A, &Nnz);
  size_t StartingConns = Net->GetConnAlloc().Size();

  for (auto _ : S) {
    Net->DoAddConnections();
    // Throw it away to keep proposal scratch and connection count from
    // unbounded growth. Rebuild rarely; AddConn's proposal phase is
    // cheap on the second visit because Bernoulli is deterministic.
    if (Net->GetConnAlloc().Size() > StartingConns + 64) {
      S.PauseTiming();
      Net.reset();
      Net = BuildNet(A, &Nnz);
      StartingConns = Net->GetConnAlloc().Size();
      S.ResumeTiming();
    }
  }
  AnnotateCounters(S, A, Nnz, Net->GetUnitAlloc().Size(),
                   Net->GetConnAlloc().Size());
}

void BM_Reset(benchmark::State &S) {
  auto A = ParseArgs(S);
  size_t Nnz = 0;
  auto Net = BuildNet(A, &Nnz);
  for (auto _ : S) {
    Net->DoResetGlobalState();
  }
  AnnotateCounters(S, A, Nnz, Net->GetUnitAlloc().Size(),
                   Net->GetConnAlloc().Size());
}

// ---------------------------------------------------------------------------
// Validation: a tiny (In=4, Hidden=3, Out=1) instance with hand-set
// weights. We bypass SparseLinearLayer's random fill by patching the
// allocator directly so the expected forward/backward/update outputs are
// trivial closed forms.
//
// Topology:
//   inputs:  i0, i1, i2, i3            (level 0)
//   hidden:  h0, h1, h2                (level 1)
//   output:  y                         (level 2)
//
// Hand-set weights (in-major):
//   i->h:  i0->h0 = 1.0   i1->h0 = 2.0   i2->h0 = 0
//          i0->h1 = 0     i1->h1 = -1.0  i3->h1 = 0.5
//          i2->h2 = 1.0   i3->h2 = -2.0
//   h->y:  h0->y  = 1.0   h1->y  = 1.0   h2->y  = -1.0
//
// Forward (x = [1, 1, 1, 1]):
//   h0 = 1*1 + 2*1                       =  3
//   h1 = -1*1 + 0.5*1                    = -0.5
//   h2 = 1*1 + -2*1                      = -1
//   y  = 1*3 + 1*(-0.5) + -1*(-1)        =  3.5
//
// Loss (target = 2.0):
//   BackwardAcc[y] = y - target          = 1.5
//
// Backward:
//   GradPreAct[y]  = 1.5  (linear apply)
//   BackwardAcc[h0] = w(h0->y) * grad[y] = 1.0 * 1.5  =  1.5
//   BackwardAcc[h1] = w(h1->y) * grad[y] = 1.0 * 1.5  =  1.5
//   BackwardAcc[h2] = w(h2->y) * grad[y] = -1.0 * 1.5 = -1.5
//   GradPreAct[h0] = 1.5; [h1] = 1.5; [h2] = -1.5
//
// UpdateConn (lr = 0.1):
//   w(h0->y) -= 0.1 * grad[y] * act[h0] = 0.1 * 1.5 *  3   = 0.45  -> 0.55
//   w(h1->y) -= 0.1 * grad[y] * act[h1] = 0.1 * 1.5 * -0.5 = -0.075 -> 1.075
//   w(h2->y) -= 0.1 * grad[y] * act[h2] = 0.1 * 1.5 * -1   = -0.15 -> -0.85
//   w(i0->h0) -= 0.1 * grad[h0] * 1     = 0.15            -> 0.85
//   w(i1->h0) -= 0.1 * grad[h0] * 1     = 0.15            -> 1.85
//   w(i1->h1) -= 0.1 * grad[h1] * 1     = 0.15            -> -1.15
//   w(i3->h1) -= 0.1 * grad[h1] * 1     = 0.15            -> 0.35
//   w(i2->h2) -= 0.1 * grad[h2] * 1     = -0.15           -> 1.15
//   w(i3->h2) -= 0.1 * grad[h2] * 1     = -0.15           -> -1.85
// ---------------------------------------------------------------------------

struct LocalEdge {
  uint32_t SrcLocal; // index inside prev layer
  uint32_t DstLocal; // index inside this layer
  float Weight;
};

// Layer builder that emits a fixed list of (src,dst,w) edges relative to
// the previous layer / this layer's local index space. Used by Validate
// so the topology matches the hand-traced expected values exactly.
struct ExplicitEdgeLayer {
  size_t NumUnits;
  std::vector<LocalEdge> Edges;

  template <typename UA, typename CA>
  plastix::UnitRange operator()(UA &UnitAlloc, CA &ConnAlloc,
                                plastix::UnitRange Prev) const {
    uint16_t SrcLevel = plastix::GetLevel(UnitAlloc, Prev.Begin);
    uint16_t NewLevel = SrcLevel + 1;
    plastix::UnitRange Units = UnitAlloc.AllocateMany(NumUnits);
    for (auto Id : Units.Ids())
      plastix::GetLevel(UnitAlloc, Id) = NewLevel;
    for (const auto &E : Edges) {
      auto Cid = ConnAlloc.Allocate();
      plastix::GetField<plastix::FromIdTag>(ConnAlloc, Cid) =
          static_cast<uint32_t>(Prev.Begin + E.SrcLocal);
      plastix::GetField<plastix::ToIdTag>(ConnAlloc, Cid) =
          static_cast<uint32_t>(Units.Begin + E.DstLocal);
      plastix::GetField<plastix::SrcLevelTag>(ConnAlloc, Cid) = SrcLevel;
      plastix::GetField<plastix::DeadTag>(ConnAlloc, Cid) = false;
      plastix::GetWeight(ConnAlloc, Cid) = E.Weight;
    }
    return Units;
  }
};

void Validate() {
  // Topology built via the public LayerBuilder API so SortConnectionsByLevel
  // runs and the Ranges array stays consistent.
  //   inputs  0..3  -> hidden 4..6  -> output 7
  //   i0->h0=1   i1->h0=2
  //   i1->h1=-1  i3->h1=0.5
  //   i2->h2=1   i3->h2=-2
  //   h0->y=1    h1->y=1   h2->y=-1
  PhaseNet Net(
      4,
      ExplicitEdgeLayer{3,
                        {{0, 0, 1.0f},
                         {1, 0, 2.0f},
                         {1, 1, -1.0f},
                         {3, 1, 0.5f},
                         {2, 2, 1.0f},
                         {3, 2, -2.0f}}},
      ExplicitEdgeLayer{1,
                        {{0, 0, 1.0f}, {1, 0, 1.0f}, {2, 0, -1.0f}}});
  auto &CA = Net.GetConnAlloc();
  auto &UA = Net.GetUnitAlloc();

  // Forward
  std::vector<float> Input(4, 1.0f);
  Net.DoForwardPass(std::span<const float>(Input));
  auto Out = Net.GetOutput();
  auto AssertNear = [](const char *Tag, float Got, float Want, float Tol) {
    if (std::abs(Got - Want) > Tol) {
      std::fprintf(stderr, "Validation %s: got=%g want=%g (tol=%g)\n", Tag,
                   Got, Want, Tol);
      std::abort();
    }
  };
  AssertNear("forward.h0", plastix::GetActivation(UA, 4), 3.0f, 1e-5f);
  AssertNear("forward.h1", plastix::GetActivation(UA, 5), -0.5f, 1e-5f);
  AssertNear("forward.h2", plastix::GetActivation(UA, 6), -1.0f, 1e-5f);
  AssertNear("forward.y", Out[0], 3.5f, 1e-5f);

  // Loss
  std::vector<float> Tgt(1, 2.0f);
  Net.DoCalculateLoss(std::span<const float>(Tgt));
  AssertNear("loss.grad_y", plastix::GetBackwardAcc(UA, 7), 1.5f, 1e-5f);

  // Backward
  Net.DoBackwardPass();
  AssertNear("bwd.grad_pre.y", plastix::GetField<GradPreActTag>(UA, 7), 1.5f,
             1e-5f);
  AssertNear("bwd.grad_pre.h0", plastix::GetField<GradPreActTag>(UA, 4), 1.5f,
             1e-5f);
  AssertNear("bwd.grad_pre.h1", plastix::GetField<GradPreActTag>(UA, 5), 1.5f,
             1e-5f);
  AssertNear("bwd.grad_pre.h2", plastix::GetField<GradPreActTag>(UA, 6), -1.5f,
             1e-5f);

  // UpdateConn (lr=0.1)
  // PhaseTraits uses default GlobalState constructed by the network; bump
  // learning rate in our local copy.
  // The Network<> owns its GlobalState; we don't get to swap it. Instead
  // adapt expected values to LearningRate=0.01 (the default).
  Net.DoUpdateConnectionState();
  // Verify a single edge to keep the table short. w(h0->y) was 1.0;
  // new w = 1.0 - 0.01 * 1.5 * 3 = 1.0 - 0.045 = 0.955.
  // Find the (4,7) connection.
  size_t Found = SIZE_MAX;
  for (size_t I = 0; I < CA.Size(); ++I)
    if (plastix::GetField<plastix::FromIdTag>(CA, I) == 4 &&
        plastix::GetField<plastix::ToIdTag>(CA, I) == 7) {
      Found = I;
      break;
    }
  if (Found == SIZE_MAX) {
    std::fprintf(stderr, "Validation: did not find h0->y edge after update\n");
    std::abort();
  }
  AssertNear("update.w_h0_y", plastix::GetWeight(CA, Found), 0.955f, 1e-5f);

  std::fprintf(stderr, "Validation OK (phase isolated)\n");
}

// ---------------------------------------------------------------------------
// Registration. Shapes mirror crossover so we can stack the per-phase
// breakdowns next to the fused forward-pass timing.
// ---------------------------------------------------------------------------

void RegisterAll() {
  // (In, Hidden) shapes.
  const std::vector<std::pair<int64_t, int64_t>> Shapes = {
      {256, 256}, {1024, 1024}, {4096, 1024}, {4096, 4096},
  };
  const std::vector<int64_t> Densities = {100, 500, 1000, 2000, 5000};
  const std::vector<int64_t> DeadFracs = {10, 25, 50, 90};

  auto Reg2 = [](const char *Name, auto Fn, int64_t In, int64_t H, int64_t D) {
    benchmark::RegisterBenchmark(Name, Fn)
        ->Args({In, H, D})
        ->Unit(benchmark::kMicrosecond)
        ->MinTime(0.25);
  };
  auto Reg3 = [](const char *Name, auto Fn, int64_t In, int64_t H, int64_t D,
                 int64_t X) {
    benchmark::RegisterBenchmark(Name, Fn)
        ->Args({In, H, D, X})
        ->Unit(benchmark::kMicrosecond)
        ->MinTime(0.25);
  };

  for (auto [In, H] : Shapes) {
    for (auto D : Densities) {
      Reg2("Forward", BM_Forward, In, H, D);
      Reg2("Loss", BM_Loss, In, H, D);
      Reg2("Backward", BM_Backward, In, H, D);
      Reg2("UpdateConn", BM_UpdateConn, In, H, D);
      Reg2("PruneConn", BM_PruneConn, In, H, D);
      Reg2("AddConn", BM_AddConn, In, H, D);
      Reg2("Reset", BM_Reset, In, H, D);
      for (auto F : DeadFracs)
        Reg3("CompactConn", BM_CompactConn, In, H, D, F);
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  Validate();
  RegisterAll();

  // Default --benchmark_out to OutputDir() if the caller didn't pass one.
  auto Default = bench_common::OutputFile("phase_isolated.json").string();
  std::vector<std::string> Args(argv, argv + argc);
  bool HasOut = false;
  for (const auto &A : Args)
    if (A.rfind("--benchmark_out=", 0) == 0) { HasOut = true; break; }
  std::string OutFlag = "--benchmark_out=" + Default;
  std::string FmtFlag = "--benchmark_out_format=json";
  std::vector<char *> NewArgv;
  for (auto &A : Args)
    NewArgv.push_back(A.data());
  if (!HasOut) {
    NewArgv.push_back(OutFlag.data());
    NewArgv.push_back(FmtFlag.data());
  }
  int NewArgc = static_cast<int>(NewArgv.size());

  benchmark::Initialize(&NewArgc, NewArgv.data());
  if (benchmark::ReportUnrecognizedArguments(NewArgc, NewArgv.data()))
    return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
