// Topology-resort cost microbenchmark.
//
// SortConnectionsByLevel + RecomputeLevels run inside DoForwardPass
// whenever the network's NeedsResort flag is set — flipped by
// DoAddConnections (on commit) or DoCompactConnections in the
// Topological model. The cost dominates the structural phase in several
// of the dynamic benches in `traditional/_results/runs.csv` (the
// `structural_ns_mean` column in 09_imprintin_learner is 4.5 ms on
// average — that is the resort).
//
// SortConnectionsByLevel is private inside plastix::Network, so we
// expose its cost indirectly with three timed measurements:
//
//   T_forward_only       = DoForwardPass on a sorted network
//   T_compact_only       = DoCompactConnections with zero dead
//                          (just a tombstone scan, no Gather)
//   T_forward_after_dirty = DoCompactConnections + DoForwardPass
//                           (compact flips NeedsResort=true; forward
//                            then triggers the resort)
//
// Resort cost in post-processing is T_forward_after_dirty
//   - T_forward_only - T_compact_only.
//
// Args per phase: (In, Layers, Hidden, density_bps)
//   Layers = number of hidden levels (controls graph depth, which is the
//            first-order driver of SortConnectionsByLevel's histogram
//            and Kahn-BFS costs).
//   Hidden = units per hidden level.
//
// All shapes share the same Plastix traits as bench_phase_isolated and
// use deterministic edge predicates / weights.

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
#include <vector>

using bench_common::EdgeExists;
using bench_common::EdgeWeight;
using bench_common::RandomVector;

namespace {

// Identity activation + the bare minimum policies needed for
// DoCompactConnections to be enabled (NetworkShrinks && HasConnAdd).
// The policies don't have to do anything useful — they just have to be
// non-No* so the static gates open.

struct IdentityFwd {
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

struct NeverPruneConn {
  PLASTIX_HD static bool ShouldPrune(auto &, size_t, size_t, auto &, size_t,
                                     auto &) {
    return false;
  }
};

struct NeverAddConn {
  PLASTIX_HD static bool ShouldAddIncomingConnection(auto &, size_t, size_t,
                                                     auto &) {
    return false;
  }
  PLASTIX_HD static bool ShouldAddOutgoingConnection(auto &, size_t, size_t,
                                                     auto &) {
    return false;
  }
  PLASTIX_HD static void InitConnection(auto &, size_t, size_t, auto &, size_t,
                                        auto &) {}
};

struct ResortTraits : plastix::DefaultNetworkTraits<> {
  using ForwardPass = IdentityFwd;
  using PruneConn = NeverPruneConn; // satisfies NetworkShrinks<Traits>
  using AddConn = NeverAddConn;     // satisfies HasConnAdd<Traits>
  static constexpr size_t UnitCapacity = 64 * 1024;
  static constexpr size_t ConnCapacity = 64 * 1024 * 1024;
};

using ResortNet = plastix::Network<ResortTraits>;

// Layer builder identical in semantics to bench_phase_isolated /
// bench_spmv_crossover.
struct SparseLinearLayer {
  size_t NumUnits;
  float Density;
  size_t *NnzOut;

  template <typename UA, typename CA>
  plastix::UnitRange operator()(UA &UnitAlloc, CA &ConnAlloc,
                                plastix::UnitRange Prev) const {
    uint16_t SrcLevel = plastix::GetLevel(UnitAlloc, Prev.Begin);
    uint16_t NewLevel = SrcLevel + 1;
    plastix::UnitRange Units = UnitAlloc.AllocateMany(NumUnits);
    for (auto Id : Units.Ids())
      plastix::GetLevel(UnitAlloc, Id) = NewLevel;

    size_t In = Prev.End - Prev.Begin;
    size_t Out = NumUnits;
    size_t Nnz = 0;
    for (size_t Src = 0; Src < In; ++Src) {
      for (size_t Dst = 0; Dst < Out; ++Dst) {
        if (!EdgeExists(Src, Dst, Out, Density))
          continue;
        auto Cid = ConnAlloc.Allocate();
        plastix::GetField<plastix::FromIdTag>(ConnAlloc, Cid) =
            static_cast<uint32_t>(Prev.Begin + Src);
        plastix::GetField<plastix::ToIdTag>(ConnAlloc, Cid) =
            static_cast<uint32_t>(Units.Begin + Dst);
        plastix::GetField<plastix::SrcLevelTag>(ConnAlloc, Cid) = SrcLevel;
        plastix::GetWeight(ConnAlloc, Cid) = EdgeWeight(Src, Dst, Out);
        ++Nnz;
      }
    }
    if (NnzOut)
      *NnzOut += Nnz;
    return Units;
  }
};

struct Args {
  size_t In;
  size_t Layers;
  size_t Hidden;
  float Density;
};

Args ParseArgs(const benchmark::State &S) {
  Args A;
  A.In = static_cast<size_t>(S.range(0));
  A.Layers = static_cast<size_t>(S.range(1));
  A.Hidden = static_cast<size_t>(S.range(2));
  A.Density = static_cast<float>(S.range(3)) / 10000.0f;
  return A;
}

void AnnotateCounters(benchmark::State &S, const Args &A, size_t Nnz,
                      size_t Units, size_t Conns) {
  S.counters["In"] = static_cast<double>(A.In);
  S.counters["Layers"] = static_cast<double>(A.Layers);
  S.counters["Hidden"] = static_cast<double>(A.Hidden);
  S.counters["density"] = A.Density;
  S.counters["nnz_init"] = static_cast<double>(Nnz);
  S.counters["units"] = static_cast<double>(Units);
  S.counters["conns_live"] = static_cast<double>(Conns);
}

// Build a deep MLP: In -> [Hidden]xLayers, each layer at its own level
// so the resort pass has Layers+1 levels to bucket connections into.
std::unique_ptr<ResortNet> BuildDeepNet(const Args &A, size_t *NnzOut) {
  if (NnzOut)
    *NnzOut = 0;
  // RandomSparseLayer's NnzOut argument expects a pointer — we accumulate
  // through it across all layers by using the same pointer everywhere.
  // Workaround: capture in a vector of builders and feed the same ptr.
  // SparseLinearLayer adds to *NnzOut on construction-time call.
  switch (A.Layers) {
  case 1:
    return std::make_unique<ResortNet>(
        A.In, SparseLinearLayer{A.Hidden, A.Density, NnzOut});
  case 2:
    return std::make_unique<ResortNet>(
        A.In, SparseLinearLayer{A.Hidden, A.Density, NnzOut},
        SparseLinearLayer{A.Hidden, A.Density, NnzOut});
  case 4:
    return std::make_unique<ResortNet>(
        A.In, SparseLinearLayer{A.Hidden, A.Density, NnzOut},
        SparseLinearLayer{A.Hidden, A.Density, NnzOut},
        SparseLinearLayer{A.Hidden, A.Density, NnzOut},
        SparseLinearLayer{A.Hidden, A.Density, NnzOut});
  case 8:
    return std::make_unique<ResortNet>(
        A.In, SparseLinearLayer{A.Hidden, A.Density, NnzOut},
        SparseLinearLayer{A.Hidden, A.Density, NnzOut},
        SparseLinearLayer{A.Hidden, A.Density, NnzOut},
        SparseLinearLayer{A.Hidden, A.Density, NnzOut},
        SparseLinearLayer{A.Hidden, A.Density, NnzOut},
        SparseLinearLayer{A.Hidden, A.Density, NnzOut},
        SparseLinearLayer{A.Hidden, A.Density, NnzOut},
        SparseLinearLayer{A.Hidden, A.Density, NnzOut});
  case 16: {
    return std::make_unique<ResortNet>(
        A.In, SparseLinearLayer{A.Hidden, A.Density, NnzOut},
        SparseLinearLayer{A.Hidden, A.Density, NnzOut},
        SparseLinearLayer{A.Hidden, A.Density, NnzOut},
        SparseLinearLayer{A.Hidden, A.Density, NnzOut},
        SparseLinearLayer{A.Hidden, A.Density, NnzOut},
        SparseLinearLayer{A.Hidden, A.Density, NnzOut},
        SparseLinearLayer{A.Hidden, A.Density, NnzOut},
        SparseLinearLayer{A.Hidden, A.Density, NnzOut},
        SparseLinearLayer{A.Hidden, A.Density, NnzOut},
        SparseLinearLayer{A.Hidden, A.Density, NnzOut},
        SparseLinearLayer{A.Hidden, A.Density, NnzOut},
        SparseLinearLayer{A.Hidden, A.Density, NnzOut},
        SparseLinearLayer{A.Hidden, A.Density, NnzOut},
        SparseLinearLayer{A.Hidden, A.Density, NnzOut},
        SparseLinearLayer{A.Hidden, A.Density, NnzOut},
        SparseLinearLayer{A.Hidden, A.Density, NnzOut});
  }
  default:
    std::fprintf(stderr, "Unsupported Layers=%zu\n", A.Layers);
    std::abort();
  }
}

// ---------------------------------------------------------------------------
// The three measurements.
// ---------------------------------------------------------------------------

void BM_ForwardOnly(benchmark::State &S) {
  auto A = ParseArgs(S);
  size_t Nnz = 0;
  auto Net = BuildDeepNet(A, &Nnz);
  auto Input = RandomVector(A.In);
  // Prime: first forward after construction is already cheap (sorted at
  // ctor time). Verify NeedsResort is false by running one pass.
  Net->DoForwardPass(std::span<const float>(Input));

  for (auto _ : S) {
    Net->DoForwardPass(std::span<const float>(Input));
    auto Out = Net->GetOutput();
    benchmark::DoNotOptimize(Out.data());
  }
  AnnotateCounters(S, A, Nnz, Net->GetUnitAlloc().Size(),
                   Net->GetConnAlloc().Size());
}

void BM_CompactOnly(benchmark::State &S) {
  auto A = ParseArgs(S);
  size_t Nnz = 0;
  auto Net = BuildDeepNet(A, &Nnz);
  // No DeadTag set anywhere — DoCompactConnections will scan every
  // connection once, find zero dead, and bail before Gather.
  for (auto _ : S) {
    Net->DoCompactConnections();
    benchmark::ClobberMemory();
  }
  AnnotateCounters(S, A, Nnz, Net->GetUnitAlloc().Size(),
                   Net->GetConnAlloc().Size());
}

void BM_ForwardAfterDirty(benchmark::State &S) {
  auto A = ParseArgs(S);
  size_t Nnz = 0;
  auto Net = BuildDeepNet(A, &Nnz);
  auto Input = RandomVector(A.In);

  // The first forward after construction may itself trigger a resort
  // (NeedsResort is false in this configuration, but warm the cache).
  Net->DoForwardPass(std::span<const float>(Input));

  for (auto _ : S) {
    // Compact flips NeedsResort=true (Topological model). The next
    // forward then pays the resort cost.
    Net->DoCompactConnections();
    Net->DoForwardPass(std::span<const float>(Input));
    auto Out = Net->GetOutput();
    benchmark::DoNotOptimize(Out.data());
  }
  AnnotateCounters(S, A, Nnz, Net->GetUnitAlloc().Size(),
                   Net->GetConnAlloc().Size());
}

// ---------------------------------------------------------------------------
// Validation: build a small 2-layer net, run the dirty path, assert the
// output matches the warm path. This proves the resort itself is
// idempotent on equilibrium topology.
// ---------------------------------------------------------------------------

void Validate() {
  Args A{16, 2, 8, 0.5f};
  size_t Nnz = 0;
  auto Net = BuildDeepNet(A, &Nnz);
  auto Input = RandomVector(A.In);

  Net->DoForwardPass(std::span<const float>(Input));
  std::vector<float> WarmOut(Net->GetOutput().begin(), Net->GetOutput().end());

  // Trigger a resort by calling DoCompactConnections; forward should
  // produce bit-identical output because no edges were marked dead.
  Net->DoCompactConnections();
  Net->DoForwardPass(std::span<const float>(Input));
  auto DirtyOut = Net->GetOutput();

  if (WarmOut.size() != DirtyOut.size()) {
    std::fprintf(stderr, "Validation: output size mismatch %zu vs %zu\n",
                 WarmOut.size(), DirtyOut.size());
    std::abort();
  }
  for (size_t I = 0; I < WarmOut.size(); ++I) {
    float D = std::abs(WarmOut[I] - DirtyOut[I]);
    float Tol = 1e-5f * std::max(1.0f, std::abs(WarmOut[I]));
    if (D > Tol) {
      std::fprintf(stderr,
                   "Validation: output %zu warm=%g dirty=%g (tol=%g)\n", I,
                   WarmOut[I], DirtyOut[I], Tol);
      std::abort();
    }
  }
  std::fprintf(stderr, "Validation OK (resort cost): %zu outputs match\n",
               WarmOut.size());
}

void RegisterAll() {
  // (In, Layers, Hidden, density_bps)
  const std::vector<std::tuple<int64_t, int64_t, int64_t>> Shapes = {
      {64, 1, 256}, {64, 2, 256}, {64, 4, 256}, {64, 8, 256}, {64, 16, 256},
      {64, 4, 512}, {64, 4, 1024},
      {256, 4, 256}, {256, 4, 1024},
  };
  const std::vector<int64_t> Densities = {500, 2000, 5000};

  auto Reg = [](const char *Name, auto Fn, int64_t In, int64_t L, int64_t H,
                int64_t D) {
    benchmark::RegisterBenchmark(Name, Fn)
        ->Args({In, L, H, D})
        ->Unit(benchmark::kMicrosecond)
        ->MinTime(0.25);
  };

  for (auto [In, L, H] : Shapes) {
    for (auto D : Densities) {
      Reg("ForwardOnly", BM_ForwardOnly, In, L, H, D);
      Reg("CompactOnly", BM_CompactOnly, In, L, H, D);
      Reg("ForwardAfterDirty", BM_ForwardAfterDirty, In, L, H, D);
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  Validate();
  RegisterAll();

  auto Default = bench_common::OutputFile("resort_cost.json").string();
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
