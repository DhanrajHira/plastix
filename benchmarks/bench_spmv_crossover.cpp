// Plastix-vs-dense SpMV/GEMV crossover benchmark, CPU only.
//
// Compares a single-layer Plastix forward pass (random unstructured
// sparsity, identity activation) against three dense baselines computing
// the same y = W x: Eigen, OpenBLAS cblas_sgemv, and a hand-rolled tiled
// loop. The same edge predicate and weight draws populate both the Plastix
// connection allocator and the dense Out x In matrix, so all four kernels
// produce mathematically identical outputs.
//
// Args per benchmark: (In, Out, density_bps)
//   density_bps = density in basis points (1 bp = 0.01 %).
//                 e.g. 5000 -> 50 %, 100 -> 1 %, 10 -> 0.1 %.

#include <benchmark/benchmark.h>

#include <Eigen/Dense>
#include <cblas.h>

#include "plastix/plastix.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// Plastix configuration: identity-activation forward pass, no other policies.
// ---------------------------------------------------------------------------

struct IdentityFwd {
  using Accumulator = float;
  PLASTIX_HD static float Map(auto &U, size_t, size_t SrcId, auto &C,
                              size_t ConnId, auto &) {
    return plastix::GetWeight(C, ConnId) * plastix::GetActivation(U, SrcId);
  }
  PLASTIX_HD static float Combine(float A, float B) { return A + B; }
  PLASTIX_HD static void Apply(auto &U, size_t Id, auto &, float Acc) {
    plastix::GetActivation(U, Id) = Acc;
  }
};

// Capacities sized for the largest shape we test. SOAAllocator mmaps with
// MAP_NORESERVE so unused pages are never committed; oversizing is cheap.
struct BenchTraits : plastix::DefaultNetworkTraits<> {
  using ForwardPass = IdentityFwd;
  static constexpr size_t UnitCapacity = 32 * 1024;        // 32 K units
  static constexpr size_t ConnCapacity = 16 * 1024 * 1024; // 16 M edges
};

using BenchNet = plastix::Network<BenchTraits>;

// ---------------------------------------------------------------------------
// Shared edge predicate and weight: identical draws across kernels.
// ---------------------------------------------------------------------------

static constexpr uint64_t kEdgeSeed = 0xC0FFEE5EED5;
static constexpr uint64_t kWeightSeed = 0xDEADBEEFC0DECAFE;
static constexpr uint64_t kInputSeed = 0xABAD1DEA;

static inline bool EdgeExists(size_t Src, size_t Dst, size_t Out,
                              float Density) {
  return plastix::Bernoulli(kEdgeSeed, Src * Out + Dst, Density);
}
static inline float EdgeWeight(size_t Src, size_t Dst, size_t Out) {
  return plastix::UniformReal(kWeightSeed, Src * Out + Dst, -1.0f, 1.0f);
}

// LayerBuilder: walks the (Src, Dst) Cartesian in source-major order and
// emits an edge whenever EdgeExists holds. Stores nnz back through a pointer
// so the benchmark can report it as a counter.
struct RandomSparseLayer {
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

// Row-major Out x In dense matrix, zero-filled then patched at live edges.
static std::vector<float> BuildDenseMatrix(size_t In, size_t Out,
                                           float Density, size_t *NnzOut) {
  std::vector<float> W(Out * In, 0.0f);
  size_t Nnz = 0;
  for (size_t Src = 0; Src < In; ++Src) {
    for (size_t Dst = 0; Dst < Out; ++Dst) {
      if (!EdgeExists(Src, Dst, Out, Density))
        continue;
      W[Dst * In + Src] = EdgeWeight(Src, Dst, Out);
      ++Nnz;
    }
  }
  if (NnzOut)
    *NnzOut = Nnz;
  return W;
}

static std::vector<float> RandomVector(size_t N) {
  std::vector<float> V(N);
  for (size_t I = 0; I < N; ++I)
    V[I] = plastix::UniformReal(kInputSeed, I, -1.0f, 1.0f);
  return V;
}

static inline std::tuple<size_t, size_t, float>
ParseArgs(const benchmark::State &S) {
  auto In = static_cast<size_t>(S.range(0));
  auto Out = static_cast<size_t>(S.range(1));
  float Density = static_cast<float>(S.range(2)) / 10000.0f;
  return {In, Out, Density};
}

static inline void AnnotateCounters(benchmark::State &S, size_t In, size_t Out,
                                    float Density, size_t Nnz) {
  S.counters["In"] = static_cast<double>(In);
  S.counters["Out"] = static_cast<double>(Out);
  S.counters["density"] = Density;
  S.counters["nnz"] = static_cast<double>(Nnz);
  // FMA cost: dense baselines do In*Out MACs regardless of sparsity;
  // sparse-aware kernels do Nnz MACs.
  S.counters["dense_macs"] = static_cast<double>(In * Out);
  S.counters["sparse_macs"] = static_cast<double>(Nnz);
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

static void BM_Plastix(benchmark::State &S) {
  auto [In, Out, Density] = ParseArgs(S);
  size_t Nnz = 0;
  BenchNet Net(In, RandomSparseLayer{Out, Density, &Nnz});
  auto Input = RandomVector(In);
  std::span<const float> InSpan(Input);

  for (auto _ : S) {
    Net.DoForwardPass(InSpan);
    auto Output = Net.GetOutput();
    benchmark::DoNotOptimize(Output.data());
  }
  AnnotateCounters(S, In, Out, Density, Nnz);
}

static void BM_Eigen(benchmark::State &S) {
  auto [In, Out, Density] = ParseArgs(S);
  size_t Nnz = 0;
  auto W = BuildDenseMatrix(In, Out, Density, &Nnz);
  auto Input = RandomVector(In);
  std::vector<float> Output(Out);

  using EMat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                             Eigen::RowMajor>;
  Eigen::Map<const EMat> MW(W.data(), static_cast<Eigen::Index>(Out),
                            static_cast<Eigen::Index>(In));
  Eigen::Map<const Eigen::VectorXf> MI(Input.data(),
                                       static_cast<Eigen::Index>(In));
  Eigen::Map<Eigen::VectorXf> MO(Output.data(),
                                 static_cast<Eigen::Index>(Out));

  for (auto _ : S) {
    MO.noalias() = MW * MI;
    benchmark::DoNotOptimize(Output.data());
  }
  AnnotateCounters(S, In, Out, Density, Nnz);
}

static void BM_OpenBLAS(benchmark::State &S) {
  auto [In, Out, Density] = ParseArgs(S);
  size_t Nnz = 0;
  auto W = BuildDenseMatrix(In, Out, Density, &Nnz);
  auto Input = RandomVector(In);
  std::vector<float> Output(Out);

  for (auto _ : S) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans, static_cast<int>(Out),
                static_cast<int>(In), 1.0f, W.data(), static_cast<int>(In),
                Input.data(), 1, 0.0f, Output.data(), 1);
    benchmark::DoNotOptimize(Output.data());
  }
  AnnotateCounters(S, In, Out, Density, Nnz);
}

// Hand-rolled GEMV. Compiled with -O3 -march=native -ffast-math; gives a
// "no vendor BLAS" floor that should auto-vectorize cleanly.
static void BM_HandRolled(benchmark::State &S) {
  auto [In, Out, Density] = ParseArgs(S);
  size_t Nnz = 0;
  auto W = BuildDenseMatrix(In, Out, Density, &Nnz);
  auto Input = RandomVector(In);
  std::vector<float> Output(Out);

  for (auto _ : S) {
    const float *WPtr = W.data();
    const float *IPtr = Input.data();
    float *OPtr = Output.data();
    for (size_t I = 0; I < Out; ++I) {
      float Acc = 0.0f;
      const float *Row = WPtr + I * In;
      for (size_t J = 0; J < In; ++J)
        Acc += Row[J] * IPtr[J];
      OPtr[I] = Acc;
    }
    benchmark::DoNotOptimize(Output.data());
  }
  AnnotateCounters(S, In, Out, Density, Nnz);
}

// ---------------------------------------------------------------------------
// Correctness check: tiny case must match across all kernels before timing.
// ---------------------------------------------------------------------------

static void Validate() {
  const size_t In = 8, Out = 4;
  const float Density = 0.5f;

  size_t NnzPlastix = 0;
  BenchNet Net(In, RandomSparseLayer{Out, Density, &NnzPlastix});
  auto Input = RandomVector(In);
  Net.DoForwardPass(std::span<const float>(Input));
  auto PlastixOut = Net.GetOutput();

  size_t NnzDense = 0;
  auto W = BuildDenseMatrix(In, Out, Density, &NnzDense);
  std::vector<float> Ref(Out, 0.0f);
  for (size_t I = 0; I < Out; ++I) {
    float Acc = 0.0f;
    for (size_t J = 0; J < In; ++J)
      Acc += W[I * In + J] * Input[J];
    Ref[I] = Acc;
  }

  if (NnzPlastix != NnzDense) {
    std::fprintf(stderr,
                 "Validation: nnz mismatch (Plastix=%zu Dense=%zu)\n",
                 NnzPlastix, NnzDense);
    std::abort();
  }
  for (size_t I = 0; I < Out; ++I) {
    float Tol = 1e-4f * std::max(1.0f, std::abs(Ref[I]));
    if (std::abs(PlastixOut[I] - Ref[I]) > Tol) {
      std::fprintf(stderr,
                   "Validation: output %zu Plastix=%g Ref=%g (tol=%g)\n", I,
                   PlastixOut[I], Ref[I], Tol);
      std::abort();
    }
  }
  std::fprintf(stderr,
               "Validation OK: In=%zu Out=%zu density=%.2f nnz=%zu\n", In,
               Out, Density, NnzPlastix);
}

// ---------------------------------------------------------------------------
// Registration. Shapes come from the design discussion: square + classifier
// tail + two thin rectangles that stress the input/output-vector bandwidth.
// Densities are basis points (1 bp = 0.01 %).
// ---------------------------------------------------------------------------

static void RegisterAll() {
  const std::vector<std::pair<int64_t, int64_t>> Shapes = {
      {256, 256},   {1024, 1024}, {4096, 4096},
      {4096, 1024}, {8192, 64},   {64, 8192},
  };
  // Coarse pass found crossover in the 1–20 % band for every shape we tested.
  // Refine inside that band; keep the outer points so the curve has shape.
  // Basis points: 1 bp = 0.01 %.
  const std::vector<int64_t> Densities = {10,   100,  200,  500,  1000, 1500,
                                          2000, 2500, 3000, 4000, 5000};

  auto Reg = [](const char *Name, auto Fn, int64_t In, int64_t Out,
                int64_t D) {
    benchmark::RegisterBenchmark(Name, Fn)
        ->Args({In, Out, D})
        ->Unit(benchmark::kMicrosecond)
        ->MinTime(0.5);
  };

  for (auto [In, Out] : Shapes) {
    for (auto D : Densities) {
      Reg("Plastix", BM_Plastix, In, Out, D);
      Reg("Eigen", BM_Eigen, In, Out, D);
      Reg("OpenBLAS", BM_OpenBLAS, In, Out, D);
      Reg("Hand", BM_HandRolled, In, Out, D);
    }
  }
}

int main(int argc, char **argv) {
  Validate();
  RegisterAll();
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv))
    return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
