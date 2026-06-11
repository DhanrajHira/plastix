// Plastix-vs-LibTorch SpMV/GEMV crossover benchmark (CPU).
//
// Companion to bench_spmv_crossover.cpp. That file pits Plastix's sparse
// forward pass against Eigen / OpenBLAS / hand-rolled GEMV; this file
// adds the two PyTorch (libtorch CPU) baselines a downstream user would
// reach for first: dense `torch::matmul` and sparse `torch::sparse_csr`
// matmul. The same edge predicate, weight draws, and input vector feed
// every kernel so outputs are mathematically identical and the crossover
// density (the band at which Plastix overtakes a vendored dense kernel)
// is directly comparable across files.
//
// Args per benchmark: (In, Out, density_bps)
//   density_bps = density in basis points (1 bp = 0.01%).
//                 e.g. 5000 -> 50%, 100 -> 1%, 10 -> 0.1%.

#include <benchmark/benchmark.h>
#include <torch/torch.h>

#include "_bench_common.hpp"
#include "plastix/plastix.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <utility>
#include <vector>

using bench_common::EdgeExists;
using bench_common::EdgeWeight;
using bench_common::RandomVector;
using bench_common::kInputSeed;

// ---------------------------------------------------------------------------
// Plastix configuration — must match bench_spmv_crossover.cpp so the two
// CSVs can be merged later without re-running the Plastix sweep.
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

struct BenchTraits : plastix::DefaultNetworkTraits<> {
  using ForwardPass = IdentityFwd;
  static constexpr size_t UnitCapacity = 32 * 1024;
  static constexpr size_t ConnCapacity = 16 * 1024 * 1024;
};

using BenchNet = plastix::Network<BenchTraits>;

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

// Build CSR for torch::sparse_csr_tensor. Row index space is Out, column
// space is In. Same walk order as BuildDenseMatrix so the per-edge weight
// assignment is identical.
struct CsrMatrix {
  std::vector<int64_t> CrowIndices;
  std::vector<int64_t> ColIndices;
  std::vector<float> Values;
  size_t Nnz = 0;
};

static CsrMatrix BuildCsrMatrix(size_t In, size_t Out, float Density) {
  CsrMatrix M;
  M.CrowIndices.assign(Out + 1, 0);
  // Two-pass: first pass counts per-row nnz, second pass fills indices/values.
  std::vector<size_t> RowNnz(Out, 0);
  for (size_t Src = 0; Src < In; ++Src) {
    for (size_t Dst = 0; Dst < Out; ++Dst) {
      if (!EdgeExists(Src, Dst, Out, Density))
        continue;
      ++RowNnz[Dst];
    }
  }
  size_t Total = 0;
  for (size_t I = 0; I < Out; ++I) {
    M.CrowIndices[I] = static_cast<int64_t>(Total);
    Total += RowNnz[I];
  }
  M.CrowIndices[Out] = static_cast<int64_t>(Total);
  M.Nnz = Total;
  M.ColIndices.assign(Total, 0);
  M.Values.assign(Total, 0.0f);

  std::vector<size_t> RowCursor(Out, 0);
  for (size_t Src = 0; Src < In; ++Src) {
    for (size_t Dst = 0; Dst < Out; ++Dst) {
      if (!EdgeExists(Src, Dst, Out, Density))
        continue;
      size_t Pos =
          static_cast<size_t>(M.CrowIndices[Dst]) + RowCursor[Dst];
      M.ColIndices[Pos] = static_cast<int64_t>(Src);
      M.Values[Pos] = EdgeWeight(Src, Dst, Out);
      ++RowCursor[Dst];
    }
  }
  return M;
}

// ---------------------------------------------------------------------------
// Argument parsing / counter annotation.
// ---------------------------------------------------------------------------

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

static void BM_TorchDense(benchmark::State &S) {
  torch::NoGradGuard NoGrad;
  auto [In, Out, Density] = ParseArgs(S);
  size_t Nnz = 0;
  auto W = BuildDenseMatrix(In, Out, Density, &Nnz);
  auto Input = RandomVector(In);

  auto Opts = torch::TensorOptions().dtype(torch::kFloat32);
  // from_blob borrows the storage; clone() so the tensor owns it.
  auto WT = torch::from_blob(W.data(),
                             {static_cast<long>(Out), static_cast<long>(In)},
                             Opts)
                .clone();
  auto IT = torch::from_blob(Input.data(), {static_cast<long>(In)}, Opts)
                .clone();
  auto OT = torch::empty({static_cast<long>(Out)}, Opts);

  for (auto _ : S) {
    OT = torch::matmul(WT, IT);
    benchmark::DoNotOptimize(OT.data_ptr<float>());
  }
  AnnotateCounters(S, In, Out, Density, Nnz);
}

static void BM_TorchSparseCsr(benchmark::State &S) {
  torch::NoGradGuard NoGrad;
  auto [In, Out, Density] = ParseArgs(S);
  auto Csr = BuildCsrMatrix(In, Out, Density);

  auto LongOpts = torch::TensorOptions().dtype(torch::kInt64);
  auto FloatOpts = torch::TensorOptions().dtype(torch::kFloat32);
  auto Crow = torch::from_blob(Csr.CrowIndices.data(),
                               {static_cast<long>(Csr.CrowIndices.size())},
                               LongOpts)
                  .clone();
  auto Col = torch::from_blob(Csr.ColIndices.data(),
                              {static_cast<long>(Csr.ColIndices.size())},
                              LongOpts)
                 .clone();
  auto Val = torch::from_blob(Csr.Values.data(),
                              {static_cast<long>(Csr.Values.size())},
                              FloatOpts)
                 .clone();
  auto W = torch::sparse_csr_tensor(
      Crow, Col, Val,
      {static_cast<long>(Out), static_cast<long>(In)}, FloatOpts);

  auto Input = RandomVector(In);
  // sparse_csr @ dense matmul needs the rhs as a [In, 1] dense matrix to
  // dispatch to the sparse CSR mm kernel. Squeeze the output back to a
  // vector after.
  auto IT = torch::from_blob(Input.data(),
                             {static_cast<long>(In), 1}, FloatOpts)
                .clone();
  torch::Tensor OT;

  for (auto _ : S) {
    OT = torch::matmul(W, IT);
    benchmark::DoNotOptimize(OT.data_ptr<float>());
  }
  AnnotateCounters(S, In, Out, Density, Csr.Nnz);
}

// ---------------------------------------------------------------------------
// Correctness check. Tiny case (In=8, Out=4, density=0.5) — same as the
// non-torch crossover bench. All three kernels must match the closed-form
// y = W x to 1e-4 before we time anything.
// ---------------------------------------------------------------------------

static void Validate() {
  torch::NoGradGuard NoGrad;
  const size_t In = 8, Out = 4;
  const float Density = 0.5f;

  // Reference: closed-form y[i] = sum_j W[i,j] * x[j].
  size_t NnzDense = 0;
  auto W = BuildDenseMatrix(In, Out, Density, &NnzDense);
  auto Input = RandomVector(In);
  std::vector<float> Ref(Out, 0.0f);
  for (size_t I = 0; I < Out; ++I) {
    float Acc = 0.0f;
    for (size_t J = 0; J < In; ++J)
      Acc += W[I * In + J] * Input[J];
    Ref[I] = Acc;
  }

  auto CheckMatch = [&](const char *Name, const float *Got, size_t N) {
    if (N != Out) {
      std::fprintf(stderr, "Validation: %s wrong size (got=%zu Out=%zu)\n",
                   Name, N, Out);
      std::abort();
    }
    for (size_t I = 0; I < Out; ++I) {
      float Tol = 1e-4f * std::max(1.0f, std::abs(Ref[I]));
      if (std::abs(Got[I] - Ref[I]) > Tol) {
        std::fprintf(stderr,
                     "Validation: %s output %zu got=%g Ref=%g (tol=%g)\n",
                     Name, I, Got[I], Ref[I], Tol);
        std::abort();
      }
    }
  };

  // Plastix
  size_t NnzPlastix = 0;
  BenchNet Net(In, RandomSparseLayer{Out, Density, &NnzPlastix});
  Net.DoForwardPass(std::span<const float>(Input));
  auto PO = Net.GetOutput();
  if (NnzPlastix != NnzDense) {
    std::fprintf(stderr, "Validation: nnz mismatch (Plastix=%zu Dense=%zu)\n",
                 NnzPlastix, NnzDense);
    std::abort();
  }
  CheckMatch("Plastix", PO.data(), PO.size());

  // Torch dense
  auto FOpts = torch::TensorOptions().dtype(torch::kFloat32);
  auto WT = torch::from_blob(W.data(),
                             {static_cast<long>(Out), static_cast<long>(In)},
                             FOpts)
                .clone();
  auto IT = torch::from_blob(Input.data(), {static_cast<long>(In)}, FOpts)
                .clone();
  auto OD = torch::matmul(WT, IT).contiguous();
  CheckMatch("TorchDense", OD.data_ptr<float>(),
             static_cast<size_t>(OD.numel()));

  // Torch sparse CSR
  auto Csr = BuildCsrMatrix(In, Out, Density);
  auto LOpts = torch::TensorOptions().dtype(torch::kInt64);
  auto Crow = torch::from_blob(Csr.CrowIndices.data(),
                               {static_cast<long>(Csr.CrowIndices.size())},
                               LOpts)
                  .clone();
  auto Col = torch::from_blob(Csr.ColIndices.data(),
                              {static_cast<long>(Csr.ColIndices.size())},
                              LOpts)
                 .clone();
  auto Val = torch::from_blob(Csr.Values.data(),
                              {static_cast<long>(Csr.Values.size())},
                              FOpts)
                 .clone();
  auto WS = torch::sparse_csr_tensor(
      Crow, Col, Val, {static_cast<long>(Out), static_cast<long>(In)},
      FOpts);
  auto ITC = torch::from_blob(Input.data(),
                              {static_cast<long>(In), 1}, FOpts)
                 .clone();
  auto OS = torch::matmul(WS, ITC).contiguous();
  CheckMatch("TorchSparseCsr", OS.data_ptr<float>(),
             static_cast<size_t>(OS.numel()));

  std::fprintf(
      stderr,
      "Validation OK: In=%zu Out=%zu density=%.2f nnz=%zu (Plastix/TorchDense/TorchSparse)\n",
      In, Out, Density, NnzPlastix);
}

// ---------------------------------------------------------------------------
// Registration. Same shape/density sweep as bench_spmv_crossover.cpp so
// the two CSVs can be joined on (In, Out, density_bps).
// ---------------------------------------------------------------------------

static void RegisterAll() {
  const std::vector<std::pair<int64_t, int64_t>> Shapes = {
      {256, 256},   {1024, 1024}, {4096, 4096},
      {4096, 1024}, {8192, 64},   {64, 8192},
  };
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
      Reg("TorchDense", BM_TorchDense, In, Out, D);
      Reg("TorchSparseCsr", BM_TorchSparseCsr, In, Out, D);
    }
  }
}

int main(int argc, char **argv) {
  torch::set_num_threads(1); // Match Plastix's single-threaded forward pass.
  Validate();
  RegisterAll();

  // Default --benchmark_out to OutputDir() so a plain `./bench_..._torch`
  // invocation drops a CSV next to all the other microbench artifacts.
  auto Default = bench_common::OutputFile("spmv_crossover_torch.json").string();
  std::vector<std::string> Args(argv, argv + argc);
  bool HasOut = false;
  for (const auto &A : Args)
    if (A.rfind("--benchmark_out=", 0) == 0) { HasOut = true; break; }
  std::vector<char *> NewArgv;
  NewArgv.reserve(Args.size() + 2);
  std::string OutFlag = "--benchmark_out=" + Default;
  std::string FmtFlag = "--benchmark_out_format=json";
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
